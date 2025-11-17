from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Attachment = Dict[str, Any]
Transaction = Dict[str, Any]

# =============================================================================
# Normalization Utilities
# =============================================================================


def normalize_ref(ref: Optional[str]) -> Optional[str]:
    """Normalize reference strings: strip, lowercase, alphanumeric only."""
    if not ref:
        return None
    cleaned = ref.replace(" ", "").lower()
    cleaned = "".join(ch for ch in cleaned if ch.isalnum()).lstrip("0")
    return cleaned or None


def normalize_name(name: Optional[str]) -> Optional[str]:
    """Normalize names for fuzzy comparison."""
    if not name:
        return None
    name = name.lower()
    name = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in name)
    return " ".join(name.split())


def parse_date(value: Optional[str]) -> Optional[datetime]:
    """Parse date in YYYY-MM-DD format."""
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d")


# =============================================================================
# Attachment Field Extractors
# =============================================================================


def att_amount(att: Attachment) -> Optional[float]:
    return att.get("data", {}).get("total_amount")


def att_reference(att: Attachment) -> Optional[str]:
    return att.get("data", {}).get("reference")


def att_counterparty(att: Attachment) -> Optional[str]:
    """Extract chosen counterparty field with deterministic priority."""
    data = att.get("data", {})
    return data.get("issuer") or data.get("recipient") or data.get("supplier")


def att_due_date(att: Attachment) -> Optional[datetime]:
    """Due date or receiving date fallback."""
    data = att.get("data", {})
    date_str = data.get("due_date") or data.get("receiving_date")
    return parse_date(date_str)


# =============================================================================
# Feature Extraction
# -----------------------------------------------------------------------------
# We collapse both transactions and attachments into the same Feature shape so
# the scoring logic can stay direction-agnostic and easy to reason about.
# =============================================================================


@dataclass(frozen=True)
class Features:
    amount: Optional[float]
    name: Optional[str]
    date: Optional[datetime]


def round_amount(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(abs(value), 2)


def tx_features(tx: Transaction) -> Features:
    return Features(
        amount=round_amount(tx.get("amount")),
        name=normalize_name(tx.get("contact")),
        date=parse_date(tx.get("date")),
    )


def att_features(att: Attachment) -> Features:
    return Features(
        amount=round_amount(att_amount(att)),
        name=normalize_name(att_counterparty(att)),
        date=att_due_date(att),
    )


# =============================================================================
# Similarity Metrics
# -----------------------------------------------------------------------------
# Lightweight token-set similarity is enough here; no need for heavy string
# metrics given the tiny dataset and deterministic scoring needs.
# =============================================================================


def token_similarity(a: Optional[str], b: Optional[str]) -> float:
    """
    Simple token-set similarity (intersection/union).
    Lightweight and deterministic.
    """
    if not a or not b:
        return 0.0
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# =============================================================================
# Scoring Constants
# -----------------------------------------------------------------------------
# Tuned so that any confident match requires at least two strong signals.
# MIN_NAME_SIMILARITY also doubles as a guardrail against vendor mixups.
# =============================================================================

AMOUNT_SCORE = 50
NAME_SCORE_1 = 30
NAME_SCORE_2 = 20
NAME_SCORE_3 = 10
DATE_SCORE_1 = 20
DATE_SCORE_2 = 10
DATE_SCORE_3 = 5
UNIQUENESS_BONUS = 10
COMBO_BONUS = 10
THRESHOLD = 60
MIN_NAME_SIMILARITY = 0.55


# =============================================================================
# Scoring Engine (functions)
# -----------------------------------------------------------------------------
# Each component returns a partial score; the aggregate must clear the global
# threshold before we accept the candidate.
# =============================================================================


def score_amount(source: Features, candidate: Features) -> float:
    if source.amount is None or candidate.amount is None:
        return 0.0
    return AMOUNT_SCORE if source.amount == candidate.amount else 0.0


def score_name(sim: float) -> float:
    if sim > 0.90:
        return NAME_SCORE_1
    if sim > 0.75:
        return NAME_SCORE_2
    if sim > 0.60:
        return NAME_SCORE_3
    return 0.0


def score_date(source: Features, candidate: Features) -> float:
    if not source.date or not candidate.date:
        return 0.0

    diff = abs((source.date - candidate.date).days)
    if diff <= 3:
        return DATE_SCORE_1
    if diff <= 7:
        return DATE_SCORE_2
    if diff <= 14:
        return DATE_SCORE_3
    return 0.0


def score_uniqueness(
    source: Features,
    candidate: Features,
    context: Sequence[Features],
    sim: float,
) -> float:
    score = 0.0

    if candidate.amount is not None:
        # Only reward amount if it uniquely appears once in the candidate set.
        same_amt_count = sum(
            1 for feat in context if feat.amount == candidate.amount
        )
        if same_amt_count == 1:
            score += UNIQUENESS_BONUS

    if candidate.name:
        # Likewise give points when the counterparty name is unique.
        same_name_count = sum(
            1 for feat in context if feat.name == candidate.name
        )
        if same_name_count == 1:
            score += UNIQUENESS_BONUS

    if (
        source.amount is not None
        and candidate.amount == source.amount
        and sim > 0.6
    ):
        candidates = [
            feat for feat in context if feat.amount == source.amount
        ]
        matching = [
            feat
            for feat in candidates
            if token_similarity(feat.name, source.name) > 0.6
        ]
        if len(matching) == 1:
            score += COMBO_BONUS

    return score


def compute_score(
    source: Features,
    candidate: Features,
    candidate_context: Sequence[Features],
) -> float:
    """Composite weighted heuristic score."""
    sim = token_similarity(source.name, candidate.name)

    amount_peers: list[Features] = []
    # Group attachments/transactions that share the same (absolute) amount
    # so we can understand whether this quantity is noisy or discriminative.
    if source.amount is not None:
        amount_peers = [
            feat for feat in candidate_context if feat.amount == source.amount
        ]

    duplicate_amount_cluster = len(amount_peers) > 1
    max_peer_similarity = 0.0
    if source.name and amount_peers:
        peer_sims = [
            token_similarity(source.name, peer.name) for peer in amount_peers
        ]
        max_peer_similarity = max(peer_sims, default=0.0)

    # Guard 1: if both sides have names but the similarity is too low, fail fast.
    if source.name and candidate.name and sim < MIN_NAME_SIMILARITY:
        return 0.0

    # Guard 2: when multiple candidates share the same amount, require a decent
    # name match within that cohort; otherwise the amount signal is too weak.
    if duplicate_amount_cluster and (
        not source.name or max_peer_similarity < max(MIN_NAME_SIMILARITY, 0.6)
    ):
        return 0.0

    return (
        score_amount(source, candidate)
        + score_name(sim)
        + score_date(source, candidate)
        + score_uniqueness(source, candidate, candidate_context, sim)
    )


# =============================================================================
# Core Matching Algorithm
# -----------------------------------------------------------------------------
# The public functions first attempt exact reference matches and then fall back
# to the heuristic scorer with deterministic tie-breaking.
# =============================================================================


def hard_reference_match(
    source_ref: Optional[str],
    candidates: Iterable[Any],
    extract_ref: Callable[[Any], Optional[str]],
) -> Optional[Any]:
    """Return candidate with identical normalized reference."""
    if not source_ref:
        return None

    src_norm = normalize_ref(source_ref)
    if not src_norm:
        return None

    for c in candidates:
        if normalize_ref(extract_ref(c)) == src_norm:
            return c

    return None


def pick_best_candidate(
    scored: List[Tuple[float, Any]]
) -> Optional[Any]:
    """Select highest-scoring unique candidate above threshold."""
    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_item = scored[0]
    if len(scored) > 1:
        second_score = scored[1][0]
        # If the margin is tiny, treat it as ambiguous instead of guessing.
        if abs(best_score - second_score) < 5:
            return None

    if best_score < THRESHOLD:
        return None

    return best_item


# =============================================================================
# Public API for Assignment
# =============================================================================


def find_attachment(
    transaction: Transaction,
    attachments: List[Attachment],
) -> Optional[Attachment]:
    """Return the best matching attachment for a given transaction."""
    tx_feat = tx_features(transaction)

    # Hard match
    direct = hard_reference_match(
        source_ref=transaction.get("reference"),
        candidates=attachments,
        extract_ref=att_reference,
    )
    if direct:
        return direct

    att_feature_pairs = [(att, att_features(att)) for att in attachments]
    att_context = [feat for _, feat in att_feature_pairs]

    scored = [
        (compute_score(tx_feat, feat, att_context), att)
        for att, feat in att_feature_pairs
    ]
    return pick_best_candidate(scored)


def find_transaction(
    attachment: Attachment,
    transactions: List[Transaction],
) -> Optional[Transaction]:
    """Return the best matching transaction for a given attachment."""
    att_feat = att_features(attachment)

    # Hard match
    direct = hard_reference_match(
        source_ref=att_reference(attachment),
        candidates=transactions,
        extract_ref=lambda tx: tx.get("reference"),
    )
    if direct:
        return direct

    tx_feature_pairs = [(tx, tx_features(tx)) for tx in transactions]
    tx_context = [feat for _, feat in tx_feature_pairs]

    scored = [
        (compute_score(att_feat, feat, tx_context), tx)
        for tx, feat in tx_feature_pairs
    ]
    return pick_best_candidate(scored)
