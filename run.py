"""This file serves as an entry point to run the matching logic against
the provided fixture data and see how well it performs.

Do not touch this file; instead, implement the matching logic in match.py.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.match import find_attachment, find_transaction

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "src" / "data"

Attachment = Dict[str, Any]
Transaction = Dict[str, Any]

EXPECTED_TX_TO_ATTACHMENT: Dict[int, Optional[int]] = {
    2001: 3001,
    2002: 3002,
    2003: 3003,
    2004: 3004,
    2005: 3005,
    2006: None,
    2007: 3006,
    2008: 3007,
    2009: None,
    2010: None,
    2011: None,
    2012: None,
}

EXPECTED_ATTACHMENT_TO_TX: Dict[int, Optional[int]] = {
    3001: 2001,
    3002: 2002,
    3003: 2003,
    3004: 2004,
    3005: 2005,
    3006: 2007,
    3007: 2008,
    3008: None,
    3009: None,
}


def _load_transactions() -> dict[int, Transaction]:
    with open(DATA_DIR / "transactions.json", "r", encoding="utf-8") as f:
        transactions_list: List[Transaction] = json.load(f)
    return {tx["id"]: tx for tx in transactions_list}


def _load_attachments() -> dict[int, Attachment]:
    with open(DATA_DIR / "attachments.json", "r", encoding="utf-8") as f:
        attachments_list: List[Attachment] = json.load(f)
    return {att["id"]: att for att in attachments_list}


def _print_row(*cols: Sequence[str]) -> None:
    print(
        " ".join(
            col.ljust(30 if idx < len(cols) - 1 else 0) for idx, col in enumerate(cols)
        )
    )


def _describe_attachment(att: Optional[Attachment]) -> str:
    if att is None:
        return "∅"
    return f"Attachment (id={att['id']})"


def _describe_transaction(tx: Optional[Transaction]) -> str:
    if tx is None:
        return "∅"
    return f"Transaction (id={tx['id']})"


def _compare_items(
    item1: Union[Transaction, Attachment],
    item2: Union[Transaction, Attachment],
) -> str:
    if item1 is None and item2 is None:
        return True

    item1_id = item1.get("id") if item1 else None
    item2_id = item2.get("id") if item2 else None
    return item1_id == item2_id


def entry():
    transactions = _load_transactions()
    attachments = _load_attachments()

    print("\nFind attachments:\n")
    _print_row("Transaction", "Expected Att", "Found Att", "Result")
    for tx_id, att_id in EXPECTED_TX_TO_ATTACHMENT.items():
        transaction = transactions[tx_id]
        expected_attachment = attachments.get(att_id, None)
        actual_attachment = find_attachment(transaction, list(attachments.values()))

        _print_row(
            _describe_transaction(transaction),
            _describe_attachment(expected_attachment),
            _describe_attachment(actual_attachment),
            "✅" if _compare_items(expected_attachment, actual_attachment) else "❌",
        )

    print("\nFind transactions:\n")
    _print_row("Attachment", "Expected Tx", "Found Tx", "Result")
    for att_id, tx_id in EXPECTED_ATTACHMENT_TO_TX.items():
        attachment = attachments[att_id]
        expected_transaction = transactions.get(tx_id, None)
        actual_transaction = find_transaction(attachment, list(transactions.values()))

        _print_row(
            _describe_attachment(attachment),
            _describe_transaction(expected_transaction),
            _describe_transaction(actual_transaction),
            "✅" if _compare_items(expected_transaction, actual_transaction) else "❌",
        )

    print()


if __name__ == "__main__":
    entry()
