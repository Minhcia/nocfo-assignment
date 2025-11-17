# Matching Engine – NOCFO Homework Assignment

## 1. Instructions to Run the App

The application uses the matching logic in `src/match.py`.  
To execute the provided report script:

```
python run.py
```

This loads the fixture data, applies the matching logic in both directions, and prints a summary of the detected matches.


## 2. Architecture and Technical Decisions

### Two-Stage Matching Pipeline

#### 1. Reference-Based Matching  
Reference numbers are normalized by removing whitespace, leading zeros, and non-alphanumeric characters.  
If both sides contain a reference and their normalized values match, the pair is immediately linked.  
This follows standard accounting practice, where references act as unique identifiers.

#### 2. Feature-Based Heuristic Scoring  
If no reference match exists, both transactions and attachments are transformed into a unified feature structure:
- amount (absolute, rounded)
- normalized counterparty name
- relevant date (transaction date or due/receiving date)

A weighted scoring function evaluates:
- exact amount equality
- fuzzy name similarity using token-based comparison
- date proximity
- uniqueness within the candidate set

A global threshold ensures that only confident matches are accepted, and tie-breaking rules prevent ambiguous selections.  
The system is deterministic and symmetric across both functions: `find_attachment` and `find_transaction`.

### Design Principles
- deterministic logic  
- fail-safe matching (avoid false positives)  
- modular separation of normalization, feature extraction, and scoring  
- readable, maintainable code  


## 3. Future Development Notes

Although the assignment only requires a single-file implementation, a production-ready version would introduce several improvements:

### Stronger Matching Signals  
- supplier alias tables or learned name mappings  
- support for IBAN, VAT ID, or registry identifiers  
- historical patterns and frequency-based matching  

### Explainability and Monitoring  
- structured scoring breakdown for auditors  
- logging and anomaly detection  
- interactive review UI for low-confidence pairs  

### Scalable Code Structure  
A more complete folder layout would separate concerns cleanly:

src/
│
├── matching/                     ← Domain logic layer
│   │
│   ├── normalizers.py            ← String, reference, and name normalization
│   ├── extractors.py             ← Raw JSON → domain field extraction
│   ├── features.py               ← Unified Feature dataclass for scoring
│   ├── similarity.py             ← Token-based similarity functions
│   ├── scoring.py                ← Amount, name, date, uniqueness scoring
│   └── engine.py                 ← Public API: find_attachment, find_transaction
│
├── data/                         ← Fixture or input data files
│   ├── attachments.json
│   └── transactions.json
│
├── api/                          ← Optional application interface layer
│   └── match_controller.py       ← REST/RPC endpoints (if expanded)
│
└── tests/                        ← Unit and integration tests
    ├── test_normalizers.py
    ├── test_scoring.py
    ├── test_engine.py
    └── test_similarity.py




This organization supports growth, testing, and maintainability without changing the core logic required in the assignment.

