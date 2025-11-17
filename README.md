# NOCFO Homework Assignment - AI Engineer

> [!NOTE]
> We recommend spending **no more than 6 hours** on this task. Focus on the essentials – a functional and clear implementation is more important than perfection.

## Objective

Your task is to write logic for matching bank transactions with potential attachments (receipts, invoices). In accounting, every transaction on a bank account must have an attachment as supporting evidence, so this is a real-world problem. The logic you implement must work in both directions. You will write two functions—`find_attachment` and `find_transaction`—and your goal is to fill in their implementations in `src/match.py`. Treat this repository as your starter template: build directly on top of it so that `run.py` continues to work without modifications.

---

## Starting point

You will receive a ready-made list of bank transactions and a list of attachments that have already been parsed and structured into JSON format. These JSON files can be found in the `/src/data` directory at the project root.

Additionally, a file named `run.py` has been provided. This file calls the functions you are required to implement. Running this file will produce a report of the successfully matched pairs. You can run it using the following command:

```py
python run.py
```

---

## What you need to implement

- The matching logic lives in `src/match.py`. Implement the `find_attachment` and `find_transaction` functions there; do not modify `run.py`.
- `find_attachment(transaction, attachments)` must return the single best candidate attachment for the provided transaction or `None` if no confident match exists.
- `find_transaction(attachment, transactions)` must do the same in the opposite direction.
- Use only the fixture data under `/src/data` and the helper report that `run.py` prints to guide your implementation.

---

## What makes a good match?

- A **reference number** match is always a 1:1 match. If a reference number match is found, the link should always be created.
  - Note that there may be variations in the format of reference numbers. Leading zeros or whitespace within the reference number should be ignored during comparison.
- **Amount**, **date**, and **counterparty** information are equally strong cues — but none of them alone are sufficient. Find a suitable combination of these signals to produce a confident match.
  - Note that the spelling of the counterparty's name may vary in the bank statement. Also, the transaction date of an invoice payment rarely matches the due date exactly — it can vary. Sometimes invoices are paid late, or bank processing may take a few days. In other cases, people pay the invoice immediately upon receiving it instead of waiting until the due date.

Keep in mind that the list of attachments includes not only receipts but also both purchase and sales invoices. Therefore, the counterparty may sometimes appear on the `recipient` field and other times on the `issuer` field.
In receipt data, the merchant information can be found in the `supplier` field.

The company whose bank transactions and attachments you are matching is **Example Company Oy**.
If this entity is mentioned in an attachment, it always refers to the company itself.

---

## Technical Requirements

- The functionality is implemented using **Python**.
- The `run.py` file must remain executable and must return an updated test report when run.
- Your implementation should be deterministic: rerunning `python run.py` with the same data should yield the same matches every time.

---

## Submission

Submit your completed app by providing a link to a **public GitHub repository**. The repository must include:

1. **Source code**: All files necessary to run the app.
2. **README.md file**, containing:

   - Instructions to run the app.
   - A brief description of the architecture and technical decisions.

Email the link to the repository to **people@nocfo.io**. The email subject must include "Homework assignment". Good luck with the assignment! :)

---

## Evaluation Criteria

1. Matching Accuracy: The implemented heuristics produce reasonable and explainable matches with minimal false positives.
2. Code Clarity: The logic is easy to read, well-structured, and includes clear comments or docstrings explaining the reasoning.
3. Edge Case Handling: The implementation behaves predictably with missing data, ambiguous cases, and noisy inputs.
4. Reusability & Design: Functions are modular and deterministic.
5. Documentation & Tests: The README and test cases clearly describe the approach, assumptions, and demonstrate correctness.

---

> [!IMPORTANT]
> If you have technical challenges with completing the task, you can contact Juho via email at **juho.enala@nocfo.io**.
