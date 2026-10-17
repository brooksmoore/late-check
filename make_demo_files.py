"""Turn the IBM sample into a realistic "Monday morning" demo for the CLI.

    python make_demo_files.py [--as-of 2013-11-01]

Writes, from data/WA_Fn-UseC_-Accounts-Receivable.csv:
    data/demo_history.csv  invoices paid before the as-of date (with paid dates)
    data/demo_open.csv     invoices issued before the as-of date but not paid yet (paid date removed)
    data/demo_answers.csv  what really happened to those open invoices (kept out of the CLI's view)
Columns are renamed to the plain names the CLI documents, so the demo looks like a freelancer's export.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "WA_Fn-UseC_-Accounts-Receivable.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2013-11-01")
    a = ap.parse_args()
    as_of = pd.Timestamp(a.as_of)
    d = pd.read_csv(SRC, dtype=str)
    for c in ("InvoiceDate", "DueDate", "SettledDate"):
        d[c] = pd.to_datetime(d[c], format="%m/%d/%Y")
    d = d.rename(columns={"customerID": "client", "invoiceNumber": "invoice_id", "InvoiceDate": "issued",
                          "DueDate": "due", "InvoiceAmount": "amount", "SettledDate": "paid", "Disputed": "disputed"})
    cols = ["client", "invoice_id", "issued", "due", "amount", "disputed"]
    hist = d[d.paid < as_of][cols + ["paid"]]
    open_ = d[(d.issued < as_of) & (d.paid >= as_of)]
    answers = open_[["invoice_id", "due", "paid"]].assign(
        days_late=lambda x: (x.paid - x.due).dt.days.clip(lower=0))
    fmt = lambda f: f.assign(**{c: f[c].dt.strftime("%Y-%m-%d") for c in f.columns  # noqa: E731
                                if pd.api.types.is_datetime64_any_dtype(f[c])})
    fmt(hist).to_csv(ROOT / "data" / "demo_history.csv", index=False)
    fmt(open_[cols]).to_csv(ROOT / "data" / "demo_open.csv", index=False)
    fmt(answers).to_csv(ROOT / "data" / "demo_answers.csv", index=False)
    print(f"as of {as_of.date()}: {len(hist)} paid invoices in history, {len(open_)} open invoices "
          f"(answers kept in data/demo_answers.csv)")


if __name__ == "__main__":
    main()
