"""Late Check: which of my open invoices will be paid late? Runs offline on your own CSVs.

Usage:
    python latecheck.py history.csv open.csv              # report -> out/report.md + out/report.html
    python latecheck.py history.csv open.csv --as-of 2013-11-01 --out out/

history.csv: invoices that are already paid (needs a paid date).
open.csv:    invoices that are not paid yet.

Columns (case-insensitive; IBM sample names also work):
    client      (customerID, customer)        who owes you
    invoice_id  (invoiceNumber, invoice)      optional, any id
    issued      (InvoiceDate, invoice_date)   date the invoice went out
    due         (DueDate, due_date)           due date
    amount      (InvoiceAmount)               invoice total
    paid        (SettledDate, paid_date)      history.csv only
    disputed    (Disputed)                    optional, yes/no

Nothing is uploaded anywhere. The model is TabPFN-v2, loaded from ./models on this machine.
Built with PriorLabs-TabPFN.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(ROOT / "models"))
os.environ["HF_HUB_OFFLINE"] = "1"          # never reach the network from here on
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
warnings.filterwarnings("ignore", message=".*Running on CPU.*")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEED = 0
VERY_LATE_DAYS = 8          # "more than a week late"
CLASSES = ["on time", f"1-{VERY_LATE_DAYS - 1} days late", f"{VERY_LATE_DAYS}+ days late"]

ALIASES = {
    "client": ["client", "customerid", "customer", "customer_id", "client_id"],
    "invoice_id": ["invoice_id", "invoicenumber", "invoice", "invoice_number", "id"],
    "issued": ["issued", "invoicedate", "invoice_date", "date", "issue_date"],
    "due": ["due", "duedate", "due_date"],
    "amount": ["amount", "invoiceamount", "invoice_amount", "total"],
    "paid": ["paid", "settleddate", "paid_date", "settled_date", "settled"],
    "disputed": ["disputed", "dispute", "is_disputed"],
}

FEATURES = [
    "amount", "amount_vs_usual", "terms_days", "disputed", "issue_month",
    "prior_n", "prior_late_rate", "prior_mean_days_late", "last_days_late",
    "open_at_issue", "overdue_at_issue", "days_since_prev",
]
FEATURES_NO_DISPUTE = [f for f in FEATURES if f != "disputed"]


# --------------------------------------------------------------------------- input
def load_invoices(path: str | Path, need_paid: bool) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    lower = {c.lower().strip(): c for c in raw.columns}
    out = pd.DataFrame(index=raw.index)
    for canon, names in ALIASES.items():
        src = next((lower[n] for n in names if n in lower), None)
        if src is not None:
            out[canon] = raw[src]
    missing = [c for c in ("client", "issued", "due", "amount") + (("paid",) if need_paid else ())
               if c not in out]
    if missing:
        raise SystemExit(f"{path}: missing column(s) {missing}. See `python latecheck.py --help`.")
    if "invoice_id" not in out:
        out["invoice_id"] = [f"row{i + 1}" for i in range(len(out))]
    if "disputed" not in out:
        out["disputed"] = "no"
    for c in ("issued", "due") + (("paid",) if "paid" in out else ()):
        out[c] = pd.to_datetime(out[c], format="mixed")
    out["amount"] = pd.to_numeric(out["amount"].str.replace(r"[$,]", "", regex=True))
    out["disputed"] = out["disputed"].astype(str).str.strip().str.lower().isin(
        ["yes", "y", "true", "1"]).astype(int)
    out["client"] = out["client"].astype(str)
    out["invoice_id"] = out["invoice_id"].astype(str)
    if not need_paid:
        out["paid"] = pd.NaT
    return out[["client", "invoice_id", "issued", "due", "amount", "paid", "disputed"]]


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["days_late"] = (df.paid - df.due).dt.days.clip(lower=0)
    df["late"] = (df.days_late > 0).astype(int)
    df["late_class"] = np.select([df.days_late <= 0, df.days_late < VERY_LATE_DAYS], [0, 1], 2)
    return df


# --------------------------------------------------------------------------- features
def build_features(rows: pd.DataFrame, book: pd.DataFrame) -> pd.DataFrame:
    """Features for each row, using ONLY what was known on that row's issue date.

    `book` is every invoice we know about (paid and unpaid). For a row issued on day t, the client's
    history is the client's invoices paid before t; "open at issue" is the client's invoices issued
    before t that were not yet paid on t.
    """
    feats = []
    by_client = {c: g.sort_values("issued") for c, g in book.groupby("client")}
    for r in rows.itertuples():
        g = by_client.get(r.client, book.iloc[:0])
        t = r.issued
        prev = g[(g.issued < t)]
        done = prev[prev.paid.notna() & (prev.paid < t)]
        open_ = prev[prev.paid.isna() | (prev.paid >= t)]
        dl = (done.paid - done.due).dt.days.clip(lower=0)
        last = done.sort_values("paid").iloc[-1] if len(done) else None
        usual = prev.amount.median() if len(prev) else np.nan
        feats.append({
            "amount": r.amount,
            "amount_vs_usual": r.amount / usual if usual and usual > 0 else np.nan,
            "terms_days": (r.due - r.issued).days,
            "disputed": r.disputed,
            "issue_month": t.month,
            "prior_n": len(done),
            "prior_late_count": int((dl > 0).sum()),
            "prior_late_rate": float((dl > 0).mean()) if len(done) else np.nan,
            "prior_mean_days_late": float(dl.mean()) if len(done) else np.nan,
            "last_days_late": float(max((last.paid - last.due).days, 0)) if last is not None else np.nan,
            "open_at_issue": len(open_),
            "overdue_at_issue": int((open_.due < t).sum()),
            "days_since_prev": float((t - prev.issued.max()).days) if len(prev) else np.nan,
        })
    return pd.DataFrame(feats, index=rows.index)


def client_rate_score(f: pd.DataFrame, base_rate: float, k: float = 2.0) -> np.ndarray:
    """Baseline: the client's own past late rate, shrunk toward the overall rate (k pseudo-invoices)."""
    late = f.prior_late_count.to_numpy(float)
    n = f.prior_n.to_numpy(float)
    return (late + k * base_rate) / (n + k)


# --------------------------------------------------------------------------- model
def tabpfn(n_context: int):
    from tabpfn import TabPFNClassifier
    ckpts = sorted((ROOT / "models").glob("tabpfn-v2-classifier*.ckpt"))
    if not ckpts:
        raise SystemExit("TabPFN-v2 weights not found in ./models. Run `python fetch_data.py model` once.")
    return TabPFNClassifier(model_path=str(ckpts[0]), device="cpu", random_state=SEED,
                            ignore_pretraining_limits=n_context > 1000)


def fit_tabpfn(X: pd.DataFrame, y: np.ndarray):
    clf = tabpfn(len(X))
    clf.fit(X.to_numpy(float), np.asarray(y))
    return clf


def late_probs(clf, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """P(any lateness), P(VERY_LATE_DAYS+ late) from the 3-class model."""
    p = clf.predict_proba(X.to_numpy(float))
    cols = list(clf.classes_)
    get = lambda k: p[:, cols.index(k)] if k in cols else np.zeros(len(p))  # noqa: E731
    return get(1) + get(2), get(2)


# --------------------------------------------------------------------------- reasons
def explain(clf, X: pd.DataFrame, ref: pd.DataFrame, features: list[str]) -> list[list[tuple[str, float]]]:
    """Per row: how much P(late) drops when one feature is reset to a typical value from the
    training rows (median). Simple one-at-a-time sensitivity, not SHAP. One batched predict call."""
    typical = ref[features].median(numeric_only=True)
    base, _ = late_probs(clf, X[features])
    blocks = []
    for f in features:
        Xf = X[features].copy()
        Xf[f] = typical[f]
        blocks.append(Xf)
    pl, _ = late_probs(clf, pd.concat(blocks, ignore_index=True))
    pl = pl.reshape(len(features), len(X))
    out = []
    for i in range(len(X)):
        deltas = [(f, float(base[i] - pl[j, i])) for j, f in enumerate(features)]
        out.append(sorted(deltas, key=lambda d: -d[1]))
    return out


def phrase(f: str, row: pd.Series) -> str:
    n = int(row.prior_n)
    if f in ("prior_late_rate", "prior_n") and n == 0:
        return "no paid invoices from this client yet"
    if f == "prior_late_rate":
        return f"paid late on {int(row.prior_late_count)} of {n} past invoices"
    if f == "prior_n":
        return f"only {n} past invoice{'s' if n != 1 else ''} to judge from" if n < 5 else f"{n} past invoices on record"
    if f == "prior_mean_days_late":
        return "no payment history" if np.isnan(row.prior_mean_days_late) else \
            f"averages {row.prior_mean_days_late:.1f} days late"
    if f == "last_days_late":
        if np.isnan(row.last_days_late):
            return "no previous payment"
        return "last invoice paid on time" if row.last_days_late == 0 else \
            f"last invoice paid {row.last_days_late:.0f} days late"
    if f == "disputed":
        return "invoice is disputed" if row.disputed else "not disputed"
    if f == "open_at_issue":
        return f"{int(row.open_at_issue)} other unpaid invoice(s) when this one went out"
    if f == "overdue_at_issue":
        return f"{int(row.overdue_at_issue)} overdue invoice(s) when this one went out"
    if f == "amount":
        return f"amount ${row.amount:,.2f}"
    if f == "amount_vs_usual":
        return "first invoice for this client" if np.isnan(row.amount_vs_usual) else \
            f"{row.amount_vs_usual:.1f}x this client's usual amount"
    if f == "issue_month":
        return f"issued in {pd.Timestamp(2000, int(row.issue_month), 1):%B}"
    if f == "days_since_prev":
        return "first invoice for this client" if np.isnan(row.days_since_prev) else \
            f"{row.days_since_prev:.0f} days since their previous invoice"
    if f == "terms_days":
        return f"{int(row.terms_days)}-day terms"
    return f


# --------------------------------------------------------------------------- the small "agent" layer
def next_action(p_late: float, p_very: float, days_to_due: int, disputed: int, overdue_days: int) -> str:
    """Plain rules, written down so you can disagree with them."""
    if disputed:
        return "Resolve the dispute first; a payment reminder won't move a disputed invoice."
    if overdue_days > 0:
        return f"Overdue {overdue_days} days: call or send a firm reminder today."
    if p_late >= 0.5 and days_to_due <= 7:
        return "Send a friendly reminder now, before the due date."
    if p_late >= 0.5:
        return f"Put a reminder on the calendar for {max(days_to_due - 5, 0)} days from now (5 days before due)."
    if p_late >= 0.3:
        return "Some risk: send a short courtesy note 3 days before the due date."
    return "No action needed; normal reminder on the due date."


def draft_email(r: pd.Series) -> str:
    return (f"Subject: Invoice {r.invoice_id} (${r.amount:,.2f}) due {r.due:%b %d}\n\n"
            f"Hi {r.client},\n\nA quick heads-up that invoice {r.invoice_id} for ${r.amount:,.2f} is due on "
            f"{r.due:%A, %B %d}. Let me know if you need anything from me to process it.\n\nThanks!")


# --------------------------------------------------------------------------- the report
def score_open(history: pd.DataFrame, open_: pd.DataFrame, as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    history = add_labels(history[history.paid < as_of])
    book = pd.concat([history, open_.assign(paid=pd.NaT)], ignore_index=True)
    Xh = build_features(history, book)
    Xo = build_features(open_, book)
    clf = fit_tabpfn(Xh[FEATURES], history.late_class.to_numpy())
    p_late, p_very = late_probs(clf, Xo[FEATURES])
    reasons = explain(clf, Xo, Xh, FEATURES)
    res = open_.copy()
    res = res.join(Xo[["prior_n", "prior_late_count"]])
    res["p_late"] = p_late
    res["p_very_late"] = p_very
    res["days_to_due"] = (res.due - as_of).dt.days
    res["overdue_days"] = (-res.days_to_due).clip(lower=0)
    res["why"] = ["; ".join(phrase(f, Xo.loc[idx]) for f, d in rs[:2] if d > 0.01) or
                  "nothing stands out; close to this book's typical invoice"
                  for idx, rs in zip(Xo.index, reasons)]
    res["action"] = [next_action(a, b, d, s, o) for a, b, d, s, o in
                     zip(res.p_late, res.p_very_late, res.days_to_due, res.disputed, res.overdue_days)]
    res["status"] = np.where(res.overdue_days > 0, "overdue", "not yet due")
    # Overdue invoices are already late, so they go first (by days overdue); then the model's ranking.
    res = res.sort_values(["overdue_days", "p_late"], ascending=[False, False])
    meta = {"as_of": str(as_of.date()), "history_invoices": int(len(history)),
            "history_late_share": round(float(history.late.mean()), 3), "open_invoices": int(len(res)),
            "overdue": int((res.overdue_days > 0).sum()), "model": "TabPFN-v2 classifier (3 classes)"}
    return res, meta


def write_report(res: pd.DataFrame, meta: dict, out: Path, top_emails: int = 3) -> None:
    out.mkdir(parents=True, exist_ok=True)
    nyd = res[res.status == "not yet due"]
    od = res[res.status == "overdue"]
    L = [f"# Late Check: {meta['open_invoices']} open invoices as of {meta['as_of']}", "",
         f"Learned from {meta['history_invoices']} paid invoices ({meta['history_late_share']:.0%} were paid late). "
         f"Model: {meta['model']}, run locally. Built with PriorLabs-TabPFN.", ""]
    if len(od):
        L += [f"## Already overdue ({len(od)}): chase these first", "",
              "| Client | Invoice | Amount | Due | Days overdue | Next step |", "|---|---|---|---|---|---|"]
        L += [f"| {r.client} | {r.invoice_id} | ${r.amount:,.2f} | {r.due:%Y-%m-%d} | {r.overdue_days} | {r.action} |"
              for r in od.itertuples()]
        L += [""]
    L += [f"## Not yet due ({len(nyd)}), riskiest first", "",
          f"| # | Client | Invoice | Amount | Due in | P(late) | P({VERY_LATE_DAYS}+ days late) | Why | Next step |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {i} | {r.client} | {r.invoice_id} | ${r.amount:,.2f} | {r.days_to_due} d | {r.p_late:.0%} | "
          f"{r.p_very_late:.0%} | {r.why} | {r.action} |" for i, r in enumerate(nyd.itertuples(), 1)]
    drafts = nyd[(nyd.p_late >= 0.5) & (nyd.disputed == 0)].head(top_emails)
    if len(drafts):
        L += ["", "## Draft reminders (templates; edit before sending, nothing is sent for you)", ""]
        for _, r in drafts.iterrows():
            L += ["```", draft_email(r), "```", ""]
    L += ["", "*Why* = the two inputs that raise this invoice's late score most (each reset to a typical value, "
          "one at a time). It is a rough guide, not a proof.", ""]
    md = "\n".join(L)
    (out / "report.md").write_text(md)
    res.to_csv(out / "report.csv", index=False)

    def tr(cells, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{html.escape(str(c))}</{tag}>" for c in cells) + "</tr>"
    rows = [tr(["#", "Client", "Invoice", "Amount", "Status", "P(late)", f"P({VERY_LATE_DAYS}+ d)", "Why",
                "Next step"], "th")]
    for i, r in enumerate(res.itertuples(), 1):
        st = f"overdue {r.overdue_days} d" if r.overdue_days else f"due in {r.days_to_due} d"
        pl = "already late" if r.overdue_days else f"{r.p_late:.0%}"
        pv = "" if r.overdue_days else f"{r.p_very_late:.0%}"
        why = "" if r.overdue_days else r.why
        rows.append(tr([i, r.client, r.invoice_id, f"${r.amount:,.2f}", st, pl, pv, why, r.action]))
    page = f"""<!doctype html><meta charset="utf-8"><title>Late Check {meta['as_of']}</title>
<style>body{{font:14px system-ui,sans-serif;margin:2em}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}th{{background:#f3f3f3}}</style>
<h1>Late Check: {meta['open_invoices']} open invoices as of {meta['as_of']}</h1>
<p>Learned from {meta['history_invoices']} paid invoices on this computer. {html.escape(meta['model'])}.
Built with PriorLabs-TabPFN.</p><table>{''.join(rows)}</table>"""
    (out / "report.html").write_text(page)
    (out / "report_meta.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("history")
    ap.add_argument("open")
    ap.add_argument("--as-of", help="YYYY-MM-DD; default = latest date in the files")
    ap.add_argument("--out", default=str(ROOT / "out"))
    a = ap.parse_args()
    hist = load_invoices(a.history, need_paid=True)
    opn = load_invoices(a.open, need_paid=False)
    as_of = pd.Timestamp(a.as_of) if a.as_of else max(hist.paid.max(), opn.issued.max()) + pd.Timedelta(days=1)
    res, meta = score_open(hist, opn, as_of)
    write_report(res, meta, Path(a.out))
    print((Path(a.out) / "report.md").read_text())
    print(f"\nWrote {a.out}/report.md, report.html, report.csv")


if __name__ == "__main__":
    main()
