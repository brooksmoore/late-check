"""Score the demo report against what really happened (data/demo_answers.csv).

    python check_demo.py   # -> out/demo_check.json

Only the open invoices that were NOT yet due on the as-of date count: overdue ones are late by
definition, so they would flatter the numbers. This is one snapshot of a few dozen invoices, so treat
it as an illustration; the holdout in out/results.md is the real test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def main() -> None:
    rep = pd.read_csv(OUT / "report.csv", dtype={"invoice_id": str})
    ans = pd.read_csv(ROOT / "data" / "demo_answers.csv", dtype={"invoice_id": str})
    meta = json.loads((OUT / "report_meta.json").read_text())
    d = rep[rep.status == "not yet due"].merge(ans[["invoice_id", "days_late"]], on="invoice_id")
    d = d.sort_values("p_late", ascending=False)
    late = (d.days_late > 0).astype(int)
    k = int(np.ceil(0.2 * len(d)))
    flagged = d.p_late >= 0.5
    res = {
        "as_of": meta["as_of"],
        "not_yet_due_invoices": int(len(d)),
        "actually_late": int(late.sum()),
        "top20pct_size": k,
        "top20pct_actually_late": int(late.iloc[:k].sum()),
        "flagged_p_late_ge_50": int(flagged.sum()),
        "flagged_actually_late": int(late[flagged].sum()),
        "late_not_flagged": int(late[~flagged].sum()),
        "mean_days_late_flagged_and_late": round(float(d.days_late[flagged & (late == 1)].mean()), 1)
        if (flagged & (late == 1)).any() else None,
        "overdue_on_as_of": int((rep.status == "overdue").sum()),
        "note": "One snapshot; small sample. The holdout in results.md is the real test.",
    }
    (OUT / "demo_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
