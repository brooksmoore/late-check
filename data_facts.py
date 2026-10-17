"""Plain facts about the sample dataset, for the README (deterministic, no model).

    python data_facts.py   # -> out/data_facts.json
"""
from __future__ import annotations

import json
from pathlib import Path

import latecheck as LC

ROOT = Path(__file__).resolve().parent


def main() -> None:
    inv = LC.add_labels(LC.load_invoices(ROOT / "data" / "WA_Fn-UseC_-Accounts-Receivable.csv", need_paid=True))
    terms = (inv.due - inv.issued).dt.days
    per_client = inv.groupby("client").size()
    facts = {
        "invoices": int(len(inv)), "clients": int(inv.client.nunique()),
        "invoices_per_client_min_median_max": [int(per_client.min()), int(per_client.median()), int(per_client.max())],
        "terms_days_values": sorted(int(t) for t in terms.unique()),
        "amount_min": round(float(inv.amount.min()), 2), "amount_max": round(float(inv.amount.max()), 2),
        "amount_median": round(float(inv.amount.median()), 2),
        "late_share": round(float(inv.late.mean()), 3),
        "days_late_median_if_late": float(inv.days_late[inv.late == 1].median()),
        "days_late_max": int(inv.days_late.max()),
        "disputed_share": round(float(inv.disputed.mean()), 3),
    }
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out" / "data_facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
