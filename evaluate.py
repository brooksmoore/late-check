"""Honest time-split check of Late Check on the IBM Accounts Receivable sample.

    python evaluate.py      # -> out/results.json, out/results.md, out/capture_curve.png

Train: invoices issued AND paid before the cutoff (only outcomes you would have known then).
Test:  every invoice issued on or after the cutoff. Features use only what was known on each
       invoice's issue date. Nothing about the test period is used to fit or tune anything.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import latecheck as LC  # sets offline env vars before torch/tabpfn load
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
DATA_FILE = ROOT / "data" / "WA_Fn-UseC_-Accounts-Receivable.csv"
CUTOFF = pd.Timestamp("2013-04-01")
TOP_SHARE = 0.20
N_BOOT = 2000


def capture(y, s, share=TOP_SHARE, w=None, seed=0):
    """Share of late invoices (or late dollars, with w) inside the top `share` of scores.
    Ties are broken at random (seeded) so a coarse score gets no free credit."""
    y = np.asarray(y)
    w = np.ones(len(y)) if w is None else np.asarray(w)
    rnd = np.random.default_rng(seed).random(len(y))
    order = np.lexsort((rnd, -np.asarray(s)))
    k = int(np.ceil(share * len(y)))
    top = order[:k]
    return float((y[top] * w[top]).sum() / (y * w).sum())


def metrics(y, p, amt):
    return {"auc": round(float(roc_auc_score(y, p)), 3),
            "brier": round(float(brier_score_loss(y, p)), 3),
            "log_loss": round(float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), 3),
            "top20_late_caught": round(capture(y, p), 3),
            "top20_late_dollars_caught": round(capture(y, p, w=amt), 3),
            "top20_precision": round(float(np.asarray(y)[np.lexsort((np.random.default_rng(0).random(len(y)),
                                     -np.asarray(p)))[:int(np.ceil(TOP_SHARE * len(y)))]].mean()), 3)}


def calibration(y, p, edges=(0, .2, .4, .6, .8, 1.0001)):
    b = pd.cut(p, edges, right=False, labels=[f"{edges[i]:.1f}-{min(edges[i+1], 1):.1f}" for i in range(len(edges) - 1)])
    t = pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b", observed=False).agg(
        n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    return [{"bin": str(i), "n": int(r.n), "mean_pred": None if r.n == 0 else round(float(r.mean_pred), 3),
             "observed_late": None if r.n == 0 else round(float(r.observed), 3)} for i, r in t.iterrows()]


def paired_boot(y, a, b, amt, seed=0):
    """95% bootstrap interval for (model a - model b) on AUC and top-20% capture."""
    rng = np.random.default_rng(seed)
    y, a, b = map(np.asarray, (y, a, b))
    d_auc, d_cap = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d_auc.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
        d_cap.append(capture(y[i], a[i]) - capture(y[i], b[i]))
    q = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]  # noqa: E731
    return {"auc_diff": round(float(np.mean(d_auc)), 3), "auc_diff_95ci": q(d_auc),
            "top20_capture_diff": round(float(np.mean(d_cap)), 3), "top20_capture_diff_95ci": q(d_cap),
            "resamples": len(d_auc)}


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit("data/ missing. Run `python fetch_data.py` once (internet needed).")
    OUT.mkdir(exist_ok=True)
    inv = LC.add_labels(LC.load_invoices(DATA_FILE, need_paid=True))
    X = LC.build_features(inv, inv)
    train = inv[(inv.issued < CUTOFF) & (inv.paid < CUTOFF)]
    test = inv[inv.issued >= CUTOFF]
    Xtr, Xte = X.loc[train.index], X.loc[test.index]
    y, amt = test.late.to_numpy(), test.amount.to_numpy()
    base_rate = float(train.late.mean())

    preds, secs = {}, {}
    preds["Everyone gets the overall late rate"] = np.full(len(test), base_rate)
    preds["Client's past late rate"] = LC.client_rate_score(Xte, base_rate)

    t0 = time.time()
    gb = HistGradientBoostingClassifier(random_state=LC.SEED).fit(Xtr[LC.FEATURES], train.late)
    preds["Gradient boosting (sklearn defaults)"] = gb.predict_proba(Xte[LC.FEATURES])[:, 1]
    secs["gradient_boosting"] = round(time.time() - t0, 1)

    gb_nd = HistGradientBoostingClassifier(random_state=LC.SEED).fit(Xtr[LC.FEATURES_NO_DISPUTE], train.late)
    preds["Gradient boosting without the dispute flag"] = gb_nd.predict_proba(Xte[LC.FEATURES_NO_DISPUTE])[:, 1]

    t0 = time.time()
    tab = LC.fit_tabpfn(Xtr[LC.FEATURES], train.late_class.to_numpy())
    p_late, p_very = LC.late_probs(tab, Xte[LC.FEATURES])
    preds["TabPFN-v2 (Late Check)"] = p_late
    secs["tabpfn"] = round(time.time() - t0, 1)

    t0 = time.time()
    tab_nd = LC.fit_tabpfn(Xtr[LC.FEATURES_NO_DISPUTE], train.late_class.to_numpy())
    preds["TabPFN-v2 without the dispute flag"] = LC.late_probs(tab_nd, Xte[LC.FEATURES_NO_DISPUTE])[0]
    secs["tabpfn_no_dispute"] = round(time.time() - t0, 1)

    tab_key, gb_key, cr_key = "TabPFN-v2 (Late Check)", "Gradient boosting (sklearn defaults)", "Client's past late rate"
    very = (test.days_late >= LC.VERY_LATE_DAYS).astype(int).to_numpy()
    results = {
        "dispute_effect": {"test_disputed_share": round(float(test.disputed.mean()), 3),
                           "test_late_share_if_disputed": round(float(test.late[test.disputed == 1].mean()), 3),
                           "test_late_share_if_not_disputed": round(float(test.late[test.disputed == 0].mean()), 3)},
        "dataset": {"file": DATA_FILE.name, "invoices": int(len(inv)), "clients": int(inv.client.nunique()),
                    "issued_from": str(inv.issued.min().date()), "issued_to": str(inv.issued.max().date()),
                    "late_share_all": round(float(inv.late.mean()), 3)},
        "split": {"cutoff": str(CUTOFF.date()),
                  "train": {"rows": int(len(train)), "late_share": round(base_rate, 3),
                            "issued": [str(train.issued.min().date()), str(train.issued.max().date())]},
                  "test": {"rows": int(len(test)), "late_share": round(float(y.mean()), 3),
                           "issued": [str(test.issued.min().date()), str(test.issued.max().date())],
                           "late_invoices": int(y.sum())},
                  "excluded_issued_before_cutoff_but_paid_after": int(((inv.issued < CUTOFF) & (inv.paid >= CUTOFF)).sum())},
        "holdout": {k: metrics(y, v, amt) for k, v in preds.items()},
        "tabpfn_very_late": {"definition": f">= {LC.VERY_LATE_DAYS} days late", "test_share": round(float(very.mean()), 3),
                             "auc": round(float(roc_auc_score(very, p_very)), 3),
                             "brier": round(float(brier_score_loss(very, p_very)), 3)},
        "tabpfn_calibration": calibration(y, p_late),
        "gradient_boosting_calibration": calibration(y, preds[gb_key]),
        "paired_bootstrap": {"tabpfn_minus_gradient_boosting": paired_boot(y, p_late, preds[gb_key], amt),
                             "tabpfn_minus_client_rate": paired_boot(y, p_late, preds[cr_key], amt),
                             "tabpfn_no_dispute_minus_gradient_boosting_no_dispute": paired_boot(
                                 y, preds["TabPFN-v2 without the dispute flag"],
                                 preds["Gradient boosting without the dispute flag"], amt),
                             "tabpfn_no_dispute_minus_client_rate": paired_boot(
                                 y, preds["TabPFN-v2 without the dispute flag"], preds[cr_key], amt)},
        "cpu_seconds_fit_and_score": secs,
        "top_share": TOP_SHARE,
        "random_chasing_expected_capture": TOP_SHARE,
        "top20_slots": int(np.ceil(TOP_SHARE * len(y))),
        "top20_capture_ceiling": round(min(1.0, np.ceil(TOP_SHARE * len(y)) / y.sum()), 3),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    (OUT / "results.md").write_text(results_md(results))
    pd.DataFrame({"invoice_id": test.invoice_id, "late": y, "days_late": test.days_late,
                  **{k: v for k, v in preds.items()}}).to_csv(OUT / "holdout_predictions.csv", index=False)
    plot_capture(y, preds)
    print(results_md(results))


def fmt2(x):
    return "" if x is None else f"{x:.2f}"


def results_md(r: dict) -> str:
    s, d = r["split"], r["dataset"]
    L = ["# Late Check: holdout results", "",
         f"Data: `{d['file']}`: {d['invoices']:,} invoices, {d['clients']} clients, issued {d['issued_from']} "
         f"to {d['issued_to']}; {d['late_share_all']:.1%} paid late overall.", "",
         f"Time split at **{s['cutoff']}**. Train: {s['train']['rows']:,} invoices issued and paid before the "
         f"cutoff ({s['train']['late_share']:.1%} late). Test: {s['test']['rows']:,} invoices issued "
         f"{s['test']['issued'][0]} to {s['test']['issued'][1]} ({s['test']['late_share']:.1%} late, "
         f"{s['test']['late_invoices']} late invoices). {s['excluded_issued_before_cutoff_but_paid_after']} "
         "invoices issued before the cutoff but paid after it are left out of both sets "
         "(their outcome wasn't known yet at the cutoff).", "",
         "| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Top 20% flagged: share of late invoices caught ↑ | "
         "…share of late dollars caught ↑ | …share of flagged that were late ↑ |",
         "|---|---|---|---|---|---|---|"]
    for k, m in r["holdout"].items():
        L.append(f"| {k} | {m['auc']:.3f} | {m['brier']:.3f} | {m['log_loss']:.3f} | {m['top20_late_caught']:.1%} | "
                 f"{m['top20_late_dollars_caught']:.1%} | {m['top20_precision']:.1%} |")
    L += ["", f"Chasing a random 20% would catch about {r['random_chasing_expected_capture']:.0%} of late invoices. "
          f"The best possible is {r['top20_capture_ceiling']:.1%}: the top 20% is {r['top20_slots']} invoices and "
          f"there are {s['test']['late_invoices']} late ones. Ties in a score are broken at random (seeded).", ""]
    v = r["tabpfn_very_late"]
    L += [f"**How late:** TabPFN's P({v['definition']}) on the test set: AUC {v['auc']:.3f}, "
          f"Brier {v['brier']:.3f} ({v['test_share']:.1%} of test invoices were that late).", ""]
    de = r["dispute_effect"]
    L += [f"**Disputes matter a lot here:** {de['test_disputed_share']:.1%} of test invoices were disputed; "
          f"{de['test_late_share_if_disputed']:.1%} of those were paid late vs {de['test_late_share_if_not_disputed']:.1%} "
          "of the rest. The dataset doesn't say *when* a dispute was raised, so the 'without the dispute flag' rows "
          "are the fairer estimate for an invoice scored on the day it goes out.", ""]
    L += ["## Paired bootstrap on the test set (2,000 resamples, 95% interval)", "",
          "| Comparison | AUC difference | Top-20% capture difference |", "|---|---|---|"]
    for k, b in r["paired_bootstrap"].items():
        L.append(f"| {k.replace('_', ' ')} | {b['auc_diff']:+.3f} [{b['auc_diff_95ci'][0]:+.3f}, "
                 f"{b['auc_diff_95ci'][1]:+.3f}] | {b['top20_capture_diff']:+.3f} "
                 f"[{b['top20_capture_diff_95ci'][0]:+.3f}, {b['top20_capture_diff_95ci'][1]:+.3f}] |")
    for name, key in (("TabPFN-v2", "tabpfn_calibration"), ("Gradient boosting", "gradient_boosting_calibration")):
        L += ["", f"## {name} calibration (test set)", "", "| Predicted P(late) | Invoices | Mean predicted | Actually late |",
              "|---|---|---|---|"]
        L += [f"| {c['bin']} | {c['n']} | {fmt2(c['mean_pred'])} | {fmt2(c['observed_late'])} |" for c in r[key]]
    c = r["cpu_seconds_fit_and_score"]
    L += ["", f"CPU time to fit and score the {s['test']['rows']} test invoices: TabPFN-v2 {c['tabpfn']} s, "
          f"TabPFN-v2 without dispute flag {c['tabpfn_no_dispute']} s, gradient boosting {c['gradient_boosting']} s "
          "(8-core box; TabPFN on CPU is not bit-for-bit repeatable between runs).", ""]
    return "\n".join(L)


def plot_capture(y, preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    shares = np.linspace(0.01, 1, 100)
    for k in ("TabPFN-v2 (Late Check)", "Gradient boosting (sklearn defaults)", "Client's past late rate"):
        ax.plot(shares * 100, [capture(y, preds[k], s) * 100 for s in shares], label=k)
    ax.plot([0, 100], [0, 100], ls="--", c="grey", label="Random order")
    ax.axvline(20, c="#bbb", lw=0.8)
    ax.set_xlabel("% of test invoices you chase (riskiest first)")
    ax.set_ylabel("% of late invoices caught")
    ax.set_title("Late Check holdout (invoices issued Apr–Dec 2013)", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.text(0.99, 0.01, "Built with PriorLabs-TabPFN", ha="right", fontsize=7, color="#666")
    fig.tight_layout()
    fig.savefig(OUT / "capture_curve.png")


if __name__ == "__main__":
    main()
