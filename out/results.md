# Late Check: holdout results

Data: `WA_Fn-UseC_-Accounts-Receivable.csv`: 2,466 invoices, 100 clients, issued 2012-01-03 to 2013-12-02; 35.6% paid late overall.

Time split at **2013-04-01**. Train: 1,500 invoices issued and paid before the cutoff (37.3% late). Test: 872 invoices issued 2013-04-01 to 2013-12-02 (30.0% late, 262 late invoices). 94 invoices issued before the cutoff but paid after it are left out of both sets (their outcome wasn't known yet at the cutoff).

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Top 20% flagged: share of late invoices caught ↑ | …share of late dollars caught ↑ | …share of flagged that were late ↑ |
|---|---|---|---|---|---|---|
| Everyone gets the overall late rate | 0.500 | 0.215 | 0.623 | 21.4% | 20.5% | 32.0% |
| Client's past late rate | 0.866 | 0.139 | 0.435 | 50.0% | 51.0% | 74.9% |
| Gradient boosting (sklearn defaults) | 0.896 | 0.120 | 0.405 | 52.3% | 55.4% | 78.3% |
| Gradient boosting without the dispute flag | 0.851 | 0.146 | 0.489 | 48.9% | 49.7% | 73.1% |
| TabPFN-v2 (Late Check) | 0.917 | 0.105 | 0.338 | 56.5% | 55.2% | 84.6% |
| TabPFN-v2 without the dispute flag | 0.875 | 0.131 | 0.408 | 51.1% | 51.7% | 76.6% |

Chasing a random 20% would catch about 20% of late invoices. The best possible is 66.8%: the top 20% is 175 invoices and there are 262 late ones. Ties in a score are broken at random (seeded).

**How late:** TabPFN's P(>= 8 days late) on the test set: AUC 0.917, Brier 0.074 (13.8% of test invoices were that late).

**Disputes matter a lot here:** 22.8% of test invoices were disputed; 62.3% of those were paid late vs 20.5% of the rest. The dataset doesn't say *when* a dispute was raised, so the 'without the dispute flag' rows are the fairer estimate for an invoice scored on the day it goes out.

## Paired bootstrap on the test set (2,000 resamples, 95% interval)

| Comparison | AUC difference | Top-20% capture difference |
|---|---|---|
| tabpfn minus gradient boosting | +0.021 [+0.010, +0.031] | +0.044 [+0.012, +0.079] |
| tabpfn minus client rate | +0.050 [+0.032, +0.069] | +0.069 [+0.034, +0.107] |
| tabpfn no dispute minus gradient boosting no dispute | +0.024 [+0.009, +0.039] | +0.020 [-0.004, +0.047] |
| tabpfn no dispute minus client rate | +0.009 [+0.001, +0.017] | +0.017 [-0.012, +0.045] |

## TabPFN-v2 calibration (test set)

| Predicted P(late) | Invoices | Mean predicted | Actually late |
|---|---|---|---|
| 0.0-0.2 | 499 | 0.05 | 0.06 |
| 0.2-0.4 | 97 | 0.29 | 0.29 |
| 0.4-0.6 | 80 | 0.50 | 0.53 |
| 0.6-0.8 | 115 | 0.69 | 0.77 |
| 0.8-1.0 | 81 | 0.92 | 0.91 |

## Gradient boosting calibration (test set)

| Predicted P(late) | Invoices | Mean predicted | Actually late |
|---|---|---|---|
| 0.0-0.2 | 518 | 0.04 | 0.07 |
| 0.2-0.4 | 67 | 0.28 | 0.34 |
| 0.4-0.6 | 46 | 0.49 | 0.61 |
| 0.6-0.8 | 80 | 0.71 | 0.60 |
| 0.8-1.0 | 161 | 0.94 | 0.80 |

CPU time to fit and score the 872 test invoices: TabPFN-v2 7.8 s, TabPFN-v2 without dispute flag 6.1 s, gradient boosting 0.2 s (8-core box; TabPFN on CPU is not bit-for-bit repeatable, so the last digit can move between runs).
