#!/usr/bin/env bash
# Full pipeline from a clean checkout. Needs internet only for step 1.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/pip install -q -r requirements.txt
fi
PY=.venv/bin/python
$PY fetch_data.py                                   # 1. dataset (sha256-checked) + TabPFN-v2 weights
$PY data_facts.py                                   # 2a. plain dataset facts -> out/data_facts.json
$PY evaluate.py                                     # 2b. time-split holdout -> out/results.md/json
$PY make_demo_files.py --as-of 2013-11-01           # 3. demo history.csv / open.csv from the sample
$PY latecheck.py data/demo_history.csv data/demo_open.csv --as-of 2013-11-01   # 4. the tool itself
$PY check_demo.py                                   # 5. demo report vs what really happened
# 6. optional: picture of the HTML report (needs Chrome/Chromium; skipped if absent)
CHROME=$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)
if [ -n "$CHROME" ]; then
  "$CHROME" --headless=new --disable-gpu --no-sandbox --hide-scrollbars --window-size=1500,1100 \
    --screenshot="$PWD/out/report.png" "file://$PWD/out/report.html" >/dev/null 2>&1 || true
fi
echo "ALL STEPS OK"
