"""One-time downloads (needs internet). Everything after this runs offline.

    python fetch_data.py          # dataset + TabPFN-v2 weights
    python fetch_data.py data     # just the dataset
    python fetch_data.py model    # just the weights

1. IBM Watson Analytics sample "Accounts Receivable" data (WA_Fn-UseC_-Accounts-Receivable.csv),
   from a public GitHub mirror, checked against a SHA-256 so you know you got the same file we tested.
2. TabPFN-v2 classifier weights (Prior-Labs/TabPFN-v2-clf on Hugging Face) into ./models.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

AR_URL = ("https://raw.githubusercontent.com/pthiagu2/DataMining/master/"
          "WA_Fn-UseC_-Accounts-Receivable.csv")
AR_FILE = DATA / "WA_Fn-UseC_-Accounts-Receivable.csv"
AR_SHA256 = "f5d488eaa4c1cc843b478b6f1068c553c5394aab4b64d587e4b9ad811d845c19"


def fetch_ar() -> None:
    DATA.mkdir(exist_ok=True)
    with urllib.request.urlopen(AR_URL, timeout=60) as r:
        raw = r.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != AR_SHA256:
        raise SystemExit(f"Checksum mismatch for {AR_URL}: got {digest}. The mirror changed; stopping.")
    AR_FILE.write_bytes(raw)
    print(f"wrote {AR_FILE} ({raw.count(b'\n') - 1} invoices, sha256 ok)")


def fetch_model() -> None:
    MODELS.mkdir(exist_ok=True)
    if sorted(MODELS.glob("tabpfn-v2-classifier*.ckpt")):
        print("TabPFN-v2 weights already in ./models")
        return
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(MODELS))
    import numpy as np
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu")
    clf.fit(np.random.rand(20, 2), np.array([0, 1] * 10))  # forces the download into ./models
    print(f"model weights: {[c.name for c in sorted(MODELS.glob('tabpfn-v2-classifier*.ckpt'))]}")


def main(argv: list[str] | None = None) -> None:
    what = set(argv if argv is not None else sys.argv[1:]) or {"data", "model"}
    if "data" in what:
        fetch_ar()
    if "model" in what:
        fetch_model()


if __name__ == "__main__":
    main()
