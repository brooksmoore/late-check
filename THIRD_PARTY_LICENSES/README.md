# Third-party pieces

| Piece | Where it comes from | License | In this repo? |
|---|---|---|---|
| TabPFN-v2 classifier weights (`tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, a content-identical alias of `tabpfn-v2-classifier-v2_default.ckpt`) | https://huggingface.co/Prior-Labs/TabPFN-v2-clf | Prior Labs License v1.1 = Apache 2.0 + attribution ("Built with PriorLabs-TabPFN"). Copy: `TabPFN-v2-clf-LICENSE.txt` | No, downloaded by `fetch_data.py` |
| `tabpfn` Python package 9.1.0 (inference code) | https://github.com/PriorLabs/TabPFN | Apache-2.0 (per the package metadata) | No, pip dependency |
| IBM Watson Analytics sample "Accounts Receivable" CSV | Originally IBM's Watson Analytics sample-data guide (page now offline; archived copy: http://web.archive.org/web/20190701221803/https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/). Mirror used: https://github.com/pthiagu2/DataMining (file `WA_Fn-UseC_-Accounts-Receivable.csv`, sha256 `f5d488ea…845c19`). Also on Kaggle as "Finance Factoring - IBM Late Payment Histories". | Not stated. IBM published it as a free sample; neither the GitHub mirror nor the Kaggle copy carries a license (Kaggle lists "Unknown"). | No, downloaded by `fetch_data.py` |

No code was copied from other projects. The TabPFN loading pattern and the repo layout follow our own
Week 1 entry (Lakefront Window, same authors, MIT).
