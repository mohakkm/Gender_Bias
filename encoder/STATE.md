# STATE: encoder team
Last updated: 2026-11-13 by Ishaan Kumar
Current task: D1 pilot runs on the dev GPU
Platform / hardware today: Windows laptop, NVIDIA RTX 4070 Laptop GPU 8 GB, CUDA 12.8 (development only; final cost numbers on a Kaggle/Colab T4)

## Milestones
- data_v1 hashes verified: yes (2026-11-13, code/data_check.py)
- NOMINATE commit pushed: no (nominated configuration: TBD)
- FINAL commit pushed: no

## Data
- Usable rows: train 17,290 / val 1,931 / test 3,756
- Val positive rate by language: en 0.197, hi 0.284, ta 0.431 (train is the same within 0.002)

## Pinned versions (dev machine, 2026-11-13; see code/requirements.txt)
torch 2.11.0+cu128, transformers 5.22.2, tokenizers 0.23.3, sentencepiece 0.2.3, scikit-learn 1.9.1, scipy 1.18.1, pandas 3.1.1, numpy 2.5.4, Python 3.13.7

## Token study (train, notok; full table in code/results/token_study.csv)
| Model | mean tokens | p99 (worst variant) | max | train truncated at 192 | posts with [UNK] |
| --- | --- | --- | --- | --- | --- |
| IndicBERT v2 | 29.6 | 87 | 172 | 0% | 2.0% |
| MuRIL | 31.3 | 93 | 149 | 0% | 24.3% |
| mBERT | 45.2 | 139 | 182 | 0% | 27.9% |
| XLM-R | 35.1 | 104 | 179 | 0% | 2.0% |
- Tamil costs mBERT about 2x the tokens of IndicBERT (59.5 vs 31.8 mean).
- MuRIL and mBERT map content to [UNK] in about a quarter of posts (likely emoji and rare symbols), IndicBERT and XLM-R in about 2%. Text is not modified (shared protocol), so report this as a possible handicap in error analysis.
- Posts with Latin letters: hi 40%, ta 20% (code-mixing proxy, B5).

## Experiments (development runs: val only; full log in code/results/runs.csv)
| ID | Date | Model | Variant | Seed | Key hparams | Val macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Test-access log (date | what was run on test | who)
(none)

## Decisions log (date | decision | reason)
- 2026-11-13 | IndicBERT = ai4bharat/IndicBERTv2-MLM-only | current AI4Bharat model, BERT-style, MIT, covers en/hi/ta, ungated; v1 is ALBERT and older. Token study: lowest token count and [UNK] rate of the four
- 2026-11-13 | max_len = smallest multiple of 8 covering >= 99% of train under both variants, cap 512, per model | rule A2.8; same length for tok and notok keeps the variants comparable
- 2026-11-13 | Metrics imported from compare/compare.py | val numbers computed with the lead's exact functions
- 2026-11-13 | max_len = 192 for every model (overrides the multiple-of-8-over-p99 rule) | covers 100% of train for all four tokenizers; dynamic padding means short posts cost nothing extra, so there is no reason to truncate 1%
- 2026-11-13 | Development on the laptop RTX 4070; final runs and cost.json on a T4 | TEAM_AGREEMENTS section 2
- 2026-11-13 | Class weights off by default | brief says optional but declared; revisit only if Tamil/English recall is lopsided

## Blockers and open problems
- Latency protocol and threshold rule not yet fixed by the lead (TEAM_AGREEMENTS section 3)
- Model split with teammate and deadline not yet set
- train.py not yet run on a GPU: the pilot is its first real test
- [UNK] rate of MuRIL/mBERT (about 25% of posts) may hurt them; not fixable without changing the shared text

## Next 3 actions
1. Pilot (200 rows) for all four models, then one full MuRIL notok run as an end-to-end check
2. Agree the model split with teammate; log pilot timings here
3. Start grid.py (seed 42) on the dev GPU
