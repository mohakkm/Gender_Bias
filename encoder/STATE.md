# STATE: encoder team
Last updated: 2026-11-13 by Ishaan Kumar
Current task: D1 setup (token study and pilot on Kaggle T4 next)
Platform / hardware today: no GPU yet (code written and data check run on CPU)

## Milestones
- data_v1 hashes verified: yes (2026-11-13, code/data_check.py)
- NOMINATE commit pushed: no (nominated configuration: TBD)
- FINAL commit pushed: no

## Data
- Usable rows: train 17,290 / val 1,931 / test 3,756
- Val positive rate by language: en 0.197, hi 0.284, ta 0.431 (train is the same within 0.002)

## Pinned versions
TBD (paste output of code/env.py from the T4 session)

## Experiments (development runs: val only; full log in code/results/runs.csv)
| ID | Date | Model | Variant | Seed | Key hparams | Val macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Test-access log (date | what was run on test | who)
(none)

## Decisions log (date | decision | reason)
- 2026-11-13 | IndicBERT = ai4bharat/IndicBERTv2-MLM-only (proposed) | current AI4Bharat model, BERT-style, MIT, covers en/hi/ta; v1 is ALBERT and older. Confirm after token study
- 2026-11-13 | max_len = smallest multiple of 8 covering >= 99% of train under both variants, cap 512, per model | rule A2.8; same length for tok and notok keeps the variants comparable
- 2026-11-13 | Metrics imported from compare/compare.py | val numbers computed with the lead's exact functions
- 2026-11-13 | Class weights off by default | brief says optional but declared; revisit only if Tamil/English recall is lopsided

## Blockers and open problems
- Latency protocol and threshold rule not yet fixed by the lead (TEAM_AGREEMENTS section 3)
- Model split with teammate and deadline not yet set
- train.py not yet run on a GPU: the pilot is its first real test

## Next 3 actions
1. On a Kaggle T4: env.py, data_check.py, token_study.py; pin requirements.txt; push results/
2. Pilot: train.py --pilot 200 for muril (and indicbert), log ms/row and the full-grid estimate here
3. Agree the model split with teammate, then start grid.py (seed 42)
