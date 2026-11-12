# CHECKLIST: encoder team
(Tick items the moment they are done. Add the output path beside each. TEAM_AGREEMENTS.md overrides the brief.)

## Setup
- [x] Repo cloned; git config pull.rebase true AND rebase.autoStash true
- [x] Hashes of train/val/test verified against SHA256SUMS.txt (code/data_check.py, PASS 2026-11-13)
- [x] Three working files created in encoder/
- [x] code/requirements.txt pinned from env.py (dev machine; re-check on the T4 before final runs)
- [x] Token-length study: 99th percentile and truncation rate per tokenizer (code/results/token_study.csv, max_len 192)
- [x] IndicBERT checkpoint confirmed and logged in STATE.md (ai4bharat/IndicBERTv2-MLM-only)
- [ ] Baseline validation scores received from the lead (D2)
- [ ] Pilot run (200 rows) timed, full-grid estimate logged in STATE.md

## Develop (train and val only, seed 42 only, no test)
- [ ] IndicBERT: tok and notok, small grid (code/results/runs.csv)
- [ ] MuRIL: tok and notok, small grid
- [ ] mBERT: tok and notok, small grid
- [ ] Optional: XLM-R tok and notok
- [ ] Optional: multi-task heads (label_1, label_2, label_3)
- [ ] Latency script in code/, re-runnable once the lead fixes the protocol

## Nominate
- [ ] Final model and configuration chosen by val macro-F1 on notok, written in STATE.md
- [ ] Commit starting with exactly 'encoder: NOMINATE', changing only encoder/, pushed before any test run

## Final run and submission (all cost numbers on a Kaggle/Colab T4)
- [ ] Seeds 43 and 44 for the nominated configuration (tok and notok)
- [ ] Nominated configuration run on test: 3 seeds x tok, notok; each test run logged in STATE.md
- [ ] encoder/final/final_predictions.csv written (raw p_pos, no tuned threshold)
- [ ] encoder/final/cost.json written (hardware e.g. "Kaggle T4")
- [ ] python compare/validate_submission.py encoder prints PASS
- [ ] FINAL commit pushed, lead told

## Wrap-up
- [ ] Error analysis (per language, unanimous vs 2-vs-1, handle token, Latin-script proxy, 30 hand-read errors)
- [ ] Checkpoints saved and linked in STATE.md
- [ ] Code runs from requirements.txt
- [ ] Written summary of error patterns (after the lead's comparison report)
- [ ] ARCHITECTURE.md, CHECKLIST.md, STATE.md final
