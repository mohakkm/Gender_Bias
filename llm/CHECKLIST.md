# CHECKLIST: LLM team
(Tick items the moment they are done. Add the output path beside each.)

## Setup
- [x] Repo cloned, git configured, hashes of train/val/test verified against SHA256SUMS.txt (train and val sanity script: code/data_check.py)
- [ ] Three working files created in llm/ and pushed
- [ ] code/requirements.txt with pinned versions committed (written 2026-11-13, not yet committed)
- [x] Pipeline scripts written and smoke-tested on a tiny random model (code/common.py, code/train_cls.py, code/e0_prompt.py)
- [x] Token-length study: 99th percentile and truncation rate per candidate tokenizer (code/token_study.py, code/token_study_results.csv)
- [x] Tokens per word on 200 Hindi and 200 Tamil posts per candidate model (same files)
- [x] Base model(s) chosen, reasons logged in STATE.md (Qwen3-1.7B and Qwen3-4B; the size is picked by val)
- [ ] Baseline validation scores received from the lead (D2)

## Develop (train and val only, no test)
- [x] E0 zero-shot and few-shot prompting on val (best: 6-shot, 0.600 tok / 0.609 notok)
- [x] 200-row pilot and timing, full-run estimate logged (plus a 1,000-row pilot: about 34 min per full epoch on a T4, STATE.md)
- [x] F1 classification head: tok seed 42, notok seed 42 (0.760 / 0.756)
- [ ] F2 generative scoring: tok seed 42, notok seed 42 (code/train_gen.py written and smoke-tested; Kaggle job queued)
- [ ] Ablations on val: LoRA rank 16 / 32, all linear layers, 3 epochs, Qwen3-4B (Kaggle jobs running 2026-11-14; rank 8 skipped for time)
- [ ] Optional: multi-label head (label_1 + label_3)

## Nominate
- [ ] Final configuration chosen by val macro-F1 on notok, written in STATE.md
- [ ] NOMINATE commit pushed (before any run on test)

## Final run and submission
- [ ] Seeds 43 and 44 for the nominated configuration (tok and notok)
- [ ] Nominated configuration run on test: 3 seeds x tok, notok
- [ ] llm/final/final_predictions.csv written
- [ ] llm/final/cost.json written (time, peak memory, trainable parameters, model size, latency at batch size 1, hardware)
- [ ] python compare/validate_submission.py llm prints PASS
- [ ] FINAL commit pushed, lead told

## Wrap-up
- [ ] Error analysis (per language, unanimous vs 2-vs-1, handle token, 30 hand-read errors)
- [ ] Adapter weights saved and linked in STATE.md
- [ ] Code runs from requirements.txt
- [ ] Written summary vs encoder team (after the lead's comparison report)
- [ ] ARCHITECTURE.md, CHECKLIST.md, STATE.md final
