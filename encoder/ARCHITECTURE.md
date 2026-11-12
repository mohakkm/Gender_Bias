# ARCHITECTURE: encoder team

## Goal
Fine-tune multilingual encoders to predict label_1 (gendered abuse directed at a person of marginalized gender) on data_v1. Compared against the LLM team under identical data and metrics by the lead's compare.py. ../TEAM_AGREEMENTS.md overrides the brief where they differ.

## Data flow
data_v1/{train,val,test}.csv (read-only, hashes in SHA256SUMS.txt)
-> drop empty label_1 rows (train 17,290, val 1,931 usable)
-> variant: tok (text as is) or notok (remove '<handle replaced>', collapse whitespace, strip)
-> tokenize (max_len = smallest multiple of 8 covering >= 99% of train under both variants, cap 512; truncation rate logged)
-> fine-tune on train, early stop on val macro-F1 (patience 2)
-> NOMINATE commit ('encoder: NOMINATE ...', chosen on val, notok)
-> nominated config: seeds 42, 43, 44 x tok, notok, then test
-> encoder/final/final_predictions.csv + encoder/final/cost.json -> FINAL commit

Loading, notok and metrics are imported from ../compare/compare.py (code/common.py), so val numbers use the same functions the lead uses on test.

## Models
- IndicBERT: ai4bharat/IndicBERTv2-MLM-only (BERT-style, 278M, MIT, covers en/hi/ta). Chosen over v1 (ai4bharat/indic-bert, ALBERT) as the current AI4Bharat model. Confirm after the token study.
- MuRIL: google/muril-base-cased
- mBERT: bert-base-multilingual-cased
- Optional: xlm-roberta-base

## Training setup (tune with seed 42 only; seeds 43 and 44 only on the nominated config)
- AutoModelForSequenceClassification, 2 labels, softmax p_pos
- AdamW (no decay on bias/LayerNorm), lr in {2e-5, 3e-5, 5e-5}, batch in {16, 32}, up to 5 epochs, early stopping patience 2 on val macro-F1
- Linear schedule, warmup 10%, weight decay 0.01, grad clip 1.0, float16 mixed precision on T4 (no bf16)
- Class weights: off by default. Any run with them is suffixed _cw and declared in STATE.md
- Raw p_pos submitted, threshold 0.5 (threshold rule still open, TEAM_AGREEMENTS section 3)
- Extension (optional): multi-task heads for label_1, label_2, label_3, loss masked for empty labels

## Code (encoder/code/)
- common.py: model checkpoints, data loading, metrics (imports compare.py read-only)
- data_check.py: hashes and row counts
- token_study.py: length percentiles, truncation, UNK rate, Latin share per tokenizer -> results/token_study.csv, token_study_maxlen.json
- train.py: one run; resumable from last.pt each epoch; --pilot N for timing -> results/runs.csv, results/val_preds/<run_id>.csv
- grid.py: the B2 grid for given models, skips finished runs
- env.py: prints pinned versions for requirements.txt
- To come before nominating: predict_test.py (refuses to run without an 'encoder: NOMINATE' commit), latency.py, cost.json writer

## How to run (Kaggle, GPU T4, internet on)
    git clone <repo> && cd Gender_Bias && git config pull.rebase true && git config rebase.autoStash true
    python encoder/code/env.py
    python encoder/code/data_check.py
    python encoder/code/token_study.py
    python encoder/code/train.py --model muril --variant notok --pilot 200 --out /kaggle/working/runs
    python encoder/code/grid.py --models muril --out /kaggle/working/runs
Checkpoints live in --out, never in the repo. Kaggle /kaggle/working is lost when an interactive session ends: copy results/ back into the repo and push before closing.

## Environment
- Development: any hardware. Final runs and every cost.json number: Kaggle or Colab T4
- Pinned versions: TBD (run env.py on the T4 session)

## Repo layout
encoder/{CLAUDE.md, ARCHITECTURE.md, CHECKLIST.md, STATE.md}
encoder/code/ scripts, requirements.txt, results/
encoder/final/ final_predictions.csv, cost.json
