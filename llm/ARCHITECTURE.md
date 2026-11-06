# ARCHITECTURE: LLM team

## Goal
Fine-tune an open-weight LLM with LoRA/QLoRA to predict label_1 (gendered abuse directed at a person of marginalized gender) on data_v1. Compared against the encoder team under identical data and metrics by the lead's compare.py.

## Data flow
data_v1/{train,val,test}.csv (read-only, hashes in SHA256SUMS.txt)
 -> drop empty label_1 rows (train and val)
 -> variant: tok (text as is) or notok (remove '<handle replaced>', collapse whitespace, strip)
 -> tokenize (max length = 99th percentile of train, truncation rate logged)
 -> train on train, early stop and tune on val
 -> NOMINATE commit (final configuration chosen on val, notok)
 -> run nominated config on test: seeds 42, 43, 44 x tok, notok
 -> llm/final/final_predictions.csv + llm/final/cost.json -> FINAL commit

## Formulations
F1 (primary): LLM + sequence-classification head, LoRA adapters (SEQ_CLS), p_pos from softmax.
F2 (secondary): instruction prompt, answer Yes/No, completion-only loss + EOS, p_pos from the two answer-token logits.

## Components (fill in as decided)
- Base model(s): TBD. Proposed in STATE.md: Qwen3-1.7B, Qwen3-4B, Sarvam-1
- Quantization: QLoRA 4-bit nf4, float16 compute on a T4
- LoRA config: r=16, alpha=32, dropout=0.05, target modules TBD
- Training: lr, batch, epochs, scheduler TBD
- Seeds: 42, 43, 44

## Environment
- Platform / hardware: TBD
- code/requirements.txt pinned versions: TBD

## Repo layout
llm/{ARCHITECTURE.md, CHECKLIST.md, STATE.md}
llm/code/    scripts, notebooks, configs, requirements.txt
llm/final/   final_predictions.csv, cost.json
