# STATE: LLM team
Last updated: 2026-11-13 by Mohakk with Claude Code
Current task: D1-D2, pipeline built and tested on a tiny random model; waiting for downloads before real runs
Platform / hardware today: local Windows, RTX 4060 Laptop 8 GB, 15.7 GB RAM, Python 3.12.6, venv %USERPROFILE%\venvs\seri-llm (system packages + peft, accelerate), versions in code/requirements.txt

## Milestones
- data_v1 hashes verified: yes, train, val and test on 2026-11-06 (hash only, test contents not read)
- NOMINATE commit pushed: no (nominated configuration: TBD)
- FINAL commit pushed: no

## Data
- Usable rows: train 17,290 / val 1,931 / test 3,756 (train and val confirmed by code/data_check.py)
- Under 5% of usable train and val rows have 2+ annotators (train 3.9%, val 4.1%), while test has 3 per post. Val selects on noisier labels than test scores.
- notok leaves no empty posts. 5 train posts have 3 characters or fewer.
- notok creates 27 repeated texts in train and 2 in val. They are unique under tok.

## Models (Hub revisions checked 2026-11-06)
| Model | Params | License | Gated | Revision | Role |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-1.7B | 2.03B (API) | Apache-2.0 | no | 70d244cc86 | Primary |
| Qwen/Qwen3-4B | 4.0B | Apache-2.0 | no | 1cfa9a7208 | Second size (brief B2) |
| sarvamai/sarvam-1 | 2.53B | Sarvam research, non-commercial | no | e960733728 | Dropped 2026-11-12, kept only as token-study reference |

Not taken forward: Qwen3.5 Small (multimodal, sequence-classification support unverified), Gemma 4 E2B (gating and language list unverified), Sarvam 30B (too large for a T4), Gemma 3 (gated).

## Token study (train only, 17,290 posts, details in code/token_study_results.csv)
| Measure | Qwen3 (1.7B and 4B) | Sarvam-1 |
| --- | --- | --- |
| Mean tokens per post, notok: en / hi / ta | 25.2 / 83.5 / 131.6 | 32.4 / 38.4 / 43.2 |
| p99, notok: en / hi / ta | 75.5 / 261 / 312 | 96 / 117.7 / 106 |
| Highest per-language p99: tok / notok | 317 / 312 | 128 / 118 |
| Truncated at highest per-language p99, pooled: tok / notok | 0.34% / 0.34% | 0.46% / 0.50% |
| Tokens per word (200 posts, seed 42): en / hi / ta | 1.39 / 3.92 / 8.40 | 1.74 / 1.80 / 2.69 |

Max length for Qwen3: 317 for both tok and notok. The pooled p99 alone would truncate 2.7-2.9% of Tamil posts. F2 prompts need extra room on top.

## Code (llm/code/, outputs in git-ignored llm/outputs/)
- common.py: data loading, tok/notok, metrics, and a hard gate: reading test.csv raises an error unless a pushed 'llm: NOMINATE' commit touching only llm/ exists. Tested: refuses today.
- train_cls.py: F1 (QLoRA 4-bit, LoRA SEQ_CLS), early stopping on val macro-F1, resume, pilot timing, latency at batch size 1 on val posts, pooling check (padded vs unpadded logits identical).
- kaggle/build_job.py: packages the scripts into one Kaggle script job (private, GPU, internet). The job checks train/val SHA256, refuses if any test.csv is attached, uses one GPU, deletes checkpoints and leaves results/summary.json, metrics, val predictions, adapters and logs. Kaggle dataset: <owner>/seri-uli-data-v1-trainval (train.csv, val.csv, DATA_CARD.md, their SHA256 lines only).
- e0_prompt.py: E0 zero-shot and few-shot on val, p_pos from the Yes/No answer-token logits, Qwen3 thinking switched off.
- Smoke-tested end to end on tiny random Qwen3 models (no 4-bit yet; bitsandbytes not installed).
- Bug found and fixed: with group_by_length=True, transformers 4.57.3 also length-sorts eval and predict batches, so trainer.predict output is NOT in dataset order. It would have attached p_pos to the wrong ids, test included. Predictions now use our own in-order loop, and the run stops if the final val macro-F1 differs from the best epoch's recorded score by more than 0.01.

## Experiments (development runs: val only)
| ID | Date | Model | Formulation | Variant | Seed | Key hparams | Val macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0-a | 2026-11-13 | Qwen3-1.7B | E0 prompt | notok | - | 0-shot | 0.517 (en .524 hi .478 ta .507) | AUROC .627; says Yes on 69% vs 31% true; only 60% of probability on Yes/No |
| E0-b | 2026-11-13 | Qwen3-1.7B | E0 prompt | tok | - | 0-shot | 0.493 (en .478 hi .451 ta .514) | AUROC .617; says Yes on 72% |
| E0-c | 2026-11-13 | Qwen3-1.7B | E0 prompt | tok | - | 6-shot | 0.600 (en .584 hi .577 ta .634) | AUROC .678; says Yes on 48%; 8.5 min |
| E0-d | 2026-11-13 | Qwen3-1.7B | E0 prompt | notok | - | 6-shot | crashed | T4 out of memory (full-vocab logits at every position); fixed with logits_to_keep=1 |
| E0-d2 | 2026-11-14 | Qwen3-1.7B | E0 prompt | notok | - | 6-shot | 0.609 (en .608 hi .608 ta .607) | AUROC .687; says Yes on 43% |
| F1-a | 2026-11-14 | Qwen3-1.7B 4-bit | F1 cls head | notok | 42 | r16 attn, lr 2e-4, 2 ep, batch 8x2 | **0.756** (en .730 hi .714 ta .763) | AUROC .858, pos-F1 .656; epoch 1 .731, epoch 2 .756 (still rising); 60 min; 3.5 GB; 98 ms/post |
| F1-b | 2026-11-14 | Qwen3-1.7B 4-bit | F1 cls head | tok | 42 | same | 0.760 (en .779 hi .680 ta .770) | AUROC .859; epoch 1 .660, epoch 2 .760; English gains .049 from the handle token, Hindi loses .034 |
| P1 | 2026-11-13 | Qwen3-1.7B 4-bit | F1 pilot | notok | 42 | 200 rows, 1 epoch, r16 attn, lr 2e-4 | 0.472 (val sample 400) | timing only; peak 3.47 GB at batch 8; latency 114 ms/post (T4, batch 1); pooling diff 0.006 |
| P2 | 2026-11-13 | Qwen3-1.7B 4-bit | F1 pilot | notok | 42 | 1,000 rows, 1 epoch | 0.520 (val sample 400) | 1.96 min train; estimate about 34 min per full epoch on a T4; 6.4M trainable params |

Sanity check on F1-a (2026-11-14): 35 val posts share normalized text with train; excluding them gives 0.754, so no leakage. Val scores are likely higher than test will be: val labels come from the same mostly single-annotator process as train, while test is a 3-annotator majority (alpha 0.462). The 0.616 ICON figure is on test with an unverified F1 definition, so it is not comparable to val.
Kaggle allows 2 GPU jobs at once and 30 GPU hours per week (refresh 2026-11-21). Running 2026-11-14: ablate_17b (F1 notok: 3 epochs; r32; all-linear), qwen4b (F1 notok Qwen3-4B). Queued by a local helper when a slot frees: f2_s42 (F2 notok and tok).
Kaggle job f1_full_s42 v1 (2026-11-14): 134 min of T4 time. Kaggle job e0_pilot v1 (2026-11-13): 20.3 min of T4 time, Kaggle image python 3.13, torch 2.11.0+cu130, our pinned transformers/peft/bitsandbytes. Results in llm/outputs/kaggle/e0_pilot_v1 (git-ignored).

## Test-access log (date | what was run on test | who)
2026-11-06 | SHA256 hash only, contents not read | Mohakk with Claude Code

## Decisions log (date | decision | reason)
2026-11-12 | Qwen3 only (1.7B primary, 4B as second size); Sarvam-1 dropped | Lead decision. Sarvam-1 restricts the model, adapters and outputs to non-commercial research. Cost: Qwen3 needs about 3x the tokens on Tamil, so runs and latency are slower.
2026-11-12 | Pilots on the local RTX 4060; final timed runs on a T4 (Kaggle or Colab) | Encoder team agreed T4 for final timed runs, so cost and latency are comparable (see ../TEAM_AGREEMENTS.md)
2026-11-12 | Max length 317 for Qwen3 | Covers 99% of train in every language
2026-11-13 | Never use trainer.predict for saved predictions | It returns length-sorted order when group_by_length=True (see Code)
2026-11-14 | Latency follows TEAM_AGREEMENTS.md 3a (first 500 val rows, notok, batch 1, raw text to p_pos, 20 warm-up, median of 3 means), via common.measure_latency | Lead decision for both teams; earlier pilot latencies (114, 98, 102 ms) used 100-200 posts and are not protocol numbers
2026-11-14 | p_pos is the model's raw probability, scored at 0.5; no post-hoc shifting; class weights allowed if declared (TEAM_AGREEMENTS.md 3b) | Lead decision for both teams. We use no class weights so far
2026-11-13 | Real runs on Kaggle (background jobs, one T4), laptop only for tiny-model code checks | Home network about 330 KB/s makes model downloads take hours; Kaggle jobs run with the laptop off and match the final hardware

## Blockers and open problems
- Latency on a T4 is about 114 ms per post at batch size 1 (pilot, tokenization included). A base-size encoder will very likely be much faster, so a tie almost certainly goes to the encoder team.
- Encoder team must accept TEAM_AGREEMENTS.md section 3 (latency protocol, p_pos rule) before either team nominates.
- Qwen3 quality on Hindi and Tamil is documented support only, not measured on this task.
- bitsandbytes 0.50.2 installed (checksum matched PyPI) and runs on the laptop GPU: a 4-bit layer and the paged 8-bit AdamW step both work. A full 4-bit Qwen3 load is still untested. The 8 GB GPU may run short on long Tamil batches.
- Home network is slow (measured 2026-11-13: about 35 KB/s from PyPI, about 330 KB/s from Hugging Face). Qwen3-1.7B weights (about 4 GB) would take about 3.5 hours. Kaggle and Colab download fast, so real runs may be quicker there.

## Next 3 actions
1. Collect ablate_17b, qwen4b and f2_s42; pick ONE configuration by val macro-F1 on notok. Treat gaps under about 0.01 to 0.02 as noise (val has 1,931 posts) and prefer the cheaper model then.
2. Once the encoder team accepts TEAM_AGREEMENTS.md section 3, write the nomination in STATE.md and push 'llm: NOMINATE ...'.
3. After NOMINATE: seeds 43 and 44 for both variants, test predictions (upload test.csv to Kaggle only then), final/ files, validator, FINAL commit.
