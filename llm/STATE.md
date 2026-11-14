# STATE: LLM team
Last updated: 2026-11-14 by Mohakk with Claude Code
Current task: A3 nominated 2026-11-14; next: final runs on Kaggle (3 seeds x tok, notok) with test predictions
Platform / hardware today: local Windows, RTX 4060 Laptop 8 GB, 15.7 GB RAM, Python 3.12.6, venv %USERPROFILE%\venvs\seri-llm (system packages + peft, accelerate), versions in code/requirements.txt

## Milestones
- data_v1 hashes verified: yes, train, val and test on 2026-11-06 (hash only, test contents not read)
- NOMINATE commit pushed: yes, 2026-11-14 (nominated configuration: A3, see Nomination)
- FINAL commit pushed: no

## Nomination (2026-11-14, approved by the lead)
A3: F1 classification head on Qwen/Qwen3-1.7B (revision 70d244cc86), 4-bit nf4 QLoRA, LoRA r16 alpha 32 dropout 0.05 on all linear layers, lr 2e-4, 2 epochs, batch 8 x grad-accum 2, warmup 0.05, max length 317, no class weights, p_pos = raw softmax probability scored at 0.5.
Final command per run: train_cls.py --model Qwen/Qwen3-1.7B --targets all --rank 16 --alpha 32 --lr 2e-4 --epochs 2 --variant <tok|notok> --seed <42|43|44> --predict-test (protocol latency on notok seed 42 only).
Why: best val macro-F1 on notok (0.770), tied with F2 (0.768) and Qwen3-4B (0.768) within bootstrap noise, and the cheapest of the three (70 min vs 105 and 111 min of T4 training). Only configuration whose lead over TF-IDF (0.747) has a 95% CI above zero (+0.002 to +0.043).
Known limits: development used seed 42 only; val is easier than test (single-annotator labels); A3's protocol latency is not yet measured (expected about 100 ms on a T4).

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
- Final stage (ready before nominating): build_job.py --job final --final-script <F1|F2 script> --final-args '<nominated settings>' --variant tok|notok builds a job only if our pushed NOMINATE commit exists and llm/code is committed; it embeds the NOMINATE sha so the Kaggle runner accepts test.csv (SHA256-checked) from a separate private dataset <owner>/seri-uli-data-v1-test, created only after NOMINATE. assemble_final.py turns the 6 runs into final/final_predictions.csv and cost.json and refuses on wrong ids, missing seeds, mixed configs or non-protocol latency. Rehearsed 2026-11-14 end to end (runner, 6 runs, assembly, compare/validate_submission.py: all PASS) with tiny models and a fake test set of 300 val rows; the real test.csv was not read. Building a real final job is refused today (no NOMINATE yet).
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
| A1 | 2026-11-14 | Qwen3-1.7B 4-bit | F1 cls head | notok | 42 | r16 attn, 3 ep | 0.758 (en .764 hi .696 ta .762) | AUROC .857; best epoch 3 but no gain over 2 epochs; 83 min |
| A2 | 2026-11-14 | Qwen3-1.7B 4-bit | F1 cls head | notok | 42 | r32 alpha 64 attn, 2 ep | 0.763 (en .748 hi .692 ta .782) | AUROC .867; 56 min; adapter 49 MB |
| A3 | 2026-11-14 | Qwen3-1.7B 4-bit | F1 cls head | notok | 42 | r16 all-linear, 2 ep | **0.770** (en .757 hi .706 ta .786) | AUROC .872, pos-F1 .677; 70 min; 3.6 GB; 17.4M trainable; adapter 67 MB |
| Q4 | 2026-11-14 | Qwen3-4B 4-bit | F1 cls head | notok | 42 | r16 attn, 2 ep | 0.768 (en .775 hi .690 ta .785) | AUROC .871; 111 min; 5.5 GB; 125 ms/post (200 posts, not protocol) |
| F2-a | 2026-11-14 | Qwen3-1.7B 4-bit | F2 generative Yes/No | notok | 42 | r16 attn, 2 ep, batch 4x4 | 0.768 (en .783 hi .709 ta .760) | AUROC .869; epoch 1 .707, epoch 2 .768; 105 min; 5.4 GB; protocol latency 148 ms/post (T4, TEAM_AGREEMENTS 3a) |
| F2-b | 2026-11-14 | Qwen3-1.7B 4-bit | F2 generative Yes/No | tok | 42 | same | 0.774 (en .802 hi .717 ta .759) | AUROC .871; epoch 1 .755, epoch 2 .774 |
| P1 | 2026-11-13 | Qwen3-1.7B 4-bit | F1 pilot | notok | 42 | 200 rows, 1 epoch, r16 attn, lr 2e-4 | 0.472 (val sample 400) | timing only; peak 3.47 GB at batch 8; latency 114 ms/post (T4, batch 1); pooling diff 0.006 |
| P2 | 2026-11-13 | Qwen3-1.7B 4-bit | F1 pilot | notok | 42 | 1,000 rows, 1 epoch | 0.520 (val sample 400) | 1.96 min train; estimate about 34 min per full epoch on a T4; 6.4M trainable params |

Lead baselines on val (2026-11-14, compare.py functions, train and val only, notok): majority per language 0.410; TF-IDF + logistic regression **0.747** (en .724 hi .681 ta .764, AUROC .843; C picked on val, so slightly optimistic). F1-a beats TF-IDF by only 0.009 overall: +.006 en, +.033 hi, -.001 ta. That gap is within val noise.

Paired bootstrap on val (2026-11-14, 2,000 resamples, notok, threshold 0.5): A3 all-linear minus F1-a +0.014 (95% CI -0.003 to +0.030); Qwen3-4B minus F1-a +0.012 (-0.005 to +0.030); A3 minus 4B +0.001 (-0.015 to +0.018). Against TF-IDF: A3 +0.023 (+0.002 to +0.043), 4B +0.021 (-0.000 to +0.043), F1-a +0.009 (-0.012 to +0.031). F2-a (notok) minus A3 -0.002 (-0.019 to +0.015); F2-a minus TF-IDF +0.020 (-0.000 to +0.042). So A3 is the only configuration whose lead over TF-IDF clears zero, narrowly, and the 4B model buys nothing over A3 at twice the training time and more latency.

Sanity check on F1-a (2026-11-14): 35 val posts share normalized text with train; excluding them gives 0.754, so no leakage. Val scores are likely higher than test will be: val labels come from the same mostly single-annotator process as train, while test is a 3-annotator majority (alpha 0.462). The 0.616 ICON figure is on test with an unverified F1 definition, so it is not comparable to val.
Kaggle allows 2 GPU jobs at once and 30 GPU hours per week (refresh 2026-11-21). Done 2026-11-14: ablate_17b v1 (213 min of runs) and qwen4b v1 (116 min), results in llm/outputs/kaggle/. f2_s42 v1 done 15:11 (221 min of runs). These jobs were built before the protocol latency code, so their latency fields are not protocol numbers.
Kaggle job f1_full_s42 v1 (2026-11-14): 134 min of T4 time. Kaggle job e0_pilot v1 (2026-11-13): 20.3 min of T4 time, Kaggle image python 3.13, torch 2.11.0+cu130, our pinned transformers/peft/bitsandbytes. Results in llm/outputs/kaggle/e0_pilot_v1 (git-ignored).

## Test-access log (date | what was run on test | who)
2026-11-06 | SHA256 hash only, contents not read | Mohakk with Claude Code

## Decisions log (date | decision | reason)
2026-11-14 | Nominate A3 (Qwen3-1.7B F1 head, LoRA r16 all-linear, 2 epochs) | Ties F2 and Qwen3-4B on val within noise at the lowest cost; lead approved
2026-11-12 | Qwen3 only (1.7B primary, 4B as second size); Sarvam-1 dropped | Lead decision. Sarvam-1 restricts the model, adapters and outputs to non-commercial research. Cost: Qwen3 needs about 3x the tokens on Tamil, so runs and latency are slower.
2026-11-12 | Pilots on the local RTX 4060; final timed runs on a T4 (Kaggle or Colab) | Encoder team agreed T4 for final timed runs, so cost and latency are comparable (see ../TEAM_AGREEMENTS.md)
2026-11-12 | Max length 317 for Qwen3 | Covers 99% of train in every language
2026-11-13 | Never use trainer.predict for saved predictions | It returns length-sorted order when group_by_length=True (see Code)
2026-11-14 | Latency follows TEAM_AGREEMENTS.md 3a (first 500 val rows, notok, batch 1, raw text to p_pos, 20 warm-up, median of 3 means), via common.measure_latency | Lead decision for both teams; earlier pilot latencies (114, 98, 102 ms) used 100-200 posts and are not protocol numbers
2026-11-14 | p_pos is the model's raw probability, scored at 0.5; no post-hoc shifting; class weights allowed if declared (TEAM_AGREEMENTS.md 3b) | Lead decision for both teams. We use no class weights so far
2026-11-13 | Real runs on Kaggle (background jobs, one T4), laptop only for tiny-model code checks | Home network about 330 KB/s makes model downloads take hours; Kaggle jobs run with the laptop off and match the final hardware

## Blockers and open problems
- TF-IDF + logistic regression is close: our best (A3, all-linear) leads it by 0.023 on val, with a bootstrap CI that only just clears zero. The honest finding may still be that the LLM adds little over a lexical baseline here, at far higher cost.
- Latency on a T4 is about 114 ms per post at batch size 1 (pilot, tokenization included). A base-size encoder will very likely be much faster, so a tie almost certainly goes to the encoder team.
- Qwen3 quality on Hindi and Tamil is documented support only, not measured on this task.
- bitsandbytes 0.50.2 installed (checksum matched PyPI) and runs on the laptop GPU: a 4-bit layer and the paged 8-bit AdamW step both work. A full 4-bit Qwen3 load is still untested. The 8 GB GPU may run short on long Tamil batches.
- Home network is slow (measured 2026-11-13: about 35 KB/s from PyPI, about 330 KB/s from Hugging Face). Qwen3-1.7B weights (about 4 GB) would take about 3.5 hours. Kaggle and Colab download fast, so real runs may be quicker there.

## Next 3 actions
1. Upload test.csv as private Kaggle dataset <owner>/seri-uli-data-v1-test (lead approves first), then push the two final jobs (build_job.py --job final, notok and tok).
2. Download the 6 runs, assemble_final.py, validate_submission.py llm, log test access, FINAL commit.
3. Error analysis on val and test (per language, unanimous vs 2-vs-1, handle token, 30 hand-read errors).
