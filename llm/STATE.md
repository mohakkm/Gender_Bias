# STATE: LLM team
Last updated: 2026-11-12 by Mohakk with Claude Code
Current task: D1, environment setup for Qwen3 (requirements.txt, 4-bit loading)
Platform / hardware today: local Windows, RTX 4060 Laptop 8 GB, 15.7 GB RAM, Python 3.12.6, torch 2.7.1+cu118, transformers 4.57.3

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

## Experiments (development runs: val only)
| ID | Date | Model | Formulation | Variant | Seed | Key hparams | Val macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Test-access log (date | what was run on test | who)
2026-11-06 | SHA256 hash only, contents not read | Mohakk with Claude Code

## Decisions log (date | decision | reason)
2026-11-12 | Qwen3 only (1.7B primary, 4B as second size); Sarvam-1 dropped | Lead decision. Sarvam-1 restricts the model, adapters and outputs to non-commercial research. Cost: Qwen3 needs about 3x the tokens on Tamil, so runs and latency are slower.
2026-11-12 | Pilots on the local RTX 4060; final timed runs on a T4 (Kaggle or Colab) | Encoder team agreed T4 for final timed runs, so cost and latency are comparable (see ../TEAM_AGREEMENTS.md)
2026-11-12 | Max length 317 for Qwen3 | Covers 99% of train in every language

## Blockers and open problems
- Latency protocol for the tie-break and the rule on shifting p_pos are undefined. Lead must decide before either team nominates.
- Qwen3 quality on Hindi and Tamil is documented support only, not measured on this task.
- 4-bit loading (bitsandbytes) on Windows is untested. The 8 GB laptop GPU may run short on long Tamil batches.

## Next 3 actions
1. Install peft, bitsandbytes and accelerate, pin code/requirements.txt, and confirm 4-bit loading of Qwen3-1.7B on the laptop (needs approval to download about 4 GB of weights).
2. Run E0 zero-shot and few-shot prompting on val.
3. Run the 200-row F1 pilot with timing, then confirm a Kaggle or Colab T4 session works.
