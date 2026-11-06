# STATE: LLM team
Last updated: 2026-11-06 by Mohakk with Claude Code
Current task: D1, token study for the proposed base models
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

## Candidate models (proposed, not chosen; revisions are the Hub commits checked 2026-11-06)
| Model | Params | License | Gated | Hindi, Tamil documented | Revision | Role |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-1.7B | 2.03B (API) | Apache-2.0 | no | yes | 70d244cc86 | First choice |
| Qwen/Qwen3-4B | 4.0B | Apache-2.0 | no | yes | 1cfa9a7208 | Second size |
| sarvamai/sarvam-1 | 2.53B (API; card says 2B) | Sarvam research, non-commercial | no | yes | e960733728 | Second family, needs lead approval |

Not taken forward for now: Qwen3.5 Small (multimodal, sequence-classification support unverified), Gemma 4 E2B (gating and language list unverified), Sarvam 30B (32B params, too large for a T4), Gemma 3 (gated).

## Experiments (development runs: val only)
| ID | Date | Model | Formulation | Variant | Seed | Key hparams | Val macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Test-access log (date | what was run on test | who)
2026-11-06 | SHA256 hash only, contents not read | Mohakk with Claude Code

## Decisions log (date | decision | reason)

## Blockers and open problems
- Token study needs tokenizer downloads: Qwen3 15.9 MB (same files for 1.7B and 4B), Sarvam-1 11.2 MB. Awaiting approval.
- Sarvam-1 terms restrict the model, adapters and outputs to non-commercial research. Lead or PI must approve before use.
- Training platform undecided. The local GPU has 8 GB, and the encoder brief names a T4 as the likely common hardware.
- Latency protocol for the tie-break and the rule on shifting p_pos are undefined. Lead must decide before either team nominates.
- Qwen3 quality on Hindi and Tamil is documented support only, not measured on this task.

## Next 3 actions
1. Download the tokenizers, then run the token-length and tokens-per-word studies on train (and val if needed).
2. Pin requirements.txt and confirm 4-bit loading on the chosen platform.
3. Run E0 zero-shot and few-shot prompting on val.
