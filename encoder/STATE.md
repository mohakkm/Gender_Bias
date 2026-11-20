# STATE: encoder team
Last updated: 2026-11-21 by Ishaan Kumar
Current task: NOMINATE pushed; final runs on a Kaggle T4 next
Platform / hardware today: Windows laptop, NVIDIA RTX 4070 Laptop GPU 8 GB, CUDA 12.8 (development only; final cost numbers on a Kaggle/Colab T4)

## Milestones
- data_v1 hashes verified: yes (2026-11-13, code/data_check.py)
- NOMINATE commit pushed: yes, 2026-11-21 (IndicBERT v2, ai4bharat/IndicBERTv2-MLM-only, lr 2e-5, batch 32, max_len 192, up to 5 epochs, patience 2)
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

## Baselines on val (computed with compare.py --baselines-val, notok)
- Majority per language: macro-F1 0.410 (en 0.446, hi 0.417, ta 0.363)
- TF-IDF + logistic regression: macro-F1 0.747, pos-F1 0.658 (en 0.724, hi 0.681, ta 0.764)

## Pilot timings (200 rows, RTX 4070 Laptop, fp16, bs 32)
| Model | ms/row | est. min/epoch | peak GPU GB |
| --- | --- | --- | --- |
| MuRIL | 12.6 | 3.6 (measured 1.6 on full run) | 5.2 |
| IndicBERT v2 | 10.9 | 3.2 | 5.8 |
| mBERT | 9.1 | 2.6 | 4.9 |
| XLM-R | 8.9 | 2.6 | 6.0 |
Pilot overestimates (warm-up cost). Full grid, 48 runs: roughly 5-7 hours on this GPU.

## Experiments (development runs: val only; full log in code/results/runs.csv)
| ID | Date | Model | Variant | Seed | Key hparams | Val macro-F1 | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 2026-11-13 | MuRIL | notok | 42 | lr 3e-5, bs 32, ml 192 | 0.7656 (ep 3/5) | en 0.749 hi 0.698 ta 0.771; 1.6 min/epoch on RTX 4070 Laptop |

## Grid result (seed 42, val, 48 runs; full table in code/results/grid_summary.md)
| Model | Best notok config | Best notok | Median notok (6 configs) | en | hi | ta | vs TF-IDF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IndicBERT v2 | lr 2e-5, bs 32 | 0.781 | 0.772 | 0.738 | 0.741 | 0.790 | +0.034 |
| XLM-R | lr 2e-5, bs 32 | 0.777 | 0.770 | 0.767 | 0.724 | 0.790 | +0.030 |
| MuRIL | lr 3e-5, bs 16 | 0.768 | 0.756 | 0.746 | 0.706 | 0.780 | +0.021 |
| mBERT | lr 5e-5, bs 32 | 0.755 | 0.748 | 0.732 | 0.670 | 0.772 | +0.008 |
- MuRIL collapsed to all-negative (0.410) at lr 5e-5 in 2 of 4 runs; no other model collapsed.
- tok vs notok: within about 0.01 for every model, no consistent direction on val.
- Grid train_min values are wall-clock on the dev laptop and include sleep pauses (e.g. 84 min for one XLM-R run); they are NOT cost numbers.

## Test-access log (date | what was run on test | who)
(none)

## Decisions log (date | decision | reason)
- 2026-11-13 | IndicBERT = ai4bharat/IndicBERTv2-MLM-only | current AI4Bharat model, BERT-style, MIT, covers en/hi/ta, ungated; v1 is ALBERT and older. Token study: lowest token count and [UNK] rate of the four
- 2026-11-13 | max_len = smallest multiple of 8 covering >= 99% of train under both variants, cap 512, per model | rule A2.8; same length for tok and notok keeps the variants comparable
- 2026-11-13 | Metrics imported from compare/compare.py | val numbers computed with the lead's exact functions
- 2026-11-13 | max_len = 192 for every model (overrides the multiple-of-8-over-p99 rule) | covers 100% of train for all four tokenizers; dynamic padding means short posts cost nothing extra, so there is no reason to truncate 1%
- 2026-11-13 | Development on the laptop RTX 4070; final runs and cost.json on a T4 | TEAM_AGREEMENTS section 2
- 2026-11-21 | NOMINATE IndicBERT v2, lr 2e-5, bs 32, max_len 192 | highest notok val macro-F1 in the grid (the pre-declared rule, brief A4); also the highest median over its 6 configs and it never collapsed. XLM-R is 0.003 behind, inside seed noise; the rule picks IndicBERT and we do not override it after seeing results
- 2026-11-16 | Seed 42 is retrained on the T4 for the final submission (tag final_t4), not reused from the grid | grid models are not saved, and all six final runs then share one hardware for cost.json; the grid seed-42 val score and the T4 one will differ slightly
- 2026-11-16 | NOMINATE subject format: 'encoder: NOMINATE <model> lr<lr> bs<bs> ml<max_len>' (e.g. 'encoder: NOMINATE muril lr3e-05 bs32 ml192') | final.py refuses to open test unless a commit with exactly these tokens exists
- 2026-11-16 | Latency and p_pos follow TEAM_AGREEMENTS section 3 (accepted 2026-11-14); code/latency.py implements 3a | tie-break rule fixed before results
- 2026-11-13 | Class weights off by default | brief says optional but declared; revisit only if Tamil/English recall is lopsided

## Blockers and open problems
- LLM team pushed NOMINATE and FINAL (Qwen3-1.7B, F1 head, QLoRA r16). The comparison waits on us
- Dev laptop C: drive was full (grid crashed 2026-11-13 on a checkpoint save); per-epoch checkpoints now opt-in
- MuRIL beats TF-IDF on val by only 0.019 macro-F1 (0.766 vs 0.747). If the tuned grid does not widen this, the encoder's advantage over a bag-of-ngrams model is marginal: say so in the paper
- [UNK] rate of MuRIL/mBERT (about 25% of posts) may hurt them; not fixable without changing the shared text

## Next 3 actions
1. Run code/final.py --model indicbert --lr 2e-5 --bs 32 on a Kaggle T4 (lead offered the LLM team's setup)
2. Validator PASS, then the FINAL commit; tell the lead
3. Error analysis (B5) on test predictions
