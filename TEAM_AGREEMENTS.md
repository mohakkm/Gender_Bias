# Team agreements (both teams)

Maintained by the lead. Read this file together with your team brief (v2, 10 Oct 2026). Where the two disagree, this file wins because it is newer. Teams never edit this file. Ask the lead for changes.

Last updated: 2026-11-12

## 1. Corrections to the v2 briefs

- **NOMINATE commit format.** The first line of the message starts with `<team>: NOMINATE`, for example `encoder: NOMINATE MuRIL lr3e-5 bs32` or `llm: NOMINATE F1 <model> r16 lr2e-4`. This matches the commit prefix rule and the brief's own examples. The sentence "a commit whose message starts with NOMINATE" in the briefs is replaced by this.
- **Only your own NOMINATE counts.** It must be pushed by your team and change files only in your own folder. The other team's NOMINATE commit does not unlock test.csv for you.
- **Seeds.** Tune with seed 42 only. Run seeds 43 and 44 only on the nominated configuration. Every reported final result uses all three.
- **notok definition.** Remove the literal string `<handle replaced>`, collapse whitespace, strip. This is exactly what compare/compare.py does.
- **Working files.** Read ARCHITECTURE.md, CHECKLIST.md and STATE.md in your folder, plus Part A of your team brief, at the start of every session. There is no BRIEF.md file.

## 2. Hardware (agreed 2026-11-12)

- **Development and pilots:** any hardware.
- **Final runs:** every number that goes into cost.json (training time, peak GPU memory, latency at batch size 1) is measured on an NVIDIA T4 from Kaggle or Colab, by both teams. This keeps the cost comparison and the latency tie-break fair.
- Write the platform in cost.json, for example `"hardware": "Kaggle T4"`.
- A T4 has no native bf16. Use float16 mixed precision.
- Free sessions disconnect and have usage limits. Save checkpoints every epoch and make training resumable.
- If a final run cannot be done on a T4, tell the lead before the FINAL commit and state it in cost.json and in the paper.

## 3. Still open: do not assume an answer

The lead will decide these before either team nominates, and will update this file.

- **Latency measurement protocol:** whether tokenization time counts, the number of warm-up posts, and how many posts are averaged. Until then, keep your latency script in your code folder so it can be re-run under the final protocol.
- **Threshold and p_pos:** whether a team may shift p_pos using a threshold tuned on val. Until then, submit the model's raw probabilities. compare.py scores at a threshold of 0.5.

## 4. Data facts from the lead's checks of train and val

- The SHA256 hashes of train.csv, val.csv and test.csv match SHA256SUMS.txt.
- Only 3.9% of usable train rows and 4.1% of usable val rows have two or more annotators. Test has three per post. Val labels are noisier than test labels, so small differences on val may be noise.
- The notok variant leaves no empty posts. It creates 27 repeated texts in train and 2 in val.
