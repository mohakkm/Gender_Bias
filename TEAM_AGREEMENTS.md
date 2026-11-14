# Team agreements (both teams)

Maintained by the lead. Read this file together with your team brief (v2, 10 Oct 2026). Where the two disagree, this file wins because it is newer. Teams never edit this file. Ask the lead for changes.

Last updated: 2026-11-14 (latency protocol and p_pos rule decided, section 3)

## 1. Corrections to the v2 briefs

- **NOMINATE commit format.** The first line of the message starts with `<team>: NOMINATE`, for example `encoder: NOMINATE MuRIL lr3e-5 bs32` or `llm: NOMINATE F1 <model> r16 lr2e-4`. This matches the commit prefix rule and the brief's own examples. The sentence "a commit whose message starts with NOMINATE" in the briefs is replaced by this.
- **Only your own NOMINATE counts.** It must be pushed by your team and change files only in your own folder. The other team's NOMINATE commit does not unlock test.csv for you.
- **Seeds.** Tune with seed 42 only. Run seeds 43 and 44 only on the nominated configuration. Every reported final result uses all three.
- **notok definition.** Remove the literal string `<handle replaced>`, collapse whitespace, strip. This is exactly what compare/compare.py does.
- **Git setup (one time, every person, every clone).** Run `git config pull.rebase true` and `git config rebase.autoStash true`. The brief lists only the first. On its own it makes `git pull` fail with "cannot pull with rebase: You have unstaged changes" whenever you have unsaved edits, which is exactly when the commit routine starts. The second setting sets your edits aside during the pull and puts them back afterwards. If they conflict with someone else's change, git says so: stop and tell the lead.
- **Working files.** Read ARCHITECTURE.md, CHECKLIST.md and STATE.md in your folder, plus Part A of your team brief, at the start of every session. There is no BRIEF.md file.

## 2. Hardware (agreed 2026-11-12)

- **Development and pilots:** any hardware.
- **Final runs:** every number that goes into cost.json (training time, peak GPU memory, latency at batch size 1) is measured on an NVIDIA T4 from Kaggle or Colab, by both teams. This keeps the cost comparison and the latency tie-break fair.
- Write the platform in cost.json, for example `"hardware": "Kaggle T4"`.
- A T4 has no native bf16. Use float16 mixed precision.
- Free sessions disconnect and have usage limits. Save checkpoints every epoch and make training resumable.
- If a final run cannot be done on a T4, tell the lead before the FINAL commit and state it in cost.json and in the paper.

## 3. Latency and p_pos rules (decided 2026-11-14)

These apply to both teams from now on and must not change after results are seen.

### 3a. Latency protocol (the tie-break number, `latency_ms_per_post_bs1` in cost.json)

- **Hardware:** one NVIDIA T4 on Kaggle or Colab, single GPU visible.
- **Model:** the nominated model exactly as used for final_predictions.csv (same precision, quantization, adapters and max length).
- **Posts:** the first 500 rows of data_v1/val.csv in file order, notok text (rows with an empty label_1 are included). Never test posts.
- **Method:** batch size 1, one post at a time. The timed part is everything from the raw text to p_pos: building any prompt, tokenizing, moving to the GPU, the forward pass and the softmax. Model loading is not timed.
- **Warm-up:** run 20 posts first, untimed.
- **Timing:** call torch.cuda.synchronize() before starting and before stopping the clock. Time all 500 posts, take the mean ms per post, repeat 3 times, and report the median of the 3 means.
- **Script:** commit the latency script in your code folder so the lead can re-run it.

### 3b. p_pos rule

- final_predictions.csv holds the model's own probability for the positive class, with no post-hoc shifting, rescaling, calibration or threshold folded in. compare.py scores at a threshold of 0.5.
- Class weights or other training-time choices are allowed but must be declared in STATE.md and in the paper.
- A threshold tuned on val may be reported separately, clearly labelled, and is never used to pick the winner.

## 4. Data facts from the lead's checks of train and val

- The SHA256 hashes of train.csv, val.csv and test.csv match SHA256SUMS.txt.
- Only 3.9% of usable train rows and 4.1% of usable val rows have two or more annotators. Test has three per post. Val labels are noisier than test labels, so small differences on val may be noise.
- The notok variant leaves no empty posts. It creates 27 repeated texts in train and 2 in val.
