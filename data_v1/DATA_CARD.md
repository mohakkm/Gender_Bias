# DATA_CARD.md — Uli core (data_v1 freeze)

This freeze contains only the Uli gendered-abuse tables. Hinglish and live comments are not included.

## Source and license

- Dataset: Uli gendered abuse (Arora et al. 2023), GitHub [tattle-made/uli_dataset](https://github.com/tattle-made/uli_dataset), used in the ICON 2023 shared task ([Vaidya et al.](https://arxiv.org/abs/2401.03677)).
- Paper: Arora, Arnav; Jinadoss, Maha; Arora, Cheshta; George, Denny; et al. 2023/2024. “The Uli Dataset: An Exercise in Experience Led Annotation of oGBV.” [arXiv:2311.09086](https://arxiv.org/abs/2311.09086), [ACL Anthology](https://aclanthology.org/2024.woah-1.16/).
- Files here: `train.csv`, `val.csv`, `test.csv` derived from `raw/uli/training/` and `raw/uli/testing/`.
- License: **CC BY 4.0**. Full text is in `raw/uli/LICENSE`. Human-readable terms: [creativecommons.org/licenses/by/4.0](https://creativecommons.org/licenses/by/4.0/).
- Attribution: this freeze copies and adapts the Uli dataset. Changes: (1) majority vote per question across annotators using only 0/1; (2) normalized-text dedupe inside each official split; (3) train-side drop of texts also in official test; (4) 10% val carve-out from cleaned train, seed 42, stratified by language and label_1. We do not claim endorsement by the licensors.

## Label questions (ICON 2023 shared-task wording)

Copied from Vaidya et al., *Overview of the 2023 ICON Shared Task on Gendered Abuse Detection in Indic Languages*:

- Label 1: Is the post a gendered abuse when directed at a person of marginalized gender?
- Label 2: Is the post a gendered abuse when it is not directed at a person of marginalized gender?
- Label 3: Does this post contain explicit/ aggressive language?

They are not mutually exclusive. Values in the source: 1 = yes, 0 = no, NL = assigned but not annotated, blank/NaN = not assigned. NL and blank are missing, not labels and not 0. Arora et al. (2023) state the same two gendered-abuse questions with the opposite numbering; this freeze follows the ICON 2023 numbering, which matches `question_1`/`l1` files.

## Aggregation

Each of label_1, label_2, label_3 is aggregated separately. Only cells `0`, `0.0`, `1`, `1.0` count as votes. Majority wins. An exact tie or zero valid votes yields an empty label, never 0. Fractional cells (`0.5`, `0.75`, `0.666…`) are ignored. Questions are never compared to each other.

## Official split and expected counts

Official (ICON/Uli): English 6,531 train / 1,107 test; Hindi 6,197 / 1,516; Tamil 6,779 / 1,135. Totals stated as en 7,638, hi 7,714, ta 7,914. Hindi 6,197 + 1,516 = 7,713, one below the stated 7,714. This freeze keeps official test assignment. Val is carved from cleaned train (not an official Uli split).

## Funnel (from work/build_report.txt)

See `work/build_report.txt` for every `step | rows in | rows out | reason` line. Summary of extra copies / drops versus official post counts:

| Reason | Split touched | en | hi | ta |
|---|---|---:|---:|---:|
| Intra-split normalized-text merge (extra copies removed; first kept) | official train | 8 | 0 | 1 |
| Intra-split normalized-text merge | official test | 0 | 2 | 0 |
| Normalized text also in official test (train only) | official train | 4 | 2 | 1 |
| Empty text | all | 0 | 0 | 0 |
| Hindi test unquoted newline joined to one post | official test | 0 | 0 extra posts (1,517 physical lines → 1,516) | 0 |
| Val carve-out (10% of cleaned train) | train → val | 652 | 620 | 678 |
| Near-dup char 3–5 gram TF-IDF cosine ≥ 0.9 vs val or test | freeze train only | 16 | 35 | 36 |

English gap of −12 versus 6,531 = 8 merged copies + 4 train posts that match test. Hindi test gap of −2 versus 1,516 = two case/spacing duplicates inside test. Hindi cleaned train is 6,196 (−1 versus 6,197) after adding one post that appears only in Hindi label-2/3 files (6,198 unique) then dropping two train/test leaks. Tamil cleaned train 6,777 (−2 versus 6,779) = one merge + one leak. After freeze QC, 87 further train rows were removed as near-duplicates of val or test (en 16, hi 35, ta 36). Test and val were not edited.

## Known limitations

- Most train labels are single-annotator (`n_ann_1 >= 2` is about 7.6% English train, 1.3% Hindi train, 5.3% Tamil train). Test is almost entirely multi-annotator.
- Seven fractional source cells were ignored, not mapped to 0 or 1 (six posts; one Tamil val post has two fractional cells). `ta_train_00597` is among them and was later dropped from freeze train by the near-duplicate check.
- Digit runs of length ≥ 8 were reviewed in `work/pii_digits_review.csv`. 23 phone-like runs (Indian mobile 10 digits starting 6–9, optional +91/0, packed 91XXXXXXXXXX, or STD+8-digit landline) were replaced with the literal token `<phone>`. 6 other runs (amounts, case/return ids) were left unchanged. By split×class: {('test', 'phone-like'): 3, ('train', 'other'): 6, ('train', 'phone-like'): 16, ('val', 'phone-like'): 4}. Handles remain the source token `<handle replaced>`.
- Hinglish offensiveness labels and unlabeled live comments are out of scope for this freeze.
- Train language mix is the Uli mix (en/hi/ta tweets), not live social-media traffic.

## Freeze extras

Fractional source cells (ignored in aggregation):
- en_train_03756 | split=train | language=en | label_3 | train_en_l3.csv en_a4=0.5
- hi_train_04164 | split=train | language=hi | label_1 | train_hi_l1.csv hi_a3=0.5
- hi_train_03837 | split=train | language=hi | label_2 | train_hi_l2.csv hi_a5=0.5
- ta_train_00597 | split=train | language=ta | label_1 | train_ta_l1.csv ta_a4=0.5 (later removed from freeze train by near-dup QC)
- ta_train_02092 | split=val | language=ta | label_1 | train_ta_l1.csv ta_a2=0.6666666666666666
- ta_train_02092 | split=val | language=ta | label_1 | train_ta_l1.csv ta_a4=0.75
- hi_test_01228 | split=test | language=hi | label_2 | test_hi_l2.csv hi_a1=0.5

Krippendorff's alpha (nominal) for label_1 on official test posts with exactly three valid 0/1 annotator votes: 0.461621 (units=3707, en=1107, hi=1514, ta=1086). NL/blank/fractions excluded.

Near-duplicate TF-IDF (char 3–5 grams, cosine ≥ 0.9): val-train pairs 23, test-train pairs 180, within-test pairs 27, train rows removed 87.
## Phone scrub
- train: 16 phone-like runs replaced with '<phone>' in 14 rows; 6 other runs unchanged
- val: 4 phone-like runs replaced with '<phone>' in 2 rows; 0 other runs unchanged
- test: 3 phone-like runs replaced with '<phone>' in 3 rows; 0 other runs unchanged
