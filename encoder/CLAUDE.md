# Team context (encoder team)

Project: SERI research comparing LLM (LoRA/QLoRA) and encoder (IndicBERT/mBERT/MuRIL) models for gendered abuse detection (target label_1, English/Hindi/Tamil) on the frozen Uli data in ../data_v1/.

Read first, every session: ARCHITECTURE.md, CHECKLIST.md and STATE.md in this folder, plus Part A of the team brief (a Word file, not in the repo; ask the lead if you do not have it). Update CHECKLIST.md and STATE.md before the session ends.

Rules:
- Edit only files in this folder. Never modify ../data_v1/ or ../compare/, and never touch the other team's folder.
- Tune on train and val only. Do not run anything on ../data_v1/test.csv until your team has pushed a commit whose message starts with exactly 'encoder: NOMINATE' and that changes files in this folder and nowhere else. The other team's NOMINATE commit does not count.
- Every experiment runs in two variants: tok (text as is) and notok (remove the literal '<handle replaced>', collapse whitespace, strip). Tune on seed 42 only. Run seeds 43 and 44 only on the nominated configuration.
- Final output: final/final_predictions.csv (id, variant, seed, p_pos) and final/cost.json. Check with python ../compare/validate_submission.py encoder.
- Commit routine: git pull, git add <this folder>, git status, commit with prefix 'encoder:', git pull, git push. Never force-push. Never commit weights, keys or .env files.
- Be blunt about problems. Tell me when a result looks too good or a rule is at risk.
