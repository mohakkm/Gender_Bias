# Team context (encoder team)

Project: SERI research comparing LLM (LoRA/QLoRA) and encoder (IndicBERT/mBERT/MuRIL) models for gendered abuse detection (target label_1, English/Hindi/Tamil) on the frozen Uli data in ../data_v1/.

Read first, every session: ARCHITECTURE.md, CHECKLIST.md, STATE.md and BRIEF.md in this folder. Update CHECKLIST.md and STATE.md before the session ends.

Rules:
- Edit only files in this folder. Never modify ../data_v1/ or ../compare/, and never touch the other team's folder.
- Tune on train and val only. Do not run anything on ../data_v1/test.csv until a git commit whose message starts with NOMINATE exists.
- Every experiment runs in two variants: tok (text as is) and notok (remove the literal '<handle replaced>', collapse whitespace). Seeds 42, 43, 44.
- Final output: final/final_predictions.csv (id, variant, seed, p_pos) and final/cost.json. Check with python ../compare/validate_submission.py encoder.
- Commit routine: git pull, git add <this folder>, git status, commit with prefix 'encoder:', git pull, git push. Never force-push. Never commit weights, keys or .env files.
- Be blunt about problems. Tell me when a result looks too good or a rule is at risk.
