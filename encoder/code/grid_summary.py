"""Summarise the seed-42 grid for the NOMINATE decision (val only).

    python encoder/code/grid_summary.py

Ranks configurations by val macro-F1 on notok (the nomination rule, brief A4 item 1), shows the matching tok
run, per-language scores and the margin over the TF-IDF floor. Writes results/grid_summary.md.
"""
import pandas as pd

from common import RESULTS, cmp

TFIDF = {"all": 0.747, "en": 0.724, "hi": 0.681, "ta": 0.764}  # compare.py --baselines-val, notok

r = pd.read_csv(RESULTS / "runs.csv")
r = r[(r["seed"] == 42) & (~r["run_id"].str.contains("pilot"))].drop_duplicates("run_id", keep="last")
key = ["model", "lr", "bs", "max_len", "class_weights"]
n = r[r["variant"] == "notok"].set_index(key)
t = r[r["variant"] == "tok"].set_index(key)
s = n[["val_macro_f1", "val_macro_f1_en", "val_macro_f1_hi", "val_macro_f1_ta", "val_pos_f1", "best_epoch", "train_min"]].copy()
s["tok_macro_f1"] = t["val_macro_f1"].reindex(s.index)
s["vs_tfidf"] = (s["val_macro_f1"] - TFIDF["all"]).round(4)
s = s.sort_values("val_macro_f1", ascending=False).reset_index()

best_per_model = s.groupby("model").head(1)
lines = ["# Seed-42 grid summary (val, notok ranks; nomination rule)", "",
         f"Runs logged: {len(r)} ({len(n)} notok, {len(t)} tok). TF-IDF floor on val: {TFIDF['all']}", "",
         "## Best configuration per model", "", cmp.md_table(best_per_model), "",
         "## All notok configurations", "", cmp.md_table(s), "",
         "Caveat: one seed per configuration. Gaps under about 0.01 between configurations are within seed noise; "
         "prefer the simpler or faster configuration when the top results are that close."]
out = RESULTS / "grid_summary.md"
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
