"""Sanity checks on train and val only. Never reads test.csv.

    python llm/code/data_check.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "<handle replaced>"


def notok(text: pd.Series) -> pd.Series:
    return (text.str.replace(TOKEN, " ", regex=False)
            .str.replace(r"\s+", " ", regex=True).str.strip())


for split in ("train", "val"):
    d = pd.read_csv(ROOT / "data_v1" / f"{split}.csv", dtype=str, keep_default_na=False, encoding="utf-8")
    print(f"\n=== {split}: {len(d)} rows")
    print("language:", d["language"].value_counts().to_dict())
    print("label_1:", d["label_1"].value_counts().to_dict())
    u = d[d["label_1"].isin(["0", "1"])].copy()
    print("usable rows:", len(u))
    u["y"] = u["label_1"].astype(int)
    u["has_token"] = u["text"].str.contains(TOKEN, regex=False)
    u["notok"] = notok(u["text"])
    print("empty after notok:", int((u["notok"] == "").sum()),
          "| 3 chars or fewer:", int((u["notok"].str.len() <= 3).sum()))
    print("distinct text repeats, tok:", int(u["text"].duplicated().sum()),
          "| notok:", int(u["notok"].duplicated().sum()))
    print("n_ann_1 >= 2 share:", round(float((pd.to_numeric(u["n_ann_1"]) >= 2).mean()), 3))
    g = u.groupby("language").agg(n=("y", "size"), pos_rate=("y", "mean"), token_share=("has_token", "mean"))
    print(g.round(3).to_string())
    print(u.groupby(["language", "has_token"])["y"].mean().round(3).to_string())
    print(u.assign(chars=u["notok"].str.len()).groupby("language")["chars"]
          .describe(percentiles=[.5, .95, .99]).round(0).to_string())
