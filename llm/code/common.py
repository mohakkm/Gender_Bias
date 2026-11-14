"""Shared helpers for the LLM team: data loading, variants, metrics and the test-set gate."""
import os

os.environ.setdefault("USE_TF", "0")  # the system TensorFlow slows every import and is never used
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

CODE = Path(__file__).resolve().parent
TEAM = CODE.parent
ROOT = TEAM.parent
DATA = ROOT / "data_v1"
OUTPUTS = TEAM / "outputs"  # git-ignored: checkpoints, adapters, predictions
TOKEN = "<handle replaced>"
SEEDS = (42, 43, 44)
LANGS = ("en", "hi", "ta")
MAX_LEN = 317  # covers 99% of train in every language for Qwen3 (see STATE.md)
NOMINATE_PREFIX = "llm: NOMINATE"


def notok(text: pd.Series) -> pd.Series:
    """Same rule as compare/compare.py: drop the token, collapse whitespace, strip."""
    return (text.str.replace(TOKEN, " ", regex=False)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def nomination_commit():
    """Return the sha of our pushed NOMINATE commit, or None.

    Rules from TEAM_AGREEMENTS.md: the subject starts with 'llm: NOMINATE', the commit is on
    origin/main (pushed), and it changes files only inside llm/.
    """
    if not (ROOT / ".git").exists():
        # Kaggle has no git checkout. kaggle/build_job.py --final only builds after checking the pushed
        # NOMINATE commit locally with this same function, and embeds its sha as SERI_NOMINATE_SHA.
        return os.environ.get("SERI_NOMINATE_SHA") or None

    def git(*args):
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout

    git("fetch", "--quiet", "origin")
    for line in git("log", "origin/main", "--format=%H%x09%s", "--", "llm").splitlines():
        sha, subject = line.split("\t", 1)
        if subject.startswith(NOMINATE_PREFIX):
            files = [f for f in git("show", "--name-only", "--format=", sha).splitlines() if f]
            if files and all(f.startswith("llm/") for f in files):
                return sha
    return None


def read_split(name: str) -> pd.DataFrame:
    if name == "test":
        sha = nomination_commit()
        if sha is None:
            raise SystemExit("REFUSED: no pushed 'llm: NOMINATE' commit touching only llm/ was found. "
                             "test.csv stays closed until then.")
        print(f"test.csv opened under NOMINATE commit {sha[:10]}. Log this run in STATE.md (test-access log).")
    return pd.read_csv(DATA / f"{name}.csv", dtype=str, keep_default_na=False, encoding="utf-8")


def load(name: str, variant: str) -> pd.DataFrame:
    """Rows with a usable label_1 (test keeps every row) and an 'input' column for the variant."""
    if variant not in ("tok", "notok"):
        raise ValueError(variant)
    d = read_split(name)
    if name != "test":
        d = d[d["label_1"].isin(["0", "1"])].copy()
        d["y"] = d["label_1"].astype(int)
    d["input"] = d["text"] if variant == "tok" else notok(d["text"])
    return d.reset_index(drop=True)


def scores(y, p, threshold=0.5) -> dict:
    """Macro-F1 (headline, as in compare.py) plus supporting metrics."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    out = {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
           "pos_f1": float(f1_score(y, pred, zero_division=0)),
           "accuracy": float((pred == y).mean()), "n": int(len(y))}
    if len(np.unique(y)) == 2 and len(np.unique(p)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    return out


def report(df: pd.DataFrame, p, threshold=0.5) -> dict:
    """Pooled and per-language scores for a labelled frame (train or val only)."""
    res = {"all": scores(df["y"], p, threshold)}
    p = np.asarray(p, float)
    for lang in LANGS:
        m = (df["language"] == lang).to_numpy()
        res[lang] = scores(df["y"].to_numpy()[m], p[m], threshold)
    return res


LATENCY_POSTS, LATENCY_WARMUP, LATENCY_REPEATS = 500, 20, 3
LATENCY_PROTOCOL = ("TEAM_AGREEMENTS.md 3a: first 500 rows of val.csv in file order, notok text, batch size 1, "
                    "raw text to p_pos timed, 20 untimed warm-up posts, median of 3 repeated means")


def latency_texts(n=LATENCY_POSTS):
    """First n rows of val.csv in file order (empty labels included), notok text."""
    return notok(read_split("val")["text"]).head(n).tolist()


def measure_latency(predict_one, texts, warmup=LATENCY_WARMUP, repeats=LATENCY_REPEATS):
    """Median over `repeats` of the mean ms per post. predict_one(text) must run raw text -> p_pos."""
    import time

    import torch
    cuda = torch.cuda.is_available()
    for t in texts[:warmup]:
        predict_one(t)
    means = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for t in texts:
            predict_one(t)
        if cuda:
            torch.cuda.synchronize()
        means.append(1000 * (time.perf_counter() - start) / len(texts))
    return float(np.median(means)), [round(m, 2) for m in means]


def fmt(res: dict) -> str:
    return " | ".join(f"{k} {v['macro_f1']:.3f}" for k, v in res.items())
