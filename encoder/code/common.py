"""Shared helpers for the encoder team. Data loading, the notok variant and the metrics are imported from
../../compare/compare.py so our validation numbers are computed exactly the way the lead scores test.
Never reads test.csv (load_train_val only touches train and val)."""
import json
import random
import sys
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
ENC = CODE.parent
ROOT = ENC.parent
RESULTS = CODE / "results"
sys.path.insert(0, str(ROOT / "compare"))
import compare as cmp  # noqa: E402  (read-only use of the lead's code)

LANGS = cmp.LANGS
TOKEN = cmp.TOKEN

# Exact checkpoints. Pin the revision in STATE.md after the first download (snapshot hash).
MODELS = {
    "indicbert": "ai4bharat/IndicBERTv2-MLM-only",
    "muril": "google/muril-base-cased",
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
}


def load_train_val():
    """Usable rows only (empty label_1 dropped). Adds y, agree, text_notok, exactly as compare.py does."""
    train = cmp.prep(cmp.read_split("train"))
    val = cmp.prep(cmp.read_split("val"))
    assert len(train) == 17290 and len(val) == 1931, (len(train), len(val))
    return train, val


def texts(df, variant):
    if variant == "tok":
        return df["text"].tolist()
    if variant == "notok":
        return df["text_notok"].tolist()
    raise ValueError(variant)


def evaluate(df, p_pos):
    """Pooled and per-language metrics at threshold 0.5, same functions as compare.py."""
    y = df["y"].to_numpy()
    out = {"all": cmp.metrics(y, p_pos)}
    for lang in LANGS:
        m = (df["language"] == lang).to_numpy()
        out[lang] = cmp.metrics(y[m], p_pos[m])
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
