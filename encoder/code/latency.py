"""Inference latency exactly as TEAM_AGREEMENTS.md section 3a (the tie-break number).

    python encoder/code/latency.py --model-dir /kaggle/working/final/<run_id>/best --max-len 192

- One T4, the nominated model as used for final_predictions.csv (fp16 autocast, same max length).
- Posts: first 500 rows of data_v1/val.csv in file order, notok text, rows with empty label_1 INCLUDED.
- Batch size 1. Timed: raw text -> tokenize -> to GPU -> forward -> softmax -> p_pos. Loading not timed.
- 20 warm-up posts untimed. torch.cuda.synchronize() before starting and before stopping the clock.
- Mean ms/post over the 500, repeated 3 times, report the median of the 3 means.
"""
import argparse
import statistics
import time

import torch

from common import ROOT, TOKEN, cmp  # noqa: F401


def notok(t):
    import re
    return re.sub(r"\s+", " ", t.replace(TOKEN, " ")).strip()


def measure(model, tok, max_len, device, n_posts=500, warmup=20, repeats=3):
    val = cmp.read_split("val")  # raw file order, empty labels kept
    posts = [notok(t) for t in val["text"].tolist()[:n_posts]]
    assert len(posts) == n_posts
    fp16 = device.type == "cuda"
    model.eval()

    @torch.no_grad()
    def one(text):
        enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = model(**enc).logits
        return torch.softmax(logits.float(), -1)[0, 1].item()  # .item() also waits for the GPU

    for t in posts[:warmup]:
        one(t)
    means = []
    for _ in range(repeats):
        if fp16:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for t in posts:
            one(t)
        if fp16:
            torch.cuda.synchronize()
        means.append((time.perf_counter() - t0) * 1000 / n_posts)
    return {"latency_ms_per_post_bs1": round(statistics.median(means), 3),
            "repeat_means_ms": [round(m, 3) for m in means],
            "gpu": torch.cuda.get_device_name(0) if fp16 else "cpu"}


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--max-len", type=int, default=192)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(a.model_dir).to(dev)
    print(measure(model, tok, a.max_len, dev))
