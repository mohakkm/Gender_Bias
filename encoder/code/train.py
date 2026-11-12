"""Fine-tune one encoder on train, early-stop and score on val. Never reads test.csv.

    # pilot: 200 train rows, 1 epoch, timing extrapolated to a full run
    python encoder/code/train.py --model muril --variant notok --pilot 200 --out /kaggle/working/runs

    # one grid run (seed 42)
    python encoder/code/train.py --model muril --variant notok --lr 3e-5 --bs 32 --out /kaggle/working/runs

Checkpoints (weights) go to --out, OUTSIDE the repo. They are never committed.
Small outputs go to encoder/code/results/: runs.csv (one row per finished run) and val_preds/<run_id>.csv.
Resumable: if --out/<run_id>/last.pt exists the run continues from the last finished epoch.
max_len defaults to the token study's recommendation (results/token_study_maxlen.json).
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import MODELS, RESULTS, LANGS, evaluate, load_train_val, set_seed, texts

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--variant", required=True, choices=["tok", "notok"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--patience", type=int, default=2)
ap.add_argument("--max-len", type=int, default=0, help="0 = token study recommendation")
ap.add_argument("--warmup", type=float, default=0.10)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--class-weights", action="store_true", help="inverse-frequency weights; must be declared in STATE.md")
ap.add_argument("--pilot", type=int, default=0, help="train on N rows for 1 epoch, report timing, log nothing to runs.csv")
ap.add_argument("--save-best", action="store_true", help="keep the best model (save_pretrained). Use for the nominated config")
ap.add_argument("--out", required=True, help="directory for checkpoints, outside the repo")
ap.add_argument("--eval-bs", type=int, default=128)
a = ap.parse_args()

if a.max_len == 0:
    rec = json.loads((RESULTS / "token_study_maxlen.json").read_text(encoding="utf-8"))
    a.max_len = rec[a.model]["max_len"]

run_id = (f"{a.model}_{a.variant}_s{a.seed}_lr{a.lr:g}_bs{a.bs}_ml{a.max_len}"
          + ("_cw" if a.class_weights else "") + (f"_pilot{a.pilot}" if a.pilot else ""))
out = Path(a.out) / run_id
out.mkdir(parents=True, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fp16 = dev.type == "cuda"  # T4: float16 mixed precision, no bf16
print(f"run {run_id} on {torch.cuda.get_device_name(0) if fp16 else 'CPU'}")

set_seed(a.seed)
train, val = load_train_val()
if a.pilot:
    train = train.sample(a.pilot, random_state=a.seed).reset_index(drop=True)
    a.epochs = 1

tok = AutoTokenizer.from_pretrained(MODELS[a.model])


def encode(df):
    enc = tok(texts(df, a.variant), truncation=True, max_length=a.max_len)
    full = tok(texts(df, a.variant), truncation=False)["input_ids"]
    trunc_rate = float(np.mean([len(x) > a.max_len for x in full]))
    return [{"input_ids": i, "attention_mask": m} for i, m in zip(enc["input_ids"], enc["attention_mask"])], trunc_rate


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
        att[i, :len(b["attention_mask"])] = 1
    out = {"input_ids": ids, "attention_mask": att}
    if "y" in batch[0]:
        out["labels"] = torch.tensor([b["y"] for b in batch])
    return out


tr_items, tr_trunc = encode(train)
for it, y in zip(tr_items, train["y"]):
    it["y"] = int(y)
va_items, va_trunc = encode(val)
print(f"train {len(tr_items)} rows (truncated {tr_trunc:.2%}), val {len(va_items)} rows (truncated {va_trunc:.2%}), max_len {a.max_len}")

g = torch.Generator().manual_seed(a.seed)
tr_dl = DataLoader(tr_items, batch_size=a.bs, shuffle=True, collate_fn=collate, generator=g)
va_dl = DataLoader(va_items, batch_size=a.eval_bs, shuffle=False, collate_fn=collate)

model = AutoModelForSequenceClassification.from_pretrained(MODELS[a.model], num_labels=2).to(dev)
no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
params = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": a.wd},
          {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0}]
opt = torch.optim.AdamW(params, lr=a.lr)
total = len(tr_dl) * a.epochs
sched = get_linear_schedule_with_warmup(opt, math.ceil(a.warmup * total), total)
scaler = torch.amp.GradScaler("cuda", enabled=fp16)
if a.class_weights:
    pos = float(train["y"].mean())
    weight = torch.tensor([0.5 / (1 - pos), 0.5 / pos], dtype=torch.float, device=dev)
else:
    weight = None
loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

state = {"epoch": 0, "best": -1.0, "best_epoch": 0, "bad": 0, "train_sec": 0.0, "history": []}
last = out / "last.pt"
if last.exists() and not a.pilot:
    ck = torch.load(last, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    scaler.load_state_dict(ck["scaler"]); state = ck["state"]; g.set_state(ck["gen"])
    torch.set_rng_state(ck["rng"])
    if fp16 and ck.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    print(f"resumed after epoch {state['epoch']} (best val macro-F1 {state['best']:.4f})")


@torch.no_grad()
def predict(dl):
    model.eval()
    ps = []
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = model(**b).logits
        ps.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


if fp16:
    torch.cuda.reset_peak_memory_stats()
done = state["bad"] >= a.patience
while state["epoch"] < a.epochs and not done:
    model.train()
    t0, losses = time.time(), []
    for step, b in enumerate(tr_dl):
        b = {k: v.to(dev) for k, v in b.items()}
        labels = b.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = model(**b).logits
        loss = loss_fn(logits.float(), labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  epoch {state['epoch'] + 1} step {step}/{len(tr_dl)} loss {np.mean(losses[-100:]):.4f}")
    if fp16:
        torch.cuda.synchronize()
    sec = time.time() - t0
    state["train_sec"] += sec
    state["epoch"] += 1

    p = predict(va_dl)
    m = evaluate(val, p)
    f1 = m["all"]["macro_f1"]
    state["history"].append({"epoch": state["epoch"], "loss": float(np.mean(losses)), "val_macro_f1": f1})
    print(f"epoch {state['epoch']}: loss {np.mean(losses):.4f}, val macro-F1 {f1:.4f} "
          f"(en {m['en']['macro_f1']:.3f} hi {m['hi']['macro_f1']:.3f} ta {m['ta']['macro_f1']:.3f}), {sec / 60:.1f} min")

    if a.pilot:
        per_row = sec / len(tr_items)
        print(f"\nPILOT: {sec:.1f}s for {len(tr_items)} rows = {per_row * 1000:.1f} ms/row. "
              f"Full epoch on 17,290 rows ~ {per_row * 17290 / 60:.1f} min; 5 epochs ~ {per_row * 17290 * 5 / 60:.0f} min "
              f"(upper bound, early stopping usually ends sooner). Pilot val scores are meaningless.")
        if fp16:
            print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        raise SystemExit(0)

    if f1 > state["best"]:
        state.update(best=f1, best_epoch=state["epoch"], bad=0, best_metrics=m)
        np.save(out / "best_val_p.npy", p)
        if a.save_best:
            model.save_pretrained(out / "best"); tok.save_pretrained(out / "best")
    else:
        state["bad"] += 1
    done = state["bad"] >= a.patience
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "state": state, "gen": g.get_state(), "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if fp16 else None}, last)

# ---- log the finished run (val only)
p = np.load(out / "best_val_p.npy")
m = state["best_metrics"]
vp = RESULTS / "val_preds"
vp.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"id": val["id"], "p_pos": np.round(p, 6)}).to_csv(vp / f"{run_id}.csv", index=False)
row = {
    "run_id": run_id, "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "model": a.model,
    "checkpoint": MODELS[a.model], "variant": a.variant, "seed": a.seed, "lr": a.lr, "bs": a.bs, "max_len": a.max_len,
    "class_weights": a.class_weights, "epochs_run": state["epoch"], "best_epoch": state["best_epoch"],
    "val_macro_f1": round(m["all"]["macro_f1"], 4), "val_pos_f1": round(m["all"]["pos_f1"], 4),
    "val_precision": round(m["all"]["precision"], 4), "val_recall": round(m["all"]["recall"], 4),
    **{f"val_macro_f1_{l}": round(m[l]["macro_f1"], 4) for l in LANGS},
    "val_auroc": round(m["all"]["auroc"], 4), "train_trunc": round(tr_trunc, 4),
    "train_min": round(state["train_sec"] / 60, 1),
    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if fp16 else "",
    "gpu": torch.cuda.get_device_name(0) if fp16 else "cpu",
}
log = RESULTS / "runs.csv"
new = not log.exists()
with open(log, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(f"\nDONE {run_id}: best val macro-F1 {row['val_macro_f1']} at epoch {row['best_epoch']}. Logged to {log}")
if not a.save_best:
    last.unlink(missing_ok=True)  # free disk; the val predictions and metrics are kept
