"""FINAL stage: the nominated configuration on test. Run ONLY after the 'encoder: NOMINATE' commit is pushed.

    python encoder/code/final.py --model muril --lr 3e-5 --bs 32 --out /kaggle/working/final

Refuses to start unless (1) git history has a commit whose subject starts with 'encoder: NOMINATE' and names this
exact configuration (model, lr, bs, max_len tokens, e.g. 'encoder: NOMINATE muril lr3e-05 bs32 ml192'), and
(2) the GPU is a T4 (TEAM_AGREEMENTS section 2), unless --allow-any-gpu is given for a dry run.

For variant in (notok, tok) and seed in (42, 43, 44): train with train.py (tag final_t4, best model kept), reload
the best model, re-predict val and check it matches the logged best-epoch macro-F1 (row-order check), then predict
test in file order. Writes encoder/final/final_predictions.csv (id, variant, seed, p_pos; raw p_pos, no
calibration, TEAM_AGREEMENTS 3b), encoder/final/cost.json (latency per 3a on the notok seed-42 model) and
encoder/code/results/final_log.json. Then runs compare/validate_submission.py encoder.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import CODE, ENC, MODELS, RESULTS, ROOT, cmp, evaluate, load_train_val, texts, write_json
from latency import measure

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--bs", type=int, required=True)
ap.add_argument("--max-len", type=int, default=0)
ap.add_argument("--out", required=True, help="model checkpoints, outside the repo")
ap.add_argument("--allow-any-gpu", action="store_true", help="dry run off a T4; numbers must not be submitted")
a = ap.parse_args()
if a.max_len == 0:
    a.max_len = json.loads((RESULTS / "token_study_maxlen.json").read_text(encoding="utf-8"))[a.model]["max_len"]

# ---- gate 1: pushed NOMINATE commit naming this configuration
want = [a.model, f"lr{a.lr:g}", f"bs{a.bs}", f"ml{a.max_len}"]
log = subprocess.run(["git", "-C", str(ROOT), "log", "--all", "--format=%H\t%s"], capture_output=True, text=True,
                     encoding="utf-8").stdout.splitlines()
noms = [l.split("\t", 1) for l in log if "\t" in l and l.split("\t", 1)[1].startswith("encoder: NOMINATE")]
match = [(h, s) for h, s in noms if all(w.lower() in s.lower().split() for w in want)]
if not match:
    sys.exit(f"REFUSED: no 'encoder: NOMINATE' commit naming {' '.join(want)}. Found: {[s for _, s in noms]}")
print(f"NOMINATE commit {match[0][0][:10]}: {match[0][1]}")

# ---- gate 2: T4
dev = torch.device("cuda")
gpu = torch.cuda.get_device_name(0)
if "T4" not in gpu and not a.allow_any_gpu:
    sys.exit(f"REFUSED: final runs must be on a T4 (found {gpu}). Use --allow-any-gpu only for a dry run.")

train, val = load_train_val()
test = cmp.prep(cmp.read_split("test"))
assert len(test) == 3756
out = Path(a.out)
runs, preds = [], []


@torch.no_grad()
def predict(model, tok, df, variant):
    enc = tok(texts(df, variant), truncation=True, max_length=a.max_len)
    items = [{"input_ids": i, "attention_mask": m} for i, m in zip(enc["input_ids"], enc["attention_mask"])]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, :len(b["input_ids"])] = torch.tensor(b["input_ids"])
            att[i, :len(b["input_ids"])] = 1
        return {"input_ids": ids, "attention_mask": att}

    model.eval()
    ps = []
    for b in DataLoader(items, batch_size=128, shuffle=False, collate_fn=collate):  # shuffle=False keeps file order
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**{k: v.to(dev) for k, v in b.items()}).logits
        ps.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(ps)


cost = {}
for variant in ("notok", "tok"):
    for seed in (42, 43, 44):
        cmd = [sys.executable, str(CODE / "train.py"), "--model", a.model, "--variant", variant, "--seed", str(seed),
               "--lr", str(a.lr), "--bs", str(a.bs), "--max-len", str(a.max_len), "--save-best", "--tag", "final_t4",
               "--out", str(out)]
        print("\n>>>", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode:
            sys.exit(f"training failed: {variant} seed {seed}")
        run_id = f"{a.model}_{variant}_s{seed}_lr{a.lr:g}_bs{a.bs}_ml{a.max_len}_final_t4"
        row = pd.read_csv(RESULTS / "runs.csv").query("run_id == @run_id").iloc[-1]
        best = out / run_id / "best"
        tok = AutoTokenizer.from_pretrained(best)
        model = AutoModelForSequenceClassification.from_pretrained(best).to(dev)

        pv = predict(model, tok, val, variant)
        f1v = evaluate(val, pv)["all"]["macro_f1"]
        ok = abs(f1v - row["val_macro_f1"]) < 0.002
        print(f"val check {run_id}: re-predicted {f1v:.4f} vs best epoch {row['val_macro_f1']:.4f} -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit("STOP: val re-prediction does not match the best epoch (row order or model loading problem)")

        t0 = time.time()
        pt = predict(model, tok, test, variant)
        preds.append(pd.DataFrame({"id": test["id"], "variant": variant, "seed": seed, "p_pos": np.round(pt, 6)}))
        runs.append({"run_id": run_id, "variant": variant, "seed": seed, "val_macro_f1": round(f1v, 4),
                     "best_epoch": int(row["best_epoch"]), "train_min": float(row["train_min"]),
                     "peak_gpu_gb": float(row["peak_gpu_gb"]), "test_predict_sec": round(time.time() - t0, 1),
                     "gpu": gpu, "test_opened_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M")})

        if variant == "notok" and seed == 42:
            cost["trainable_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            weights = [f for f in best.iterdir() if f.suffix in (".safetensors", ".bin")]
            cost["model_size_mb"] = round(sum(f.stat().st_size for f in weights) / 2**20, 1)
            cost["train_time_min"] = float(row["train_min"])
            cost.update(measure(model, tok, a.max_len, dev))
        del model
        torch.cuda.empty_cache()
        for f in sorted(best.glob("*")):
            if not (variant == "notok" and seed == 42):
                f.unlink()  # keep only the latency model on disk

fin = ENC / "final"
fin.mkdir(exist_ok=True)
pd.concat(preds).to_csv(fin / "final_predictions.csv", index=False, encoding="utf-8")
r = pd.DataFrame(runs)
cost_json = {
    "team": "encoder",
    "model": f"{MODELS[a.model]} (full fine-tune), lr {a.lr:g}, batch {a.bs}, max_len {a.max_len}",
    "formulation": "sequence classification head, cross-entropy, early stopping on val macro-F1 (patience 2, max 5 epochs)",
    "hardware": f"Kaggle {gpu} (one GPU), fp16 autocast",
    "train_time_min": cost["train_time_min"],
    "peak_gpu_mem_gb": float(r["peak_gpu_gb"].max()),
    "trainable_params": cost["trainable_params"],
    "model_size_mb": cost["model_size_mb"],
    "latency_ms_per_post_bs1": cost["latency_ms_per_post_bs1"],
    "notes": (f"train_time_min and latency from the notok seed 42 run; train time mean over 6 runs "
              f"{r['train_min'].mean():.1f} min; peak memory is the max over 6 runs; model_size_mb in MiB (2^20 bytes) "
              f"of the weight files; latency per TEAM_AGREEMENTS 3a, repeat means {cost['repeat_means_ms']} ms"),
}
write_json(fin / "cost.json", cost_json)
summary = r.groupby("variant")["val_macro_f1"].agg(["mean", "std"]).round(4).to_dict("index")
write_json(RESULTS / "final_log.json", {"nominate_commit": match[0][0], "nominate_subject": match[0][1],
                                       "runs": runs, "val_macro_f1_mean_std": summary, "cost": cost_json})
print(json.dumps(cost_json, indent=2), "\nval macro-F1 over seeds:", summary)
subprocess.run([sys.executable, str(ROOT / "compare" / "validate_submission.py"), "encoder"])
