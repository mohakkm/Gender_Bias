"""Run the brief's small grid (B2) for one or more models, seed 42, skipping runs already in results/runs.csv.
Safe to re-run after a disconnect: finished runs are skipped, an interrupted run resumes from its last epoch.

    python encoder/code/grid.py --models muril indicbert --out /kaggle/working/runs
    python encoder/code/grid.py --models mbert --variants notok --lrs 3e-5 --bss 32 --out ...   (subset)
"""
import argparse
import itertools
import json
import subprocess
import sys

import pandas as pd

from common import CODE, MODELS, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
ap.add_argument("--variants", nargs="+", default=["notok", "tok"])
ap.add_argument("--lrs", nargs="+", type=float, default=[2e-5, 3e-5, 5e-5])
ap.add_argument("--bss", nargs="+", type=int, default=[16, 32])
ap.add_argument("--out", required=True)
a = ap.parse_args()

maxlen = json.loads((RESULTS / "token_study_maxlen.json").read_text(encoding="utf-8"))
log = RESULTS / "runs.csv"
done = set(pd.read_csv(log)["run_id"]) if log.exists() else set()
jobs = list(itertools.product(a.models, a.variants, a.lrs, a.bss))
for i, (m, v, lr, bs) in enumerate(jobs, 1):
    rid = f"{m}_{v}_s42_lr{lr:g}_bs{bs}_ml{maxlen[m]['max_len']}"
    if rid in done:
        print(f"[{i}/{len(jobs)}] skip {rid} (already logged)")
        continue
    print(f"[{i}/{len(jobs)}] run {rid}", flush=True)
    r = subprocess.run([sys.executable, str(CODE / "train.py"), "--model", m, "--variant", v, "--seed", "42",
                        "--lr", str(lr), "--bs", str(bs), "--out", a.out])
    if r.returncode:
        sys.exit(f"stopped: {rid} failed (exit {r.returncode})")
