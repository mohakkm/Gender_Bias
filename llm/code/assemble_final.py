"""Build llm/final/final_predictions.csv and llm/final/cost.json from the 6 final runs
(nominated configuration x variant tok/notok x seed 42/43/44), then run the validator yourself:

    python llm/code/assemble_final.py --runs <run dir> [<run dir> ...] --base-size-mb <MB>
    python compare/validate_submission.py llm

Each run dir holds metrics.json and test_predictions.csv (from train_cls.py / train_gen.py --predict-test).
Refuses on: missing or duplicate (variant, seed), ids that differ from data_v1/test.csv, p_pos outside
[0, 1], runs with different hyperparameters, or a cost run whose latency did not follow TEAM_AGREEMENTS 3a.
Reading test ids goes through common.read_split, so it only works after the NOMINATE commit.
"""
import common  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS, SEEDS = ("tok", "notok"), common.SEEDS
PER_RUN = {"variant", "seed", "predict_test", "latency_posts", "resume", "tag"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", nargs="+", required=True)
    a.add_argument("--base-size-mb", type=float, required=True,
                   help="size on disk of the base model checkpoint the adapters are applied to")
    a.add_argument("--hardware", default="Kaggle T4")
    args = a.parse_args()

    test_ids = common.read_split("test")["id"].tolist()  # gated by the NOMINATE commit
    runs = {}
    for d in map(Path, args.runs):
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        key = (m["args"]["variant"], int(m["args"]["seed"]))
        if key in runs:
            raise SystemExit(f"duplicate run for {key}: {d}")
        runs[key] = (d, m)
    missing = [(v, s) for v in VARIANTS for s in SEEDS if (v, s) not in runs]
    if missing or len(runs) != 6:
        raise SystemExit(f"need exactly the 6 (variant, seed) runs; missing {missing}, have {sorted(runs)}")

    configs = {json.dumps({k: v for k, v in m["args"].items() if k not in PER_RUN}, sort_keys=True)
               for _, m in runs.values()}
    if len(configs) != 1:
        raise SystemExit("runs do not share one configuration:\n" + "\n".join(configs))

    frames = []
    for (v, s), (d, m) in sorted(runs.items()):
        p = pd.read_csv(d / "test_predictions.csv", dtype={"id": str}, keep_default_na=False, encoding="utf-8")
        if list(p.columns) != ["id", "p_pos"] or len(p) != len(test_ids) or set(p["id"]) != set(test_ids):
            raise SystemExit(f"{d}: test_predictions.csv does not cover exactly the test ids")
        pp = pd.to_numeric(p["p_pos"], errors="coerce")
        if pp.isna().any() or (pp < 0).any() or (pp > 1).any():
            raise SystemExit(f"{d}: p_pos missing or outside [0, 1]")
        p = p.set_index("id").loc[test_ids].reset_index()
        frames.append(pd.DataFrame({"id": p["id"], "variant": v, "seed": s, "p_pos": p["p_pos"].astype(float).to_numpy()}))
        print(f"{v:5s} seed {s}: {len(p)} test rows | predicted positive rate {(p['p_pos'].astype(float) >= 0.5).mean():.1%} "
              f"| val macro-F1 {m['val']['all']['macro_f1']:.3f}")

    final = common.TEAM / "final"
    final.mkdir(exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(final / "final_predictions.csv", index=False, encoding="utf-8")

    d, m = runs[("notok", 42)]
    if m.get("latency_protocol") != common.LATENCY_PROTOCOL or m.get("latency_ms_per_post_bs1_val") is None:
        raise SystemExit(f"{d}: latency was not measured with the agreed protocol (TEAM_AGREEMENTS.md 3a)")
    ar = m["args"]
    short = ar["model"].split("/")[-1]
    cost = {
        "team": "llm",
        "model": f"{short} (4-bit nf4) + LoRA r{ar['rank']} {ar['targets']}" if not ar.get("no_4bit") else f"{short} + LoRA r{ar['rank']}",
        "formulation": m["formulation"],
        "hardware": args.hardware,
        "train_time_min": m["train_time_min"],
        "peak_gpu_mem_gb": m["peak_gpu_mem_gb"],
        "trainable_params": m["trainable_params"],
        "model_size_mb": round(args.base_size_mb + m["adapter_size_mb"], 1),
        "latency_ms_per_post_bs1": m["latency_ms_per_post_bs1_val"],
        "notes": (f"cost figures from the notok seed 42 run; model_size_mb = base checkpoint {args.base_size_mb} MB "
                  f"+ adapter {m['adapter_size_mb']} MB; latency per TEAM_AGREEMENTS.md 3a, repeat means "
                  f"{m.get('latency_repeat_means_ms')} ms; train time mean over 6 runs "
                  f"{np.mean([r[1]['train_time_min'] for r in runs.values()]):.1f} min"),
    }
    (final / "cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    print(f"wrote {final / 'final_predictions.csv'} and {final / 'cost.json'}\n"
          f"next: python compare/validate_submission.py llm")


if __name__ == "__main__":
    main()
