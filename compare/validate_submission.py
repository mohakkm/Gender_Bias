"""
validate_submission.py -- checks a team's final submission. Changes nothing.

    python compare/validate_submission.py llm
    python compare/validate_submission.py encoder

It must print only PASS lines before you make your FINAL commit.
Files checked: <team>/final/final_predictions.csv and <team>/final/cost.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {"tok", "notok"}
SEEDS = {42, 43, 44}
PRED_COLS = ["id", "variant", "seed", "p_pos"]
COST_TEXT = ["team", "model", "formulation", "hardware"]
COST_NUM = ["train_time_min", "peak_gpu_mem_gb", "trainable_params", "model_size_mb", "latency_ms_per_post_bs1"]


def check(team):
    """Return (ok, messages)."""
    msgs, state = [], {"ok": True}

    def rec(cond, text):
        msgs.append(("PASS  " if cond else "FAIL  ") + text)
        state["ok"] = state["ok"] and bool(cond)

    test_path = ROOT / "data_v1" / "test.csv"
    pred_path = ROOT / team / "final" / "final_predictions.csv"
    cost_path = ROOT / team / "final" / "cost.json"
    rec(test_path.exists(), "data_v1/test.csv found")
    rec(pred_path.exists(), f"{team}/final/final_predictions.csv found")
    rec(cost_path.exists(), f"{team}/final/cost.json found")
    if not (test_path.exists() and pred_path.exists()):
        return False, msgs

    test_ids = pd.read_csv(test_path, dtype=str, keep_default_na=False, encoding="utf-8")["id"]
    p = pd.read_csv(pred_path, dtype=str, keep_default_na=False, encoding="utf-8")
    rec(list(p.columns) == PRED_COLS, f"columns are exactly {PRED_COLS} (found {list(p.columns)})")
    if not set(PRED_COLS) <= set(p.columns):
        return False, msgs

    pp = pd.to_numeric(p["p_pos"], errors="coerce")
    rec(bool(pp.notna().all()), "p_pos is numeric in every row")
    rec(bool(((pp >= 0) & (pp <= 1)).all()), "p_pos is between 0 and 1 in every row")
    rec(set(p["variant"]) == VARIANTS, f"variants are exactly tok and notok (found {sorted(set(p['variant']))})")
    seeds = pd.to_numeric(p["seed"], errors="coerce")
    rec(bool(seeds.notna().all()) and set(seeds.dropna().astype(int)) == SEEDS,
        f"seeds are exactly 42, 43, 44 (found {sorted(set(p['seed']))})")
    have = {(v, int(s)) for v, s in zip(p["variant"], seeds.fillna(-1))}
    want = {(v, s) for v in VARIANTS for s in SEEDS}
    rec(have == want, "all 6 combinations of variant (tok, notok) and seed (42, 43, 44) are present"
        + (f" (missing: {sorted(want - have)})" if have != want else ""))
    rec(not p.duplicated(["id", "variant", "seed"]).any(), "no duplicate (id, variant, seed) rows")
    expected = len(test_ids) * len(VARIANTS) * len(SEEDS)
    rec(len(p) == expected, f"row count is {len(p)} (expected {expected})")

    bad = []
    wanted = set(test_ids)
    for (v, s), g in p.groupby(["variant", "seed"]):
        if len(g) != len(wanted) or set(g["id"]) != wanted:
            bad.append(f"variant={v} seed={s}")
    rec(not bad, "every (variant, seed) covers exactly the test ids once" + (f" (problem in: {', '.join(bad)})" if bad else ""))

    if cost_path.exists():
        try:
            cost = json.loads(cost_path.read_text(encoding="utf-8"))
        except Exception as e:
            cost = None
            rec(False, f"cost.json is valid JSON ({e})")
        if cost is not None:
            missing = [k for k in COST_TEXT + COST_NUM if k not in cost]
            rec(not missing, "cost.json has all required keys" + (f" (missing: {missing})" if missing else ""))
            nonnum = [k for k in COST_NUM if k in cost and not isinstance(cost[k], (int, float))]
            rec(not nonnum, "cost.json numeric fields are numbers" + (f" (not numbers: {nonnum})" if nonnum else ""))
            rec(cost.get("team") == team, f"cost.json team is '{team}'")
    return state["ok"], msgs


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("llm", "encoder"):
        print("usage: python compare/validate_submission.py llm|encoder")
        sys.exit(2)
    ok, messages = check(sys.argv[1])
    print("\n".join(messages))
    print("\nRESULT:", "PASS, you can make the FINAL commit" if ok else "FAIL, fix the lines above")
    sys.exit(0 if ok else 1)
