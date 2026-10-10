"""
compare.py -- final comparison of the LLM team and the encoder team. RUN BY THE PROJECT LEAD ONLY.

    python compare/compare.py                  final comparison on test (needs llm/final and encoder/final)
    python compare/compare.py --baselines-val  prints the two baselines' VALIDATION scores (send to teams on D2)
    python compare/compare.py --boot 2000      fewer bootstrap resamples (quick check). Default 10000.

Writes compare/comparison_report.md and compare/comparison_summary.csv. Changes nothing else.
Rules implemented (see section A3b of the team briefs):
  headline = macro-F1 on label_1, full test set, 'notok' variant, mean of seeds 42/43/44
  winner   = paired bootstrap 95% CI of the difference excludes 0 on BOTH full test and the unanimous subset
  tie-break = lower inference latency (bs 1); if within 10%, lower training time
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import validate_submission as vs  # noqa: E402

ROOT = vs.ROOT
TEAMS = ["llm", "encoder"]
SEEDS = sorted(vs.SEEDS)
VARIANTS = ["notok", "tok"]
LANGS = ["en", "hi", "ta"]
TOKEN = "<handle replaced>"
OUT_MD = HERE / "comparison_report.md"
OUT_CSV = HERE / "comparison_summary.csv"


# ---------------------------------------------------------------- data
def read_split(name):
    return pd.read_csv(ROOT / "data_v1" / f"{name}.csv", dtype=str, keep_default_na=False, encoding="utf-8")


def prep(d):
    d = d.copy()
    d["y"] = pd.to_numeric(d["label_1"], errors="coerce")
    d = d[d["y"].isin([0, 1])].copy()
    d["y"] = d["y"].astype(int)
    d["agree"] = pd.to_numeric(d["agree_1"], errors="coerce")
    d["text_notok"] = (d["text"].str.replace(TOKEN, " ", regex=False)
                       .str.replace(r"\s+", " ", regex=True).str.strip())
    return d.reset_index(drop=True)


# ---------------------------------------------------------------- metrics
def macro_f1_counts(c):
    """c[..., 4] = TN, FP, FN, TP. Returns macro-F1 (mean of the two class F1s), 0 when undefined."""
    tn, fp, fn, tp = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        f_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f_pos + f_neg) / 2


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = np.bincount(2 * y + pred, minlength=4)
    out = {
        "macro_f1": float(macro_f1_counts(np.array([tn, fp, fn, tp]))),
        "pos_f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "accuracy": float((tn + tp) / len(y)),
    }
    if len(np.unique(y)) == 2 and len(np.unique(p)) > 1:
        out["auroc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["auroc"] = out["pr_auc"] = float("nan")
    return out


def subsets(test):
    m = {"full": np.ones(len(test), bool),
         "unanimous": (test["agree"] == 1).to_numpy(),
         "two_vs_one": (test["agree"] < 1).to_numpy()}
    for lang in LANGS:
        m[f"lang_{lang}"] = (test["language"] == lang).to_numpy()
    return m


def summarize(name, preds, test, variants):
    """preds[(variant, seed)] -> p_pos array. Mean and std over seeds. One row per metric."""
    rows = []
    y = test["y"].to_numpy()
    for v in variants:
        for sname, mask in subsets(test).items():
            ms = [metrics(y[mask], preds[(v, s)][mask]) for s in SEEDS if (v, s) in preds]
            for k in ms[0]:
                vals = np.array([m[k] for m in ms], float)
                rows.append(dict(system=name, variant=v, subset=sname, n=int(mask.sum()), metric=k,
                                 mean=float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan"),
                                 std=float(np.nanstd(vals)) if not np.all(np.isnan(vals)) else float("nan")))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- bootstrap
def boot_means(codes_a, codes_b, n_boot, rng, chunk=500):
    """Paired bootstrap: the SAME resampled posts are used for both teams.
    codes_*: (n_seeds, N) arrays of 2*y+pred. Returns per-resample mean-over-seeds macro-F1 for A and B."""
    n = codes_a.shape[1]
    ma, mb = [], []
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(m, n))
        offs = (4 * np.arange(m))[:, None]

        def mean_f1(codes):
            vals = []
            for s in range(codes.shape[0]):
                flat = (codes[s][idx] + offs).ravel()
                c = np.bincount(flat, minlength=4 * m).reshape(m, 4)
                vals.append(macro_f1_counts(c))
            return np.mean(vals, axis=0)

        ma.append(mean_f1(codes_a))
        mb.append(mean_f1(codes_b))
    return np.concatenate(ma), np.concatenate(mb)


def paired_diff(codes_a, codes_b, mask, n_boot, rng):
    a, b = codes_a[:, mask], codes_b[:, mask]
    ma, mb = boot_means(a, b, n_boot, rng)
    d = ma - mb
    point = float(np.mean(macro_f1_counts(np.stack([np.bincount(c, minlength=4) for c in a])))
                  - np.mean(macro_f1_counts(np.stack([np.bincount(c, minlength=4) for c in b]))))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(point=point, lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0))


# ---------------------------------------------------------------- baselines
def fit_baselines(train, val, target):
    """Majority class per language, and TF-IDF (word 1-2 + char 2-5) + logistic regression (C picked on val).
    Both use the notok text. Returns {name: p_pos on target}."""
    maj = train.groupby("language")["y"].agg(lambda s: int(s.mean() >= 0.5)).to_dict()
    out = {"majority_per_language": target["language"].map(maj).fillna(0).astype(float).to_numpy()}
    w = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200000)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000)
    xtr = hstack([w.fit_transform(train["text_notok"]), c.fit_transform(train["text_notok"])]).tocsr()

    def feats(d):
        return hstack([w.transform(d["text_notok"]), c.transform(d["text_notok"])]).tocsr()

    xva, xta = feats(val), feats(target)
    best, best_c = -1, 1.0
    for C in (0.3, 1.0, 3.0):
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear")
        lr.fit(xtr, train["y"])
        score = metrics(val["y"].to_numpy(), lr.predict_proba(xva)[:, 1])["macro_f1"]
        if score > best:
            best, best_c = score, C
    lr = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000, solver="liblinear")
    lr.fit(xtr, train["y"])
    out["tfidf_logreg"] = lr.predict_proba(xta)[:, 1]
    return out


# ---------------------------------------------------------------- misc
def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(r[c] if isinstance(r[c], str) else f"{r[c]}" for c in cols) + " |")
    return "\n".join(lines)


def fmt(m, s=None):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "n/a"
    return f"{m:.3f}" if s is None or np.isnan(s) else f"{m:.3f} +/- {s:.3f}"


def git_lines(args):
    try:
        r = subprocess.run(["git", "-C", str(ROOT)] + args, capture_output=True, text=True, timeout=30)
        return r.stdout.split() if r.returncode == 0 else None
    except Exception:
        return None


def nomination_check(team):
    noms = git_lines(["log", "--all", "--grep=NOMINATE", "--format=%ct", "--", team])
    adds = git_lines(["log", "--all", "--diff-filter=A", "--format=%ct", "--", f"{team}/final/final_predictions.csv"])
    if noms is None or adds is None:
        return "git history not available here, check the NOMINATE commit by hand"
    if not noms:
        return "WARNING: no commit with NOMINATE in its message was found for this team"
    if not adds:
        return "git has no commit adding final_predictions.csv yet (commit it, then re-run)"
    nom, first = min(int(t) for t in noms), min(int(t) for t in adds)
    when = lambda t: datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "OK" if nom <= first else "VIOLATION (test predictions were committed before the nomination)"
    return f"{status}: first NOMINATE commit {when(nom)}; final predictions first committed {when(first)}"


def tie_break(cost_a, cost_b):
    la, lb = cost_a["latency_ms_per_post_bs1"], cost_b["latency_ms_per_post_bs1"]
    if abs(la - lb) / max(la, lb) > 0.10:
        return "llm" if la < lb else "encoder", "lower inference latency per post (bs 1)"
    ta, tb = cost_a["train_time_min"], cost_b["train_time_min"]
    if ta == tb:
        return None, "latency within 10% and equal training time: lead decides"
    return ("llm" if ta < tb else "encoder"), "latency within 10%, so lower training time"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines-val", action="store_true")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    train, val, test = (prep(read_split(n)) for n in ("train", "val", "test"))

    if a.baselines_val:
        bl = fit_baselines(train, val, val)
        print("BASELINES on VALIDATION (notok text). Floor to beat, not a target.\n")
        for name, p in bl.items():
            m = metrics(val["y"].to_numpy(), p)
            per = {l: metrics(val["y"].to_numpy()[(val['language'] == l).to_numpy()], p[(val['language'] == l).to_numpy()])["macro_f1"] for l in LANGS}
            print(f"{name}: macro-F1 {m['macro_f1']:.3f} | pos-F1 {m['pos_f1']:.3f} | per language " +
                  ", ".join(f"{l} {v:.3f}" for l, v in per.items()))
        return

    # validate both submissions first
    bad = False
    for t in TEAMS:
        ok, msgs = vs.check(t)
        if not ok:
            bad = True
            print(f"\n{t}: submission FAILED validation\n" + "\n".join(msgs))
    if bad:
        sys.exit(1)

    ids = test["id"]
    preds, costs, codes = {}, {}, {}
    y = test["y"].to_numpy()
    for t in TEAMS:
        p = pd.read_csv(ROOT / t / "final" / "final_predictions.csv", dtype=str, keep_default_na=False, encoding="utf-8")
        p["p_pos"] = p["p_pos"].astype(float)
        p["seed"] = p["seed"].astype(int)
        preds[t] = {(v, s): g.set_index("id")["p_pos"].reindex(ids).to_numpy() for (v, s), g in p.groupby(["variant", "seed"])}
        costs[t] = json.loads((ROOT / t / "final" / "cost.json").read_text(encoding="utf-8"))
        for v in VARIANTS:
            codes[(t, v)] = np.stack([2 * y + (preds[t][(v, s)] >= 0.5).astype(int) for s in SEEDS])

    rng = np.random.default_rng(a.seed)
    masks = subsets(test)
    res = {}
    for v in VARIANTS:
        for sub in ("full", "unanimous"):
            res[(v, sub)] = paired_diff(codes[("llm", v)], codes[("encoder", v)], masks[sub], a.boot, rng)

    # decision (headline = notok)
    f, u = res[("notok", "full")], res[("notok", "unanimous")]
    if f["excludes_zero"] and u["excludes_zero"] and np.sign(f["point"]) == np.sign(u["point"]):
        winner = "llm" if f["point"] > 0 else "encoder"
        decision = f"**{winner.upper()} team wins.** The 95% CI of the macro-F1 difference excludes 0 on the full test set and on the unanimous subset."
        tb = None
    else:
        winner = None
        tb_team, tb_why = tie_break(costs["llm"], costs["encoder"])
        decision = ("**No significant difference** (the CI includes 0 on the full test set or on the unanimous subset, or the signs disagree). "
                    + (f"Pre-registered tie-break: **{tb_team.upper()}** goes forward ({tb_why})." if tb_team else f"Tie-break: {tb_why}."))

    summ = pd.concat([summarize(t, preds[t], test, VARIANTS) for t in TEAMS])
    bl = fit_baselines(train, val, test)
    bl_summ = pd.concat([summarize(n, {(v, s): p for v in VARIANTS for s in SEEDS}, test, ["notok"]) for n, p in bl.items()])
    summ_all = pd.concat([summ, bl_summ])
    summ_all.to_csv(OUT_CSV, index=False, encoding="utf-8")

    def val_of(system, variant, subset, metric):
        r = summ_all[(summ_all.system == system) & (summ_all.variant == variant) & (summ_all.subset == subset) & (summ_all.metric == metric)]
        return (r["mean"].iloc[0], r["std"].iloc[0]) if len(r) else (None, None)

    L = []
    L.append("# Comparison report: LLM team vs encoder team\n")
    L.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Target: label_1. Test posts: {len(test)} "
             f"(unanimous subset: {int(masks['unanimous'].sum())}). Seeds {SEEDS}. Bootstrap resamples: {a.boot} (paired).\n")
    L.append("## 1. Decision\n")
    L.append(decision + "\n")
    rows = []
    for v in VARIANTS:
        for sub in ("full", "unanimous"):
            r = res[(v, sub)]
            rows.append({"variant": v, "eval set": sub,
                         "LLM macro-F1": fmt(*val_of("llm", v, sub, "macro_f1")),
                         "Encoder macro-F1": fmt(*val_of("encoder", v, sub, "macro_f1")),
                         "difference (LLM - encoder)": f"{r['point']:+.3f}",
                         "95% CI": f"[{r['lo']:+.3f}, {r['hi']:+.3f}]",
                         "excludes 0": "yes" if r["excludes_zero"] else "no"})
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append("Headline rows are the `notok` variant. `tok` is reported next to it.\n")

    L.append("## 2. Nomination check (from git history)\n")
    for t in TEAMS:
        L.append(f"- {t}: {nomination_check(t)}")
    L.append("")

    L.append("## 3. Supporting metrics (mean +/- std over 3 seeds, full test set)\n")
    rows = []
    for system in TEAMS + list(bl.keys()):
        for v in (VARIANTS if system in TEAMS else ["notok"]):
            r = {"system": system, "variant": v}
            for m in ("macro_f1", "pos_f1", "precision", "recall", "accuracy", "auroc", "pr_auc"):
                r[m] = fmt(*val_of(system, v, "full", m))
            rows.append(r)
    L.append(md_table(pd.DataFrame(rows)) + "\n")

    L.append("## 4. Macro-F1 by language and by label agreement (notok)\n")
    rows = []
    for system in TEAMS + list(bl.keys()):
        r = {"system": system}
        for sub, label in [("lang_en", "en"), ("lang_hi", "hi"), ("lang_ta", "ta"), ("unanimous", "unanimous"), ("two_vs_one", "2-vs-1")]:
            r[label] = fmt(*val_of(system, "notok", sub, "macro_f1"))
        rows.append(r)
    L.append(md_table(pd.DataFrame(rows)) + "\n")
    L.append("Accuracy on unanimous versus 2-vs-1 posts (notok):\n")
    rows = []
    for system in TEAMS:
        rows.append({"system": system, "unanimous": fmt(*val_of(system, "notok", "unanimous", "accuracy")),
                     "2-vs-1": fmt(*val_of(system, "notok", "two_vs_one", "accuracy"))})
    L.append(md_table(pd.DataFrame(rows)) + "\n")

    L.append("## 5. Reliance on the handle token (macro-F1, full test)\n")
    rows = []
    for system in TEAMS:
        t_, n_ = val_of(system, "tok", "full", "macro_f1")[0], val_of(system, "notok", "full", "macro_f1")[0]
        rows.append({"system": system, "tok": fmt(t_), "notok": fmt(n_), "drop (tok - notok)": f"{t_ - n_:+.3f}"})
    L.append(md_table(pd.DataFrame(rows)) + "\n")

    L.append("## 6. Cost (reported side by side, no composite score)\n")
    keys = ["model", "formulation", "hardware", "train_time_min", "peak_gpu_mem_gb", "trainable_params", "model_size_mb", "latency_ms_per_post_bs1"]
    L.append(md_table(pd.DataFrame([{"field": k, "llm": costs["llm"][k], "encoder": costs["encoder"][k]} for k in keys])) + "\n")
    L.append("Tie-break rule (fixed in advance): lower inference latency per post at batch size 1; if within 10%, lower training time.\n")

    L.append("## 7. Notes\n")
    L.append("- Human agreement on this task is low (Krippendorff alpha 0.462 on test label_1), so small gaps are not meaningful.")
    L.append("- Baselines are trained on train, text with the handle token removed, with C chosen on val. They are a floor, not a competitor.")
    L.append("- The headline compares only the two nominated configurations. Any other runs are descriptive.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"{decision}\n")
    for v in VARIANTS:
        for sub in ("full", "unanimous"):
            r = res[(v, sub)]
            print(f"{v:6s} {sub:9s} diff {r['point']:+.3f}  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  excludes 0: {r['excludes_zero']}")
    print(f"\nWrote {OUT_MD}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
