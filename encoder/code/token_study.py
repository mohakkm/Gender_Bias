"""D1 token-length study (rule A2.8). Train only, never test.

    python encoder/code/token_study.py                 all four tokenizers
    python encoder/code/token_study.py muril mbert     a subset

For each tokenizer and variant: length percentiles (special tokens included), pooled and per language;
truncation rate at candidate max lengths; [UNK] rate per language (mBERT and MuRIL can map emoji and rare
characters to [UNK]); share of posts with Latin letters (code-mixing proxy).
Recommended max_len = smallest multiple of 8 covering >= 99% of train under BOTH variants, capped at 512.
Decision 2026-11-13: token_study_maxlen.json was then set by hand to 192 for every model (covers 100% of train,
see STATE.md decisions log). Re-running this script overwrites that file, so re-apply the decision afterwards.
Writes encoder/code/results/token_study.csv and token_study_maxlen.json.
"""
import math
import re
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from common import MODELS, RESULTS, LANGS, load_train_val, texts, write_json

LATIN = re.compile(r"[A-Za-z]")
CANDIDATES = [64, 128, 192, 256, 384, 512]

names = sys.argv[1:] or list(MODELS)
train, _ = load_train_val()
lang = train["language"].to_numpy()
rows, rec = [], {}

for name in names:
    tk = AutoTokenizer.from_pretrained(MODELS[name])
    print(f"\n===== {name}: {MODELS[name]} (vocab {tk.vocab_size}, fast={tk.is_fast})")
    need = 0
    for v in ("tok", "notok"):
        enc = tk(texts(train, v), add_special_tokens=True, truncation=False)["input_ids"]
        n = np.array([len(x) for x in enc])
        unk = np.array([x.count(tk.unk_token_id) for x in enc]) if tk.unk_token_id is not None else np.zeros(len(n))
        p99 = math.ceil(np.percentile(n, 99))
        need = max(need, p99)
        print(f"{v}: pooled p99 {p99}, max {n.max()}, " +
              ", ".join(f"trunc@{c} {(n > c).mean():.2%}" for c in CANDIDATES))
        for scope in ["all"] + LANGS:
            m = np.ones(len(n), bool) if scope == "all" else lang == scope
            x, u = n[m], unk[m]
            r = dict(tokenizer=name, checkpoint=MODELS[name], variant=v, scope=scope, n=int(m.sum()),
                     mean=round(float(x.mean()), 1), p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)),
                     p99=float(np.percentile(x, 99)), max=int(x.max()),
                     unk_per_post=round(float(u.mean()), 3), posts_with_unk=round(float((u > 0).mean()), 4))
            r.update({f"trunc_at_{c}": round(float((x > c).mean()), 4) for c in CANDIDATES})
            if scope != "all":
                r["posts_with_latin"] = round(float(train.loc[m, "text_notok"].map(lambda t: bool(LATIN.search(t))).mean()), 3)
            rows.append(r)
            print(f"  {scope:3s} n={r['n']:5d} mean={r['mean']:6.1f} p99={r['p99']:6.1f} max={r['max']:4d} "
                  f"unk/post={r['unk_per_post']:.3f} posts_with_unk={r['posts_with_unk']:.1%}")
    max_len = min(512, int(math.ceil(need / 8) * 8))
    rec[name] = {"checkpoint": MODELS[name], "max_len": max_len, "p99_max_over_variants": need}
    print(f"-> recommended max_len for {name}: {max_len}")

RESULTS.mkdir(parents=True, exist_ok=True)
out = RESULTS / "token_study.csv"
old = pd.read_csv(out) if out.exists() else pd.DataFrame()
if len(old):
    old = old[~old["tokenizer"].isin(names)]
pd.concat([old, pd.DataFrame(rows)]).to_csv(out, index=False, encoding="utf-8")
prev = {}
jp = RESULTS / "token_study_maxlen.json"
if jp.exists():
    import json
    prev = json.loads(jp.read_text(encoding="utf-8"))
prev.update(rec)
write_json(jp, prev)
print(f"\nWrote {out} and {jp}")
