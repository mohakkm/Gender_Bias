"""Token-length and tokens-per-word study. Reads train.csv only, never test.csv.

    python llm/code/token_study.py <tokenizers_dir>

<tokenizers_dir> holds one folder per tokenizer (qwen3/, sarvam1/), each with tokenizer.json:
    Qwen/Qwen3-1.7B    revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e (same files as Qwen3-4B)
    sarvamai/sarvam-1  revision e9607337286ddf496d4a2562b194e489dcf3feea

Lengths include the tokenizer's own special tokens. Prompts for F2 add more on top.
Writes llm/code/token_study_results.csv.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "token_study_results.csv"
TOKEN = "<handle replaced>"
LANGS = ["en", "hi", "ta"]
NAMES = ["qwen3", "sarvam1"]
LATIN = re.compile(r"[A-Za-z]")


def notok(s):
    return s.str.replace(TOKEN, " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()


def lengths(tk, texts, special=True):
    return np.array([len(e.ids) for e in tk.encode_batch(list(texts), add_special_tokens=special)])


d = pd.read_csv(ROOT / "data_v1" / "train.csv", dtype=str, keep_default_na=False, encoding="utf-8")
d = d[d["label_1"].isin(["0", "1"])].reset_index(drop=True)
assert len(d) == 17290, len(d)
d["tok"], d["notok"] = d["text"], notok(d["text"])
samples = {l: d[d["language"] == l].sample(200, random_state=42) for l in LANGS}

rows = []
for name in NAMES:
    tk = Tokenizer.from_file(str(Path(sys.argv[1]) / name / "tokenizer.json"))
    unk = tk.token_to_id("<unk>")
    print(f"\n===== {name} (vocab {tk.get_vocab_size()}, unk id {unk})")

    for v in ("tok", "notok"):
        n = lengths(tk, d[v])
        per = {l: n[(d["language"] == l).to_numpy()] for l in LANGS}
        p99_all = math.ceil(np.percentile(n, 99))
        p99_max = max(math.ceil(np.percentile(x, 99)) for x in per.values())
        print(f"{v}: pooled p99 {p99_all} | highest per-language p99 {p99_max} | max {n.max()} | "
              f"truncated at pooled p99 {(n > p99_all).mean():.2%}, at highest per-language p99 {(n > p99_max).mean():.2%}")
        for scope, x in [("all", n)] + list(per.items()):
            r = dict(tokenizer=name, variant=v, scope=scope, n=len(x), mean=round(float(x.mean()), 1),
                     p50=float(np.percentile(x, 50)), p95=float(np.percentile(x, 95)),
                     p99=float(np.percentile(x, 99)), max=int(x.max()),
                     trunc_at_pooled_p99=round(float((x > p99_all).mean()), 4),
                     trunc_at_max_lang_p99=round(float((x > p99_max).mean()), 4))
            rows.append(r)
            print(f"  {scope:3s} n={r['n']:5d} mean={r['mean']:6.1f} p50={r['p50']:5.0f} p95={r['p95']:5.0f} "
                  f"p99={r['p99']:6.1f} max={r['max']:4d} trunc@pooled={r['trunc_at_pooled_p99']:.2%}")

    for l in LANGS:
        s = samples[l]
        words = s["notok"].str.split().str.len().to_numpy()
        toks = lengths(tk, s["notok"], special=False)
        latin = s["notok"].map(lambda t: bool(LATIN.search(t))).mean()
        unk_n = sum(e.ids.count(unk) for e in tk.encode_batch(list(s["notok"]), add_special_tokens=False)) if unk is not None else 0
        tpw = toks.sum() / words.sum()
        rows.append(dict(tokenizer=name, variant="notok", scope=f"tpw_sample_{l}", n=len(s), tokens_per_word=round(float(tpw), 2),
                         unk_tokens=int(unk_n), posts_with_latin_share=round(float(latin), 3)))
        print(f"  tokens per word {l}: {tpw:.2f} (200 posts, seed 42, notok; unk tokens {unk_n}; posts with Latin letters {latin:.0%})")

pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
print(f"\nWrote {OUT}")
