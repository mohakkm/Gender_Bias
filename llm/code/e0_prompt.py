"""E0: zero-shot and few-shot prompting with no training. Validation only (never reads test.csv).

p_pos comes from the model's scores for the answer tokens 'Yes' and 'No' at the answer position,
so there is nothing to parse and no parse failures.

    python llm/code/e0_prompt.py --variant notok --shots 0
    python llm/code/e0_prompt.py --variant notok --shots 6

Outputs go to llm/outputs/E0_<model>_<variant>_<k>shot/ (git-ignored).
"""
import common  # noqa: F401  (sets environment variables before the heavy imports)

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REVISIONS = {"Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
             "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c"}
SYSTEM = ("You annotate social media posts in English, Hindi and Tamil for a research study on "
          "online gendered abuse.")
QUESTION = ("Is this post gendered abuse when directed at a person of marginalized gender? "
            "Answer with one word: Yes or No.")


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="Qwen/Qwen3-1.7B")
    a.add_argument("--revision", default=None)
    a.add_argument("--variant", choices=["tok", "notok"], required=True)
    a.add_argument("--shots", type=int, default=0, help="examples from train, balanced over language and label")
    a.add_argument("--bs", type=int, default=8)
    a.add_argument("--limit-val", type=int, default=0)
    a.add_argument("--four-bit", action="store_true")
    a.add_argument("--seed", type=int, default=42)
    return a.parse_args()


def pick_shots(train, k, seed):
    """k examples cycling over (language, label) cells, so every cell is covered once k >= 6."""
    if k == 0:
        return train.iloc[0:0]
    cells = [(l, y) for l in common.LANGS for y in (1, 0)]
    rows = []
    for i in range(k):
        lang, y = cells[i % len(cells)]
        pool = train[(train["language"] == lang) & (train["y"] == y)]
        rows.append(pool.sample(1, random_state=seed + i))
    return pd.concat(rows)


def clip(tok, text, max_len):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text if len(ids) <= max_len else tok.decode(ids[:max_len])


def build_prompt(tok, text, shots):
    msgs = [{"role": "system", "content": SYSTEM}]
    for _, s in shots.iterrows():
        msgs.append({"role": "user", "content": f"{QUESTION}\n\nPost: {s['input']}"})
        msgs.append({"role": "assistant", "content": "Yes" if s["y"] == 1 else "No"})
    msgs.append({"role": "user", "content": f"{QUESTION}\n\nPost: {text}"})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def main():
    a = parse()
    is_hub = not common.Path(a.model).exists()
    rev = a.revision or REVISIONS.get(a.model)
    if is_hub and rev is None:
        raise SystemExit(f"Pin a revision for {a.model} with --revision <commit sha>.")
    short = a.model.rstrip("/\\").replace("\\", "/").split("/")[-1]
    run_dir = common.OUTPUTS / f"E0_{short}_{a.variant}_{a.shots}shot"
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model, revision=rev if is_hub else None)
    tok.padding_side = "left"  # the answer position must be the last column for every row
    yes, no = (tok(w, add_special_tokens=False)["input_ids"] for w in ("Yes", "No"))
    if len(yes) != 1 or len(no) != 1:
        raise SystemExit(f"'Yes'/'No' are not single tokens: {yes} {no}")

    train, val = common.load("train", a.variant), common.load("val", a.variant)
    if a.limit_val:
        val = val.sample(a.limit_val, random_state=a.seed).reset_index(drop=True)
    shots = pick_shots(train, a.shots, a.seed)
    shots = shots.assign(input=[clip(tok, t, 120) for t in shots["input"]])  # keep few-shot prompts short
    prompts = [build_prompt(tok, clip(tok, t, common.MAX_LEN), shots) for t in val["input"]]

    kw = dict(revision=rev if is_hub else None, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    if a.four_bit:
        kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                         bnb_4bit_compute_dtype=torch.float16), device_map={"": 0})
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if not a.four_bit and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    dev = next(model.parameters()).device

    order = np.argsort([len(p) for p in prompts])  # similar lengths per batch, less padding
    p_pos, mass = np.zeros(len(prompts)), np.zeros(len(prompts))
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(order), a.bs):
            idx = order[i:i + a.bs]
            enc = tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(dev)
            # logits_to_keep=1: only the last position. Full logits (batch x length x 151k vocab) ran a
            # T4 out of memory on 6-shot prompts in the first Kaggle job.
            last = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            probs = torch.softmax(last, -1)
            two = torch.softmax(last[:, [yes[0], no[0]]], -1)
            p_pos[idx] = two[:, 0].cpu().numpy()
            mass[idx] = (probs[:, yes[0]] + probs[:, no[0]]).cpu().numpy()
    minutes = (time.perf_counter() - t0) / 60

    res = common.report(val, p_pos)
    print(f"E0 {a.shots}-shot {a.variant}: VAL macro-F1 (threshold 0.5): {common.fmt(res)}")
    print(f"AUROC {res['all'].get('auroc', float('nan')):.3f} | predicted positive rate {(p_pos >= 0.5).mean():.2%} "
          f"vs true {val['y'].mean():.2%} | mean probability on Yes+No {mass.mean():.2%} | {minutes:.1f} min")
    pd.DataFrame({"id": val["id"], "language": val["language"], "y": val["y"], "p_pos": p_pos}).to_csv(
        run_dir / "val_predictions.csv", index=False, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps({
        "run": run_dir.name, "args": vars(a), "revision": rev, "val": res, "minutes": round(minutes, 2),
        "predicted_positive_rate": float((p_pos >= 0.5).mean()), "true_positive_rate": float(val["y"].mean()),
        "mean_yes_no_mass": float(mass.mean()), "shot_ids": shots["id"].tolist(),
        "prompt": {"system": SYSTEM, "question": QUESTION}}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
