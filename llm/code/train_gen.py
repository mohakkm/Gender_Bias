"""F2 (secondary): instruction-style generative classification with QLoRA.
The model reads the E0 zero-shot prompt and is trained to answer 'Yes' or 'No' followed by the
end-of-turn token. Loss is on the answer only (completion-only). p_pos comes from the 'Yes' and
'No' token logits at the answer position, so there is no text parsing.

    python llm/code/train_gen.py --variant notok --seed 42
    pilot: add --limit-train 200 --limit-val 200 --epochs 1

Same safeguards as train_cls.py: in-order predictions, explicit best-epoch reload with a score
check, test.csv only through the NOMINATE gate. Outputs: llm/outputs/F2_<...>/ (git-ignored).
"""
import common  # noqa: F401  (sets environment variables before the heavy imports)

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch
import transformers
from peft import (LoraConfig, TaskType, get_peft_model, load_peft_weights, prepare_model_for_kbit_training,
                  set_peft_model_state_dict)
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from e0_prompt import build_prompt, clip
from train_cls import REVISIONS, TARGETS, dir_mb

FULL_TRAIN_ROWS = 17290


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="Qwen/Qwen3-1.7B")
    a.add_argument("--revision", default=None)
    a.add_argument("--variant", choices=["tok", "notok"], required=True)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--rank", type=int, default=16)
    a.add_argument("--alpha", type=int, default=32)
    a.add_argument("--dropout", type=float, default=0.05)
    a.add_argument("--targets", choices=sorted(TARGETS), default="attn")
    a.add_argument("--lr", type=float, default=2e-4)
    a.add_argument("--epochs", type=int, default=2)
    a.add_argument("--bs", type=int, default=8)
    a.add_argument("--grad-accum", type=int, default=2)
    a.add_argument("--warmup", type=float, default=0.05)
    a.add_argument("--max-len", type=int, default=common.MAX_LEN, help="max tokens of the post itself")
    a.add_argument("--no-4bit", action="store_true")
    a.add_argument("--limit-train", type=int, default=0)
    a.add_argument("--limit-val", type=int, default=0)
    a.add_argument("--latency-posts", type=int, default=common.LATENCY_POSTS,
                   help="TEAM_AGREEMENTS.md 3a uses 500; 0 skips (development runs)")
    a.add_argument("--resume", action="store_true")
    a.add_argument("--tag", default="")
    a.add_argument("--predict-test", action="store_true", help="ONLY after the NOMINATE commit")
    return a.parse_args()


class Scorer:
    """p_pos from the Yes/No logits at the answer position, in input order."""

    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len
        self.yes = tok("Yes", add_special_tokens=False)["input_ids"]
        self.no = tok("No", add_special_tokens=False)["input_ids"]
        if len(self.yes) != 1 or len(self.no) != 1:
            raise SystemExit(f"'Yes'/'No' are not single tokens: {self.yes} {self.no}")

    def prompts(self, texts):
        empty = pd.DataFrame(columns=["input", "y"])
        return [build_prompt(self.tok, clip(self.tok, t, self.max_len), empty) for t in texts]

    @torch.no_grad()
    def __call__(self, model, texts, bs=16):
        was_training = model.training
        model.eval()
        dev = next(model.parameters()).device
        prompts = self.prompts(texts)
        order = np.argsort([len(p) for p in prompts])
        out = np.zeros(len(prompts), dtype=np.float32)
        side = self.tok.padding_side
        self.tok.padding_side = "left"  # answer position = last column for every row
        try:
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                enc = self.tok([prompts[j] for j in idx], return_tensors="pt", padding=True).to(dev)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    last = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
                out[idx] = torch.softmax(last[:, [self.yes[0], self.no[0]]], -1)[:, 0].cpu().numpy()
        finally:
            self.tok.padding_side = side
        if was_training:
            model.train()
        return out


def encode_train(tok, scorer, df):
    """input_ids = prompt + answer + end-of-turn; labels = -100 on the prompt (completion-only loss)."""
    end = tok.convert_tokens_to_ids("<|im_end|>")
    rows = []
    for prompt, y in zip(scorer.prompts(df["input"]), df["y"]):
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = (scorer.yes if y == 1 else scorer.no) + [end]
        rows.append({"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids})
    return rows


class Collate:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        return {"input_ids": ids, "attention_mask": att, "labels": lab}


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class ValCallback(TrainerCallback):
    """Score val at each epoch end, keep the best adapter, stop after one epoch without improvement."""

    def __init__(self, scorer, val, best_dir, bs):
        self.scorer, self.val, self.best_dir, self.bs = scorer, val, best_dir, bs
        self.best, self.history, self.bad = -1.0, [], 0

    def on_epoch_end(self, args, state, control, model=None, **kw):
        p = self.scorer(model, self.val["input"].tolist(), self.bs)
        f1 = common.scores(self.val["y"], p)["macro_f1"]
        self.history.append({"epoch": round(state.epoch, 2), "val_macro_f1": f1})
        print(f"epoch {state.epoch:.2f}: val macro-F1 {f1:.4f}", flush=True)
        if f1 > self.best:
            self.best, self.bad = f1, 0
            model.save_pretrained(str(self.best_dir))
        else:
            self.bad += 1
            if self.bad >= 1:
                control.should_training_stop = True
        return control


def main():
    a = parse()
    is_hub = not common.Path(a.model).exists()
    rev = a.revision or REVISIONS.get(a.model)
    if is_hub and rev is None:
        raise SystemExit(f"Pin a revision for {a.model} with --revision <commit sha>.")
    four_bit = not a.no_4bit
    set_seed(a.seed)
    short = a.model.rstrip("/\\").replace("\\", "/").split("/")[-1]
    name = f"F2_{short}_{a.variant}_s{a.seed}_r{a.rank}_{a.targets}_lr{a.lr:g}"
    name += f"_pilot{a.limit_train}" if a.limit_train else ""
    run_dir = common.OUTPUTS / (name + (f"_{a.tag}" if a.tag else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run: {run_dir.name}")

    train, val = common.load("train", a.variant), common.load("val", a.variant)
    if a.limit_train:
        train = train.sample(a.limit_train, random_state=a.seed).reset_index(drop=True)
    if a.limit_val:
        val = val.sample(a.limit_val, random_state=a.seed).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(a.model, revision=rev if is_hub else None)
    tok.padding_side = "right"
    scorer = Scorer(tok, a.max_len)
    rows = encode_train(tok, scorer, train)
    lens = [len(r["input_ids"]) for r in rows]
    print(f"rows train {len(train)} val {len(val)} | train sequence length p50 {np.median(lens):.0f} max {max(lens)}")

    kw = dict(revision=rev if is_hub else None)
    if four_bit:
        kw.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16), device_map={"": 0}, dtype=torch.float16)
    else:
        kw.update(dtype=torch.float32)
    model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    elif torch.cuda.is_available():
        model = model.cuda()
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=a.rank, lora_alpha=a.alpha,
                                             lora_dropout=a.dropout, target_modules=TARGETS[a.targets]))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params {trainable:,} of {total:,} ({trainable / total:.2%})")

    best_dir = run_dir / "adapter"
    cb = ValCallback(scorer, val, best_dir, a.bs * 2)
    cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"), per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, num_train_epochs=a.epochs,
        lr_scheduler_type="cosine", warmup_ratio=a.warmup, eval_strategy="no", save_strategy="epoch",
        save_total_limit=1, fp16=cuda, logging_steps=20, report_to="none", seed=a.seed, data_seed=a.seed,
        group_by_length=True, length_column_name="length", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, remove_unused_columns=False,
        optim="paged_adamw_8bit" if four_bit else "adamw_torch", dataloader_num_workers=0)
    for r, n in zip(rows, lens):
        r["length"] = n
    trainer = Trainer(model=model, args=args, train_dataset=ListDataset(rows),
                      data_collator=Collate(tok.pad_token_id), callbacks=[cb])

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    has_ckpt = any((run_dir / "checkpoints").glob("checkpoint-*"))
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=True if (a.resume and has_ckpt) else None)
    train_min = (time.perf_counter() - t0) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0

    set_peft_model_state_dict(model, load_peft_weights(str(best_dir), device=str(next(model.parameters()).device)))
    pv = scorer(model, val["input"].tolist(), a.bs * 2)
    res = common.report(val, pv)
    print(f"VAL macro-F1 (threshold 0.5): {common.fmt(res)}")
    if cb.best >= 0 and abs(res["all"]["macro_f1"] - cb.best) > 0.01:
        raise SystemExit(f"Best adapter not restored: final {res['all']['macro_f1']:.4f} vs best {cb.best:.4f}")
    pd.DataFrame({"id": val["id"], "language": val["language"], "y": val["y"], "p_pos": pv}).to_csv(
        run_dir / "val_predictions.csv", index=False, encoding="utf-8")

    lat, lat_runs = None, None
    if a.latency_posts:
        lat, lat_runs = common.measure_latency(lambda t: scorer(model, [t], 1), common.latency_texts(a.latency_posts))

    metrics = {
        "run": run_dir.name, "formulation": "F2 generative Yes/No scoring", "args": vars(a), "revision": rev,
        "val": res, "val_history": cb.history, "train_time_min": round(train_min, 2), "peak_gpu_mem_gb": round(peak_gb, 2),
        "trainable_params": trainable, "total_params": total, "adapter_size_mb": round(dir_mb(best_dir), 1),
        "latency_ms_per_post_bs1_val": None if lat is None else round(lat, 2),
        "latency_repeat_means_ms": lat_runs,
        "latency_protocol": common.LATENCY_PROTOCOL if a.latency_posts == common.LATENCY_POSTS
        else f"NOT the agreed protocol: {a.latency_posts} posts",
        "hardware": torch.cuda.get_device_name(0) if cuda else platform.processor(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    if a.limit_train:
        metrics["full_run_estimate_min_per_epoch"] = round(train_min / (len(train) * max(1, len(cb.history))) * FULL_TRAIN_ROWS, 1)
    if a.predict_test:
        test = common.load("test", a.variant)  # refuses without the NOMINATE commit
        pt = scorer(model, test["input"].tolist(), a.bs * 2)
        pd.DataFrame({"id": test["id"], "p_pos": pt}).to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {run_dir / 'metrics.json'} | train {train_min:.1f} min | peak GPU {peak_gb:.2f} GB | "
          f"latency {metrics['latency_ms_per_post_bs1_val']} ms/post")


if __name__ == "__main__":
    main()
