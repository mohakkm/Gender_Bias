"""F1 (primary): Qwen3 + sequence-classification head + LoRA, QLoRA 4-bit by default.
Trains on train, early-stops and reports on val. Never reads test.csv unless --predict-test is
given AND our pushed 'llm: NOMINATE' commit exists (enforced in common.read_split).

    pilot:  python llm/code/train_cls.py --variant notok --seed 42 --limit-train 200 --limit-val 200 --epochs 1
    full:   python llm/code/train_cls.py --variant notok --seed 42
    resume: add --resume (continues from the last epoch checkpoint after a disconnect)

Outputs go to llm/outputs/<run>/ (git-ignored): metrics.json, val_predictions.csv, adapter/.
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
from datasets import Dataset
from peft import (LoraConfig, TaskType, get_peft_model, load_peft_weights, prepare_model_for_kbit_training,
                  set_peft_model_state_dict)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments,
                          set_seed)

REVISIONS = {"Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
             "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c"}
TARGETS = {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
           "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
FULL_TRAIN_ROWS = 17290


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Hub id or local folder")
    a.add_argument("--revision", default=None, help="Hub commit; required for Hub models not in REVISIONS")
    a.add_argument("--variant", choices=["tok", "notok"], required=True)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--rank", type=int, default=16)
    a.add_argument("--alpha", type=int, default=32)
    a.add_argument("--dropout", type=float, default=0.05)
    a.add_argument("--targets", choices=sorted(TARGETS), default="attn")
    a.add_argument("--lr", type=float, default=2e-4)
    a.add_argument("--epochs", type=float, default=2)
    a.add_argument("--bs", type=int, default=8, help="per-device batch size")
    a.add_argument("--grad-accum", type=int, default=2, help="effective batch = bs x grad-accum")
    a.add_argument("--warmup", type=float, default=0.05)
    a.add_argument("--max-len", type=int, default=common.MAX_LEN)
    a.add_argument("--class-weights", choices=["none", "balanced"], default="none")
    a.add_argument("--no-4bit", action="store_true", help="full precision load (smoke tests, large GPUs)")
    a.add_argument("--no-grad-ckpt", action="store_true")
    a.add_argument("--limit-train", type=int, default=0, help="pilot: sample this many train rows")
    a.add_argument("--limit-val", type=int, default=0)
    a.add_argument("--latency-posts", type=int, default=common.LATENCY_POSTS,
                   help="TEAM_AGREEMENTS.md 3a uses 500; 0 skips (development runs)")
    a.add_argument("--resume", action="store_true")
    a.add_argument("--tag", default="")
    a.add_argument("--predict-test", action="store_true", help="ONLY after the NOMINATE commit")
    return a.parse_args()


def run_name(a):
    short = a.model.rstrip("/\\").replace("\\", "/").split("/")[-1]
    name = f"F1_{short}_{a.variant}_s{a.seed}_r{a.rank}_{a.targets}_lr{a.lr:g}"
    if a.limit_train:
        name += f"_pilot{a.limit_train}"
    return name + (f"_{a.tag}" if a.tag else "")


def to_dataset(tok, df, max_len):
    full = tok(list(df["input"]), add_special_tokens=True)["input_ids"]
    trunc_rate = float(np.mean([len(x) > max_len for x in full]))
    enc = tok(list(df["input"]), truncation=True, max_length=max_len)
    cols = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if "y" in df:
        cols["labels"] = df["y"].tolist()
    return Dataset.from_dict(cols), trunc_rate


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kw):
        super().__init__(*args, **kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        out = model(**inputs)
        logits = out.logits.float()
        w = None if self.class_weights is None else self.class_weights.to(logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=w)
        return (loss, out) if return_outputs else loss


def p_pos(logits):
    logits = logits[0] if isinstance(logits, tuple) else logits
    z = torch.tensor(np.asarray(logits, dtype=np.float32))
    return torch.softmax(z, -1)[:, 1].numpy()


@torch.no_grad()
def predict_probs(model, tok, texts, max_len, bs=32):
    """p_pos in the SAME order as texts. Do not use trainer.predict: with group_by_length=True,
    transformers 4.57 also length-sorts eval/predict batches, so its outputs are not in dataset order."""
    model.eval()
    dev = next(model.parameters()).device
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts), dtype=np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            logits = model(**enc).logits
        out[idx] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return out


@torch.no_grad()
def check_pooling(model, tok, texts):
    """The head must read the last real token: a post's logits must not change when it is padded."""
    model.eval()
    dev = next(model.parameters()).device
    alone = [model(**tok([t], return_tensors="pt").to(dev)).logits.float() for t in texts]
    batch = model(**tok(texts, return_tensors="pt", padding=True).to(dev)).logits.float()
    diff = max(float((batch[i] - alone[i][0]).abs().max()) for i in range(len(texts)))
    print(f"pooling check: max logit difference padded vs unpadded = {diff:.4f}")
    if diff > 0.05:
        raise SystemExit("Pooling check FAILED: the classification head is reading pad tokens.")
    return diff


def dir_mb(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


def main():
    a = parse()
    is_hub = not common.Path(a.model).exists()
    rev = a.revision or REVISIONS.get(a.model)
    if is_hub and rev is None:
        raise SystemExit(f"Pin a revision for {a.model} with --revision <commit sha>.")
    four_bit = not a.no_4bit
    set_seed(a.seed)
    run_dir = common.OUTPUTS / run_name(a)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run: {run_dir.name}")

    train, val = common.load("train", a.variant), common.load("val", a.variant)
    if a.limit_train:
        train = train.sample(a.limit_train, random_state=a.seed).reset_index(drop=True)
    if a.limit_val:
        val = val.sample(a.limit_val, random_state=a.seed).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(a.model, revision=rev if is_hub else None)
    tok.padding_side = "right"
    if tok.pad_token is None:
        raise SystemExit("Tokenizer has no pad token; set one explicitly before training.")
    train_ds, trunc_train = to_dataset(tok, train, a.max_len)
    val_ds, trunc_val = to_dataset(tok, val, a.max_len)
    print(f"rows train {len(train)} val {len(val)} | truncated at {a.max_len}: train {trunc_train:.2%} val {trunc_val:.2%}")

    kw = dict(num_labels=2, revision=rev if is_hub else None)
    if four_bit:
        kw.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16), device_map={"": 0}, dtype=torch.float16)
    else:
        kw.update(dtype=torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, **kw)
    model.config.pad_token_id = tok.pad_token_id
    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    elif torch.cuda.is_available():
        model = model.cuda()
    model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=a.rank, lora_alpha=a.alpha,
                                             lora_dropout=a.dropout, target_modules=TARGETS[a.targets]))
    trainable, total = model.get_nb_trainable_parameters()
    print(f"trainable params {trainable:,} of {total:,} ({trainable / total:.2%})")
    pooling_diff = check_pooling(model, tok, val["input"].head(4).tolist())

    weights = None
    if a.class_weights == "balanced":
        counts = np.bincount(train["y"], minlength=2)
        weights = torch.tensor(len(train) / (2 * counts), dtype=torch.float32)
        print(f"class weights {weights.tolist()}")

    cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"), per_device_train_batch_size=a.bs,
        per_device_eval_batch_size=a.bs * 2, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, num_train_epochs=a.epochs, lr_scheduler_type="cosine", warmup_ratio=a.warmup,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, fp16=cuda, logging_steps=20,
        report_to="none", seed=a.seed, data_seed=a.seed, group_by_length=True,
        gradient_checkpointing=not a.no_grad_ckpt, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if four_bit else "adamw_torch", dataloader_num_workers=0)
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok,
        data_collator=DataCollatorWithPadding(tok), class_weights=weights,
        compute_metrics=lambda ev: {"macro_f1": common.scores(ev.label_ids, p_pos(ev.predictions))["macro_f1"]},
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] if a.epochs > 1 else None)

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    has_ckpt = any((run_dir / "checkpoints").glob("checkpoint-*"))
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=True if (a.resume and has_ckpt) else None)
    train_min = (time.perf_counter() - t0) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0

    # Load the best epoch's adapter explicitly, predict in input order, and check the score matches
    # what the Trainer recorded for that epoch (guards against misaligned or wrong-epoch predictions).
    best = trainer.state.best_model_checkpoint
    if best:
        set_peft_model_state_dict(model, load_peft_weights(best, device=str(next(model.parameters()).device)))
    pv = predict_probs(model, tok, val["input"].tolist(), a.max_len, a.bs * 2)
    res = common.report(val, pv)
    if best and abs(res["all"]["macro_f1"] - trainer.state.best_metric) > 0.01:
        raise SystemExit(f"Best checkpoint not restored: final val macro-F1 {res['all']['macro_f1']:.4f} "
                         f"vs best epoch {trainer.state.best_metric:.4f}")
    print(f"VAL macro-F1 (threshold 0.5): {common.fmt(res)}")
    pd.DataFrame({"id": val["id"], "language": val["language"], "y": val["y"], "p_pos": pv}).to_csv(
        run_dir / "val_predictions.csv", index=False, encoding="utf-8")

    adapter = run_dir / "adapter"
    model.save_pretrained(str(adapter))
    lat, lat_runs = None, None
    if a.latency_posts:
        lat, lat_runs = common.measure_latency(lambda t: predict_probs(model, tok, [t], a.max_len, 1),
                                               common.latency_texts(a.latency_posts))

    metrics = {
        "run": run_dir.name, "formulation": "F1 classification head", "args": vars(a), "revision": rev,
        "val": res, "train_time_min": round(train_min, 2), "peak_gpu_mem_gb": round(peak_gb, 2),
        "trainable_params": trainable, "total_params": total, "adapter_size_mb": round(dir_mb(adapter), 1),
        "latency_ms_per_post_bs1_val": None if lat is None else round(lat, 2),
        "latency_repeat_means_ms": lat_runs,
        "latency_protocol": common.LATENCY_PROTOCOL if a.latency_posts == common.LATENCY_POSTS
        else f"NOT the agreed protocol: {a.latency_posts} posts",
        "truncation": {"train": trunc_train, "val": trunc_val}, "pooling_check_max_diff": pooling_diff,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "hardware": torch.cuda.get_device_name(0) if cuda else platform.processor(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    if a.limit_train:
        per_row = train_min / (len(train) * a.epochs)
        metrics["full_run_estimate_min_per_epoch"] = round(per_row * FULL_TRAIN_ROWS, 1)
        print(f"pilot estimate: {metrics['full_run_estimate_min_per_epoch']} min per full epoch on this hardware "
              f"(rough: small pilots overstate per-row overhead)")

    if a.predict_test:
        test = common.load("test", a.variant)  # refuses without the NOMINATE commit
        pt = predict_probs(model, tok, test["input"].tolist(), a.max_len, a.bs * 2)
        pd.DataFrame({"id": test["id"], "p_pos": pt}).to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8")
        metrics["test_predictions"] = "test_predictions.csv written (no labels read here)"

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {run_dir / 'metrics.json'} | train {train_min:.1f} min | peak GPU {peak_gb:.2f} GB | "
          f"latency {metrics['latency_ms_per_post_bs1_val']} ms/post")


if __name__ == "__main__":
    main()
