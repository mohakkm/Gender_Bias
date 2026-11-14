"""Package llm/code into one self-contained Kaggle script job.

    python llm/code/kaggle/build_job.py --job e0_pilot --owner <kaggle-username> --out <folder>
    kaggle kernels push -p <folder>            (then: kaggle kernels status / output)

The generated run.py writes our scripts into /kaggle/working/repo/llm/code, copies train.csv and
val.csv from the private Kaggle dataset into repo/data_v1 after checking their SHA256 against
data_v1/SHA256SUMS.txt, refuses to run if any test.csv is present, runs the job's commands on ONE
GPU (same as the agreed single-T4 final runs), deletes checkpoints, and leaves small results in
/kaggle/working/results (metrics, val predictions, adapters, logs, summary.json).

Local test with tiny models (no Kaggle):
    set SERI_WORK, SERI_DATA_DIR, SERI_SKIP_PIP=1, SERI_MODEL_CLS, SERI_MODEL_LM, then python run.py
"""
import argparse
import base64
import json
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parents[1]
DATASET_SLUG = "seri-uli-data-v1-trainval"
SHIP = ["common.py", "train_cls.py", "e0_prompt.py", "train_gen.py"]
PINS = ["transformers==4.57.3", "peft==0.21.2", "accelerate==1.15.0", "bitsandbytes==0.50.2"]

E0 = ["llm/code/e0_prompt.py", "--model", "{MODEL_LM}", "--bs", "16"]
F1 = ["llm/code/train_cls.py", "--model", "{MODEL_CLS}"]
JOBS = {
    # D1/D2: prompting baseline on val, plus 4-bit F1 pilots for timing, memory and the pooling check.
    "e0_pilot": [
        E0 + ["--variant", "notok", "--shots", "0"],
        E0 + ["--variant", "notok", "--shots", "6"],
        E0 + ["--variant", "tok", "--shots", "0"],
        E0 + ["--variant", "tok", "--shots", "6"],
        F1 + ["--variant", "notok", "--seed", "42", "--limit-train", "200", "--limit-val", "400",
              "--epochs", "1", "--latency-posts", "100"],
        F1 + ["--variant", "notok", "--seed", "42", "--limit-train", "1000", "--limit-val", "400",
              "--epochs", "1", "--latency-posts", "0"],
    ],
    # D3: the E0 run that ran out of memory, then full F1 seed 42 for both variants (about 75 min each on a T4).
    "f1_full_s42": [
        E0 + ["--variant", "notok", "--shots", "6"],
        F1 + ["--variant", "notok", "--seed", "42", "--epochs", "2", "--latency-posts", "200"],
        F1 + ["--variant", "tok", "--seed", "42", "--epochs", "2", "--latency-posts", "200"],
    ],
    # E3 ablations on val, notok seed 42 (the nomination variant). Baseline: r16 attn, 2 epochs = 0.756.
    "ablate_17b": [
        F1 + ["--variant", "notok", "--seed", "42", "--epochs", "3", "--latency-posts", "0", "--tag", "ep3"],
        F1 + ["--variant", "notok", "--seed", "42", "--epochs", "2", "--rank", "32", "--alpha", "64", "--latency-posts", "0"],
        F1 + ["--variant", "notok", "--seed", "42", "--epochs", "2", "--targets", "all", "--latency-posts", "0"],
    ],
    # E2: F2 generative scoring, seed 42, both variants. Batch 4 x 4 (full-vocab logits in training).
    "f2_s42": [
        ["llm/code/train_gen.py", "--variant", "notok", "--seed", "42", "--epochs", "2", "--bs", "4",
         "--grad-accum", "4"],
        ["llm/code/train_gen.py", "--variant", "tok", "--seed", "42", "--epochs", "2", "--bs", "4",
         "--grad-accum", "4", "--latency-posts", "0"],
    ],
    # E3 second model size: Qwen3-4B, same F1 recipe, notok seed 42.
    "qwen4b": [
        ["llm/code/train_cls.py", "--model", "Qwen/Qwen3-4B", "--variant", "notok", "--seed", "42",
         "--epochs", "2", "--latency-posts", "200"],
    ],
}

RUNNER = r'''
import base64, glob, hashlib, json, os, platform, shutil, subprocess, sys, time
JOB = json.loads(base64.b64decode("__JOB__").decode("utf-8"))
WORK = os.environ.get("SERI_WORK", "/kaggle/working")
REPO = os.path.join(WORK, "repo")
RESULTS = os.path.join(WORK, "results")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one GPU, like the agreed single-T4 final runs
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
SUBS = {"{MODEL_CLS}": os.environ.get("SERI_MODEL_CLS", "Qwen/Qwen3-1.7B"),
        "{MODEL_LM}": os.environ.get("SERI_MODEL_LM", "Qwen/Qwen3-1.7B")}

def find_data():
    if os.environ.get("SERI_DATA_DIR"):
        return os.environ["SERI_DATA_DIR"]
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if not hits:
        raise SystemExit("train.csv not found under /kaggle/input")
    return os.path.dirname(hits[0])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

t_start = time.time()
os.makedirs(RESULTS, exist_ok=True)
for rel, text in JOB["files"].items():
    path = os.path.join(REPO, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

src = find_data()
if glob.glob(os.path.join(src, "**", "test*.csv"), recursive=True) or glob.glob("/kaggle/input/**/test.csv", recursive=True):
    raise SystemExit("REFUSED: a test.csv is attached. Test data must not be on Kaggle before the NOMINATE commit.")
os.makedirs(os.path.join(REPO, "data_v1"), exist_ok=True)
for name, want in JOB["sha256"].items():
    dst = os.path.join(REPO, "data_v1", name)
    shutil.copyfile(os.path.join(src, name), dst)
    got = sha256(dst)
    if got != want:
        raise SystemExit(f"SHA256 mismatch for {name}: {got} != {want}")
print("data hashes OK:", ", ".join(JOB["sha256"]), flush=True)

if not os.environ.get("SERI_SKIP_PIP"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *JOB["pins"]], check=True)

def version(mod):
    try:
        return __import__(mod).__version__
    except Exception as e:
        return f"unavailable ({type(e).__name__})"

env = {"python": platform.python_version(), "torch": version("torch"), "transformers": version("transformers"),
       "peft": version("peft"), "bitsandbytes": version("bitsandbytes")}
try:
    import torch
    env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    env["gpu_count_visible"] = torch.cuda.device_count()
except Exception as e:
    env["gpu"] = f"unavailable ({e})"
print("environment:", env, flush=True)

runs = []
os.makedirs(os.path.join(RESULTS, "logs"), exist_ok=True)
for i, cmd in enumerate(JOB["commands"]):
    cmd = [SUBS.get(c, c) for c in cmd]
    log = os.path.join(RESULTS, "logs", f"{i:02d}_{os.path.basename(cmd[0]).replace('.py', '')}.log")
    print(f"\n=== [{i}] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen([sys.executable, *cmd], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            lf.write(line)
            if any(k in line for k in ("VAL", "E0 ", "AUROC", "pooling", "pilot estimate", "wrote", "rows ",
                                       "trainable", "Error", "REFUSED", "FAILED", "Traceback")):
                print(line, end="", flush=True)
        rc = p.wait()
    runs.append({"cmd": cmd, "returncode": rc, "minutes": round((time.time() - t0) / 60, 2), "log": os.path.relpath(log, RESULTS)})
    print(f"=== [{i}] exit {rc} after {runs[-1]['minutes']} min", flush=True)

out_dir = os.path.join(REPO, "llm", "outputs")
for ck in glob.glob(os.path.join(out_dir, "*", "checkpoints")):
    shutil.rmtree(ck, ignore_errors=True)  # keep the download small: adapters and metrics only
if os.path.isdir(out_dir):
    shutil.copytree(out_dir, os.path.join(RESULTS, "outputs"), dirs_exist_ok=True)
metrics = {}
for m in glob.glob(os.path.join(RESULTS, "outputs", "*", "metrics.json")):
    with open(m, encoding="utf-8") as f:
        metrics[os.path.basename(os.path.dirname(m))] = json.load(f)
summary = {"job": JOB["name"], "built_from_commit": JOB["commit"], "environment": env, "runs": runs,
           "metrics": metrics, "total_minutes": round((time.time() - t_start) / 60, 2)}
with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
shutil.rmtree(REPO, ignore_errors=True)
print("\nDONE exit codes:", json.dumps({f"[{i}]": r["returncode"] for i, r in enumerate(runs)}), flush=True)
'''


def expected_hashes():
    want = {}
    for line in (ROOT / "data_v1" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] in ("train.csv", "val.csv"):
            want[parts[1]] = parts[0]
    assert set(want) == {"train.csv", "val.csv"}, want
    return want


def git_commit():
    import subprocess
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "llm/code"], capture_output=True, text=True)
    return r.stdout.strip() + ("+uncommitted" if dirty.stdout.strip() else "")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--job", choices=sorted(JOBS), required=True)
    a.add_argument("--owner", required=True, help="Kaggle username that owns the kernel and the dataset")
    a.add_argument("--out", required=True)
    a.add_argument("--machine", default="NvidiaTeslaT4")
    args = a.parse_args()

    files = {f"llm/code/{n}": (CODE / n).read_text(encoding="utf-8") for n in SHIP}
    job = {"name": args.job, "commit": git_commit(), "files": files, "sha256": expected_hashes(),
           "pins": PINS, "commands": JOBS[args.job]}
    payload = base64.b64encode(json.dumps(job).encode("utf-8")).decode("ascii")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(RUNNER.replace("__JOB__", payload), encoding="utf-8", newline="\n")
    slug = f"seri-llm-{args.job.replace('_', '-')}"
    meta = {"id": f"{args.owner}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": args.machine, "dataset_sources": [f"{args.owner}/{DATASET_SLUG}"],
            "competition_sources": [], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {out / 'run.py'} ({len(payload) // 1024} KB) from {job['commit']} | kernel {meta['id']}")


if __name__ == "__main__":
    main()
