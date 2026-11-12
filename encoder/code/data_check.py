"""D1 data check: SHA256 of the frozen files against SHA256SUMS.txt, plus row and label counts vs the brief.
Reads test.csv only to hash it and count rows (no model touches it, no test label stats printed).

    python encoder/code/data_check.py
"""
import hashlib
import sys

from common import ROOT, cmp

D = ROOT / "data_v1"
ok = True
want = dict(line.split()[::-1] for line in (D / "SHA256SUMS.txt").read_text().splitlines() if line.strip())
for name in ("train.csv", "val.csv", "test.csv"):
    got = hashlib.sha256((D / name).read_bytes()).hexdigest()
    match = got == want.get(name)
    ok &= match
    print(f"{'PASS' if match else 'FAIL'}  sha256 {name} {got[:16]}...")

expect = {"train": (17455, 17290), "val": (1950, 1931), "test": (3756, 3756)}
for split, (raw_n, usable_n) in expect.items():
    raw = cmp.read_split(split)
    use = cmp.prep(raw)
    match = len(raw) == raw_n and len(use) == usable_n
    ok &= match
    rate = "" if split == "test" else f", positive rate by language {use.groupby('language')['y'].mean().round(3).to_dict()}"
    print(f"{'PASS' if match else 'FAIL'}  {split}: {len(raw)} rows, {len(use)} usable{rate}")

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
