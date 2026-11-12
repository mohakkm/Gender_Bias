"""Print the exact library versions and GPU of this session in requirements.txt format.
Run once on the platform used for final runs and paste the output into requirements.txt and STATE.md.

    python encoder/code/env.py
"""
import importlib.metadata as md

for pkg in ["torch", "transformers", "tokenizers", "sentencepiece", "scikit-learn", "scipy", "pandas", "numpy"]:
    try:
        print(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"# {pkg} not installed")
try:
    import torch
    print(f"# GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}, CUDA {torch.version.cuda}")
except ImportError:
    pass
