#!/usr/bin/env python
"""Kaggle GPU kernel: train all 4 layout collections (Phase 3). v3

Runs AS-IS in a Kaggle notebook/script kernel with:
  - Accelerator: GPU (T4/P100)   - Internet: ON
  - Competition data attached: predict-ai-model-runtime

What it does:
  1. pip-installs torch-geometric; clones the public project repo.
  2. Builds the int8 capped-pool config shards into /kaggle/working (approved
     P=2000 spec; ~6.6 GiB, ~15-30 min one-off).
  3. Trains layout_{xla,nlp}_{random,default} with GST on GPU (configs patched
     for Kaggle paths), each saving best.pt under /kaggle/working/artifacts/.
  4. Prints a summary; checkpoints persist as kernel output for download.

Each collection is wrapped in try/except so one failure never kills the rest.
Rerunning skips collections whose best.pt already exists (idempotent-ish).
"""
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO = "https://github.com/Bobksl/Predict-AI-Model-Runtime.git"
WORK = Path("/kaggle/working")
REPO_DIR = WORK / "prj"
SHARDS = WORK / "config_shards"
COLLECTIONS = ["layout_xla_random", "layout_xla_default",
               "layout_nlp_random", "layout_nlp_default"]


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    # 1. deps + code
    sh(f"{sys.executable} -m pip install --quiet torch-geometric pyyaml")
    if not REPO_DIR.exists():
        sh(f"git clone --depth 1 {REPO} {REPO_DIR}")
    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))

    import torch
    import yaml
    print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)

    # 1b. locate the competition data; fall back to kagglehub download if the
    # competition attach did not mount anything under /kaggle/input.
    kin = Path("/kaggle/input")
    print("/kaggle/input contents:", sorted(p.name for p in kin.iterdir()) if kin.exists() else "MISSING", flush=True)
    from src.data.paths import resolve_data_root
    try:
        root = resolve_data_root()
    except FileNotFoundError:
        print("No mounted data root - downloading via kagglehub ...", flush=True)
        import kagglehub
        dl = Path(kagglehub.competition_download("predict-ai-model-runtime"))
        print("kagglehub path:", dl, flush=True)
        # the archive root should contain npz_all/npz/{tile,layout}
        cand = [dl / "npz_all" / "npz", dl / "npz", dl]
        root = next(p for p in cand if (p / "layout").is_dir() or (p / "tile").is_dir())
    os.environ["TPUGRAPHS_DATA_ROOT"] = str(root)
    print("DATA ROOT:", root, flush=True)

    # 2. shards (built once into /kaggle/working; reused on session restart)
    sh(f"{sys.executable} scripts/make_config_shards.py --out_root {SHARDS}",
       env={**os.environ})

    # 3. patched configs -> /kaggle/working/configs
    kcfg_dir = WORK / "configs"
    kcfg_dir.mkdir(exist_ok=True)
    results = {}
    for name in COLLECTIONS:
        ckpt = WORK / "artifacts" / "checkpoints" / f"{name}_sage" / "best.pt"
        if ckpt.exists():
            print(f"[skip] {name}: checkpoint already exists", flush=True)
            continue
        with open(REPO_DIR / "configs" / f"{name}.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["data"]["cache_dir"] = str(WORK / "cache" / name)
        cfg["data"]["num_workers"] = 2
        cfg["shards"]["root"] = str(SHARDS)
        kpath = kcfg_dir / f"{name}.yaml"
        with open(kpath, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        # 4. train (checkpoints under /kaggle/working/artifacts)
        try:
            sh(f"{sys.executable} scripts/train_layout.py --config {kpath} "
               f"--out_root {WORK / 'artifacts' / 'checkpoints'}")
            results[name] = "OK"
        except Exception:
            traceback.print_exc()
            results[name] = "FAILED"

    print("\n===== SUMMARY =====", flush=True)
    for name in COLLECTIONS:
        ckpt = WORK / "artifacts" / "checkpoints" / f"{name}_sage" / "best.pt"
        print(f"{name}: {'checkpoint saved' if ckpt.exists() else results.get(name, 'skipped')}")


if __name__ == "__main__":
    main()
