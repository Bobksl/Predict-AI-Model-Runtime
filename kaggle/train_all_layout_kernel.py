#!/usr/bin/env python
"""Kaggle GPU kernel: train all 4 layout collections (Phase 3). v5

Runs AS-IS in a Kaggle SCRIPT kernel with: GPU on, Internet on, competition data
`predict-ai-model-runtime` attached.

Pipeline: pin a working torch/PyG pair -> preflight -> clone repo -> build int8
config shards into /kaggle/working -> train layout_{xla,nlp}_{random,default} with
GST on GPU. Each collection is failure-isolated; reruns skip finished checkpoints.

Environment lesson (v1-v4): Kaggle's default torch 2.10+cu128 (a) has NO sm_60
kernels for the P100 Kaggle often assigns and (b) breaks torch-geometric's dynamo
guards. Fix: pin torch 2.4.1+cu118 (has sm_60/sm_75) + torch-geometric 2.6.1, and
NEVER import torch in this parent process (its sys.modules would cache the stale
build) -- all torch work happens in fresh subprocesses.
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
CKPTS = WORK / "artifacts" / "checkpoints"
COLLECTIONS = ["layout_xla_random", "layout_xla_default",
               "layout_nlp_random", "layout_nlp_default"]


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def main():
    if not REPO_DIR.exists():
        sh(f"git clone --depth 1 {REPO} {REPO_DIR}")
    os.chdir(REPO_DIR)

    # 1. Pin a torch/PyG pair that has sm_60 kernels AND matches torch-geometric.
    #    --no-deps stops torch-geometric from re-upgrading torch back to 2.10.
    sh(f"{sys.executable} -m pip -q uninstall -y torch torchvision torchaudio")
    sh(f"{sys.executable} -m pip -q install torch==2.4.1 "
       f"--index-url https://download.pytorch.org/whl/cu118")
    sh(f"{sys.executable} -m pip -q install --no-deps torch-geometric==2.6.1")

    # 2. Preflight in a FRESH process: fail fast (~2 min) if torch/PyG/CUDA is
    #    wrong, before the ~20-min shard build.
    sh(f'{sys.executable} -c "'
       f"import torch; from torch_geometric.nn import SAGEConv; "
       f"print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
       f"torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); "
       f"x=torch.randn(8,8,device='cuda'); print('gpu matmul ok', float((x@x).sum()))"
       f'"')

    # 3. Config shards (numpy only; data root resolved by src/data/paths.py, which
    #    now includes the /kaggle/input/competitions/... mount).
    sh(f"{sys.executable} scripts/make_config_shards.py --out_root {SHARDS}")

    # 4. Train each collection in its own subprocess (fresh, correct torch).
    import yaml
    kcfg = WORK / "configs"
    kcfg.mkdir(exist_ok=True)
    results = {}
    for name in COLLECTIONS:
        if (CKPTS / f"{name}_sage" / "best.pt").exists():
            print(f"[skip] {name}: checkpoint already exists", flush=True)
            continue
        with open(REPO_DIR / "configs" / f"{name}.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["data"]["cache_dir"] = str(WORK / "cache" / name)
        cfg["data"]["num_workers"] = 2
        cfg["shards"]["root"] = str(SHARDS)
        kpath = kcfg / f"{name}.yaml"
        with open(kpath, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        try:
            sh(f"{sys.executable} scripts/train_layout.py --config {kpath} "
               f"--out_root {CKPTS}")
            results[name] = "OK"
        except Exception:
            traceback.print_exc()
            results[name] = "FAILED"

    print("\n===== SUMMARY =====", flush=True)
    for name in COLLECTIONS:
        ok = (CKPTS / f"{name}_sage" / "best.pt").exists()
        print(f"{name}: {'checkpoint saved' if ok else results.get(name, 'skipped')}",
              flush=True)


if __name__ == "__main__":
    main()
