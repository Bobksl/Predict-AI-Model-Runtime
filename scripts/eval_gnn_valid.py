#!/usr/bin/env python
"""GNN valid OPA on FULL config lists, apples-to-apples with eval_baselines.

Scores every config of every valid graph with the checkpointed model, then
computes OPA on the identical seeded <=256-config subsample protocol used by
scripts/eval_baselines.py (P2 review recommendation c: the training-loop OPA is
estimated on k=16 sampled configs; this gives the directly comparable number).

Usage:
    python scripts/eval_gnn_valid.py --checkpoint artifacts/checkpoints/tile_sage_listmle/best.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.paths import resolve_data_root, list_npz      # noqa: E402
from src.data.cache import read_bundle                      # noqa: E402
from src.data.normalize import NodeFeatNormalizer            # noqa: E402
from src.models import build_model                           # noqa: E402
from src.inference.predict import score_all_configs          # noqa: E402
from src.training.metrics import opa                         # noqa: E402

MAX_CONFIGS = 256  # must match eval_baselines.MAX_CONFIGS
SEED = 0           # must match eval_baselines._load_pairs seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    normalizer = NodeFeatNormalizer.load(cfg["data"]["normalizer"])

    root = resolve_data_root(args.data_root)
    files = list_npz(root, cfg["data"]["collection"], "valid")
    rng = np.random.default_rng(SEED)   # same subsample stream as baselines
    vals = []
    for i, fp in enumerate(files):
        b = read_bundle(str(fp))
        y = b["config_runtime"]
        scores = score_all_configs(
            model, str(fp), normalizer, device,
            add_reverse_edges=cfg["data"].get("add_reverse_edges", True))
        if len(y) > MAX_CONFIGS:
            idx = rng.choice(len(y), size=MAX_CONFIGS, replace=False)
            y, scores = y[idx], scores[idx]
        if len(y) < 2:
            continue
        v = opa(torch.as_tensor(scores), torch.as_tensor(y))
        if not torch.isnan(v):
            vals.append(float(v))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)}  running OPA={np.mean(vals):.4f}")

    print(f"\nFull-config GNN valid OPA ({Path(args.checkpoint).parent.name}, "
          f"<= {MAX_CONFIGS}-config subsample, seed {SEED}): "
          f"{np.mean(vals):.4f}  over {len(vals)} graphs")


if __name__ == "__main__":
    main()
