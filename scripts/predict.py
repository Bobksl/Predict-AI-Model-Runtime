#!/usr/bin/env python
"""Run inference over the tile:xla test split, save per-file rankings.

Usage:
    python scripts/predict.py --checkpoint artifacts/checkpoints/tile_sage_hinge/best.pt \
        --out artifacts/submissions/tile_rankings.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.paths import resolve_data_root, list_npz    # noqa: E402
from src.data.normalize import NodeFeatNormalizer          # noqa: E402
from src.models import build_model                         # noqa: E402
from src.inference.predict import score_all_configs, rank_configs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default="artifacts/submissions/tile_rankings.json")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: best_val_opa={ckpt['best_val_opa']:.4f} "
          f"@ epoch {ckpt['best_epoch']}")

    normalizer = NodeFeatNormalizer.load(cfg["data"]["normalizer"])
    root = resolve_data_root(args.data_root)
    test_files = list_npz(root, cfg["data"]["collection"], "test")
    print(f"Scoring {len(test_files)} test files ...")

    rankings = {}
    for i, fp in enumerate(test_files):
        scores = score_all_configs(
            model, str(fp), normalizer, device,
            add_reverse_edges=cfg["data"].get("add_reverse_edges", True))
        rankings[fp.stem] = rank_configs(scores)
        if (i + 1) % 100 == 0 or i + 1 == len(test_files):
            print(f"  {i+1}/{len(test_files)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rankings, f)
    print(f"Wrote {out}  ({len(rankings)} graphs)")


if __name__ == "__main__":
    main()
