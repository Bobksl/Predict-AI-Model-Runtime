#!/usr/bin/env python
"""Assemble + validate the tile:xla submission CSV — ONE command end-to-end.

Usage (one-command, gate P2):
    python scripts/make_submission.py --checkpoint artifacts/checkpoints/tile_sage_listmle/best.pt
Or from precomputed rankings:
    python scripts/make_submission.py --rankings artifacts/submissions/tile_rankings.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.paths import resolve_data_root, list_npz          # noqa: E402
from src.data.cache import read_bundle                          # noqa: E402
from src.inference.submission import assemble_submission, validate_submission  # noqa: E402


def _rankings_from_checkpoint(checkpoint: str, data_root):
    """Score the whole test split from a checkpoint (single-command path)."""
    import torch
    from src.data.normalize import NodeFeatNormalizer
    from src.models import build_model
    from src.inference.predict import score_all_configs, rank_configs

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    normalizer = NodeFeatNormalizer.load(cfg["data"]["normalizer"])
    root = resolve_data_root(data_root)
    files = list_npz(root, cfg["data"]["collection"], "test")
    print(f"Scoring {len(files)} test files from {checkpoint} ...")
    rankings = {}
    for i, fp in enumerate(files):
        scores = score_all_configs(
            model, str(fp), normalizer, device,
            add_reverse_edges=cfg["data"].get("add_reverse_edges", True))
        rankings[fp.stem] = rank_configs(scores)
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print(f"  {i+1}/{len(files)}")
    return rankings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", default=None, help="precomputed rankings json")
    ap.add_argument("--checkpoint", default=None,
                    help="one-command mode: predict + assemble from best.pt")
    ap.add_argument("--out", default="artifacts/submissions/submission_tile.csv")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--collection", default="tile:xla")
    args = ap.parse_args()

    if (args.rankings is None) == (args.checkpoint is None):
        ap.error("provide exactly one of --rankings or --checkpoint")
    if args.checkpoint:
        rankings = _rankings_from_checkpoint(args.checkpoint, args.data_root)
    else:
        with open(args.rankings, "r", encoding="utf-8") as f:
            rankings = json.load(f)

    root = resolve_data_root(args.data_root)
    test_files = list_npz(root, args.collection, "test")
    n_configs_by_stem = {}
    for fp in test_files:
        b = read_bundle(str(fp))
        n_configs_by_stem[fp.stem] = int(b["config_runtime"].shape[0])

    assemble_submission(rankings, args.collection, args.out)
    validate_submission(args.out, list(n_configs_by_stem.keys()), args.collection,
                        n_configs_by_stem)
    print(f"Wrote + validated {args.out}  ({len(n_configs_by_stem)} rows)")


if __name__ == "__main__":
    main()
