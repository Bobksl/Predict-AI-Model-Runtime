#!/usr/bin/env python
"""Assemble + validate one layout-collection submission CSV from a checkpoint.

Usage:
    python scripts/make_layout_submission.py \
        --checkpoint artifacts/checkpoints/layout_xla_random_sage/best.pt \
        --out artifacts/submissions/submission_layout_xla_random.csv

    # smoke: score only the first 2 test files
    python scripts/make_layout_submission.py --checkpoint ... --out ... --limit_files 2
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                       # noqa: E402

from src.data.paths import resolve_data_root, list_npz             # noqa: E402
from src.data.normalize import NodeFeatNormalizer                  # noqa: E402
from src.models import build_model                                 # noqa: E402
from src.inference.predict_layout import score_all_configs_layout, rank_configs  # noqa: E402
from src.inference.submission import assemble_submission, validate_submission    # noqa: E402

# Task E chunk-size guidance (brief §Task E): xla's worst graph (N=43,615) needs a
# small chunk to stay under ~2 GB; nlp graphs are ~2x smaller so a larger chunk is
# still cheap. Overridable via --chunk_size.
_CHUNK_BY_SOURCE = {"xla": 32, "nlp": 128}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_size", type=int, default=None,
                    help="default: 32 for xla, 128 for nlp")
    ap.add_argument("--limit_files", type=int, default=None,
                    help="smoke: score only the first N test files")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    collection = cfg["data"]["collection"]
    source = collection.split(":")[1]
    chunk_size = args.chunk_size or _CHUNK_BY_SOURCE.get(source, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint: best_val_opa={ckpt['best_val_opa']:.4f} "
          f"@ epoch {ckpt['best_epoch']}  collection={collection}")

    normalizer = NodeFeatNormalizer.load(cfg["data"]["normalizer"])
    root = resolve_data_root(args.data_root)
    test_files = list_npz(root, collection, "test")
    if args.limit_files:
        test_files = test_files[: args.limit_files]
    print(f"Scoring {len(test_files)} test files (chunk_size={chunk_size}) ...")

    rankings = {}
    n_configs_by_stem = {}
    for i, fp in enumerate(test_files):
        scores = score_all_configs_layout(
            model, str(fp), normalizer, device, chunk_size=chunk_size,
            add_reverse_edges=cfg["data"].get("add_reverse_edges", True))
        rankings[fp.stem] = rank_configs(scores)
        n_configs_by_stem[fp.stem] = len(scores)
        if (i + 1) % 5 == 0 or i + 1 == len(test_files):
            print(f"  {i+1}/{len(test_files)}")

    assemble_submission(rankings, collection, args.out)
    validate_submission(args.out, list(n_configs_by_stem.keys()), collection,
                        n_configs_by_stem)
    print(f"Wrote + validated {args.out}  ({len(n_configs_by_stem)} rows)")


if __name__ == "__main__":
    main()
