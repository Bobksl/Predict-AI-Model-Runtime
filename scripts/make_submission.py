#!/usr/bin/env python
"""Assemble + validate the tile:xla submission CSV from predict.py's rankings.

Usage:
    python scripts/make_submission.py \
        --rankings artifacts/submissions/tile_rankings.json \
        --out artifacts/submissions/submission_tile.csv
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--out", default="artifacts/submissions/submission_tile.csv")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--collection", default="tile:xla")
    args = ap.parse_args()

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
