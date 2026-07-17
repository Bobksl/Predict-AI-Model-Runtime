#!/usr/bin/env python
"""Train one run from a YAML config.

Usage:
    python scripts/train.py --config configs/tile_smoke.yaml
    python scripts/train.py --config configs/tile_sage_hinge.yaml
    python scripts/train.py --config configs/tile_sage_hinge.yaml --cv_fold 2
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import train_from_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cv_fold", type=int, default=None,
                    help="override cv.enabled=true, cv.fold=N")
    ap.add_argument("--limit_files", type=int, default=None,
                    help="override data.limit_files (smoke mode)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.cv_fold is not None:
        cfg.setdefault("cv", {})
        cfg["cv"]["enabled"] = True
        cfg["cv"]["fold"] = args.cv_fold
    if args.limit_files is not None:
        cfg["data"]["limit_files"] = args.limit_files

    result = train_from_config(cfg)
    print("\nRESULT:", result)


if __name__ == "__main__":
    main()
