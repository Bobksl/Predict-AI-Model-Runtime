#!/usr/bin/env python
"""Trivial baselines the GNN must beat (brief T5).

(a) Random permutation — expected OPA ~0.50.
(b) Best single raw config_feat column (both signs), selected on TRAIN, reported
    on VALID — the bar the GNN must clear by a margin.

Usage:
    python scripts/eval_baselines.py [--data_root DIR] [--k_configs 16]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.paths import resolve_data_root, list_npz          # noqa: E402
from src.data.cache import read_bundle                          # noqa: E402
from src.training.metrics import opa                            # noqa: E402
import torch                                                    # noqa: E402


def _load_pairs(files):
    """Return list of (config_feat [c,24], y int64 [c]) per file."""
    out = []
    for f in files:
        b = read_bundle(str(f))
        out.append((b["config_feat"].astype(np.float64), b["config_runtime"]))
    return out


def random_baseline(pairs, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _cf, y in pairs:
        n = len(y)
        if n < 2:
            continue
        s = rng.permutation(n).astype(np.float64)
        v = opa(torch.as_tensor(s), torch.as_tensor(y))
        if not torch.isnan(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def best_column_baseline(train_pairs, valid_pairs):
    n_cols = train_pairs[0][0].shape[1]
    best_col, best_sign, best_train_opa = None, 1, -1.0
    for col in range(n_cols):
        for sign in (1, -1):
            vals = []
            for cf, y in train_pairs:
                if len(y) < 2:
                    continue
                s = sign * cf[:, col]
                v = opa(torch.as_tensor(s), torch.as_tensor(y))
                if not torch.isnan(v):
                    vals.append(float(v))
            m = float(np.mean(vals)) if vals else -1.0
            if m > best_train_opa:
                best_train_opa, best_col, best_sign = m, col, sign

    vals = []
    for cf, y in valid_pairs:
        if len(y) < 2:
            continue
        s = best_sign * cf[:, best_col]
        v = opa(torch.as_tensor(s), torch.as_tensor(y))
        if not torch.isnan(v):
            vals.append(float(v))
    valid_opa = float(np.mean(vals)) if vals else float("nan")
    return best_col, best_sign, best_train_opa, valid_opa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    train_files = list_npz(root, "tile:xla", "train")
    valid_files = list_npz(root, "tile:xla", "valid")
    print(f"train files: {len(train_files)}  valid files: {len(valid_files)}")

    train_pairs = _load_pairs(train_files)
    valid_pairs = _load_pairs(valid_files)

    r_opa = random_baseline(valid_pairs)
    print(f"\n(a) random baseline: valid OPA = {r_opa:.4f}  (expected ~0.50)")

    col, sign, train_opa, valid_opa = best_column_baseline(train_pairs, valid_pairs)
    print(f"(b) best single config_feat column: col={col} sign={sign:+d}  "
          f"train OPA = {train_opa:.4f}  valid OPA = {valid_opa:.4f}")

    bar = max(r_opa, valid_opa) + 0.10
    print(f"\nGate-P2 bar for the GNN: valid OPA >= 0.70 AND >= {bar:.4f} "
          f"(best-trivial {max(r_opa, valid_opa):.4f} + 0.10)")


if __name__ == "__main__":
    main()
