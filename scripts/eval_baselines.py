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


MAX_CONFIGS = 256  # exact pairwise OPA is O(c^2); subsample large config lists


def _load_pairs(files, seed=0):
    """Return list of (config_feat [c',24], y int64 [c']) per file.

    Configs are subsampled to MAX_CONFIGS with a fixed seed: full lists reach
    ~20k configs and an exact [c,c] pair matrix at that size is ~3 GB. 256
    configs give 32k pairs per graph — ample for a trivial-baseline estimate.
    """
    rng = np.random.default_rng(seed)
    out = []
    for f in files:
        b = read_bundle(str(f))
        cf = b["config_feat"].astype(np.float64)
        y = b["config_runtime"]
        if len(y) > MAX_CONFIGS:
            idx = rng.choice(len(y), size=MAX_CONFIGS, replace=False)
            cf, y = cf[idx], y[idx]
        out.append((cf, y))
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


def _column_opas(pairs):
    """Per-file-mean OPA for every (column, sign) ranker, vectorised.

    For each file builds the [c,c] label-sign matrix once and the [c,c,24]
    score-sign tensor once, so all 48 rankers are evaluated in a single pass.
    Same OPA definition as src.training.metrics: valid pairs are dy!=0; a
    score-tie on a valid pair counts as discordant.
    """
    n_cols = pairs[0][0].shape[1]
    sums_pos = np.zeros(n_cols)
    sums_neg = np.zeros(n_cols)
    n_files = 0
    for cf, y in pairs:
        if len(y) < 2:
            continue
        dy = np.sign(y[:, None] - y[None, :]).astype(np.int8)       # [c,c]
        valid = dy != 0
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        ds = np.sign(cf[:, None, :] - cf[None, :, :]).astype(np.int8)  # [c,c,24]
        same = (ds == dy[:, :, None]) & valid[:, :, None]
        opposite = (ds == -dy[:, :, None]) & valid[:, :, None]
        sums_pos += same.sum(axis=(0, 1)) / n_valid
        sums_neg += opposite.sum(axis=(0, 1)) / n_valid
        n_files += 1
    return sums_pos / n_files, sums_neg / n_files


def best_column_baseline(train_pairs, valid_pairs):
    opa_pos, opa_neg = _column_opas(train_pairs)
    if opa_pos.max() >= opa_neg.max():
        best_col, best_sign = int(np.argmax(opa_pos)), 1
        best_train_opa = float(opa_pos[best_col])
    else:
        best_col, best_sign = int(np.argmax(opa_neg)), -1
        best_train_opa = float(opa_neg[best_col])

    v_pos, v_neg = _column_opas(valid_pairs)
    valid_opa = float(v_pos[best_col] if best_sign == 1 else v_neg[best_col])
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
