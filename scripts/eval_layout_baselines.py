#!/usr/bin/env python
"""Trivial layout baselines the LayoutRanker must beat (gate P3 item 2).

(a) Random permutation — expected OPA ~0.50.
(b) Summed-config heuristic: score(config) = sum over configurable nodes and the
    18 config features of `node_config_feat` (both signs; sign chosen on train,
    reported on the SAME files used for model CV evaluation — the train split,
    since baselines have no fitted parameters beyond the sign).

Memory-safe: reads sampled config rows via shards when available, else the
streaming reader; never materialises the full [c,nc,18] tensor. Configs are
subsampled to MAX_CONFIGS per graph (exact pairwise OPA is O(c^2)).

Usage:
    python scripts/eval_layout_baselines.py [--collections ...] [--max_files N]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.paths import resolve_data_root, list_npz            # noqa: E402
from src.data.cache import read_bundle                            # noqa: E402
from src.data.configs import read_node_config_feat_rows           # noqa: E402
from src.data.config_shards import find_shard                     # noqa: E402
from src.training.metrics import opa                              # noqa: E402

LAYOUT_COLLECTIONS = ["layout:xla:random", "layout:xla:default",
                      "layout:nlp:random", "layout:nlp:default"]
MAX_CONFIGS = 128   # pair-exact OPA per graph; 128 -> ~8k pairs
SEED = 0


def _sampled(path, n_configs, rng, shard_root="data/cache/config_shards",
             collection=None, split=None, stem=None, pool=2000):
    """(config ids, ncf [m,nc,18] float64) for <=MAX_CONFIGS sampled configs."""
    m = min(n_configs, MAX_CONFIGS)
    ids = np.sort(rng.choice(n_configs, size=m, replace=False)).astype(np.int64)
    shard = find_shard(shard_root, collection, split, stem, path, pool)
    if shard is not None:
        data_f, idx_f = shard
        pool_ids = np.load(idx_f)                       # ascending original ids
        # map: keep only sampled ids that are in the pool (resample within pool)
        mm = np.load(data_f, mmap_mode="r")
        pos = rng.choice(len(pool_ids), size=min(m, len(pool_ids)), replace=False)
        pos = np.sort(pos)
        return pool_ids[pos], np.asarray(mm[pos]).astype(np.float64)
    return ids, read_node_config_feat_rows(path, ids).astype(np.float64)


def eval_collection(coll, root, max_files=None):
    files = list_npz(root, coll, "train")
    if max_files:
        files = files[:max_files]
    rng = np.random.default_rng(SEED)
    rand_vals, sum_pos, sum_neg = [], [], []
    for fp in files:
        b = read_bundle(str(fp))
        y_all = b["config_runtime"]
        if len(y_all) < 2:
            continue
        ids, ncf = _sampled(str(fp), len(y_all), rng, collection=coll,
                            split="train", stem=fp.stem)
        y = torch.as_tensor(y_all[ids])
        # (a) random
        s = torch.as_tensor(rng.permutation(len(ids)).astype(np.float64))
        v = opa(s, y)
        if not torch.isnan(v):
            rand_vals.append(float(v))
        # (b) summed config heuristic, both signs
        score = torch.as_tensor(ncf.sum(axis=(1, 2)))
        for sign, acc in ((1, sum_pos), (-1, sum_neg)):
            v = opa(sign * score, y)
            if not torch.isnan(v):
                acc.append(float(v))
    r = float(np.mean(rand_vals))
    p, n = float(np.mean(sum_pos)), float(np.mean(sum_neg))
    sign = "+" if p >= n else "-"
    best = max(p, n)
    print(f"{coll:20s} files={len(files):3d}  random={r:.4f}  "
          f"summed-cfg({sign})={best:.4f}   -> model must clearly beat {best:.4f}")
    return {"collection": coll, "random": r, "summed": best, "sign": sign}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--collections", nargs="*", default=LAYOUT_COLLECTIONS)
    ap.add_argument("--max_files", type=int, default=None)
    args = ap.parse_args()
    root = resolve_data_root(args.data_root)
    for coll in args.collections:
        eval_collection(coll, root, args.max_files)


if __name__ == "__main__":
    main()
