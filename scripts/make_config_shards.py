#!/usr/bin/env python
"""Build capped-pool int8 memmap config shards for LAYOUT train+valid.

Approved spec (gate-P1 review, docs/TEAM.md §Verified facts):
- Per graph, two bare .npy files under
  data/cache/config_shards/layout/<source>/<search>/<split>/ :
    <stem>.<sig>.pool<P>.int8.npy  int8 [P', nc, 18]   (P' = min(n_configs, P))
    <stem>.<sig>.pool<P>.idx.npy   int64 [P']          original config ids, ASC
- Pool selection deterministic per (GLOBAL_SEED, stem); identity when n_configs<=P.
- Keyed by the CompactCache signature (size+mtime); rebuild = delete the dir.
- Labels are NEVER duplicated into shards (runtimes stay in the original bundle).
- Build streams the compressed member row-by-row, casting to int8 on the fly:
  peak RAM ~ one row (nc*18 floats), guardrail #3 holds during the build.

Usage:
    python scripts/make_config_shards.py [--pool 2000] [--collections ...] [--limit N]
"""
from __future__ import annotations
import argparse
import hashlib
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.paths import resolve_data_root, list_npz, parse_collection  # noqa: E402
from src.data.cache import _signature                                     # noqa: E402
from src.data import configs as C                                         # noqa: E402

GLOBAL_SEED = 0
LAYOUT_COLLECTIONS = ["layout:xla:random", "layout:xla:default",
                      "layout:nlp:random", "layout:nlp:default"]


def pool_indices(n_configs: int, pool: int, stem: str) -> np.ndarray:
    """Deterministic ascending pool of config ids for one graph."""
    if n_configs <= pool:
        return np.arange(n_configs, dtype=np.int64)
    h = int.from_bytes(hashlib.sha1(stem.encode()).digest()[:8], "little")
    rng = np.random.default_rng((GLOBAL_SEED, h))
    return np.sort(rng.choice(n_configs, size=pool, replace=False)).astype(np.int64)


def build_one(src_path: Path, out_dir: Path, pool: int) -> str:
    """Stream-build one graph's shard pair. Returns 'built'|'exists'."""
    sig = _signature(src_path)
    stem = src_path.stem
    data_f = out_dir / f"{stem}.{sig}.pool{pool}.int8.npy"
    idx_f = out_dir / f"{stem}.{sig}.pool{pool}.idx.npy"
    if data_f.exists() and idx_f.exists():
        return "exists"
    out_dir.mkdir(parents=True, exist_ok=True)

    shape, dtype = C.read_npy_shape(src_path, "node_config_feat")  # [c, nc, 18]
    n_configs, nc, f = int(shape[0]), int(shape[1]), int(shape[2])
    idx = pool_indices(n_configs, pool, stem)                       # ascending

    tmp = data_f.with_suffix(".tmp.npy")
    mm = open_memmap(tmp, mode="w+", dtype=np.int8, shape=(len(idx), nc, f))
    row_bytes = nc * f * dtype.itemsize
    # one forward streaming pass over the compressed member (ids are ascending)
    with zipfile.ZipFile(src_path) as zf, C._open_member(zf, "node_config_feat") as fh:
        C._read_npy_header(fh)
        pos = 0
        for j, ci in enumerate(idx):
            C._skip(fh, int(ci - pos) * row_bytes)
            row = np.frombuffer(C._read_exact(fh, row_bytes), dtype=dtype)
            row = row.reshape(nc, f)
            r8 = row.astype(np.int8)
            if not np.array_equal(r8.astype(dtype), row):   # exact integrality
                raise ValueError(f"non-integer config value in {src_path}")
            mm[j] = r8
            pos = int(ci) + 1
    mm.flush()
    del mm
    tmp.replace(data_f)
    np.save(idx_f, idx)
    return "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out_root", default="data/cache/config_shards")
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--collections", nargs="*", default=LAYOUT_COLLECTIONS)
    ap.add_argument("--splits", nargs="*", default=["train", "valid"])
    ap.add_argument("--limit", type=int, default=None, help="files per coll/split (smoke)")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    t0, n_built, n_exist, total_bytes = time.perf_counter(), 0, 0, 0
    for coll in args.collections:
        family, source, search = parse_collection(coll)
        assert family == "layout", "shards are layout-only (tile has no RAM bomb)"
        for split in args.splits:                     # NEVER test (spec)
            files = list_npz(root, coll, split)
            if args.limit:
                files = files[: args.limit]
            out_dir = Path(args.out_root) / family / source / search / split
            for fp in files:
                status = build_one(fp, out_dir, args.pool)
                n_built += status == "built"
                n_exist += status == "exists"
            done = sum(f.stat().st_size for f in out_dir.glob("*.npy"))
            total_bytes += done
            print(f"  {coll:20s} {split:5s}: {len(files)} files "
                  f"({done/2**30:.2f} GiB)  [{time.perf_counter()-t0:.0f}s]")
    print(f"\nBuilt {n_built}, already existed {n_exist}. "
          f"Total shard size {total_bytes/2**30:.2f} GiB "
          f"in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
