#!/usr/bin/env python
"""Fit train-only node-feature normalisation statistics.

Usage:
    python scripts/fit_norm.py [--data_root DIR] [--out PATH] [--per_collection N]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.paths import COLLECTIONS, resolve_data_root, list_npz, writable_dir  # noqa: E402
from src.data.normalize import NodeFeatNormalizer                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default=None, help="json path (default <writable>/norm_stats.json)")
    ap.add_argument("--per_collection", type=int, default=150,
                    help="max train files per collection used to fit (speed)")
    args = ap.parse_args()

    root = resolve_data_root(args.data_root)
    files = []
    for coll in COLLECTIONS:
        fs = list_npz(root, coll, "train")[: args.per_collection]
        files.extend(fs)
        print(f"  {coll:22s} train -> using {len(fs)} files")
    print(f"\nFitting on {len(files)} train files (TRAIN ONLY) ...")

    norm = NodeFeatNormalizer.fit(files)
    out = Path(args.out) if args.out else (writable_dir() / "norm_stats.json")
    norm.save(out)
    # Record how the fit was performed (review C4: document the subsampling).
    import json
    with open(out, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["fit_meta"] = {
        "split": "train_only",
        "per_collection_cap": args.per_collection,
        "n_files_used": len(files),
        "collections": COLLECTIONS,
        "note": "global normaliser pooled across collections; per-collection stats are a Phase-4 option",
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    n_log = int(norm.log_mask.sum())
    print(f"Wrote {out}")
    print(f"  features={norm.mean.shape[0]}  log1p columns={n_log}  "
          f"log_cols_idx={list(np.where(norm.log_mask)[0])[:20]}{' ...' if n_log>20 else ''}")

    # verify finite after applying on a few held files
    import numpy as _np
    bad = 0
    for fp in files[:5] + files[-5:]:
        with _np.load(fp) as z:
            x = norm.transform(z["node_feat"])
        if not _np.isfinite(x).all():
            bad += 1
    assert bad == 0, "non-finite features after normalisation!"
    print("Gate check: node_feat finite after applying normaliser -> OK")


if __name__ == "__main__":
    main()
