"""Reproducible per-file inventory of the dataset.

Walks the five collections x three splits and records one row per NPZ file. Shapes
of the giant ``node_config_feat`` are read **from the npy header only** (never
decompressed). Float arrays (``node_feat``, tile ``config_feat``) are scanned for
NaN/Inf; ``edge_index`` is checked for out-of-bounds; test files are flagged as
having placeholder runtimes. Writes a single parquet.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .paths import COLLECTIONS, SPLITS, resolve_data_root, list_npz, parse_collection
from .configs import read_npy_shape


def _schema_hash(z) -> str:
    items = sorted((k, str(z[k].dtype) if hasattr(z[k], "dtype") else "?") for k in z.files)
    # avoid materialising arrays: use the NpzFile's zip headers for dtype via a peek
    return hashlib.sha1(str([(k) for k, _ in items]).encode()).hexdigest()[:12]


def inventory_one(path: Path, collection: str, split: str) -> dict:
    family, source, search = parse_collection(collection)
    is_tile = family == "tile"
    row = {
        "collection": collection, "split": split, "source": source,
        "search": search or "", "family": family,
        "file_path": str(path), "stem": path.stem,
        "bytes": int(os.path.getsize(path)),
        "n_nodes": None, "n_edges": None, "n_configs": None,
        "n_config_nodes": None, "n_subgraphs": None, "opcode_max": None,
        "schema_keys": "", "has_nan": False, "has_inf": False,
        "edge_oob": False, "runtime_is_placeholder": False,
    }
    with np.load(path) as z:
        row["schema_keys"] = ",".join(sorted(z.files))
        # shapes via headers where the body is large; small arrays read directly
        nf = z["node_feat"]
        row["n_nodes"] = int(nf.shape[0])
        row["has_nan"] = bool(np.isnan(nf).any())
        row["has_inf"] = bool(np.isinf(nf).any())
        edge = z["edge_index"]
        row["n_edges"] = int(edge.shape[0] if edge.shape[1] == 2 else edge.shape[1])
        n_nodes = row["n_nodes"]
        if edge.size:
            row["edge_oob"] = bool(edge.min() < 0 or edge.max() >= n_nodes)
        row["opcode_max"] = int(z["node_opcode"].max())
        rt = z["config_runtime"]
        row["n_configs"] = int(rt.shape[0])
        if split == "test":
            row["runtime_is_placeholder"] = bool(np.all(rt == 0))
        if is_tile:
            cf = z["config_feat"]
            row["has_nan"] = row["has_nan"] or bool(np.isnan(cf).any())
            row["has_inf"] = row["has_inf"] or bool(np.isinf(cf).any())
    if not is_tile:
        # header-only shape reads (do NOT decompress node_config_feat)
        shp, _ = read_npy_shape(path, "node_config_feat")     # [c, nc, 18]
        row["n_config_nodes"] = int(shp[1])
        ss, _ = read_npy_shape(path, "node_splits")           # [1, s]
        row["n_subgraphs"] = int(ss[1]) if len(ss) == 2 else int(ss[0])
    return row


def build_inventory(data_root=None, collections: Optional[List[str]] = None,
                    splits: Optional[List[str]] = None,
                    limit: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
    root = resolve_data_root(data_root)
    collections = collections or COLLECTIONS
    splits = splits or SPLITS
    rows = []
    for coll in collections:
        for split in splits:
            files = list_npz(root, coll, split)
            if limit is not None:
                files = files[:limit]
            if verbose:
                print(f"  {coll:22s} {split:6s} -> {len(files)} files")
            for fp in files:
                rows.append(inventory_one(fp, coll, split))
    return pd.DataFrame(rows)
