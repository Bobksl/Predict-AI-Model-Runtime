"""Cheap, rebuildable per-graph cache of everything *except* the RAM-bomb tensor.

Bundles the small/structural arrays a sample needs — ``node_feat``,
``node_opcode``, ``edge_index``, ``config_runtime`` (+ tile ``config_feat`` /
``config_runtime_normalizers``, + layout ``node_config_ids`` / ``node_splits``) —
into a single **uncompressed** ``.npz`` in a writable directory. It deliberately
**excludes the giant ``node_config_feat``** (layout), which stays sample-on-read.

Effect: a warm ``__getitem__`` loads **one uncompressed pickle** (no zip
central-directory or DEFLATE overhead) instead of opening and decompressing the
original ``.npz``, so warm reads of the compact arrays are markedly faster. Keyed
by a cheap file signature (size + mtime); delete the directory to rebuild.

Note: this accelerates the *compact-array* path. For layout, the per-graph cost is
dominated by streaming sampled ``node_config_feat`` rows (uncacheable here by
design); pre-sampling those offline is a Phase-2 optimisation.
"""
from __future__ import annotations
import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

# Everything a sample needs EXCEPT node_config_feat (the >1 GB layout tensor).
_ALWAYS = ["node_feat", "node_opcode", "edge_index", "config_runtime"]
_OPTIONAL = ["config_feat", "config_runtime_normalizers", "node_config_ids", "node_splits"]


def _signature(path) -> str:
    st = os.stat(path)
    raw = f"{Path(path).name}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def read_bundle(path) -> Dict[str, np.ndarray]:
    """Cold read of all sample arrays except ``node_config_feat`` (single pass)."""
    out: Dict[str, np.ndarray] = {}
    with np.load(path) as z:
        for k in _ALWAYS:
            out[k] = np.ascontiguousarray(z[k])
        for k in _OPTIONAL:
            if k in z.files:
                out[k] = np.ascontiguousarray(z[k])
    return out


class CompactCache:
    def __init__(self, cache_dir, enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, path) -> Path:
        return self.cache_dir / f"{Path(path).stem}.{_signature(path)}.pkl"

    def get(self, path) -> Dict[str, np.ndarray]:
        """Return the sample bundle for ``path``, populating the cache on miss."""
        if not self.enabled:
            return read_bundle(path)
        cp = self._cache_path(path)
        if cp.exists():
            with open(cp, "rb") as f:
                return pickle.load(f)
        arrays = read_bundle(path)
        tmp = cp.with_suffix(".tmp.pkl")
        with open(tmp, "wb") as f:
            pickle.dump(arrays, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cp)
        return arrays


# Backwards-compatible alias.
read_compact = read_bundle
