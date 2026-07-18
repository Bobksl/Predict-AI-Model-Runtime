"""Config-shard consumer: read epoch-aware config samples from prebuilt shards.

Shards (built by ``scripts/make_config_shards.py``, approved gate-P1 spec) are a
capped POOL of ``P`` configs per layout graph, stored as bare ``.npy`` pairs under
``data/cache/config_shards/layout/<source>/<search>/<split>/``:

    ``<stem>.<sig>.pool<P>.int8.npy``  int8 ``[P', nc, 18]``  (P' = min(n_configs, P))
    ``<stem>.<sig>.pool<P>.idx.npy``   int64 ``[P']``         original config ids, ASC

keyed by the :func:`~src.data.cache._signature` (size+mtime) of the source
``.npz``. On a signature or file mismatch the caller MUST fall back to streaming
(``src.data.dataset.TpugraphsDataset.__getitem__`` does this) — shards are a
read-speed optimisation, never a correctness dependency. Train+valid only (test
streams sequentially at inference; ``find_shard`` returns ``None`` for split
``"test"``). Labels are NEVER read from shards — callers always index
``config_runtime`` from the original bundle by the returned ``orig_ids``
(guardrail: labels never duplicated into shards).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .cache import _signature
from .configs import sample_config_indices
from .paths import parse_collection


def shard_dir(shard_root, collection: str, split: str) -> Path:
    """``<shard_root>/layout/<source>/<search>/<split>/`` for one collection."""
    family, source, search = parse_collection(collection)
    if family != "layout":
        raise ValueError("config shards are layout-only (tile has no RAM-bomb array)")
    return Path(shard_root) / family / source / search / split


def find_shard(shard_root, collection: str, split: str, stem: str, src_path,
               pool: int) -> Optional[Tuple[Path, Path]]:
    """Return ``(data_path, idx_path)`` iff a signature-matching shard pair exists.

    Signature mismatch (source file changed) or a missing pair -> ``None``, which
    callers must treat as "fall back to streaming" (never an error).
    """
    if split == "test":
        return None  # shards are train+valid only (approved spec)
    d = shard_dir(shard_root, collection, split)
    sig = _signature(src_path)
    data_f = d / f"{stem}.{sig}.pool{pool}.int8.npy"
    idx_f = d / f"{stem}.{sig}.pool{pool}.idx.npy"
    if data_f.exists() and idx_f.exists():
        return data_f, idx_f
    return None


def read_shard_sample(data_path, idx_path, k: int, rng: np.random.Generator
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample exactly ``k`` pool positions and return ``(feats [k,nc,18] int8,
    orig_ids [k] int64)``.

    Opens the pool array as a read-only memmap (never loads the full ``[P',nc,18]``
    pool) — only the ``k`` selected rows are ever materialised in memory.
    ``orig_ids`` are the ORIGINAL config indices (not pool positions); callers must
    index labels from the source bundle's ``config_runtime`` by these ids, never
    from the shard itself.
    """
    idx = np.load(idx_path)                        # [P'] int64, ascending, cheap
    p_prime = idx.shape[0]
    # Same k-selection convention as the streaming path (configs.sample_config_indices):
    # exact k, without replacement when p_prime>=k, padded with replacement otherwise.
    pos = sample_config_indices(p_prime, k, rng)    # [k] positions into the pool
    mm = np.load(data_path, mmap_mode="r")          # [P', nc, 18] int8 memmap, no full load
    feats = np.ascontiguousarray(mm[pos])           # only k rows touched
    orig_ids = np.ascontiguousarray(idx[pos])
    return feats, orig_ids
