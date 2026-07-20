"""Path resolution and collection/split bookkeeping.

Path-portable across the local Windows checkout and the Kaggle/Colab mounts.
Resolution order for the *data root* (the directory that contains ``tile/`` and
``layout/``):

1. an explicit argument,
2. the ``TPUGRAPHS_DATA_ROOT`` environment variable,
3. a list of known candidate locations (local checkout, Kaggle mounts).

Outputs (inventory, cache, stats) go to a **writable** directory resolved by
:func:`writable_dir` (``/kaggle/working`` on Kaggle, else local ``artifacts``/cache).
"""
from __future__ import annotations
import os
import glob
from pathlib import Path
from typing import List, Optional

# The five competition collections.  "layout:source:search" or "tile:source".
COLLECTIONS: List[str] = [
    "tile:xla",
    "layout:xla:random",
    "layout:xla:default",
    "layout:nlp:random",
    "layout:nlp:default",
]
SPLITS: List[str] = ["train", "valid", "test"]

# Candidate data roots (each should directly contain ``tile/`` and ``layout/``).
_CANDIDATE_ROOTS = [
    "data/npz/tpugraphs",
    # current Kaggle mount layout (verified 2026-07-19 via kagglehub fallback)
    "/kaggle/input/competitions/predict-ai-model-runtime/npz_all/npz",
    "/kaggle/input/predict-ai-model-runtime/npz/tpugraphs",
    "/kaggle/input/predict-ai-model-runtime/npz_all/npz",
    "/kaggle/input/predict-ai-model-runtime/npz",
]


def resolve_data_root(explicit: Optional[str] = None) -> Path:
    """Return the first existing data root, raising if none is found."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("TPUGRAPHS_DATA_ROOT")
    if env:
        candidates.append(env)
    candidates.extend(_CANDIDATE_ROOTS)
    for c in candidates:
        p = Path(c)
        if (p / "tile").is_dir() or (p / "layout").is_dir():
            return p
    raise FileNotFoundError(
        "Could not resolve a TPUGraphs data root. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ". Set TPUGRAPHS_DATA_ROOT or pass data_root explicitly."
    )


def parse_collection(collection: str):
    """('tile','xla',None) or ('layout','xla','random') from a collection id."""
    parts = collection.split(":")
    if parts[0] == "tile":
        return "tile", parts[1], None
    if parts[0] == "layout":
        return "layout", parts[1], parts[2]
    raise ValueError(f"Unknown collection: {collection!r}")


def collection_dir(data_root, collection: str) -> Path:
    """Directory holding the train/valid/test subdirs for a collection."""
    family, source, search = parse_collection(collection)
    root = Path(data_root)
    if family == "tile":
        return root / "tile" / source
    return root / "layout" / source / search


def split_dir(data_root, collection: str, split: str) -> Path:
    return collection_dir(data_root, collection) / split


def list_npz(data_root, collection: str, split: str) -> List[Path]:
    """Sorted list of NPZ files for one collection/split (empty if missing)."""
    d = split_dir(data_root, collection, split)
    if not d.is_dir():
        return []
    return [Path(p) for p in sorted(glob.glob(str(d / "*.npz")))]


def writable_dir(prefer_kaggle: bool = True) -> Path:
    """Resolve a writable output directory (Kaggle working dir, env, or local)."""
    env = os.environ.get("TPUGRAPHS_OUT")
    if env:
        p = Path(env)
    elif prefer_kaggle and Path("/kaggle/working").is_dir():
        p = Path("/kaggle/working")
    else:
        p = Path("artifacts")
    p.mkdir(parents=True, exist_ok=True)
    return p
