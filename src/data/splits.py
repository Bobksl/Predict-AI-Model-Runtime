"""Grouped-by-graph cross-validation scaffold (helper only; training is Phase 2).

Each graph is its own group, so a leakage-free CV is simply a deterministic
partition of the **unique graph stems** into folds — no configuration of a graph
can appear in two folds because a graph lives entirely in one file/fold.
"""
from __future__ import annotations
from typing import Dict, List, Sequence

import numpy as np


def assign_graph_folds(stems: Sequence[str], n_folds: int = 5,
                       seed: int = 0) -> Dict[str, int]:
    """Map each unique stem to a fold id in ``[0, n_folds)`` deterministically."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    uniq: List[str] = sorted(set(stems))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    folds = np.empty(len(uniq), dtype=int)
    for rank, pos in enumerate(order):
        folds[pos] = rank % n_folds
    return {stem: int(folds[i]) for i, stem in enumerate(uniq)}
