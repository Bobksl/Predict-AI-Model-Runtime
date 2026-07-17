"""Grouped-by-graph cross-validation utilities (protocol: brief T1).

Primary evaluation = the provided valid split (early stopping, model selection,
gate numbers). Additionally, 5-fold grouped CV **within train only** proves
reproducibility: folds partition train stems via ``assign_graph_folds``; the
provided valid split never enters any fold.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

from src.data.splits import assign_graph_folds


def make_fold_files(train_files: List[str], n_folds: int, fold: int,
                    seed: int = 0) -> Tuple[List[str], List[str]]:
    """Split file paths into (fit_files, heldout_files) for one grouped fold."""
    stems = [Path(f).stem for f in train_files]
    fold_of = assign_graph_folds(stems, n_folds=n_folds, seed=seed)
    fit = [f for f, s in zip(train_files, stems) if fold_of[s] != fold]
    held = [f for f, s in zip(train_files, stems) if fold_of[s] == fold]
    return fit, held
