"""Train-only node-feature normalisation (guardrail #4).

Computes per-column statistics for ``node_feat [N,140]`` **from the train split
only**, with a per-column ``log1p`` policy derived from the data: heavy-tailed,
non-negative columns are ``log1p``-transformed before standardisation (the size /
shape features reach ~2.6e7 and cannot all be log-transformed because some columns
contain negatives — min ~-2 on xla).

Transform: ``x -> standardise(log1p(x) if log_col else x)`` with zero-variance
columns guarded, yielding finite features.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


class NodeFeatNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, log_mask: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.log_mask = np.asarray(log_mask, dtype=bool)
        self._std_safe = np.where(self.std < 1e-6, 1.0, self.std)

    # ---- fitting -------------------------------------------------------
    @classmethod
    def fit(cls, files: Sequence, max_files: Optional[int] = None,
            log_max_threshold: float = 1000.0) -> "NodeFeatNormalizer":
        """Fit per-column stats by streaming ``node_feat`` over training files.

        One pass accumulates raw and ``log1p`` first/second moments plus min/max
        per column (140 columns -> cheap). A column is ``log1p``-transformed iff it
        is non-negative and heavy-tailed (``max > log_max_threshold``).
        """
        files = list(files)
        if max_files is not None:
            files = files[:max_files]
        if not files:
            raise ValueError("no training files provided to fit normaliser")

        d = None
        s_raw = sq_raw = s_log = sq_log = cmin = cmax = None
        count = 0
        for fp in files:
            with np.load(fp) as z:
                nf = z["node_feat"].astype(np.float64)  # [N,140] — not the RAM bomb
            if d is None:
                d = nf.shape[1]
                s_raw = np.zeros(d); sq_raw = np.zeros(d)
                s_log = np.zeros(d); sq_log = np.zeros(d)
                cmin = np.full(d, np.inf); cmax = np.full(d, -np.inf)
            nlog = np.log1p(np.maximum(nf, 0.0))
            s_raw += nf.sum(0); sq_raw += (nf * nf).sum(0)
            s_log += nlog.sum(0); sq_log += (nlog * nlog).sum(0)
            cmin = np.minimum(cmin, nf.min(0)); cmax = np.maximum(cmax, nf.max(0))
            count += nf.shape[0]

        mean_raw = s_raw / count
        var_raw = np.maximum(sq_raw / count - mean_raw ** 2, 0.0)
        mean_log = s_log / count
        var_log = np.maximum(sq_log / count - mean_log ** 2, 0.0)

        log_mask = (cmin >= 0.0) & (cmax > log_max_threshold)
        mean = np.where(log_mask, mean_log, mean_raw)
        std = np.sqrt(np.where(log_mask, var_log, var_raw))
        return cls(mean, std, log_mask)

    # ---- applying ------------------------------------------------------
    def transform(self, node_feat) -> np.ndarray:
        x = np.asarray(node_feat, dtype=np.float64)
        if self.log_mask.any():
            x = x.copy()
            x[:, self.log_mask] = np.log1p(np.maximum(x[:, self.log_mask], 0.0))
        out = (x - self.mean) / self._std_safe
        return np.ascontiguousarray(out, dtype=np.float32)

    # ---- (de)serialisation --------------------------------------------
    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "log_mask": self.log_mask.astype(int).tolist(),
            "n_features": int(self.mean.shape[0]),
        }

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path) -> "NodeFeatNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(np.array(d["mean"]), np.array(d["std"]),
                   np.array(d["log_mask"], dtype=bool))
