"""Ranking metrics: Ordered-Pair Accuracy (OPA).

OPA over one list: among pairs with distinct labels, the fraction ordered
concordantly by the scores — concordant iff ``(y_i - y_j) * (s_i - s_j) > 0``.
Ties in y are excluded from the denominator; score-ties on a valid pair count as
discordant (they fail to order the pair). With no ties anywhere,
``OPA == (kendall_tau + 1) / 2``.

Labels come in as exact int64; comparisons happen on the raw values (no float cast
of y — guardrail #1).
"""
from __future__ import annotations

import torch


def opa(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """OPA for one list. ``scores`` float [k], ``y`` int64/float [k]. Returns scalar.

    Lists with no valid pair (all labels equal) return NaN — callers should skip.
    """
    s = scores.view(-1)
    yy = y.view(-1)
    ds = s.unsqueeze(0) - s.unsqueeze(1)          # [k,k] score diffs
    dy = yy.unsqueeze(0) - yy.unsqueeze(1)        # [k,k] label diffs (int-exact)
    valid = dy != 0
    concordant = (ds > 0) & (dy > 0) | (ds < 0) & (dy < 0)
    n_valid = valid.sum()
    if n_valid == 0:
        return torch.tensor(float("nan"), device=s.device)
    return (concordant & valid).sum().to(torch.float64) / n_valid.to(torch.float64)


def opa_from_batch(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean OPA over a [B, k] batch, skipping graphs with no valid pair."""
    vals = []
    for b in range(scores.shape[0]):
        v = opa(scores[b], y[b])
        if not torch.isnan(v):
            vals.append(v)
    if not vals:
        return torch.tensor(float("nan"), device=scores.device)
    return torch.stack(vals).mean()
