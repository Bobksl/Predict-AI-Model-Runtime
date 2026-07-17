"""Ranking losses: pairwise hinge and ListMLE.

Convention: higher score = slower (scores imitate runtime order). Labels are exact
int64 runtimes; they are used only for *ordering* (comparisons / argsort), never
cast to float32 (guardrail #1 — float32 would collapse distinct values into ties).
"""
from __future__ import annotations

import torch


def pairwise_hinge_loss(scores: torch.Tensor, y: torch.Tensor,
                        margin: float = 0.1) -> torch.Tensor:
    """Mean hinge over all ordered pairs of each list in a [B, k] batch.

    For a pair with ``y_i < y_j`` (i truly faster ⇒ j must score higher):
    ``max(0, margin - (s_j - s_i))``. Label-ties (which include duplicate sampled
    configs from the n<k replacement pad) are masked out.
    """
    ds = scores.unsqueeze(-1) - scores.unsqueeze(-2)     # [B,k,k]: s_i - s_j
    dy = y.unsqueeze(-1) - y.unsqueeze(-2)               # [B,k,k]: y_i - y_j (int)
    faster = dy < 0                                      # y_i < y_j
    # want s_j - s_i >= margin  <=>  penalise margin - (s_j - s_i) = margin + ds
    per_pair = torch.clamp(margin + ds, min=0.0)
    n = faster.sum()
    if n == 0:
        return scores.sum() * 0.0
    return (per_pair * faster).sum() / n


def listmle_loss(scores: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ListMLE (Plackett–Luce NLL) over a [B, k] batch.

    Ground-truth permutation sorts by **descending runtime** (slowest first),
    consistent with higher-score = slower. Loss per list:
    ``-sum_i [ s_pi(i) - logcumsumexp_{j>=i} s_pi(j) ]``, computed stably with
    ``logcumsumexp`` from the tail; a per-list max-shift guards extreme scores.
    Ties in y: stable argsort; residual ambiguity is negligible at k~16.
    """
    order = torch.argsort(y, dim=-1, descending=True, stable=True)   # int64 sort — exact
    s = torch.gather(scores, -1, order)                              # [B,k] in pi* order
    s = s - s.max(dim=-1, keepdim=True).values                       # stability shift
    # logcumsumexp over the suffix: flip, cumulate, flip back
    tail_lse = torch.logcumsumexp(s.flip(-1), dim=-1).flip(-1)       # [B,k]
    nll = (tail_lse - s).sum(dim=-1)                                 # per list
    return nll.mean()
