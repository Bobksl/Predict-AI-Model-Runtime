"""Graph Segment Training (GST): bounded backward memory on large layout graphs.

One full-graph forward pass runs under ``torch.no_grad`` (stop-gradient) and
covers every node — cheap, since no backward graph is retained. A second pass runs
WITH gradients, but only over the induced subgraph of a contiguous per-graph
window of ``max_keep_nodes`` nodes (``sampled_edge_index`` = edges with both
endpoints kept). The two are merged with ``torch.where`` so pooling still sees a
value for every node, but autograd only ever retains activations for the kept
segment: the ``[N,k,H]``-per-layer tensor the L-layer SAGE stack would otherwise
need to keep for backward is bounded by ``max_keep_nodes`` (not the full,
up to-43,615-node, graph). Matches the starter GST recipe referenced in
docs/TEAM.md / docs/briefs/phase3_brief.md (§3 "Starter GST", cell 37):
full-graph stop-gradient pass + sampled-edge grad pass + ``where`` merge,
contiguous kept nodes, edge survives iff both endpoints kept.

Eval / inference NEVER call into this module (guardrail: eval = full graph, no
GST, ``no_grad``) — only ``LayoutRanker.forward(..., gst=True)`` during training.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch


def select_gst_window(ptr: torch.Tensor, max_keep_nodes: int, seed: int,
                      epoch: int) -> torch.Tensor:
    """Per-graph contiguous node window -> boolean ``is_selected [N]``.

    ``ptr`` is the PyG ``Batch.ptr`` (``[B+1]`` node-offset boundaries). Each graph
    keeps ``min(n_i, max_keep_nodes)`` CONTIGUOUS nodes (node order ~topological,
    per the brief) so the induced subgraph stays dense/informative rather than a
    scattered random subset. The window start is deterministic per
    ``(seed, epoch, graph_index)`` — a new epoch draws a new window, mirroring the
    config-sampling convention in ``TpugraphsDataset.set_epoch``. Graphs no larger
    than ``max_keep_nodes`` keep every node.
    """
    ptr_list = [int(v) for v in ptr.tolist()]
    n_total = ptr_list[-1]
    is_selected = torch.zeros(n_total, dtype=torch.bool)
    for b in range(len(ptr_list) - 1):
        lo, hi = ptr_list[b], ptr_list[b + 1]
        n = hi - lo
        if n <= max_keep_nodes:
            is_selected[lo:hi] = True
            continue
        rng = np.random.default_rng((seed, epoch, b))
        start = int(rng.integers(0, n - max_keep_nodes + 1))
        is_selected[lo + start: lo + start + max_keep_nodes] = True
    return is_selected


def gst_forward(model, batch, max_keep_nodes: int, seed: int = 0, epoch: int = 0,
                debug: bool = False):
    """Two-pass GST forward. Returns ``(scores [B,k], is_selected [N])`` — plus a
    debug dict of internal facts (used by the GST correctness tests) when
    ``debug=True``.

    ``model`` must expose the four :class:`~src.models.layout_gnn.LayoutRanker`
    pieces: ``encode_base``, ``scatter_config``, ``message_pass``,
    ``pool_and_score``. Kept generic (no import of ``LayoutRanker`` here) to avoid
    a ``src.training`` <-> ``src.models`` import cycle.
    """
    x, op, edge_index = batch.x, batch.op, batch.edge_index
    node_config_feat, cfg_node_index = batch.node_config_feat, batch.cfg_node_index
    batch_vec, num_graphs = batch.batch, batch.num_graphs
    n_nodes = x.shape[0]
    k = node_config_feat.shape[1]
    device = x.device

    is_selected = select_gst_window(batch.ptr, max_keep_nodes, seed, epoch).to(device)

    h0 = model.encode_base(x, op)                                          # [N,H] grad
    cfg = model.scatter_config(node_config_feat, cfg_node_index, n_nodes)  # [N,k,H] grad

    # ---- pass 1: full graph, STOP-GRADIENT --------------------------------
    with torch.no_grad():
        x_full = model.message_pass(h0.detach(), cfg.detach(), edge_index, n_nodes, k)

    # ---- pass 2: kept-node induced subgraph, WITH grad ---------------------
    kept_idx = is_selected.nonzero(as_tuple=False).view(-1)                # [K]
    n_kept = kept_idx.numel()
    local_of = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
    local_of[kept_idx] = torch.arange(n_kept, device=device)
    mask_e = is_selected[edge_index[0]] & is_selected[edge_index[1]]       # both endpoints kept
    ei_local = local_of[edge_index[:, mask_e]]                             # local (compact) ids

    h0_kept = h0[kept_idx]
    cfg_kept = cfg[kept_idx]
    x_seg = model.message_pass(h0_kept, cfg_kept, ei_local, n_kept, k)     # [K,k,H] grad

    H = x_full.shape[-1]
    x_seg_full = x_full.new_zeros(n_nodes, k, H).index_copy(0, kept_idx, x_seg)
    sel = is_selected.view(n_nodes, 1, 1).expand(n_nodes, k, H)
    x_merged = torch.where(sel, x_seg_full, x_full)                        # grad only via kept rows

    scores = model.pool_and_score(x_merged, cfg_node_index, batch_vec, num_graphs, k)
    if debug:
        return scores, is_selected, {
            "x_full_requires_grad": bool(x_full.requires_grad),
            "n_kept": int(n_kept), "n_total": int(n_nodes),
        }
    return scores, is_selected
