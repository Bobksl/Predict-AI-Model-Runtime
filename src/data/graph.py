"""Per-graph PyG object construction and the pinned graph conventions.

SINGLE SOURCE OF TRUTH FOR GRAPH CONVENTIONS
--------------------------------------------
* ``edge_index`` is stored **raw as ``[n_edges, 2]``** (a list of ``(src, dst)``
  pairs). We **transpose to ``[2, n_edges]``** for PyG and assert every index is
  in ``[0, n_nodes)``.
* **Directionality.** A raw edge ``(u, v)`` is the *producer -> consumer* ("feed")
  edge: operation ``u`` produces a tensor consumed by ``v``. Message passing in the
  default orientation therefore sends messages *downstream* (producer -> consumer).
  The official starter notes that the **transpose** of this adjacency carries the
  natural "information flow"; most GNNs here treat the graph as effectively
  undirected, so :func:`build_pyg_data` exposes:
    - ``add_reverse_edges`` -> also add ``(v, u)`` (undirected message passing),
    - ``add_self_loops``    -> also add ``(i, i)`` (GCN-style self connection).
  Phase 1 only *builds* the structure; choosing the orientation is a Phase-2
  modelling decision. Defaults keep the raw directed edges only.

* Labels: ``config_runtime`` is an **ordering signal only** and is never clipped,
  normalised, or reordered (guardrail #1). It is carried **exact, as int64**:
  runtimes exceed 2^24, so a float32 cast silently merges distinct values into
  ties (measured: 1852 of 29010 uniques collapse on the alexnet layout file).
  Losses/metrics must cast to float only *after* pairwise differencing / rank
  extraction, never the raw labels.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data


class TPUData(Data):
    """PyG ``Data`` with explicit batching rules for our custom fields.

    Custom per-graph tensors and how they batch (``B`` graphs, ``k`` sampled
    configs, ``N`` nodes, ``nc`` configurable nodes):

    ====================  =====================  ============================
    field                 single-graph shape     batched shape
    ====================  =====================  ============================
    ``x``                 ``[N, 140]``           ``[sum N, 140]``
    ``op``                ``[N]``                ``[sum N]``
    ``edge_index``        ``[2, M]``             ``[2, sum M]`` (offset)
    ``y``                 ``[1, k]``             ``[B, k]``
    ``config_feat``       ``[1, k, 24]`` (tile)  ``[B, k, 24]``
    ``node_config_feat``  ``[nc, k, 18]``        ``[sum nc, k, 18]``
    ``cfg_node_index``    ``[nc]``               ``[sum nc]`` (offset by N)
    ====================  =====================  ============================
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "cfg_node_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("y", "config_feat", "node_config_feat", "cfg_node_index"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def _to_edge_index(edge_raw: np.ndarray, n_nodes: int,
                   add_reverse_edges: bool, add_self_loops: bool) -> torch.Tensor:
    e = np.asarray(edge_raw)
    if e.ndim != 2 or 2 not in e.shape:
        raise ValueError(f"edge_index must be 2-D with a size-2 axis, got {e.shape}")
    if e.shape[1] == 2 and e.shape[0] != 2:
        e = e.T  # [n_edges, 2] -> [2, n_edges]
    elif e.shape[0] == 2 and e.shape[1] == 2:
        e = e.T  # ambiguous 2x2: raw is [n_edges, 2]
    ei = torch.as_tensor(np.ascontiguousarray(e), dtype=torch.long)
    if ei.numel() > 0:
        lo, hi = int(ei.min()), int(ei.max())
        if lo < 0 or hi >= n_nodes:
            raise IndexError(f"edge index out of bounds: [{lo},{hi}] vs n_nodes={n_nodes}")
    if add_reverse_edges and ei.numel() > 0:
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    if add_self_loops:
        loops = torch.arange(n_nodes, dtype=torch.long).repeat(2, 1)
        ei = torch.cat([ei, loops], dim=1)
    return ei


def build_pyg_data(
    *,
    node_feat: np.ndarray,            # [N, 140] (already normalised)
    node_opcode: np.ndarray,          # [N]
    edge_index: np.ndarray,           # [n_edges, 2] raw
    runtimes_sampled: np.ndarray,     # [k]  ordering signal only
    is_placeholder: bool,
    collection: str,
    split: str,
    stem: str,
    config_feat: Optional[np.ndarray] = None,        # tile: [k, 24]
    node_config_feat: Optional[np.ndarray] = None,   # layout: [k, nc, 18] int8
    cfg_node_index: Optional[np.ndarray] = None,      # layout: [nc] node ids
    add_reverse_edges: bool = False,
    add_self_loops: bool = False,
) -> TPUData:
    """Assemble one :class:`TPUData` graph with its sampled configuration list."""
    n_nodes = int(node_feat.shape[0])
    x = torch.as_tensor(np.ascontiguousarray(node_feat), dtype=torch.float32)
    op = torch.as_tensor(np.ascontiguousarray(node_opcode), dtype=torch.long)
    ei = _to_edge_index(edge_index, n_nodes, add_reverse_edges, add_self_loops)
    # int64, exact — see module docstring (float32 would collapse distinct runtimes)
    y = torch.as_tensor(np.asarray(runtimes_sampled), dtype=torch.int64).view(1, -1)

    data = TPUData(x=x, edge_index=ei)
    data.op = op
    data.num_nodes = n_nodes
    data.y = y
    data.is_placeholder = torch.tensor([bool(is_placeholder)])
    # non-tensor metadata (collected into per-graph lists when batched)
    data.collection = collection
    data.split = split
    data.stem = stem

    if config_feat is not None:  # tile
        data.config_feat = torch.as_tensor(
            np.ascontiguousarray(config_feat), dtype=torch.float32).unsqueeze(0)  # [1,k,24]
    if node_config_feat is not None:  # layout
        ncf = np.asarray(node_config_feat)            # [k, nc, 18]
        ncf = np.ascontiguousarray(np.transpose(ncf, (1, 0, 2)))  # [nc, k, 18]
        data.node_config_feat = torch.as_tensor(ncf, dtype=torch.int8)
        cni = torch.as_tensor(np.ascontiguousarray(cfg_node_index), dtype=torch.long)
        if cni.numel() and (int(cni.min()) < 0 or int(cni.max()) >= n_nodes):
            raise IndexError("cfg_node_index out of bounds")
        data.cfg_node_index = cni
    return data
