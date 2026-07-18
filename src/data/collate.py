"""Variable-size batching collator (works at batch > 1 for every collection).

Wraps PyG's :meth:`Batch.from_data_list`, which — together with the
:class:`~src.data.graph.TPUData` batching rules — concatenates variable-size
graphs into one disconnected ``Batch`` (offsetting ``edge_index`` and
``cfg_node_index``, building the ``batch`` vector, stacking the per-graph sampled
config lists and labels).

No Graph Segment Training / segment dropout here — that is Phase 3.
"""
from __future__ import annotations
from typing import List

import torch
from torch_geometric.data import Batch

from .graph import TPUData


def collate(data_list: List[TPUData]) -> Batch:
    """Collate a list of :class:`TPUData` into a single PyG ``Batch``."""
    return Batch.from_data_list(data_list)


def scatter_node_config_feat(batch: Batch) -> torch.Tensor:
    """Scatter sampled layout config features onto nodes, masking non-configurable.

    Returns a dense ``[sum_N, k, 18]`` ``float32`` tensor: row ``node`` holds that
    node's sampled config features if it is configurable, else zeros. Built on
    demand (kept out of the batch by default to bound memory).
    """
    if not hasattr(batch, "node_config_feat"):
        raise ValueError("batch has no node_config_feat (not a layout batch)")
    n_nodes = batch.num_nodes
    ncf = batch.node_config_feat                 # [sum_nc, k, 18] int8
    idx = batch.cfg_node_index                   # [sum_nc] global node ids
    k, f = ncf.shape[1], ncf.shape[2]
    dense = torch.zeros((n_nodes, k, f), dtype=torch.float32)
    dense[idx] = ncf.to(torch.float32)
    return dense
