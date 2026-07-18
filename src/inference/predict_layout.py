"""Chunked all-config layout inference (test graphs have only ~1,000-1,001 configs
per docs/TEAM.md, so inference is cheap even though training is the memory risk).

No GST, no_grad, full graph — the eval/inference convention pinned in
docs/briefs/phase3_brief.md §2. Encodes the config-independent ``h0`` ONCE, then
streams config chunks straight from the original ``.npz`` via
``src.data.configs.read_node_config_feat_rows`` (never the shard path — shards are
train+valid only) and reuses :class:`~src.models.layout_gnn.LayoutRanker`'s
``scatter_config`` / ``message_pass`` / ``pool_and_score`` pieces per chunk. Peak
memory is ``O(chunk_size)``, independent of ``n_configs``. Never touches the
placeholder test labels (guardrail #2).
"""
from __future__ import annotations
from typing import List

import numpy as np
import torch

from src.data.cache import read_bundle
from src.data.configs import read_node_config_feat_rows
from src.data.graph import _to_edge_index
from src.data.normalize import NodeFeatNormalizer


@torch.no_grad()
def score_all_configs_layout(model, file_path: str, normalizer: NodeFeatNormalizer,
                             device, chunk_size: int = 32,
                             add_reverse_edges: bool = True) -> np.ndarray:
    """Return a float32 score per configuration for one layout NPZ file, ascending
    score = fastest predicted. Length equals that file's ``n_configs``.
    """
    bundle = read_bundle(file_path)
    x = torch.as_tensor(normalizer.transform(bundle["node_feat"]),
                        dtype=torch.float32, device=device)
    op = torch.as_tensor(np.ascontiguousarray(bundle["node_opcode"]),
                         dtype=torch.long, device=device)
    # Single source of truth for edge conventions (transpose/bounds/reverse):
    # src.data.graph._to_edge_index — never re-implement this logic (P2 review).
    ei = _to_edge_index(bundle["edge_index"], int(x.shape[0]),
                        add_reverse_edges=add_reverse_edges,
                        add_self_loops=False).to(device)
    cfg_node_index = torch.as_tensor(np.ascontiguousarray(bundle["node_config_ids"]),
                                     dtype=torch.long, device=device)
    n_nodes = x.shape[0]
    batch_vec = torch.zeros(n_nodes, dtype=torch.long, device=device)

    model.eval()
    h0 = model.encode_base(x, op)  # [N,H] config-independent, computed once

    n_configs = int(bundle["config_runtime"].shape[0])
    out = np.empty(n_configs, dtype=np.float32)
    for start in range(0, n_configs, chunk_size):
        end = min(start + chunk_size, n_configs)
        chunk_ids = np.arange(start, end)
        # sampled-on-read: never materialises the full [n_configs, nc, 18] tensor
        ncf = read_node_config_feat_rows(file_path, chunk_ids)      # [c, nc, 18] int8
        # graph.py convention: node_config_feat is stored/consumed [nc, k, 18]
        # (nc-major) — transpose the same way build_pyg_data does for training.
        ncf = np.ascontiguousarray(np.transpose(ncf, (1, 0, 2)))    # [nc, c, 18]
        ncf_t = torch.as_tensor(ncf, dtype=torch.int8, device=device)
        c = ncf_t.shape[1]
        cfg = model.scatter_config(ncf_t, cfg_node_index, n_nodes)  # [N, c, H]
        h = model.message_pass(h0, cfg, ei, n_nodes, c)             # [N, c, H]
        s = model.pool_and_score(h, cfg_node_index, batch_vec, 1, c)  # [1, c]
        out[start:end] = s.squeeze(0).cpu().numpy()
    return out


def rank_configs(scores: np.ndarray) -> List[int]:
    """Fastest-first permutation of config indices (ascending score)."""
    return np.argsort(scores, kind="stable").tolist()
