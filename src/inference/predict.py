"""Chunked all-config inference for tile:xla (test graphs have up to ~10k configs).

Encodes each graph once, then scores its configs in chunks so peak memory stays
bounded regardless of ``n_configs``. Returns scores only — never touches the
placeholder test labels (guardrail #2).
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch

from src.data.cache import read_bundle
from src.data.graph import _to_edge_index
from src.data.normalize import NodeFeatNormalizer


@torch.no_grad()
def score_all_configs(model, file_path: str, normalizer: NodeFeatNormalizer,
                      device, chunk_size: int = 512,
                      add_reverse_edges: bool = True) -> np.ndarray:
    """Return a float32 score per configuration for one tile NPZ file, ascending
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
    batch_vec = torch.zeros(x.shape[0], dtype=torch.long, device=device)

    model.eval()
    graph_emb = model.encode_graph(x, op, ei, batch_vec)  # [1, hidden]

    config_feat = np.ascontiguousarray(bundle["config_feat"]).astype(np.float32)
    n_configs = config_feat.shape[0]
    out = np.empty(n_configs, dtype=np.float32)
    for start in range(0, n_configs, chunk_size):
        end = min(start + chunk_size, n_configs)
        cf = torch.as_tensor(config_feat[start:end], dtype=torch.float32,
                             device=device).unsqueeze(0)          # [1, chunk, 24]
        s = model.score_configs(graph_emb, cf)                    # [1, chunk]
        out[start:end] = s.squeeze(0).cpu().numpy()
    return out


def rank_configs(scores: np.ndarray) -> List[int]:
    """Fastest-first permutation of config indices (ascending score)."""
    return np.argsort(scores, kind="stable").tolist()
