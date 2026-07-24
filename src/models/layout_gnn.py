"""LayoutRanker — per-config SAGE GNN scoring layout configurations at scale.

Shapes (batch B graphs, k sampled configs/graph, N = sum of nodes, nc = sum of
configurable nodes):

1. Base encode ONCE, config-independent: ``op [N]`` -> ``Embedding(119,32)``,
   concat ``x [N,140]`` -> ``Linear`` -> ``h0 [N,H]`` (:meth:`encode_base`).
2. Config injection WITHOUT a dense ``[N,k,18]`` intermediate: compact
   ``node_config_feat [nc,k,18]`` (int8) -> ``config_proj: 18->H`` -> ``[nc,k,H]``,
   then ``index_add`` onto zeros ``[N,k,H]`` at ``cfg_node_index`` — touches only
   the ``nc`` configurable rows (:meth:`scatter_config`). The dense
   ``collate.scatter_node_config_feat`` materialisation stays a CPU-smoke-only
   fallback, never used here.
3. Per-config message passing (:meth:`message_pass`): ``x = h0[:,None,:] +
   scattered_cfg -> [N,k,H]``; run L ``SAGEConv`` layers with weights SHARED
   across the k configs by reshaping to ``[N*k,H]`` over a k-fold, node-offset
   replica of ``edge_index`` (config c's nodes live at ``[c*N,(c+1)*N)``) — this
   is exactly k independent message-passing runs over the same topology with
   different node features, batched as one bigger disconnected graph.
4. Pool + score (:meth:`pool_and_score`): mean over ALL nodes + mean over only the
   configurable nodes, per (graph, config) pair, summed, then an MLP head ->
   ``[B,k]`` (higher score = slower, project-wide convention).

Graph Segment Training (bounded backward memory on graphs up to N=43,615) lives in
``src/training/gst.py`` and calls back into :meth:`encode_base`, :meth:`scatter_config`,
:meth:`message_pass`, :meth:`pool_and_score` directly — ``forward(..., gst=True)``
delegates there. Eval / inference never use GST (``forward`` defaults to the plain
full pass); ``src/inference/predict_layout.py`` reuses the same four pieces under
``no_grad`` with a chunked config axis instead of the GST windowed node axis.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.pool import global_mean_pool  # noqa: F401  (kept for parity/reference)
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter

from .tile_gnn import OPCODE_VOCAB


def _replicate_edge_index(edge_index: torch.Tensor, n_nodes: int, k: int) -> torch.Tensor:
    """Replicate ``[2,E]`` edges into ``k`` node-offset copies -> ``[2, k*E]``.

    Config ``c``'s nodes occupy the block ``[c*n_nodes, (c+1)*n_nodes)`` of the
    ``[k*n_nodes, H]`` feature tensor used by :meth:`LayoutRanker.message_pass`, so
    each replica's edges must be offset the same way — this keeps the k per-config
    message-passing runs fully disconnected from one another (weight-shared, but
    not feature-shared).
    """
    if k == 1:
        return edge_index
    E = edge_index.shape[1]
    offsets = (torch.arange(k, device=edge_index.device, dtype=edge_index.dtype)
              * n_nodes).view(k, 1, 1)                      # [k,1,1]
    rep = edge_index.view(1, 2, E) + offsets                # [k,2,E]
    return rep.permute(1, 0, 2).reshape(2, k * E)


class LayoutRanker(nn.Module):
    def __init__(self, opcode_emb_dim: int = 32, hidden_dim: int = 64,
                 n_layers: int = 3, config_proj_dim: Optional[int] = None,
                 dropout: float = 0.1, node_feat_dim: int = 140,
                 config_feat_dim: int = 18, config_encoder: str = "linear",
                 use_config_attn: bool = False, attn_heads: int = 4):
        super().__init__()
        config_proj_dim = config_proj_dim or hidden_dim
        if config_proj_dim != hidden_dim:
            # scatter_config's output is added elementwise onto h0 in message_pass
            # (`h0[:, None, :] + scattered_cfg`) -> the two must share a width.
            raise ValueError("config_proj_dim must equal hidden_dim (added onto h0)")
        self.hidden_dim = hidden_dim
        self.op_emb = nn.Embedding(OPCODE_VOCAB, opcode_emb_dim)
        self.node_proj = nn.Linear(node_feat_dim + opcode_emb_dim, hidden_dim)
        # Config encoder (P4 #3): "linear" (Phase-3) or a 2-layer "mlp".
        if config_encoder == "mlp":
            self.config_proj = nn.Sequential(
                nn.Linear(config_feat_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim))
        else:
            self.config_proj = nn.Linear(config_feat_dim, config_proj_dim)
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        # Cross-configuration attention (P4 #2): configs of a graph attend to each
        # other on the pooled [B,k,H] tensor before scoring — "comparing configs is
        # easier than predicting absolute runtime" (TGraph, arXiv:2405.16623).
        self.use_config_attn = use_config_attn
        if use_config_attn:
            self.config_attn = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=attn_heads,
                dim_feedforward=2 * hidden_dim, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    # ---- pieces (shared by the plain forward, GST, and chunked inference) ----
    def encode_base(self, x: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        """Node features -> config-independent base embedding ``h0 [N, H]``."""
        return self.node_proj(torch.cat([x, self.op_emb(op)], dim=-1))

    def scatter_config(self, node_config_feat: torch.Tensor,
                       cfg_node_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """``[Σnc,k,18]`` (int8/float) + ``[Σnc]`` node ids -> scattered ``[N,k,H]``.

        ``index_add`` only ever touches the ``Σnc`` configurable rows — the dense
        ``[N,k,18]`` (or ``[N,k,H]`` pre-scatter) tensor implied by a naive
        broadcast is never materialised (guardrail: never materialise the full
        per-config node tensor).
        """
        k = node_config_feat.shape[1]
        cfg = self.config_proj(node_config_feat.float())          # [Σnc,k,H]
        H = cfg.shape[-1]
        out = cfg.new_zeros((n_nodes, k, H))
        out.index_add_(0, cfg_node_index, cfg)
        return out

    def message_pass(self, h0: torch.Tensor, scattered_cfg: torch.Tensor,
                     edge_index: torch.Tensor, n_nodes: int, k: int) -> torch.Tensor:
        """``h0 [N,H]`` + ``scattered_cfg [N,k,H]`` -> per-config node embeddings
        ``[N,k,H]`` after L weight-shared ``SAGEConv`` layers over a k-fold,
        node-offset replica of ``edge_index`` (see :func:`_replicate_edge_index`).
        """
        x = h0.unsqueeze(1) + scattered_cfg                        # [N,k,H]
        H = x.shape[-1]
        x = x.permute(1, 0, 2).reshape(k * n_nodes, H)              # config-major
        ei_rep = _replicate_edge_index(edge_index, n_nodes, k)
        for conv in self.convs:
            x = x + self.dropout(torch.relu(conv(x, ei_rep)))
        return x.view(k, n_nodes, H).permute(1, 0, 2)                # back to [N,k,H]

    def pool_and_score(self, x: torch.Tensor, cfg_node_index: torch.Tensor,
                       batch_vec: torch.Tensor, num_graphs: int, k: int) -> torch.Tensor:
        """``[N,k,H]`` per-config node embeddings -> ``[B,k]`` scores.

        Pools per ``(graph, config)`` pair (never mixing configs across the k axis
        or graphs across the batch) via ``torch_geometric.utils.scatter`` over a
        flattened ``graph_id * k + config_id`` index: once over ALL nodes, once
        over only the configurable nodes (``cfg_node_index``); the two pools are
        summed before the head.
        """
        n_nodes, k_, H = x.shape
        device = x.device

        def _pool(vals: torch.Tensor, graph_ids: torch.Tensor) -> torch.Tensor:
            n = vals.shape[0]
            g = graph_ids.unsqueeze(1).expand(n, k_)
            c = torch.arange(k_, device=device).unsqueeze(0).expand(n, k_)
            flat_idx = (g * k_ + c).reshape(-1)
            flat_vals = vals.reshape(-1, H)
            return scatter(flat_vals, flat_idx, dim=0, dim_size=num_graphs * k_,
                          reduce="mean")

        full_pool = _pool(x, batch_vec)                             # [B*k, H]
        cfg_pool = _pool(x[cfg_node_index], batch_vec[cfg_node_index])  # [B*k, H]
        pooled = (full_pool + cfg_pool).view(num_graphs, k_, H)     # [B,k,H]
        if self.use_config_attn:
            # configs of a graph attend to each other along the k axis
            pooled = self.config_attn(pooled)                       # [B,k,H]
        return self.head(pooled).squeeze(-1)                        # [B,k]

    # ---- training / eval forward -----------------------------------------
    def forward(self, batch, gst: bool = False, max_keep_nodes: Optional[int] = None,
               seed: int = 0, epoch: int = 0) -> torch.Tensor:
        """Score ``batch.node_config_feat``'s k configs per graph -> ``[B,k]``.

        ``gst=False`` (default, used by eval/inference): one plain full-graph
        forward pass — no windowing, safe under ``no_grad``. ``gst=True``
        (training only) delegates to :func:`src.training.gst.gst_forward`, which
        bounds backward memory to a ``max_keep_nodes``-node window per graph.
        """
        if gst:
            from src.training.gst import gst_forward  # local import: avoids a
            # src.models <-> src.training import cycle (train_loop imports
            # build_model from src.models at module scope).
            if max_keep_nodes is None:
                raise ValueError("gst=True requires max_keep_nodes")
            scores, _ = gst_forward(self, batch, max_keep_nodes=max_keep_nodes,
                                    seed=seed, epoch=epoch)
            return scores

        n_nodes = batch.x.shape[0]
        k = batch.node_config_feat.shape[1]
        h0 = self.encode_base(batch.x, batch.op)
        cfg = self.scatter_config(batch.node_config_feat, batch.cfg_node_index, n_nodes)
        h = self.message_pass(h0, cfg, batch.edge_index, n_nodes, k)
        return self.pool_and_score(h, batch.cfg_node_index, batch.batch, batch.num_graphs, k)
