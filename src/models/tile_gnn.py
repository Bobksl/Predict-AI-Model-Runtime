"""TileRanker — small SAGEConv GNN scoring tile configurations.

Shapes (batch B graphs, k configs/graph, ΣN total nodes):

1. ``op [ΣN]`` → Embedding(119, 32) → concat ``x [ΣN,140]`` → Linear → ``[ΣN,64]``
2. 3 × SAGEConv(64,64) + ReLU + residual + dropout → ``[ΣN,64]``
   (SAGE: robust on directed/irregular graphs, no degree-normalisation/self-loop
   fuss; dataset should be built with ``add_reverse_edges=True``.)
3. ``global_mean_pool`` → ``[B,64]``
4. ``config_feat [B,k,24]`` → log1p → Linear(24,32)+ReLU → ``[B,k,32]``
5. concat broadcast graph emb → ``[B,k,96]`` → MLP 96→64→1 → scores ``[B,k]``
   (higher score = slower).

≈45k parameters. Inference encodes the graph once (``encode_graph``) and scores
configs in chunks (``score_configs``) — test graphs have up to 10k configs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

OPCODE_VOCAB = 119  # global opcode_max=118 (inventory) + 1


class TileRanker(nn.Module):
    def __init__(self, opcode_emb_dim: int = 32, hidden_dim: int = 64,
                 n_layers: int = 3, config_proj_dim: int = 32,
                 dropout: float = 0.1, node_feat_dim: int = 140,
                 config_feat_dim: int = 24):
        super().__init__()
        self.op_emb = nn.Embedding(OPCODE_VOCAB, opcode_emb_dim)
        self.node_proj = nn.Linear(node_feat_dim + opcode_emb_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.config_proj = nn.Sequential(
            nn.Linear(config_feat_dim, config_proj_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + config_proj_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    # ---- pieces (used separately at inference) --------------------------
    def encode_graph(self, x: torch.Tensor, op: torch.Tensor,
                     edge_index: torch.Tensor,
                     batch_vec: torch.Tensor) -> torch.Tensor:
        """Node features → pooled graph embedding [B, hidden]."""
        h = self.node_proj(torch.cat([x, self.op_emb(op)], dim=-1))
        h = torch.relu(h)
        for conv in self.convs:
            h = h + self.dropout(torch.relu(conv(h, edge_index)))
        return global_mean_pool(h, batch_vec)

    def score_configs(self, graph_emb: torch.Tensor,
                      config_feat: torch.Tensor) -> torch.Tensor:
        """graph_emb [B,hidden] + config_feat [B,k,24] → scores [B,k]."""
        c = self.config_proj(torch.log1p(config_feat.clamp_min(0.0)))
        g = graph_emb.unsqueeze(1).expand(-1, c.shape[1], -1)
        return self.head(torch.cat([g, c], dim=-1)).squeeze(-1)

    # ---- training forward ------------------------------------------------
    def forward(self, batch) -> torch.Tensor:
        g = self.encode_graph(batch.x, batch.op, batch.edge_index, batch.batch)
        return self.score_configs(g, batch.config_feat)
