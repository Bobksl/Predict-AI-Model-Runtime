# src/loader.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass(frozen=True)
class GraphSample:
    id_str: str
    collection: str
    split: str
    node_feat: torch.Tensor          # [n, 140]
    node_opcode: torch.Tensor        # [n]
    edge_index: torch.Tensor         # [2, m] (PyG convention)
    config_feat: Optional[torch.Tensor]      # tile: [c, 24]
    node_config_ids: Optional[torch.Tensor]  # layout: [nc]
    node_config_feat: Optional[torch.Tensor] # layout: [c, nc, 18]
    runtime: Optional[torch.Tensor]          # [c]
    runtime_norm: Optional[torch.Tensor]     # tile: [c]

def _npz_to_sample(path: Path, collection: str, split: str) -> GraphSample:
    d = np.load(path)

    node_feat = torch.from_numpy(d["node_feat"].astype(np.float32))
    node_opcode = torch.from_numpy(d["node_opcode"].astype(np.int64))

    edge = d["edge_index"].astype(np.int64)  # [m, 2]
    edge_index = torch.from_numpy(edge.T)    # [2, m]

    stem = path.stem
    if collection == "tile:xla":
        config_feat = torch.from_numpy(d["config_feat"].astype(np.float32))
        rt = torch.from_numpy(d["config_runtime"].astype(np.float64))
        rn = torch.from_numpy(d["config_runtime_normalizers"].astype(np.float64))
        id_str = f"tile:xla:{stem}"
        return GraphSample(
            id_str=id_str, collection=collection, split=split,
            node_feat=node_feat, node_opcode=node_opcode, edge_index=edge_index,
            config_feat=config_feat, node_config_ids=None, node_config_feat=None,
            runtime=rt, runtime_norm=rn,
        )
    else:
        node_config_ids = torch.from_numpy(d["node_config_ids"].astype(np.int64))
        node_config_feat = torch.from_numpy(d["node_config_feat"].astype(np.float32))
        rt = torch.from_numpy(d["config_runtime"].astype(np.float64))
        # collection is like "layout:xla:random"
        id_str = f"{collection}:{stem}"
        return GraphSample(
            id_str=id_str, collection=collection, split=split,
            node_feat=node_feat, node_opcode=node_opcode, edge_index=edge_index,
            config_feat=None, node_config_ids=node_config_ids, node_config_feat=node_config_feat,
            runtime=rt, runtime_norm=None,
        )

class TpugraphsDataset(Dataset):
    def __init__(self, paths: list[Path], collection: str, split: str):
        self.paths = paths
        self.collection = collection
        self.split = split

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> GraphSample:
        return _npz_to_sample(self.paths[idx], self.collection, self.split)
