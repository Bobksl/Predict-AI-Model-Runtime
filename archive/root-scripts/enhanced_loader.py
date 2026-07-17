"""
Enhanced Dataset Loader for Predict AI Model Runtime Competition

Fixed version with correct data structure:
- edge_index: [n_edges, 2] format (NOT [2, n_edges])
- config_runtime: int64 dtype (NOT float64)
- Added node_splits support for layout files
"""

import os
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class GraphSample:
    """
    Represents a single graph sample from the dataset.
    
    Attributes:
        id_str: Unique identifier for this sample
        collection: Collection name (e.g., "layout:xla:random")
        split: Split name ("train", "valid", or "test")
        node_feat: Node features tensor [n_nodes, 140]
        node_opcode: Operation code tensor [n_nodes]
        edge_index: Edge connectivity tensor [2, n_edges] (PyG convention) OR [n_edges, 2] (raw)
        config_feat: Configuration features for tile [n_configs, 24]
        node_config_ids: Configurable node IDs for layout
        node_config_feat: Node configuration features for layout [n_configs, n_config_nodes, 18]
        node_splits: Subgraph splits for layout [1, n_subgraphs]
        runtime: Runtime values [n_configs]
        runtime_norm: Normalizers for tile [n_configs]
    """
    id_str: str
    collection: str
    split: str
    node_feat: torch.Tensor
    node_opcode: torch.Tensor
    edge_index: torch.Tensor  # Raw: [n_edges, 2]
    config_feat: Optional[torch.Tensor] = None
    node_config_ids: Optional[torch.Tensor] = None
    node_config_feat: Optional[torch.Tensor] = None
    node_splits: Optional[torch.Tensor] = None
    runtime: Optional[torch.Tensor] = None
    runtime_norm: Optional[torch.Tensor] = None


def get_collection_paths(data_root: str, collection: str) -> Dict[str, str]:
    """
    Get paths for a collection.
    
    Args:
        data_root: Root directory containing npz files
        collection: Collection name (e.g., "tile:xla" or "layout:xla:random")
        
    Returns:
        Dictionary with keys 'train', 'valid', 'test' mapping to directories
    """
    parts = collection.split(":")
    
    if parts[0] == "tile":
        source = parts[1]
        base = os.path.join(data_root, "npz", "tile", source)
    elif parts[0] == "layout":
        source = parts[1]
        search = parts[2]
        base = os.path.join(data_root, "npz", "layout", source, search)
    else:
        raise ValueError(f"Unknown collection type: {collection}")
    
    return {
        "train": os.path.join(base, "train"),
        "valid": os.path.join(base, "valid"),
        "test": os.path.join(base, "test"),
    }


def find_npz_files(directory: str, pattern: str = "*.npz") -> List[str]:
    """Find all NPZ files in a directory."""
    if not os.path.exists(directory):
        return []
    return sorted(glob.glob(os.path.join(directory, pattern)))


def npz_to_sample(path: str, collection: str, split: str) -> GraphSample:
    """
    Load a single NPZ file and convert to GraphSample.
    
    Args:
        path: Path to NPZ file
        collection: Collection name
        split: Split name
        
    Returns:
        GraphSample object
    """
    d = np.load(path)
    
    # Load basic components
    node_feat = torch.from_numpy(d["node_feat"].astype(np.float32))
    node_opcode = torch.from_numpy(d["node_opcode"].astype(np.int64))
    
    # Edge index is [n_edges, 2] in the raw data
    # We keep it as-is but note that PyG expects [2, n_edges]
    edge = d["edge_index"].astype(np.int64)
    # edge shape: [n_edges, 2]
    edge_index = torch.from_numpy(edge)
    
    stem = Path(path).stem
    is_tile = collection.startswith("tile")
    
    if is_tile:
        # Tile-specific loading
        config_feat = torch.from_numpy(d["config_feat"].astype(np.float32))
        # Runtime is int64, representing microseconds
        runtime = torch.from_numpy(d["config_runtime"].astype(np.int64))
        runtime_norm = torch.from_numpy(d["config_runtime_normalizers"].astype(np.int64))
        
        return GraphSample(
            id_str=f"tile:xla:{stem}",
            collection=collection,
            split=split,
            node_feat=node_feat,
            node_opcode=node_opcode,
            edge_index=edge_index,
            config_feat=config_feat,
            node_config_ids=None,
            node_config_feat=None,
            node_splits=None,
            runtime=runtime,
            runtime_norm=runtime_norm,
        )
    else:
        # Layout-specific loading
        node_config_ids = torch.from_numpy(d["node_config_ids"].astype(np.int64))
        node_config_feat = torch.from_numpy(d["node_config_feat"].astype(np.float32))
        node_splits = torch.from_numpy(d["node_splits"].astype(np.int64))
        # Runtime is int64, representing microseconds
        runtime = torch.from_numpy(d["config_runtime"].astype(np.int64))
        
        return GraphSample(
            id_str=f"{collection}:{stem}",
            collection=collection,
            split=split,
            node_feat=node_feat,
            node_opcode=node_opcode,
            edge_index=edge_index,
            config_feat=None,
            node_config_ids=node_config_ids,
            node_config_feat=node_config_feat,
            node_splits=node_splits,
            runtime=runtime,
            runtime_norm=None,
        )


class TpugraphsDataset(Dataset):
    """
    PyTorch Dataset for TPUGraphs data.
    
    Supports both layout and tile collections with efficient loading.
    """
    
    def __init__(
        self,
        data_root: str,
        collection: str,
        split: str = "train",
        max_nodes: Optional[int] = None,
        max_configs: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_root: Root directory containing npz files
            collection: Collection name (e.g., "tile:xla" or "layout:xla:random")
            split: Split name ("train", "valid", or "test")
            max_nodes: Maximum number of nodes (for sampling large graphs)
            max_configs: Maximum number of configs (for sampling)
        """
        self.data_root = data_root
        self.collection = collection
        self.split = split
        self.max_nodes = max_nodes
        self.max_configs = max_configs
        
        # Get paths for this collection
        paths = get_collection_paths(data_root, collection)
        
        # Find files for this split
        self.files = find_npz_files(paths[split])
        
        if not self.files:
            print(f"Warning: No files found in {paths[split]}")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> GraphSample:
        path = self.files[idx]
        sample = npz_to_sample(path, self.collection, self.split)
        
        # Apply node sampling if enabled
        if self.max_nodes and sample.node_feat.shape[0] > self.max_nodes:
            sample = self._sample_nodes(sample)
        
        # Apply config sampling if enabled
        if self.max_configs and sample.runtime is not None:
            if sample.runtime.shape[0] > self.max_configs:
                sample = self._sample_configs(sample)
        
        return sample
    
    def _sample_nodes(self, sample: GraphSample) -> GraphSample:
        """Sample nodes from a large graph."""
        n_nodes = sample.node_feat.shape[0]
        indices = torch.randperm(n_nodes)[:self.max_nodes]
        
        return GraphSample(
            id_str=sample.id_str,
            collection=sample.collection,
            split=sample.split,
            node_feat=sample.node_feat[indices],
            node_opcode=sample.node_opcode[indices],
            edge_index=sample.edge_index,
            config_feat=sample.config_feat,
            node_config_ids=sample.node_config_ids,
            node_config_feat=sample.node_config_feat,
            node_splits=sample.node_splits,
            runtime=sample.runtime,
            runtime_norm=sample.runtime_norm,
        )
    
    def _sample_configs(self, sample: GraphSample) -> GraphSample:
        """Sample configs from a graph."""
        n_configs = sample.runtime.shape[0]
        indices = torch.randperm(n_configs)[:self.max_configs]
        
        return GraphSample(
            id_str=sample.id_str,
            collection=sample.collection,
            split=sample.split,
            node_feat=sample.node_feat,
            node_opcode=sample.node_opcode,
            edge_index=sample.edge_index,
            config_feat=sample.config_feat[indices] if sample.config_feat is not None else None,
            node_config_ids=sample.node_config_ids,
            node_config_feat=sample.node_config_feat[:, indices] if sample.node_config_feat is not None else None,
            node_splits=sample.node_splits,
            runtime=sample.runtime[indices],
            runtime_norm=sample.runtime_norm[indices] if sample.runtime_norm is not None else None,
        )


def create_dataloaders(
    data_root: str,
    collection: str,
    batch_size: int = 16,
    num_workers: int = 0,
    train_split: str = "train",
    valid_split: str = "valid",
    max_nodes: Optional[int] = None,
    max_configs: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for a collection.
    
    Args:
        data_root: Root directory containing npz files
        collection: Collection name
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Training split name
        valid_split: Validation split name
        max_nodes: Maximum number of nodes
        max_configs: Maximum number of configs
        
    Returns:
        Tuple of (train_loader, valid_loader)
    """
    train_dataset = TpugraphsDataset(
        data_root=data_root,
        collection=collection,
        split=train_split,
        max_nodes=max_nodes,
        max_configs=max_configs,
    )
    
    valid_dataset = TpugraphsDataset(
        data_root=data_root,
        collection=collection,
        split=valid_split,
        max_nodes=max_nodes,
        max_configs=max_configs,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, valid_loader


def get_default_collections() -> List[str]:
    """Get list of all available collections."""
    return [
        "tile:xla",
        "layout:xla:random",
        "layout:xla:default",
        "layout:nlp:random",
        "layout:nlp:default",
    ]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TPUGraphs dataset loader")
    parser.add_argument("--data_root", type=str, 
                       default="./data/tpugraphs",
                       help="Root directory containing npz files")
    parser.add_argument("--collection", type=str,
                       default="tile:xla",
                       help="Collection to load (e.g., tile:xla, layout:xla:random)")
    parser.add_argument("--split", type=str, default="train",
                       help="Split to load")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    
    args = parser.parse_args()
    
    print(f"Loading {args.collection}/{args.split} from {args.data_root}")
    
    dataset = TpugraphsDataset(
        data_root=args.data_root,
        collection=args.collection,
        split=args.split,
    )
    
    print(f"Found {len(dataset)} files")
    
    if len(dataset) > 0:
        # Test loading a sample
        sample = dataset[0]
        print(f"\nSample ID: {sample.id_str}")
        print(f"  Collection: {sample.collection}")
        print(f"  Split: {sample.split}")
        print(f"  Node features: {sample.node_feat.shape}")
        print(f"  Node opcodes: {sample.node_opcode.shape}")
        print(f"  Edge index: {sample.edge_index.shape} (format: [n_edges, 2])")
        
        if sample.config_feat is not None:
            print(f"  Config features: {sample.config_feat.shape}")
        if sample.node_config_ids is not None:
            print(f"  Node config IDs: {sample.node_config_ids.shape}")
        if sample.node_config_feat is not None:
            print(f"  Node config feat: {sample.node_config_feat.shape}")
        if sample.node_splits is not None:
            print(f"  Node splits: {sample.node_splits.shape}")
        if sample.runtime is not None:
            print(f"  Runtime: {sample.runtime.shape}, dtype: {sample.runtime.dtype}")
            print(f"  Runtime range: {sample.runtime.min().item()} - {sample.runtime.max().item()} us")
        if sample.runtime_norm is not None:
            print(f"  Runtime norm: {sample.runtime_norm.shape}")
