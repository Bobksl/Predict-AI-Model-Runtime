"""TPUGraphs Phase-1 data pipeline.

Importable components that turn raw TPUGraphs NPZ files into correctly-batched
PyG graphs for all five collections at batch > 1, plus inventory, normalisation
statistics, a cheap rebuildable cache, and grouped-by-graph CV splits.

See ``docs/src/study_guide.md`` (theory) and ``WORKFLOW.pdf`` (Phase 1) for context.
"""
from .paths import (
    COLLECTIONS, SPLITS, resolve_data_root, collection_dir, split_dir,
    list_npz, parse_collection, writable_dir,
)
from .graph import build_pyg_data, TPUData
from .configs import sample_config_indices, read_runtimes, read_tile_config_feat, read_node_config_feat_rows
from .collate import collate, scatter_node_config_feat
from .dataset import TpugraphsDataset, make_loader
from .normalize import NodeFeatNormalizer
from .splits import assign_graph_folds
from .config_shards import find_shard, read_shard_sample, shard_dir

__all__ = [
    "COLLECTIONS", "SPLITS", "resolve_data_root", "collection_dir", "split_dir",
    "list_npz", "parse_collection", "writable_dir",
    "build_pyg_data", "TPUData",
    "sample_config_indices", "read_runtimes", "read_tile_config_feat",
    "read_node_config_feat_rows",
    "collate", "scatter_node_config_feat",
    "TpugraphsDataset", "make_loader",
    "NodeFeatNormalizer", "assign_graph_folds",
    "find_shard", "read_shard_sample", "shard_dir",
]
