"""
Data Exploration and Statistics Module for Predict AI Model Runtime Competition

This module provides comprehensive data exploration capabilities including:
- Dataset structure analysis
- Statistical summaries
- Distribution visualization
- Data quality checks

Dataset Overview:
-----------------
The competition dataset contains NPZ files representing neural network computation graphs.
It is organized into two main collections:

1. LAYOUT Collection (npz/layout/):
   - Source: 'xla' or 'nlp'
   - Search strategy: 'random' or 'default'
   - Structure: [source]/[search]/[split]/ - split is one of: train, valid, test
   
2. TILE Collection (npz/tile/):
   - Source: 'xla' only
   - Structure: [source]/[split]/ - split is one of: train, valid, test

Each NPZ file contains:
- node_feat: Node features [n_nodes, 140] for layout, [n_nodes, ?] for tile
- node_opcode: Operation codes [n_nodes]
- edge_index: Edge connectivity [n_edges, 2] or [2, n_edges]
- config_feat: Configuration features for tile [n_configs, 24]
- node_config_ids: Configurable node IDs for layout
- node_config_feat: Node configuration features for layout [n_configs, n_config_nodes, 18]
- config_runtime: Runtime values [n_configs]
- config_runtime_normalizers: Normalizers for tile [n_configs]
"""

import os
import glob
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd


@dataclass
class NpzStats:
    """Statistics for a single NPZ file."""
    path: str
    collection: str  # e.g., "layout:xla:random" or "tile:xla"
    split: str      # "train", "valid", or "test"
    file_size_bytes: int
    
    # Graph structure
    n_nodes: int
    n_node_features: int
    n_edges: int
    
    # Configuration stats
    n_configs: int
    
    # Runtime statistics
    runtime_mean: float = 0.0
    runtime_std: float = 0.0
    runtime_min: float = 0.0
    runtime_max: float = 0.0
    runtime_p10: float = 0.0
    runtime_p50: float = 0.0
    runtime_p90: float = 0.0
    runtime_cv: float = 0.0  # Coefficient of variation
    
    # Tile-specific statistics
    ratio_mean: float = 0.0  # runtime/normalizer
    ratio_p50: float = 0.0
    ratio_cv: float = 0.0
    
    # Data quality flags
    has_nan: bool = False
    has_inf: bool = False
    has_zero_normalizer: bool = False
    
    # Schema fingerprint
    keys: str = ""
    dtypes: str = ""


def parse_collection_path(path: str) -> Tuple[str, str, str]:
    """
    Parse a file path to extract collection, source, and search info.
    
    Example:
        "npz/layout/xla/random/train/file.npz" -> ("layout", "xla", "random")
        "npz/tile/xla/train/file.npz" -> ("tile", "xla", "")
    """
    parts = Path(path).parts
    
    if "layout" in parts:
        layout_idx = parts.index("layout")
        source = parts[layout_idx + 1]
        search = parts[layout_idx + 2]
        collection = f"layout:{source}:{search}"
    elif "tile" in parts:
        tile_idx = parts.index("tile")
        source = parts[tile_idx + 1]
        collection = f"tile:{source}"
    else:
        collection = "unknown"
        source = "unknown"
        search = "unknown"
    
    return collection, source, search


def get_split_from_path(path: str) -> str:
    """Extract split (train/valid/test) from file path."""
    parts = Path(path).parts
    for split in ["train", "valid", "test"]:
        if split in parts:
            return split
    return "unknown"


def analyze_npz_file(path: str) -> Optional[NpzStats]:
    """
    Analyze a single NPZ file and extract comprehensive statistics.
    
    Args:
        path: Path to the NPZ file
        
    Returns:
        NpzStats object with all extracted information, or None if file cannot be read
    """
    try:
        d = np.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
    
    collection, source, search = parse_collection_path(path)
    split = get_split_from_path(path)
    
    # Basic structure
    node_feat = d["node_feat"]
    edge_index = d["edge_index"]
    
    stats = NpzStats(
        path=path,
        collection=collection,
        split=split,
        file_size_bytes=os.path.getsize(path),
        n_nodes=int(node_feat.shape[0]),
        n_node_features=int(node_feat.shape[1]),
        n_edges=int(edge_index.shape[0]),
        keys=",".join(sorted(d.keys())),
        dtypes=",".join([f"{k}:{d[k].dtype}" for k in sorted(d.keys())])
    )
    
    # Data quality checks
    stats.has_nan = bool(np.isnan(node_feat).any() or 
                        (hasattr(d, 'config_runtime') and np.isnan(d['config_runtime']).any()))
    stats.has_inf = bool(np.isinf(node_feat).any())
    
    # Configuration and runtime analysis
    if "tile" in collection:
        # Tile-specific analysis
        config_feat = d["config_feat"]
        stats.n_configs = int(config_feat.shape[0])
        
        if "config_runtime" in d and "config_runtime_normalizers" in d:
            rt = d["config_runtime"].astype(np.float64)
            rn = d["config_runtime_normalizers"].astype(np.float64)
            
            # Check for zero normalizers
            stats.has_zero_normalizer = bool(np.any(rn == 0))
            
            # Compute safe ratio
            rn_safe = np.where(rn <= 0, np.nan, rn)
            ratio = rt / rn_safe
            
            # Runtime stats
            valid_rt = rt[np.isfinite(rt)]
            if len(valid_rt) > 0:
                stats.runtime_mean = float(np.mean(valid_rt))
                stats.runtime_std = float(np.std(valid_rt))
                stats.runtime_min = float(np.min(valid_rt))
                stats.runtime_max = float(np.max(valid_rt))
                stats.runtime_cv = float(stats.runtime_std / (stats.runtime_mean + 1e-12))
            
            valid_ratio = ratio[np.isfinite(ratio)]
            if len(valid_ratio) > 0:
                stats.ratio_mean = float(np.mean(valid_ratio))
                stats.ratio_p50 = float(np.quantile(valid_ratio, 0.50))
                stats.ratio_cv = float(np.std(valid_ratio) / (stats.ratio_mean + 1e-12))
            
            stats.runtime_p10 = float(np.quantile(rt, 0.10))
            stats.runtime_p50 = float(np.quantile(rt, 0.50))
            stats.runtime_p90 = float(np.quantile(rt, 0.90))
    else:
        # Layout-specific analysis
        if "config_runtime" in d:
            rt = d["config_runtime"].astype(np.float64)
            stats.n_configs = int(rt.shape[0])
            
            valid_rt = rt[np.isfinite(rt)]
            if len(valid_rt) > 0:
                stats.runtime_mean = float(np.mean(valid_rt))
                stats.runtime_std = float(np.std(valid_rt))
                stats.runtime_min = float(np.min(valid_rt))
                stats.runtime_max = float(np.max(valid_rt))
                stats.runtime_cv = float(stats.runtime_std / (stats.runtime_mean + 1e-12))
            
            stats.runtime_p10 = float(np.quantile(rt, 0.10))
            stats.runtime_p50 = float(np.quantile(rt, 0.50))
            stats.runtime_p90 = float(np.quantile(rt, 0.90))
    
    return stats


def scan_directory(data_root: str, pattern: str = "**/*.npz") -> List[str]:
    """
    Recursively find all NPZ files in a directory.
    
    Args:
        data_root: Root directory to scan
        pattern: Glob pattern for file matching
        
    Returns:
        List of paths to NPZ files
    """
    root = Path(data_root)
    files = sorted(root.glob(pattern))
    return [str(f) for f in files if f.is_file()]


def explore_dataset(data_root: str, collections: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Explore the entire dataset and return statistics for all files.
    
    Args:
        data_root: Root directory containing npz files
        collections: Optional list of collections to include (e.g., ["layout:xla:random", "tile:xla"])
        
    Returns:
        DataFrame with statistics for each file
    """
    print(f"Scanning {data_root} for NPZ files...")
    all_files = scan_directory(data_root)
    print(f"Found {len(all_files)} NPZ files")
    
    stats_list = []
    for i, path in enumerate(all_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(all_files)}...")
        
        stats = analyze_npz_file(path)
        if stats is not None:
            # Filter by collection if specified
            if collections is None or stats.collection in collections:
                stats_list.append(asdict(stats))
    
    df = pd.DataFrame(stats_list)
    print(f"Successfully analyzed {len(df)} files")
    
    return df


def compute_collection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each collection.
    
    Args:
        df: DataFrame with individual file statistics
        
    Returns:
        DataFrame with collection-level summaries
    """
    summary = df.groupby("collection").agg({
        "path": "count",
        "file_size_bytes": ["sum", "mean", "min", "max"],
        "n_nodes": ["mean", "std", "min", "max"],
        "n_edges": ["mean", "std", "min", "max"],
        "n_configs": ["mean", "std", "min", "max"],
        "runtime_mean": ["mean", "std"],
        "runtime_cv": ["mean", "std"],
        "has_nan": "sum",
        "has_inf": "sum",
    }).round(2)
    
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    return summary.reset_index()


def compute_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for each split within each collection.
    
    Args:
        df: DataFrame with individual file statistics
        
    Returns:
        DataFrame with split-level summaries
    """
    summary = df.groupby(["collection", "split"]).agg({
        "path": "count",
        "file_size_bytes": "sum",
        "n_nodes": ["mean", "sum"],
        "n_edges": ["mean", "sum"],
        "n_configs": ["mean", "sum"],
        "runtime_mean": "mean",
    }).round(2)
    
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
    return summary.reset_index()


def analyze_opcode_distribution(df: pd.DataFrame, sample_files: List[str]) -> Dict[str, np.ndarray]:
    """
    Analyze the distribution of operation codes across sample files.
    
    Args:
        df: DataFrame with file statistics
        sample_files: List of file paths to sample for opcode analysis
        
    Returns:
        Dictionary mapping collection to opcode counts
    """
    opcode_counts = defaultdict(lambda: defaultdict(int))
    
    for path in sample_files:
        try:
            d = np.load(path)
            if "node_opcode" in d:
                opcodes = d["node_opcode"]
                collection, _, _ = parse_collection_path(path)
                
                for opcode in np.unique(opcodes):
                    opcode_counts[collection][int(opcode)] += 1
        except Exception as e:
            print(f"Error analyzing {path}: {e}")
    
    return dict(opcode_counts)


def detect_anomalies(df: pd.DataFrame, 
                     runtime_cv_threshold: float = 2.0,
                     node_count_threshold: int = 100000,
                     edge_count_threshold: int = 1000000) -> pd.DataFrame:
    """
    Detect potential anomalies in the dataset.
    
    Args:
        df: DataFrame with file statistics
        runtime_cv_threshold: Threshold for high coefficient of variation
        node_count_threshold: Threshold for unusually large node counts
        edge_count_threshold: Threshold for unusually large edge counts
        
    Returns:
        DataFrame with flagged anomalous files
    """
    anomalies = []
    
    # High runtime variance
    high_var = df[df["runtime_cv"] > runtime_cv_threshold]
    for _, row in high_var.iterrows():
        anomalies.append({
            "path": row["path"],
            "collection": row["collection"],
            "split": row["split"],
            "issue": f"High runtime CV: {row['runtime_cv']:.2f}",
            "severity": "warning"
        })
    
    # Large node counts
    large_nodes = df[df["n_nodes"] > node_count_threshold]
    for _, row in large_nodes.iterrows():
        anomalies.append({
            "path": row["path"],
            "collection": row["collection"],
            "split": row["split"],
            "issue": f"Unusually large: {row['n_nodes']} nodes",
            "severity": "info"
        })
    
    # Large edge counts
    large_edges = df[df["n_edges"] > edge_count_threshold]
    for _, row in large_edges.iterrows():
        anomalies.append({
            "path": row["path"],
            "collection": row["collection"],
            "split": row["split"],
            "issue": f"Unusually large: {row['n_edges']} edges",
            "severity": "info"
        })
    
    # Data quality issues
    nan_issues = df[df["has_nan"] == True]
    for _, row in nan_issues.iterrows():
        anomalies.append({
            "path": row["path"],
            "collection": row["collection"],
            "split": row["split"],
            "issue": "Contains NaN values",
            "severity": "error"
        })
    
    inf_issues = df[df["has_inf"] == True]
    for _, row in inf_issues.iterrows():
        anomalies.append({
            "path": row["path"],
            "collection": row["collection"],
            "split": row["split"],
            "issue": "Contains Inf values",
            "severity": "error"
        })
    
    return pd.DataFrame(anomalies)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore TPUGraphs dataset")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing NPZ files")
    parser.add_argument("--collections", type=str, nargs="+", default=None,
                       help="List of collections to analyze (e.g., layout:xla:random)")
    parser.add_argument("--output", type=str, default="exploration_results.parquet",
                       help="Output file for results")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of files to process (for testing)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TPUGRAPHS DATASET EXPLORATION")
    print("=" * 60)
    
    # Explore dataset
    df = explore_dataset(args.data_root, args.collections)
    
    if args.limit:
        df = df.head(args.limit)
    
    # Compute summaries
    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)
    collection_summary = compute_collection_summary(df)
    print(collection_summary.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    split_summary = compute_split_summary(df)
    print(split_summary.to_string(index=False))
    
    # Detect anomalies
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION")
    print("=" * 60)
    anomalies = detect_anomalies(df)
    print(f"Found {len(anomalies)} potential issues")
    if len(anomalies) > 0:
        print(anomalies.head(20).to_string(index=False))
    
    # Save results
    df.to_parquet(args.output, index=False)
    print(f"\nResults saved to {args.output}")
