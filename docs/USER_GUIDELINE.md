# TPUGraphs Dataset User Guideline

## Predict AI Model Runtime Competition - Data Processing Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Machine Learning Principle](#machine-learning-principle)
4. [Getting Started](#getting-started)
5. [Data Exploration](#data-exploration)
6. [Data Cleaning](#data-cleaning)
7. [Dataset Loading](#dataset-loading)
8. [Code Examples](#code-examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This guideline provides comprehensive instructions for processing the **Google - Fast or Slow? Predict AI Model Runtime** competition dataset. The dataset contains neural network computation graphs from TPU (Tensor Processing Unit) workloads, where the goal is to predict and rank the runtime of different compilation configurations.

### Competition Objective
Predict the relative runtime of different configurations for compiling AI models on TPU hardware. The task is to rank configurations from fastest to slowest, enabling automatic optimization of neural network compilation.

---

## Dataset Description

### Dataset Structure

The dataset is organized into **NPZ files** containing graph representations of neural network computations. The data is split into two main collections:

```
data/tpugraphs/npz/
├── layout/          # Layout compilation configurations
│   ├── nlp/        # NLP model graphs
│   │   ├── default/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   └── random/
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   └── xla/        # XLA compiler graphs
│       ├── default/
│       └── random/
└── tile/           # Tile-level optimization configurations
    └── xla/
        ├── train/
        ├── valid/
        └── test/
```

### Collections

| Collection | Description | Available Splits |
|------------|-------------|------------------|
| `layout:xla:random` | XLA graphs with random configurations | train, valid, test |
| `layout:xla:default` | XLA graphs with default configurations | train, valid, test |
| `layout:nlp:random` | NLP model graphs with random configs | train, valid, test |
| `layout:nlp:default` | NLP model graphs with default configs | train, valid, test |
| `tile:xla` | XLA tile-level optimizations | train, valid, test |

### NPZ File Schema

Each NPZ file contains the following arrays:

#### Layout Collection
```python
{
    'node_feat': np.ndarray,      # Shape: [n_nodes, 140] - Node features
    'node_opcode': np.ndarray,     # Shape: [n_nodes] - Operation type IDs
    'edge_index': np.ndarray,      # Shape: [n_edges, 2] or [2, n_edges] - Graph edges
    'node_config_ids': np.ndarray, # Shape: [n_config_nodes] - IDs of configurable nodes
    'node_config_feat': np.ndarray,# Shape: [n_configs, n_config_nodes, 18] - Config features
    'config_runtime': np.ndarray,  # Shape: [n_configs] - Runtime for each config
}
```

#### Tile Collection
```python
{
    'node_feat': np.ndarray,                    # Shape: [n_nodes, ?] - Node features
    'node_opcode': np.ndarray,                  # Shape: [n_nodes] - Operation type IDs
    'edge_index': np.ndarray,                   # Shape: [n_edges, 2] or [2, n_edges]
    'config_feat': np.ndarray,                   # Shape: [n_configs, 24] - Config features
    'config_runtime': np.ndarray,                # Shape: [n_configs] - Runtime values
    'config_runtime_normalizers': np.ndarray,    # Shape: [n_configs] - Normalization factors
}
```

---

## Machine Learning Principle

### Problem Type
This is a **learning-to-rank** problem in the context of graph neural networks (GNNs).

### Key Concepts

1. **Graph Representation Learning**: Each neural network computation is represented as a directed graph where:
   - **Nodes** represent operations (e.g., matrix multiplications, activations)
   - **Edges** represent data dependencies between operations
   - **Node features** capture operation attributes (shape, dtype, etc.)
   - **Config features** represent different compilation strategies

2. **Learning Objective**: Given a graph with multiple configurations, predict the relative ordering of runtimes. The model learns to score configurations such that faster configurations receive higher scores.

3. **Evaluation Metric**: The competition uses **OPA (Order Accuracy)** or similar ranking metrics to evaluate how well the predicted rankings match actual runtime orderings.

4. **Approaches**:
   - **Graph Neural Networks (GNN)**: Use message passing to aggregate information from neighbors
   - **Feature Engineering**: Normalize node/config features, handle opcode embeddings
   - **Ranking Losses**: Use listwise or pairwise ranking objectives

---

## Getting Started

### Prerequisites

```bash
# Install required packages
pip install numpy pandas torch tqdm matplotlib

# Optional: For GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Download the Dataset

```bash
# Using Kaggle CLI
kaggle competitions download -c predict-ai-model-runtime -p ./data/raw

# Or using kagglehub
python -c "import kagglehub; kagglehub.competition_download('predict-ai-model-runtime', output_dir='./data')"
```

### Directory Structure

```
SC4000/
├── data/
│   └── tpugraphs/
│       └── npz/
│           ├── layout/
│           └── tile/
├── data_exploration.py    # Dataset exploration tool
├── data_cleaning.py       # Data cleaning utilities
├── enhanced_loader.py      # PyTorch dataset loader
├── pytorch_loader.py       # Basic PyTorch loader
├── manifest-driven_aggregation.py
├── inventory_generation.py
└── USER_GUIDELINE.md      # This file
```

---

## Data Exploration

### Using the Exploration Tool

```bash
# Explore entire dataset
python data_exploration.py --data_root ./data/tpugraphs --output exploration_results.parquet

# Explore specific collections
python data_exploration.py --data_root ./data/tpugraphs \
    --collections layout:xla:random tile:xla \
    --output tile_xla_results.parquet

# Limit number of files (for testing)
python data_exploration.py --data_root ./data/tpugraphs --limit 100
```

### What the Exploration Tool Provides

1. **File Statistics**: Number of nodes, edges, configurations per file
2. **Runtime Statistics**: Mean, std, percentiles of runtimes
3. **Data Quality Checks**: Detection of NaN, Inf values
4. **Schema Consistency**: Verification of consistent data formats
5. **Anomaly Detection**: Identification of unusual files

### Sample Exploration Output

```
============================================================
TPUGRAPHS DATASET EXPLORATION
============================================================
Scanning ./data/tpugraphs for NPZ files...
Found 1000 NPZ files
Processing file 1/1000...
...

============================================================
COLLECTION SUMMARY
============================================================
  collection    count    mean_nodes    mean_edges    mean_configs
  layout:xla:random    250    1500.5    8000.2    256.0
  layout:xla:default   200    1200.3    6500.1    128.0
  tile:xla            550    3000.7    15000.3    512.0

============================================================
ANOMALY DETECTION
============================================================
Found 5 potential issues
```

---

## Data Cleaning

### Cleaning Configuration

The cleaning module handles common data quality issues:

| Issue | Strategy | Description |
|-------|----------|-------------|
| NaN values | REPLACE | Replace with configurable fill value |
| Inf values | REPLACE | Replace with configurable fill value |
| Zero normalizers | REPLACE | Replace with 1.0 (safe default) |
| Outliers | CLIP | Clip to percentile-based bounds |

### Usage Examples

```python
# Basic usage - clean all files in a directory
from data_cleaning import DataCleaner, CleaningConfig

config = CleaningConfig(
    nan_fill_value=0.0,
    inf_fill_value=0.0,
    zero_normalizer_fill_value=1.0,
    clip_runtime_ratio=True,
    outlier_std_threshold=5.0,
)

cleaner = DataCleaner(config)
report = cleaner.clean_directory(
    data_root="./data/tpugraphs",
    output_dir="./data/cleaned"
)
report.save("cleaning_report.json")
```

```bash
# Command line usage
python data_cleaning.py --input ./data/tpugraphs --output ./data/cleaned --report report.json
```

### Verification

```python
from data_cleaning import verify_cleaned_file

is_clean, issues = verify_cleaned_file("cleaned_file.npz")
if is_clean:
    print("File is clean!")
else:
    print(f"Issues found: {issues}")
```

---

## Dataset Loading

### Basic Loading

```python
from enhanced_loader import TpugraphsDataset, create_dataloaders

# Load a single collection
dataset = TpugraphsDataset(
    data_root="./data/tpugraphs",
    collection="tile:xla",
    split="train"
)

print(f"Found {len(dataset)} samples")
sample = dataset[0]
print(f"Sample shape: {sample.node_feat.shape}")
```

### Creating DataLoaders

```python
# Create train and validation loaders
train_loader, valid_loader = create_dataloaders(
    data_root="./data/tpugraphs",
    collection="layout:xla:random",
    batch_size=16,
    num_workers=4,
    max_nodes=10000,  # Limit nodes for large graphs
    max_configs=100,  # Limit configs for memory efficiency
)

# Iterate
for batch in train_loader:
    node_feat = batch["node_feat"]
    runtime = batch["config_runtime"]
    # ... training code
```

### Multi-Collection Loading

```python
from enhanced_loader import MultiCollectionDataset

# Combine multiple collections
dataset = MultiCollectionDataset(
    data_root="./data/tpugraphs",
    collections=[
        "layout:xla:random",
        "layout:xla:default",
        "layout:nlp:random",
        "layout:nlp:default",
        "tile:xla",
    ],
    split="train",
    sample_weights=[1.0, 1.0, 1.0, 1.0, 2.0],  # Weight tile more
)
```

### Custom Collator for Ranking

```python
from enhanced_loader import RankingCollator
from torch.utils.data import DataLoader

collator = RankingCollator()
loader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=collator
)
```

---

## Code Examples

### Example 1: Complete Data Pipeline

```python
from enhanced_loader import TpugraphsDataset, create_dataloaders
from data_cleaning import DataCleaner, CleaningConfig
from data_exploration import explore_dataset
import pandas as pd

# Step 1: Explore the data
print("Exploring dataset...")
df = explore_dataset("./data/tpugraphs")
print(f"Found {len(df)} files")

# Step 2: Clean the data
print("\nCleaning data...")
config = CleaningConfig()
cleaner = DataCleaner(config)
report = cleaner.clean_directory(
    "./data/tpugraphs",
    output_dir="./data/cleaned"
)
print(f"Cleaned {report.files_modified} files")

# Step 3: Create dataloaders
print("\nCreating dataloaders...")
train_loader, valid_loader = create_dataloaders(
    "./data/cleaned",
    collection="tile:xla",
    batch_size=16,
    num_workers=4,
)

# Step 4: Iterate through data
print("\nSample iteration:")
for i, batch in enumerate(train_loader):
    if i >= 2:
        break
    print(f"Batch {i}: node_feat shape = {batch['node_feat'].shape}")
```

### Example 2: Analyzing a Single File

```python
import numpy as np
from pathlib import Path

def analyze_npz(path):
    """Analyze a single NPZ file."""
    d = np.load(path)
    
    print(f"File: {path}")
    print(f"Keys: {list(d.keys())}")
    
    for key in d.keys():
        arr = d[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Check for NaN/Inf
        if arr.dtype in [np.float32, np.float64]:
            has_nan = np.isnan(arr).any()
            has_inf = np.isinf(arr).any()
            print(f"    NaN: {has_nan}, Inf: {has_inf}")
    
    # Compute runtime statistics
    if 'config_runtime' in d:
        rt = d['config_runtime']
        print(f"\nRuntime statistics:")
        print(f"  Mean: {np.mean(rt):.2f}")
        print(f"  Std: {np.std(rt):.2f}")
        print(f"  Min: {np.min(rt):.2f}")
        print(f"  Max: {np.max(rt):.2f}")
        
    d.close()

# Example usage
analyze_npz("./data/tpugraphs/npz/tile/xla/train/alexnet_train_batch_32_-1bae27a41d70f4dc.npz")
```

### Example 3: Custom Dataset with Augmentation

```python
import torch
from enhanced_loader import TpugraphsDataset

class AugmentedDataset(TpugraphsDataset):
    """Dataset with augmentation for training."""
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # Apply node feature normalization
        mean = sample.node_feat.mean(dim=0, keepdim=True)
        std = sample.node_feat.std(dim=0, keepdim=True) + 1e-8
        sample.node_feat = (sample.node_feat - mean) / std
        
        # Compute safe runtime ratio (for tile data)
        if sample.runtime_norm is not None:
            safe_norm = torch.where(
                sample.runtime_norm > 0,
                sample.runtime_norm,
                torch.ones_like(sample.runtime_norm)
            )
            sample.runtime_ratio = sample.runtime / safe_norm
        else:
            # Normalize layout runtime
            sample.runtime = sample.runtime / (sample.runtime.mean() + 1e-8)
        
        return sample

# Usage
dataset = AugmentedDataset(
    data_root="./data/tpugraphs",
    collection="tile:xla",
    split="train",
    max_nodes=5000,
)
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```python
# Solution: Limit graph size
dataset = TpugraphsDataset(
    data_root="./data/tpugraphs",
    collection="tile:xla",
    max_nodes=5000,  # Reduce from default
    max_configs=50,
)
```

#### 2. Slow Data Loading
```python
# Solution: Use multiple workers
loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,  # Increase workers
    pin_memory=True,  # Faster GPU transfer
)
```

#### 3. NaN Values in Features
```python
# Solution: Clean data first
from data_cleaning import DataCleaner, CleaningConfig

cleaner = DataCleaner(CleaningConfig(
    nan_fill_value=0.0,
    inf_fill_value=0.0,
))
cleaner.clean_directory("./data/tpugraphs", output_dir="./data/cleaned")
```

#### 4. Inconsistent Schema
```python
# Solution: Check schema consistency
from data_exploration import explore_dataset

df = explore_dataset("./data/tpugraphs")
# Check for different schemas
schema_counts = df['keys'].value_counts()
print(schema_counts)
```

### File Path Issues (Windows)

If you encounter path issues on Windows, use raw strings or forward slashes:

```python
# Wrong
path = "C:\Users\data"  # Backslash escape issues

# Correct
path = r"C:\Users\data"  # Raw string
path = "C:/Users/data"   # Forward slashes
```

---

## Summary

This guideline covered:

1. **Dataset Structure**: Layout and tile collections with train/valid/test splits
2. **ML Principle**: Learning-to-rank using graph neural networks
3. **Data Exploration**: Comprehensive statistics and quality checks
4. **Data Cleaning**: Handling NaN, Inf, outliers, and schema issues
5. **Dataset Loading**: PyTorch integration with batching and sampling
6. **Code Examples**: Complete pipelines and common use cases

For more details, refer to the individual module docstrings and the starter notebook in `starter-notebook-fast-or-slow-with-tensorflow-gnn.ipynb`.

---

## Contact & Support

For issues or questions about this data processing toolkit, please refer to the competition discussion forum at:
https://www.kaggle.com/competitions/predict-ai-model-runtime/discussion
