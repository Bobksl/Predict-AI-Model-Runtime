# TPUGraphs Dataset Structure Documentation

## Overview

This document describes the actual data structure of the **Predict AI Model Runtime** competition dataset, based on inspection of real NPZ files.

---

## Directory Structure

```
data/tpugraphs/npz/
├── tile/
│   └── xla/
│       ├── train/          # Training files
│       ├── valid/          # Validation files
│       └── test/           # Test files (no runtime labels)
│
├── layout/
│   ├── xla/
│   │   ├── random/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   └── default/
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   └── nlp/
│       ├── random/
│       │   ├── train/
│       │   ├── valid/
│       │   └── test/
│       └── default/
│           ├── train/
│           ├── valid/
│           └── test/
```

---

## Collection Types

The dataset contains **5 collections**:

| Collection ID | Description | Location |
|--------------|-------------|----------|
| `tile:xla` | Tile-level optimizations for XLA | `tile/xla/*/` |
| `layout:xla:random` | XLA layout with random search | `layout/xla/random/*/` |
| `layout:xla:default` | XLA layout with default search | `layout/xla/default/*/` |
| `layout:nlp:random` | NLP model layout with random search | `layout/nlp/random/*/` |
| `layout:nlp:default` | NLP model layout with default search | `layout/nlp/default/*/` |

---

## NPZ File Schema

### 1. TILE Collection (`tile:xla`)

Files contain **tile-level optimization data** where the tile configuration affects runtime.

#### Keys and Shapes

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `node_feat` | `[n_nodes, 140]` | float32 | Node features (140 dimensions) |
| `node_opcode` | `[n_nodes]` | uint8 | Operation type IDs (0-255) |
| `edge_index` | `[n_edges, 2]` | int64 | Edge list `[src, dst]` format |
| `config_feat` | `[n_configs, 24]` | float32 | Configuration feature vector |
| `config_runtime` | `[n_configs]` | int64 | Runtime in microseconds |
| `config_runtime_normalizers` | `[n_configs]` | int64 | Normalization factors |

#### Example (alexnet)

```
node_feat:                   (12, 140)     - 12 nodes, 140 features each
node_opcode:                 (12,)         - 12 operation codes
edge_index:                  (11, 2)       - 11 edges
config_feat:                 (266, 24)     - 266 configurations
config_runtime:              (266,)        - 266 runtime values
config_runtime_normalizers:  (266,)        - 266 normalizers
```

---

### 2. LAYOUT Collection (`layout:xla:*` and `layout:nlp:*`)

Files contain **layout configuration data** where node positioning affects runtime.

#### Keys and Shapes

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `node_feat` | `[n_nodes, 140]` | float32 | Node features (140 dimensions) |
| `node_opcode` | `[n_nodes]` | uint8 | Operation type IDs (0-255) |
| `edge_index` | `[n_edges, 2]` | int64 | Edge list `[src, dst]` format |
| `node_config_ids` | `[n_configurable_nodes]` | int64 | IDs of configurable nodes |
| `node_config_feat` | `[n_configs, n_configurable_nodes, 18]` | float32 | Config features per node |
| `node_splits` | `[1, n_subgraphs]` | int64 | Subgraph split boundaries |
| `config_runtime` | `[n_configs]` | int64 | Runtime in microseconds |

#### Example (bert_classifier)

```
node_feat:              (40332, 140)      - 40332 nodes
node_opcode:            (40332,)          - 40332 opcodes
edge_index:             (71912, 2)        - 71912 edges
node_config_ids:        (2251,)           - 2251 configurable nodes
node_config_feat:       (10760, 2251, 18) - 10760 configs × 2251 nodes × 18 features
node_splits:            (1, 469)          - 469 subgraphs
config_runtime:         (10760,)          - 10760 runtime values
```

---

## Detailed Field Descriptions

### Node Features (`node_feat`)

- **Shape**: `[n_nodes, 140]` for all collections
- **Dtype**: float32
- **Content**: 140-dimensional feature vector per operation node
- **Value Range**: Typically 0 to millions (shapes, sizes, etc.)

### Node Opcodes (`node_opcode`)

- **Shape**: `[n_nodes]`
- **Dtype**: uint8
- **Content**: Operation type ID (0-255)
- **Common Values**: Various operation types like matmul, add, relu, etc.

### Edge Index (`edge_index`)

- **Shape**: `[n_edges, 2]` (NOT `[2, n_edges]`)
- **Dtype**: int64
- **Format**: Each row is `[source_node_id, target_node_id]`
- **Value Range**: 0 to n_nodes-1

### Configuration Features (`config_feat` - Tile only)

- **Shape**: `[n_configs, 24]`
- **Dtype**: float32
- **Content**: 24-dimensional feature vector per configuration

### Node Configuration Features (`node_config_feat` - Layout only)

- **Shape**: `[n_configs, n_configurable_nodes, 18]`
- **Dtype**: float32
- **Content**: 18-dimensional features for each configurable node in each config

### Node Configuration IDs (`node_config_ids` - Layout only)

- **Shape**: `[n_configurable_nodes]`
- **Dtype**: int64
- **Content**: Indices of nodes that are configurable

### Node Splits (`node_splits` - Layout only)

- **Shape**: `[1, n_subgraphs]`
- **Dtype**: int64
- **Content**: Cumulative node counts defining subgraph boundaries

### Runtime Values (`config_runtime`)

- **Shape**: `[n_configs]`
- **Dtype**: int64 (NOT float64)
- **Unit**: Microseconds
- **Note**: No NaN or Inf values in the data

### Runtime Normalizers (`config_runtime_normalizers` - Tile only)

- **Shape**: `[n_configs]`
- **Dtype**: int64 (NOT float64)
- **Content**: Normalization factors for runtime comparison

---

## Path Parsing Logic

### Collection Identification

```python
def parse_collection(path: str) -> str:
    """Parse collection from file path."""
    parts = Path(path).parts
    
    if "tile" in parts and "xla" in parts:
        return "tile:xla"
    elif "layout" in parts:
        idx = parts.index("layout")
        source = parts[idx + 1]   # 'xla' or 'nlp'
        search = parts[idx + 2]   # 'random' or 'default'
        return f"layout:{source}:{search}"
    
    return "unknown"
```

### Split Identification

```python
def get_split(path: str) -> str:
    """Extract split from path."""
    for split in ["train", "valid", "test"]:
        if f"/{split}/" in path or path.endswith(f"/{split}"):
            return split
    return "unknown"
```

### Example Parsed Values

| File Path | Collection | Split |
|-----------|------------|-------|
| `.../tile/xla/train/alexnet.npz` | `tile:xla` | `train` |
| `.../layout/xla/random/train/bert.npz` | `layout:xla:random` | `train` |
| `.../layout/nlp/default/valid/albert.npz` | `layout:nlp:default` | `valid` |

---

## Data Loading Patterns

### Tile Data Loading

```python
d = np.load("tile_file.npz")
node_feat = d["node_feat"]           # [n_nodes, 140]
node_opcode = d["node_opcode"]       # [n_nodes]
edge_index = d["edge_index"]         # [n_edges, 2]
config_feat = d["config_feat"]       # [n_configs, 24]
runtime = d["config_runtime"]        # [n_configs]
normalizer = d["config_runtime_normalizers"]  # [n_configs]
```

### Layout Data Loading

```python
d = np.load("layout_file.npz")
node_feat = d["node_feat"]           # [n_nodes, 140]
node_opcode = d["node_opcode"]       # [n_nodes]
edge_index = d["edge_index"]         # [n_edges, 2]
node_config_ids = d["node_config_ids"]  # [n_configurable_nodes]
node_config_feat = d["node_config_feat"]  # [n_configs, n_nodes, 18]
node_splits = d["node_splits"]       # [1, n_subgraphs]
runtime = d["config_runtime"]         # [n_configs]
```

---

## Important Notes

1. **Edge Format**: The actual edge_index is `[n_edges, 2]`, NOT `[2, n_edges]` as might be expected for PyG.

2. **Dtype**: Runtime values are `int64`, not `float64`. They represent microseconds.

3. **No Missing Values**: Based on inspection, there are NO NaN or Inf values in the data.

4. **Configurable Nodes**: Only a subset of nodes in layout graphs are configurable (stored in `node_config_ids`).

5. **Subgraphs**: Layout files may contain multiple subgraphs, indicated by `node_splits`.

6. **File Naming**: Files are named after models (alexnet, bert_classifier, resnet, etc.) with hash suffixes.

---

## File Statistics Summary

| Collection | Typical Nodes | Typical Edges | Typical Configs |
|------------|---------------|---------------|-----------------|
| tile:xla | 10-50 | 10-50 | 200-5000 |
| layout:xla:* | 1000-40000 | 1000-70000 | 10000-100000 |
| layout:nlp:* | 1000-10000 | 1000-50000 | 30000-100000 |

---

## Model Examples in Dataset

### Tile (XLA)
- `alexnet_train_batch_32_*.npz`
- `bert_classifier.2x2.fp32_*.npz`
- `bert_pretraining.2x2.fp16_*.npz`
- `resnet50.*_*.npz`

### Layout (XLA & NLP)
- `alexnet_train_batch_32.npz`
- `bert_classifier.2x2.fp32.npz`
- `albert_en_base_batch_size_16_*.npz`
- `dlrm_*.npz`
