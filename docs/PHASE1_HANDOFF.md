# Phase 1 — Data Pipeline & EDA: Implementation Handoff (Planner → Orchestrator)

> **Role split.** This brief is written by the **Phase-1 planner/EDA analyst**. You (the
> orchestrator) own **implementation + execution monitoring**. Build exactly the components
> below, prove **gate P1**, then surface results back to the planner for EDA review. Do **not**
> change scope (no model/training code — that's Phase 2+). Full reasoning lives in
> `WORKFLOW.pdf` §"Phase 1" and the approved plan at
> `~/.claude/plans/you-are-an-experienced-snoopy-quasar.md`.

## 0. What you're building (one sentence)

Importable, tested data-pipeline code in `src/data/` that turns raw TPUGraphs NPZ files into
correctly-batched PyG graphs for **all 5 collections at batch > 1**, plus a reproducible
inventory, train-only normalisation stats, a cheap rebuildable cache, and an EDA **notebook** —
satisfying **quality gate P1**.

## 1. Stack & environment (decided)

- **PyTorch + PyTorch Geometric** (finalised in Phase 0). Do not introduce TensorFlow.
- **Primary runtime = Kaggle / Colab notebooks** (GPU). Code must be **path-portable**:
  resolve `data_root` from config/env, supporting both the local Windows checkout
  (`data/npz/tpugraphs/...`) and the Kaggle read-only mount
  (`/kaggle/input/predict-ai-model-runtime/npz/tpugraphs/...`). Write all outputs/cache to a
  **writable, ephemeral** dir (`/kaggle/working` on Kaggle, `artifacts/`/`data/cache` locally).
  Do not assume a fixed `num_workers`; make it configurable (default 2) and ensure the dataset
  is picklable for spawn-based workers.

## 2. Hard guardrails (violating these silently corrupts results)

1. **Never alter `config_runtime` ordering** — no clipping/normalising/"cleaning" of labels.
   The archived `archive/root-scripts/data_cleaning.py` clips runtimes and **must not be reused**.
2. **Test files have placeholder labels** (`config_runtime` all-zeros; tile-test normalizers
   all-ones). Never feed test "runtimes" to any loss/metric; flag them in the inventory. Real
   labels exist on **train and valid**.
3. **Do not materialise the full `node_config_feat`** (`[n_configs, nc, 18]`) — one file blows
   up to **>1 GB RAM**. Always slice only the sampled config rows (sample-on-read).
4. **Fit normalisation on the train split only** (no valid/test leakage).
5. **Edge index is `[n_edges, 2]` raw** → transpose to `[2, m]` before use; assert in-bounds.

## 3. Verified dataset facts (use these; they correct `docs/DATA_STRUCTURE.md`)

| Collection | train / valid / test files | max file (MB, compressed) |
|---|---|---|
| `tile:xla` | 5709 / 676 / 844 | 0.3 |
| `layout:xla:random` | 69 / 7 / 8 | 24 |
| `layout:xla:default` | 61 / 7 / 8 | 25 |
| `layout:nlp:random` | 207 / 20 / 17 | 24 |
| `layout:nlp:default` | 198 / 20 / 17 | 24 |

- Schema (all graphs): `node_feat [n,140] f32`, `node_opcode [n] uint8`, `edge_index [m,2] i64`,
  `config_runtime [c] i64` (µs, **ordering signal only**).
  Tile-only: `config_feat [c,24] f32`, `config_runtime_normalizers [c] i64`.
  Layout-only: `node_config_ids [nc] i64`, `node_config_feat [c,nc,18] f32`, `node_splits [1,s] i64`.
- `node_config_feat` values ∈ **{−1,0,1,2,3}** → store sampled slices as **int8** (4× smaller);
  `−1` ≈ padding/not-applicable.
- `node_feat` has **negatives** (min −2 on xla) and reaches ~2.6e7 → **cannot `log1p` the whole
  matrix**; derive a per-column policy from data.
- **Op-code max ≈ 118** (not 255). Compute global max in the inventory; embedding size = max+1.
- **No NaN/Inf** anywhere → no cleaning needed.
- Config counts: tile 8–~6.9k; layout up to 100040 (capped). Configurable nodes (`nc`): 20–3143.
- No `sample_submission.csv` on disk — record each **test** file's `stem` as its graph id so the
  Phase-2 submission assembler has the id universe.

## 4. Reuse (read, don't run blindly)

- `archive/ref-toolkit/pytorch_loader.py` — clean `GraphSample`/`_npz_to_sample`; **transposes
  edge_index correctly**. Good skeleton, but it loads the full `node_config_feat` → replace with
  sample-on-read.
- `archive/root-scripts/enhanced_loader.py` — reuse collection/split path parsing; **discard its
  batching** (no custom collate → breaks at batch>1) and its `[n_edges,2]` edge convention.
- `docs/DATA_STRUCTURE.md` `parse_collection` / `get_split` helpers — correct, reuse.
- `archive/ref-toolkit/inventory_generation.py` — docstring **spec only** (no code).

## 5. Tasks (build in this order)

**T1 — Paths + inventory** → `src/data/paths.py`, `src/data/inventory.py`, `scripts/make_inventory.py`
- Resolve `data_root` (local + Kaggle). Walk 5 collections × 3 splits. Per file record:
  `collection, split, file_path, stem, bytes, n_nodes, n_edges, n_configs, n_config_nodes,
  n_subgraphs, opcode_max, schema_hash(sorted keys+dtypes), has_nan, has_inf, edge_oob,
  runtime_is_placeholder`. Read shapes/maxes **lazily** (don't load `node_config_feat`). Write
  `artifacts/inventory.parquet`. Skip label checks on test.

**T2 — Graph construction + pinned conventions** → `src/data/graph.py`
- Build PyG `Data`: `edge_index → [2,m] i64`; `node_feat f32`; `node_opcode long`;
  `config_runtime` int64 (ordering only). Pin **directionality** in a module docstring as the
  single source of truth: default to the `feed` orientation used for message passing, with
  configurable `add_reverse_edges` and `add_self_loops` flags (the official starter notes the
  *transpose* of the feed adjacency carries information flow).

**T3 — Lazy config sampling** → `src/data/configs.py`
- `sample_configs(npz_handle, k, rng)`: choose `k` config indices, slice only those rows of
  `config_runtime` and `config_feat`/`node_config_feat` (return layout config feats as **int8**).
  Provide a chunked **keep-all** path for inference. Never build the full `[c,nc,18]` tensor.

**T4 — Variable-size batching collator** → `src/data/collate.py`
- Concatenate variable-size graphs into one disconnected PyG `Batch` with a `batch` vector;
  offset `edge_index`; stack per-graph sampled config lists + labels for a ranking list.
  Layout: scatter sampled `node_config_feat` onto `node_config_ids`, mask non-configurable
  nodes. **Must work at batch > 1 for every collection.** (No GST/segment-dropout here — Phase 3.)

**T5 — Cheap rebuildable cache** → `src/data/cache.py`
- Cache compact arrays only (`edge_index, node_feat, node_opcode, node_config_ids, node_splits`),
  keyed by file hash, to the writable dir; tile may cache fully. **Not** the giant config tensor.
  Must be rebuildable cheaply and measurably faster than cold reads.

**T6 — Normalisation stats** → `src/data/normalize.py`, `scripts/fit_norm.py`
- Train-split-only per-column `node_feat` mean/std + a **log1p column mask** (heavy-tailed
  non-negative columns). Save `artifacts/norm_stats.json`; apply at load → finite features.

**T7 — Grouped-by-graph CV scaffold** → `src/data/splits.py`
- Group-by-graph fold assignment (no config leakage). Helper only; training stays in Phase 2.

**T8 — Tests** → `tests/test_data.py`
- Per-collection loader shapes; `edge_index` bounds `0 ≤ idx < n_nodes`; finite features after
  normalisation; **batch > 1 works for all 5 collections**; sampling returns exactly `k`; cache
  round-trip equals cold read.

**T9 — EDA notebook** → `notebooks/01_eda.ipynb`
- Import the above and render: per-collection counts; distributions of `n_nodes/n_edges/
  n_configs/n_config_nodes`; **per-graph** runtime ranges (log scale; note cross-graph scales are
  incomparable); opcode histogram + vocab size; `node_feat` per-column min/max flagging negative
  + heavy-tailed columns; visual confirmation that test labels are placeholders. Save figures to
  `artifacts/figures/`. Hand back to the planner for review.

## 6. Gate P1 — Definition of done (prove all)

1. `python scripts/make_inventory.py` → `artifacts/inventory.parquet`, all 5 collections; counts
   match §3; zero `edge_oob`; test rows flagged `runtime_is_placeholder`.
2. `python scripts/fit_norm.py` → `norm_stats.json`; all `node_feat` finite after applying.
3. `pytest -q tests/test_data.py` green (esp. **batch>1 every collection** + edge bounds).
4. On a 5-file smoke subset and a small layout subset: cached epoch markedly faster than cold;
   peak RAM bounded (no >1 GB config-tensor materialisation).
5. `notebooks/01_eda.ipynb` runs end-to-end; figures render.

**Smoke-test every new component on a 5-file subset (seconds) before touching the full data.**
When done, report the gate-P1 evidence + the EDA figures to the planner.
