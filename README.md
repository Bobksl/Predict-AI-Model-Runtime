# Predict AI Model Runtime — TPUGraphs (GNN learning-to-rank)

A PyTorch + PyTorch Geometric solution to the Kaggle competition
[**Google – Fast or Slow? Predict AI Model Runtime**](https://www.kaggle.com/competitions/predict-ai-model-runtime/).

**The task.** Each example is a TPU/XLA neural-network **computation graph** carrying many
candidate **compiler configurations**. For every graph we must **rank its configurations
from fastest to slowest** — a graph learning-to-rank problem, not runtime regression. The
five collections are `tile:xla` and `layout:{xla,nlp}:{random,default}`; layout dominates the
overall score.

**The approach.** A small **Graph Neural Network** ranker (SAGEConv message passing over the
op graph, op-code embeddings, per-configuration scoring) trained with **pairwise-hinge /
ListMLE** ranking losses and evaluated with **Ordered-Pair Accuracy (OPA)**. Layout graphs
reach tens of thousands of nodes, so training uses **Graph Segment Training** and a
memory-safe streaming/sharded config reader.

## Repository layout

```
src/
  data/        NPZ loaders, inventory, streaming config reader, PyG batching,
               train-only normalisation, cache, config shards, grouped CV
  models/      TileRanker + LayoutRanker (SAGEConv GNN rankers)
  training/    OPA metric, pairwise-hinge & ListMLE losses, GST, train loop
  inference/   chunked all-config scoring + submission assembly
configs/       one YAML per experiment (tile_*, layout_*)
scripts/       CLIs: make_inventory, make_config_shards, fit_norm, train,
               train_layout, predict, make_submission, combine_submission, eval_*
notebooks/     01_eda.ipynb, kaggle_submission.ipynb (produces submission.csv)
kaggle/        GPU training kernel (train_all_layout_kernel.py)
tests/         pytest suite (data pipeline, metrics, losses, model, submission)
artifacts/     norm_stats.json + generated figures/checkpoints/submissions
data/          downloaded NPZ (not committed; see below)
```

## Quick start

```bash
pip install -r requirements.txt
# 1. download the competition data (Kaggle CLI; needs competition rules accepted)
kaggle competitions download -c predict-ai-model-runtime
# 2. build inventory + train-only normalisation stats
python scripts/make_inventory.py
python scripts/fit_norm.py
# 3. train the tile baseline (CPU-feasible; tile graphs are small)
python scripts/train.py --config configs/tile_sage_listmle.yaml
# 4. layout models need a GPU — see kaggle/ and notebooks/kaggle_submission.ipynb
pytest -q tests/          # 33 tests
```

## Results so far

- **`tile:xla`** — GNN ranker validation **OPA ≈ 0.878** (ListMLE) vs 0.709 best trivial
  baseline; 5-fold grouped CV 0.864 ± 0.006, exactly reproducible.
- **`layout:*`** — models trained on GPU via `kaggle/train_all_layout_kernel.py` (Graph
  Segment Training); combined 5-collection submission built by
  `notebooks/kaggle_submission.ipynb`.

## Submitting on Kaggle

With all five checkpoints under `artifacts/checkpoints/`, build the combined CSV and submit
it with the Kaggle CLI (the submission is a **CSV**, not a model file):

```bash
python scripts/make_submission.py --checkpoint artifacts/checkpoints/tile_sage_listmle/best.pt \
    --out artifacts/submissions/submission_tile.csv
for c in xla_random xla_default nlp_random nlp_default; do
  python scripts/make_layout_submission.py \
    --checkpoint artifacts/checkpoints/layout_${c}_sage/best.pt \
    --out artifacts/submissions/submission_layout_${c}.csv
done
python scripts/combine_submission.py \
    --inputs artifacts/submissions/submission_tile.csv artifacts/submissions/submission_layout_*.csv \
    --out artifacts/submissions/submission_5col.csv
kaggle competitions submit -c predict-ai-model-runtime \
    -f artifacts/submissions/submission_5col.csv -m "GNN 5-collection late submission"
```

## Data note

Data is not committed. All five collections come from the competition download; loaders
resolve the data root automatically for a local checkout or the Kaggle mount
(`src/data/paths.py`). Runtimes are an **ordering signal only** — never rescaled or clipped.
