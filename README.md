# Predict AI Model Runtime — TPUGraphs (solo GNN restart)

A clean restart of the Kaggle competition
[**Google – Fast or Slow? Predict AI Model Runtime**](https://www.kaggle.com/competitions/predict-ai-model-runtime/).
The task is **graph learning-to-rank**: for each neural-network computation graph, rank its
candidate compiler configurations from fastest to slowest. The competitive approach is a
**Graph Neural Network (GNN)** — this repo is organised around building one.

## Start here

| Document | What it is |
|---|---|
| **[`WORKFLOW.pdf`](WORKFLOW.pdf)** | The quality-first, phase-gated project plan (P0→P6). Read first. |
| **[`STUDY_GUIDE.pdf`](STUDY_GUIDE.pdf)** | In-depth theory companion (ranking, graphs, GNNs, scaling, dataset). |
| [`docs/DATA_STRUCTURE.md`](docs/DATA_STRUCTURE.md) | Exact NPZ schema, verified by inspection. |
| [`docs/USER_GUIDELINE.md`](docs/USER_GUIDELINE.md) | Original dataset-handling notes (reference). |

## Repository layout

```
data/        downloaded NPZ (never edited by code; partial — see note below)
docs/        dataset references + docs/src/ (markdown sources + build_pdf.py)
notebooks/   official TF-GNN starter + future EDA notebooks
src/         GNN code — data/ features/ models/ training/ inference/ utils/  (scaffold)
configs/     one YAML per experiment              scripts/   CLI entrypoints
artifacts/   figures/ checkpoints/ submissions/   tests/     pytest
archive/     original team scripts + the older "ref" toolkit (reference only)
```

## Agent team

Research is supported by a 5-agent planning team (definitions in `.claude/agents/`,
shared contract in [`docs/TEAM.md`](docs/TEAM.md)): `tpu-phase1-data`,
`tpu-phase2-baseline`, `tpu-phase3-layout`, `tpu-phase4-quality`,
`tpu-phase5-ensemble` — one per workflow phase. Agents plan, analyse, and review;
the **orchestrator (main session) approves and implements**. Agents never write repo
files, train, or submit.

## Status

- **Done:** competition understood; folder cleaned and restructured; `WORKFLOW.pdf` and
  `STUDY_GUIDE.pdf` produced.
- **Decisions:** pivot to a **GNN** ranker; framework leaning **PyTorch + PyTorch Geometric**
  (finalised in workflow Phase 0); tackle **`tile:xla` first, then layout**.
- **Data:** all five collections present on disk (~7.2k tile + ~660 layout graphs).
- **Next:** Phase 1 — build the NPZ inventory and correct, cached, variable-size graph
  loaders in `src/data/` (this is the data-loading stage).

## Data note

**All five collections are present** under `data/` — `tile/xla`,
`layout/nlp/{random,default}`, and `layout/xla/{random,default}`. Phase 0 of the workflow
runs an integrity audit (file counts, schema hashes, finite-value checks). If you ever need a
fresh copy:

```
kaggle competitions download -c predict-ai-model-runtime
```

## Rebuilding the PDFs

The two PDFs are generated from Markdown in `docs/src/` (math via MathJax, rendered through
headless Chrome):

```
python docs/src/build_pdf.py docs/src/workflow.md    WORKFLOW.pdf    "Project Workflow — Quality-First GNN Restart"
python docs/src/build_pdf.py docs/src/study_guide.md STUDY_GUIDE.pdf "Competition Study Guide"
```
