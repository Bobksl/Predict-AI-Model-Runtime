# PROGRESS — Google "Fast or Slow? Predict AI Model Runtime" (TPUGraphs)

*Last updated: 2026-07-17. Companion file: [SETUP.MD](SETUP.MD) (what you need to
provide). Full plan: [WORKFLOW.pdf](WORKFLOW.pdf) · theory: [STUDY_GUIDE.pdf](STUDY_GUIDE.pdf).*

## 1. The project in one paragraph

The competition is **graph learning-to-rank**: each example is a TPU/XLA computation
graph with many candidate compiler configurations, and we must rank configurations
fastest→slowest per graph (5 collections: `tile:xla` + 4 `layout:*`; layout dominates
the score). Every competitive solution is a **GNN ranker** — we pivoted the project to
PyTorch + PyTorch Geometric, listwise/pairwise ranking losses, and OPA/Kendall
evaluation. The contest is closed, so submissions are scored **late** (learning-grade,
no prizes). Work is organised as quality-gated phases (P0–P6) executed by the
orchestrator (main session) with a 5-agent planning team (`.claude/agents/`,
contract in `docs/TEAM.md`).

## 2. Phase status board

| Phase | Scope | Gate | Status |
|---|---|---|---|
| **P0** Foundation | Clean repo, stack decision, data audit | Fresh install runs; 5 collections on disk | ✅ **DONE** |
| **P1** Data pipeline & EDA | `src/data/` loaders, inventory, norm stats, cache, EDA | Batch>1 all collections; finite feats; tests green | ✅ **DONE + signed off** (APPROVE-WITH-CHANGES; all 4 changes applied) |
| **P2** Tile baseline | CV, OPA, hinge/ListMLE, small GNN, first submission | GNN valid-OPA ≥ 0.70 and ≥ trivial+0.10; valid tile submission by one command | ✅ **DONE + signed off** (APPROVE-WITH-CHANGES; required change applied). ListMLE **0.8780** valid OPA (full-config apples-to-apples **0.8794** vs trivial 0.7090); hinge 0.8757; 5-fold CV 0.8636±0.0060, fold-0 rerun exact; one-command 844-row submission validated |
| **P3** Layout at scale | Config shards, Graph Segment Training, 4 layout models, full 5-collection submission | No OOM; beats trivial per collection; LB score | 🔶 **STARTED** — brief being drafted by tpu-phase3-layout; int8 config shards building (builder verified vs streaming reader) |
| **P4** Quality push | Top-solution research, features, SAGE/GAT/GPS ladder | Meaningful CV gain on layout | ⬜ blocked by P3 |
| **P5** Tuning & ensembling | HP search, rank-averaging ensembles, CV↔LB hardening | Ensemble > best single on CV **and** LB | ⬜ blocked by P4 |
| **P6** Finalise | Reproducibility pass, report, figures | One-command repro; report done | ⬜ |

## 3. What has been delivered so far

**Workspace & docs (P0)**
- Folder redesigned: `src/ scripts/ tests/ configs/ artifacts/ docs/ notebooks/ archive/`;
  legacy toolkits preserved under `archive/` (reference only — the old cleaner *clips
  labels*, never reuse).
- `WORKFLOW.pdf` (quality-first phased plan) + `STUDY_GUIDE.pdf` (in-depth theory,
  undergrad-CS level) — sources in `docs/src/`, rebuildable via `docs/src/build_pdf.py`.

**Data pipeline (P1) — `src/data/`, 13 pytest green**
- Inventory of all **7868 NPZ files** (`artifacts/inventory.parquet`): counts verified,
  `edge_oob=0`, no NaN/Inf, opcode_max=118 (embedding 119), all 894 test files flagged
  **placeholder labels** (never fed to losses).
- **Memory-safe streaming config reader** — the layout `node_config_feat` tensor
  decompresses to up to **2.62 GB/file**; we stream only sampled rows as int8
  (54 MB peak vs 2.62 GB).
- Correct variable-size **PyG batching for all 5 collections at batch>1**, pinned
  edge conventions (`[n_edges,2]`→`[2,m]`, bounds asserted), layout config scatter.
- Train-only normalisation (`artifacts/norm_stats.json`, 25 log1p columns, fit
  documented in `fit_meta`), pickle compact-cache (×1.3–3.0), grouped-by-graph CV
  scaffold, EDA notebook `notebooks/01_eda.ipynb` (6 figures in `artifacts/figures/`).

**Agent team**
- 5 phase agents defined in `.claude/agents/tpu-phase{1..5}-*.md` + shared contract
  `docs/TEAM.md`. Role split: agents plan/review; orchestrator implements/executes.
- **Phase-1 review returned APPROVE-WITH-CHANGES** and caught a real bug (F1): our
  float32 label cast collapsed 1852/29010 distinct runtimes into ties. **Fixed** —
  labels now int64 end-to-end; sampling made epoch-aware (`set_epoch`); test file
  pinned; fit metadata + figure units corrected. All fixes verified by tests.
- **Config-shard decision APPROVED (amended):** capped-pool int8 memmap shards
  (P=2000, train+valid layout only, ≈6.6 GiB; full-axis 99 GiB variant rejected).
  Mandatory before Phase-3 layout training.
- **Phase-2 brief delivered & approved**: grouped CV + provided-valid early stopping,
  OPA with hand-worked unit tests, pairwise-hinge→ListMLE, ~45k-param SAGEConv tile
  ranker, trivial baselines, fastest-first `ID,TopConfigs` submission,
  YAML-config training loop.

## 4. Action plan from here (concrete next steps)

1. **P2 implement (orchestrator, now):** `src/training/{metrics,losses,cv,train_loop}.py`,
   `src/models/tile_gnn.py`, `src/inference/{predict,submission}.py`,
   `scripts/{train,predict,make_submission,eval_baselines}.py`, `configs/tile_*.yaml`,
   tests → CPU smoke run → **full GPU run on Kaggle/Colab** (see SETUP.MD) → gate-P2
   evidence → `tpu-phase2-baseline` reviews results.
2. **P2→P3 bridge:** build config shards (`scripts/make_config_shards.py`, ≈6.6 GiB,
   one-time <1 h) per the approved spec.
3. **P3:** spawn `tpu-phase3-layout` for its brief → implement Graph Segment Training +
   per-collection layout models → **first full 5-collection late submission** → LB score.
4. **P4:** spawn `tpu-phase4-quality` → reproduce top-solution ideas one at a time under
   identical CV (kill criteria mandatory).
5. **P5:** spawn `tpu-phase5-ensemble` → HP search + rank-averaged ensembles →
   CV↔LB correlation log.
6. **P6:** reproducibility pass + final report (course deliverable).

## 5. Known risks / facts to keep in mind

- Layout **xla valid splits are only 7 files** — per-collection valid OPA is noisy;
  rely on grouped CV over train (review flag #1).
- **nlp runtime ranges are narrow** (tie-dense) — loss/margin choice matters there.
- Layout epochs are streaming-bound until shards exist (~0.04–1.3 s/file).
- Kaggle GPU quota and late-submission cadence are the practical bottlenecks from P2 on.
- Local machine is **CPU-only torch** — full training must run on Kaggle/Colab.
