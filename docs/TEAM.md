# Agent Team Charter — TPUGraphs (Fast or Slow)

Read this FIRST if you are a phase agent. It is the shared contract for the 5-agent
research team supporting the solo Kaggle effort on
`predict-ai-model-runtime` (graph learning-to-rank, GNN pivot).

## Roles

| Agent | Owns (planning/analysis for) |
|---|---|
| `tpu-phase1-data` | Data pipeline & EDA (workflow Phase 1) |
| `tpu-phase2-baseline` | Tile ranking baseline: CV, OPA, losses, first submission (Phase 2) |
| `tpu-phase3-layout` | Layout collections at scale: GST/segment training, full submission (Phase 3) |
| `tpu-phase4-quality` | Solution-quality push: top solutions, features, architectures (Phase 4) |
| `tpu-phase5-ensemble` | Tuning, ensembling, CV↔LB validation hardening (Phase 5) |

**The orchestrator (main session) reviews, approves, implements, and executes.**
Phase agents **plan, analyse, verify, and review** — they produce briefs, reviews, and
decision recommendations. Phase agents do **not** write/edit repo files, run training,
or make submissions. Bash is for **read-only inspection only** (probing NPZ files,
checking artifacts, timing reads). If a plan requires code, specify it; the orchestrator
writes it.

## Model routing (subagent dispatch)

Adopted from the WorldQuant alpha-research pattern; applies to every `Agent` tool
call the orchestrator makes for `tpu-phase*` agents (and any other subagent spun up
for this project).

- **Orchestrator model.** The host session — planning, decomposition, cross-agent
  integration, judging gate verdicts. Uses whatever model the user has set for the
  session (no override); default to the most capable available if unspecified.
- **Opus** — reasoning-heavy dispatches: gate-review verdicts (e.g. the Phase-1
  float32-label-collapse finding), phase-brief drafting where real architecture/
  algorithm decisions are made (model design, loss choice, CV protocol, shard specs),
  failure diagnosis, quantitative/data-interpretation work, risk analysis.
- **Sonnet** — mechanical dispatches: implementing an already-decided spec, batch
  file/dataset probes, straightforward code-spec generation, simple refactors,
  monitoring/polling a running job, formatting.
- **Haiku** — low-stakes dispatches: WebSearch/WebFetch retrieval sweeps (e.g. Phase-4
  "list top Kaggle solutions" before synthesis), summarisation, simple lookups, or as
  a fallback when Claude usage limits are tight.
- **Efficiency principle.** Never spend Opus on what Sonnet can handle. When genuinely
  unsure between Opus and Sonnet, dispatch Sonnet first and escalate only if the
  result is inadequate. In practice: gate reviews and brief-drafting for `tpu-phase*`
  agents default to **Opus** (they are architecture-decision / data-interpretation
  tasks by nature); narrow follow-ups on an already-approved brief default to
  **Sonnet**; raw research gathering defaults to **Haiku**.
- **Transparency.** If asked, state which model is running the current turn and give
  a one-line reason for the routing choice made for any subagent dispatch.

## Deliverable format (briefs)

A phase brief must contain: **(0)** one-sentence goal; **(1)** environment/stack
constraints; **(2)** hard guardrails; **(3)** verified facts with how they were verified;
**(4)** reuse pointers into the existing repo; **(5)** ordered tasks with target file
paths; **(6)** the phase quality gate as a checkable definition-of-done. A review must
end with an explicit verdict: **APPROVE**, **APPROVE-WITH-CHANGES** (listed), or
**REVISE** (blocking issues listed).

## Non-negotiable dataset guardrails

1. **Never alter `config_runtime` ordering** — no clipping/normalising labels.
   (`archive/root-scripts/data_cleaning.py` clips runtimes; reference only, never reuse.)
2. **Test labels are placeholders** (runtime all-zero; tile test normalizers all-one).
   Never feed them to a loss/metric. Real labels: train + valid only.
3. **Never materialise the full layout `node_config_feat` `[c, nc, 18]`** (>1 GB, up to
   2.6 GB). Sample-on-read via `src/data/configs.py` streaming; sampled slices as int8.
4. **Normalisation stats fit on the train split only.**
5. **`edge_index` is `[n_edges, 2]` raw** → transpose to `[2, m]`; assert in-bounds.
6. CV is **grouped by graph** — configs of one graph never straddle folds.

## Ground truth & key paths

- Workflow (phases + gates): `docs/src/workflow.md` (rendered: `WORKFLOW.pdf`)
- Theory companion: `docs/src/study_guide.md` (rendered: `STUDY_GUIDE.pdf`)
- Dataset schema: `docs/DATA_STRUCTURE.md`; inventory: `artifacts/inventory.parquet`
- Pipeline code (Phase 1, DONE): `src/data/` + `tests/test_data.py` (12 green)
- Norm stats: `artifacts/norm_stats.json`; EDA: `notebooks/01_eda.ipynb`,
  figures in `artifacts/figures/`
- Data root: `data/npz/tpugraphs/` — all 5 collections present; **read-only**
- Stack: **PyTorch + PyG** (no TensorFlow); primary runtime Kaggle/Colab GPU;
  code must stay path-portable (`src/data/paths.py` resolves roots)

## Verified facts you may rely on (re-verify only if suspicious)

- Counts (train/valid/test): tile 5709/676/844; layout xla:random 69/7/8,
  xla:default 61/7/8, nlp:random 207/20/17, nlp:default 198/20/17. Total 7868.
- Global `opcode_max` = 118 → embedding size 119. No NaN/Inf anywhere. `edge_oob` = 0.
- `node_config_feat` values ∈ {−1,0,1,2,3} (int8-lossless). `node_feat` has negative
  columns; 25 columns are log1p-flagged in `norm_stats.json` (global fit, 100
  train files/collection — see its `fit_meta`).
- **Labels are int64 end-to-end** (gate-P1 review finding F1): runtimes exceed 2^24,
  so a float32 cast collapses distinct values into spurious ties. `batch.y` is int64;
  losses/metrics cast to float only after differencing/rank extraction.
- Config sampling is **epoch-aware**: call `TpugraphsDataset.set_epoch(e)` each epoch
  (same epoch ⇒ reproducible, different epoch ⇒ new k-subset).
- Known perf fact: layout `__getitem__` is dominated by streaming sampled config rows
  (~0.04–1.3 s/file); compact-array cache gives ×1.3–3.0 on that path only.
- **Config shard: APPROVED by tpu-phase1-data (gate-P1 review)** — capped-pool int8
  memmap shards, default P=2000 pooled configs/file, layout **train+valid only**
  (test streams sequentially at inference), bare `.npy` pairs (`.pool2000.int8.npy`
  `[P',nc,18]` + `.idx.npy` sorted original ids) under `data/cache/config_shards/…`,
  keyed by the CompactCache signature; loader falls back to streaming on mismatch;
  labels never duplicated into shards. Disk ≈ 6.6 GiB. Full-axis variant rejected
  (≈99 GiB). **Mandatory before the first layout training run (Phase 3).**

## Competition scoring (what we optimise)

- Tile: top-K slowdown (only the top of the ranking matters).
- Layout: Kendall-tau-style full-ordering score; layout dominates the overall score.
- Offline metric: OPA per collection; train losses: ListMLE / pairwise hinge.
- Contest is closed → late submissions score on the LB but win nothing; goal is the
  best defensible late-submission score.
