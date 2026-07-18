# BRIEF — Phase 3: Layout collections at scale (GST) + full 5-collection submission

*Author: tpu-phase3-layout (Opus). Orchestrator-approved 2026-07-18 with one
correction: there is NO `sample_submission.csv` on disk — all submission
validation uses the inventory-derived ID universe (test stems + per-file
n_configs from `artifacts/inventory.parquet`).*

**(0) Goal.** Train four per-collection layout rankers under Graph Segment Training
(no OOM on 43k-node graphs), beat trivial baselines per collection on grouped CV, and
assemble one validated 5-collection `submission.csv` that scores on the LB with an
understood CV→LB gap.

**(1) Stack/env.** PyTorch + PyG, reusing the Phase-1/2 pipeline unchanged. CPU for
smoke + small runs; Kaggle/Colab GPU (16 GB) for full layout runs. Config shards
(P=2000 int8 memmap, train+valid only) exist per the approved spec — this brief
specifies the *consumer* side.

**(2) Hard guardrails (inherit TEAM.md §Non-negotiable).**
- Never materialise full `node_config_feat [c,nc,18]`; only sampled/streamed
  `[k,nc,18]` int8, cast to float at point of use.
- Every new component states its peak-memory budget; the `[N,k,H]` per-config node
  tensor is the driver — bounded at train by GST (`MAX_KEEP_NODES`) and `k`, at
  inference by config-chunk size.
- Labels int64; float only after differencing/argsort. Test labels placeholder.
- Eval/inference = full graph, no GST, `no_grad`.

**(3) Verified facts.**
- Worst layout graphs: xla max N=43,615 / nc=3,143 / edges=73,881; worst train file
  `efficientnet_b7_eval_batch_1` (xla:random & xla:default, N=43,615, nc=1,158);
  densest config file `inference_mlperf_ssd_1200_batch_1` (xla, nc=3,143, N=25,544).
  nlp max N=21,919 / nc=848; worst `talking-heads_large_batch_size_32_train`.
- Train/valid `n_configs` reach the 100,040 cap; **test files have only ~1,000–1,001
  configs** → inference is cheap; training is the memory risk.
- nlp: 2× smaller graphs, tie-dense (narrow runtime ranges) → margin/loss/k matter.
- Starter GST (cell 37): full-graph pass on stop-gradient inputs, second pass over
  `sampled_` edges with grad, merge `x = where(is_selected, x_backprop, x_full)`;
  contiguous kept nodes; MAX_KEEP_NODES=1000; edge survives iff both endpoints kept.
  Combine (cells 44/45) = concatenation of per-collection CSVs (strip repeat headers).

**(4) Reuse (never clone).** `graph._to_edge_index` + `TPUData.__inc__/__cat_dim__`;
`collate.scatter_node_config_feat` (CPU-smoke fallback only);
`configs.read_node_config_feat_rows`, `sample_config_indices`; `dataset.set_epoch`;
`src/training/{losses,metrics,cv,train_loop}`; `src/inference/{predict,submission}`;
`scripts/make_submission.py`.

## (5) Design + ordered tasks (each with CPU smoke)

### Task A — `src/models/layout_gnn.py` (`layout_sage` in registry)
- Base encode once (config-independent): `h0 = proj(concat(node_feat, op_emb(op)))`
  → `[N,H]`.
- Config injection WITHOUT dense `[N,k,18]` on GPU: encode compact
  `node_config_feat [Σnc,k,18]` int8→float via `config_proj: 18→Hc`, then
  `index_add` the encoded `[Σnc,k,Hc]` onto zeros `[ΣN,k,Hc]` at `cfg_node_index`
  (scatters only nc rows). Dense `scatter_node_config_feat` = CPU-smoke fallback.
- Per-config message passing: `x = h0[:,None,:] + scattered_cfg` → `[N,k,H]`; run L
  SAGE layers sharing weights across k (reshape `[N·k,H]` with k-replicated,
  node-offset `edge_index`). Pool over configurable nodes + global mean → head →
  `[B,k]` scores.
- Memory (worst xla N=43,615, k=8, H=64): layer tensor 89 MB; full stop-grad pass
  keeps ~2–3 live tensors ≈0.3 GB; B=4 ≈1.2 GB — fine on 16 GB. CPU smoke: N≈3k,
  k=4, H=32.
- Smoke: forward on 2-graph batch of 5-file xla:random subset; `[B,k]` finite;
  scores change when config feats are permuted.

### Task B — `src/training/gst.py` + `layout_gnn.forward(batch, gst=True)`
- Segment: contiguous window of `MAX_KEEP_NODES` node indices per graph (node order
  ~topological); window start per (seed, epoch, graph); `is_selected [ΣN]`;
  `sampled_edge_index` = edges with both endpoints selected.
- Two passes: `x_full = encode(...).detach()` (full edges, no tape);
  `x_seg = encode(...)` (sampled edges, grad); merge with `torch.where`; pool/score.
- Sizing: backward acts ≈ MAX_KEEP·k·H·4·~6·B → MAX_KEEP=1000, k=8, H=64, B=4 →
  ~0.4 GB. Default MAX_KEEP_NODES=1000 (xla may take ~4,000); CPU smoke 512.
- Correctness asserts: loss finite; grads only via kept segment; moving the window
  changes grads while detached pass is grad-free; `gst=False` reproduces full-graph
  scores.

### Task C — `src/data/config_shards.py` + shard branch in `dataset.__getitem__`
- On shard hit (CompactCache signature + pool match): epoch-aware sample of pool
  positions in `[0,P')`; `feats = memmap[pool_pos] [k,nc,18]`;
  `orig_ids = shard_idx[pool_pos]`;
  `runtimes = bundle["config_runtime"][orig_ids]` (labels NEVER from shards).
  Signature mismatch → fall back to streaming. Test split: no shards, stream.

### Task D — `scripts/train_layout.py` + `configs/layout_{xla,nlp}_{random,default}.yaml`
- GST training loop extending `train_loop.py` conventions.
- Budgets: CPU smoke 5 files/2–3 epochs/MAX_KEEP 512/k=4 (minutes). GPU full:
  ~2–4 GPU-h/collection, max_epochs 50–100, patience 8–10.
- Model selection: PRIMARY = 5-fold grouped CV over train (xla valid is 7 files —
  never alone); deployable checkpoint = train on full train, early-stop on provided
  valid but cap max_epochs at CV-observed best epoch.
- k / loss per collection: xla k=8–16, try ListMLE + hinge; nlp k=16–32, hinge
  (margin 0.1) first. Pick per-collection winner by CV; keep both for P5 ensembles.
- Per-collection SEPARATE models by default; sharing only if CV shows a pooled
  xla (or nlp) model matches/beats separate.

### Task E — `src/inference/predict_layout.py`
- No GST, no_grad; encode `h0` once; stream ALL configs chunked via
  `read_node_config_feat_rows(path, chunk_ids)`; scatter, forward, score.
- Chunk sizing: xla N=43,615 → chunk 32–64 (≤~2 GB); nlp → chunk 128. Test files
  ~1,000 configs → few chunks. `rank_configs` ascending = fastest-first.

### Task F — `scripts/make_layout_submission.py` + `scripts/combine_submission.py`
- Per-collection assembly (`layout:xla:random:<stem>` etc.) via existing
  `assemble_submission`; combine = ONE header + data rows of 4 layout CSVs + tile
  CSV (strip repeated headers); validate combined vs inventory ID universe
  (every row a full permutation of that file's range(n_configs)).

### Task G — `tests/test_layout.py`
- All smoke checks above + batch>1 collation regression for layout subsets.

## (6) Gate P3 — definition of done
1. **No OOM via GST:** smoke then GPU epoch including `efficientnet_b7_eval_batch_1`
   (xla, N=43,615), `inference_mlperf_ssd_1200_batch_1` (nc=3,143), and
   `talking-heads_large_batch_size_32_train` (nlp) at MAX_KEEP_NODES=1000; GST
   correctness asserts pass.
2. **Beats trivial baselines per collection** on 5-fold grouped CV: (a) random
   (~0.5); (b) rank by summed `node_config_feat` over configurable nodes. Clear
   margin on all 4 layout collections.
3. **Full 5-collection submission:** 4 × `make_layout_submission.py --checkpoint`
   + tile + `combine_submission.py --out artifacts/submissions/submission_5col.csv`;
   validator passes (header once, valid permutations, IDs match inventory universe).
4. **CV→LB:** log per-collection CV-OPA (mean±std) + LB score in the artifacts
   leaderboard log; submit once, record the gap; confirm CV ordering of loss/k
   choices is directionally consistent with LB before trusting P4 tuning.

**Create:** `src/models/layout_gnn.py`, `src/training/gst.py`,
`src/data/config_shards.py`, `src/inference/predict_layout.py`,
`scripts/train_layout.py`, `scripts/make_layout_submission.py`,
`scripts/combine_submission.py`, `configs/layout_*.yaml`, `tests/test_layout.py`.
**Edit:** `src/models/__init__.py`, `src/data/dataset.py` (shard branch).
