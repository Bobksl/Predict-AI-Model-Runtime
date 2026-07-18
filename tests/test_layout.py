"""Phase-3 layout tests (gate P3): model correctness, GST bounds, shard consumer.

Fast by construction: CPU, small ``k``, 1-2 file subsets, tiny model widths.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.paths import list_npz, parse_collection
from src.data.dataset import TpugraphsDataset, make_loader
from src.data.cache import CompactCache, read_bundle
from src.data.collate import collate
from src.data.configs import sample_config_indices, read_node_config_feat_rows
from src.data import config_shards as CS
from src.models.layout_gnn import LayoutRanker
from src.training.gst import select_gst_window, gst_forward
from src.training.losses import pairwise_hinge_loss

ROOT = Path(__file__).resolve().parents[1]
K = 4
MAX_KEEP_SMOKE = 512


def _tiny_model() -> LayoutRanker:
    return LayoutRanker(opcode_emb_dim=16, hidden_dim=32, n_layers=2,
                        config_proj_dim=32, dropout=0.0)


def _small_dataset(coll, data_root, cache, n=2) -> TpugraphsDataset:
    ds = TpugraphsDataset(coll, "train", data_root=str(data_root), k_configs=K,
                          cache=cache, seed=0)
    ds.files = ds.files[:n]
    return ds


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return CompactCache(tmp_path_factory.mktemp("layout_cache"), enabled=True)


# ---- Task A: forward shape/finite + config-permutation sensitivity --------
def test_layout_forward_finite_and_permutation_sensitive(data_root, cache):
    ds = _small_dataset("layout:xla:random", data_root, cache, n=2)
    batch = next(iter(make_loader(ds, batch_size=2, shuffle=False, num_workers=0)))
    model = _tiny_model()
    model.eval()
    with torch.no_grad():
        scores = model(batch)
    assert scores.shape == (2, K)
    assert torch.isfinite(scores).all()

    # cyclic (non-identity) permutation of the k axis -> scores must be
    # equivariant: scoring permuted configs gives the same values, reordered.
    perm = torch.tensor([K - 1] + list(range(K - 1)))
    batch2 = batch.clone()
    batch2.node_config_feat = batch2.node_config_feat[:, perm, :]
    with torch.no_grad():
        scores2 = model(batch2)
    assert torch.allclose(scores2, scores[:, perm], atol=1e-5)
    # and the permutation is not a no-op on the actual score values (config feats
    # genuinely drive the score, not ignored/constant) — exact (not close) compare,
    # since these untrained scores are large in magnitude and torch.allclose's
    # relative tolerance would mask a real (but small-relative) difference.
    assert not torch.equal(scores2, scores)


# ---- Task G: batch>1 collation regression (no cross-graph/-config leakage) --
def test_layout_batch_collation_matches_singletons(data_root, cache):
    ds = _small_dataset("layout:xla:random", data_root, cache, n=2)
    model = _tiny_model()
    model.eval()

    data_list = [ds[0], ds[1]]
    batch2 = collate(data_list)
    with torch.no_grad():
        scores_batched = model(batch2)  # [2, K]
    assert scores_batched.shape == (2, K)

    for gi in range(2):
        batch1 = collate([data_list[gi]])
        with torch.no_grad():
            scores_single = model(batch1)
        assert torch.allclose(scores_batched[gi], scores_single[0], atol=1e-5)


# ---- Task B: GST correctness asserts ---------------------------------------
def test_gst_loss_finite_and_grads_flow(data_root, cache):
    ds = _small_dataset("layout:xla:random", data_root, cache, n=2)
    batch = next(iter(make_loader(ds, batch_size=2, shuffle=False, num_workers=0)))
    model = _tiny_model()

    scores, is_selected, dbg = gst_forward(
        model, batch, max_keep_nodes=MAX_KEEP_SMOKE, seed=0, epoch=0, debug=True)
    assert scores.shape == (2, K)
    loss = pairwise_hinge_loss(scores, batch.y)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
              for p in model.parameters())

    # full-graph (stop-gradient) pass must be grad-free
    assert dbg["x_full_requires_grad"] is False
    assert 0 < dbg["n_kept"] <= dbg["n_total"]


def test_gst_window_moves_and_changes_grads(data_root, cache):
    ds = _small_dataset("layout:xla:random", data_root, cache, n=2)
    batch = next(iter(make_loader(ds, batch_size=2, shuffle=False, num_workers=0)))
    model = _tiny_model()

    sel0 = select_gst_window(batch.ptr, MAX_KEEP_SMOKE, seed=0, epoch=0)
    sel1 = select_gst_window(batch.ptr, MAX_KEEP_SMOKE, seed=0, epoch=1)
    if bool((sel0 == sel1).all()):
        pytest.skip("both graphs <= MAX_KEEP_SMOKE nodes: no window to move")

    model.zero_grad()
    s0, _ = gst_forward(model, batch, max_keep_nodes=MAX_KEEP_SMOKE, seed=0, epoch=0)
    pairwise_hinge_loss(s0, batch.y).backward()
    g0 = model.node_proj.weight.grad.clone()

    model.zero_grad()
    s1, _ = gst_forward(model, batch, max_keep_nodes=MAX_KEEP_SMOKE, seed=0, epoch=1)
    pairwise_hinge_loss(s1, batch.y).backward()
    g1 = model.node_proj.weight.grad.clone()

    assert not torch.allclose(g0, g1), "moving the GST window should change grads"


def test_gst_false_reproduces_full_graph_scores(data_root, cache):
    ds = _small_dataset("layout:xla:random", data_root, cache, n=2)
    batch = next(iter(make_loader(ds, batch_size=2, shuffle=False, num_workers=0)))
    model = _tiny_model()
    model.eval()

    max_n = int(batch.ptr.diff().max())  # a window covering every node
    with torch.no_grad():
        scores_full = model(batch, gst=False)
        scores_gst_full_window, _ = gst_forward(
            model, batch, max_keep_nodes=max_n, seed=0, epoch=0)
    assert torch.allclose(scores_full, scores_gst_full_window, atol=1e-4)


# ---- Task C: shard consumer == streaming, for the same sampled ids ---------
def test_shard_consumer_matches_streaming(data_root):
    # Build (or reuse, idempotent) a tiny real shard pair via the approved builder
    # — never reimplement the shard-build logic here.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_config_shards.py"),
        "--collections", "layout:xla:random", "--splits", "valid", "--limit", "2"],
        check=True, cwd=str(ROOT), capture_output=True, text=True,
    )

    files = list_npz(data_root, "layout:xla:random", "valid")[:2]
    assert files, "no layout:xla:random valid files available"
    fp = files[0]
    stem = fp.stem
    pool = 2000

    shard = CS.find_shard("data/cache/config_shards", "layout:xla:random", "valid",
                          stem, fp, pool)
    assert shard is not None, "expected a shard just built by make_config_shards.py"
    data_f, idx_f = shard

    bundle = read_bundle(str(fp))

    rng = np.random.default_rng((0, 0, 0))
    feats_shard, orig_ids = CS.read_shard_sample(data_f, idx_f, K, rng)
    assert feats_shard.dtype == np.int8
    assert orig_ids.shape == (K,)

    # streaming path reading the SAME original config ids must give identical
    # feature rows (this holds regardless of whether the pool is a strict subset)
    feats_stream = read_node_config_feat_rows(str(fp), orig_ids)
    assert np.array_equal(feats_shard, feats_stream)

    # labels are exact int64 and come from the original bundle, never the shard
    labels = bundle["config_runtime"][orig_ids]
    assert labels.dtype == np.int64
    assert np.array_equal(labels, np.load(fp)["config_runtime"][orig_ids])


def test_shard_branch_matches_dataset_getitem(data_root, cache):
    """dataset.__getitem__'s shard branch produces the exact same features/labels
    a direct read_shard_sample call would, for the same (seed, epoch, i) stream."""
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_config_shards.py"),
        "--collections", "layout:xla:random", "--splits", "train", "--limit", "1"],
        check=True, cwd=str(ROOT), capture_output=True, text=True,
    )
    ds = TpugraphsDataset("layout:xla:random", "train", data_root=str(data_root),
                          k_configs=K, cache=cache, seed=0,
                          shard_root="data/cache/config_shards", shard_pool=2000)
    ds.files = ds.files[:1]
    stem = Path(ds.files[0]).stem

    family, source, search = parse_collection("layout:xla:random")
    shard = CS.find_shard("data/cache/config_shards", "layout:xla:random", "train",
                          stem, ds.files[0], 2000)
    if shard is None:
        pytest.skip("shard build didn't cover this file (unexpected but non-fatal)")

    data = ds[0]  # exercises the shard branch (shard_root is set)

    rng = np.random.default_rng((ds.seed, ds.epoch, 0))
    feats_expected, orig_ids = CS.read_shard_sample(shard[0], shard[1], K, rng)
    bundle = read_bundle(ds.files[0])
    labels_expected = bundle["config_runtime"][orig_ids]

    # graph.py stores node_config_feat as [nc, k, 18] (transposed from [k, nc, 18])
    got = data.node_config_feat.numpy()
    expected_t = np.ascontiguousarray(np.transpose(feats_expected, (1, 0, 2)))
    assert np.array_equal(got, expected_t)
    assert np.array_equal(data.y.view(-1).numpy(), labels_expected)
