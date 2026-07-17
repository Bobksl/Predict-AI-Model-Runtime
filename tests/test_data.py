"""Phase-1 data-pipeline tests (gate P1).

Fast by construction: every collection is exercised on a 2-file subset.
"""
import pickle

import numpy as np
import pytest
import torch

from src.data.paths import COLLECTIONS, list_npz
from src.data.configs import (
    sample_config_indices, read_npy_rows, read_node_config_feat_rows,
)
from src.data.cache import CompactCache, read_compact
from src.data.normalize import NodeFeatNormalizer
from src.data.splits import assign_graph_folds
from src.data.dataset import TpugraphsDataset, make_loader
from src.data.collate import scatter_node_config_feat

K = 6


def _small_dataset(coll, data_root, cache, n=2, normalizer=None):
    ds = TpugraphsDataset(coll, "train", data_root=str(data_root), k_configs=K,
                          cache=cache, normalizer=normalizer, seed=0)
    ds.files = ds.files[:n]
    return ds


@pytest.fixture(scope="session")
def cache(tmp_path_factory):
    return CompactCache(tmp_path_factory.mktemp("cache"), enabled=True)


# ---- T1/T3: sampling -------------------------------------------------
def test_sampling_returns_exactly_k():
    rng = np.random.default_rng(0)
    assert sample_config_indices(100, K, rng).shape[0] == K          # n > k
    assert sample_config_indices(K, K, rng).shape[0] == K            # n == k
    out = sample_config_indices(3, K, rng)                           # n < k -> pad
    assert out.shape[0] == K and out.max() < 3


# ---- T8: batch > 1 for EVERY collection ------------------------------
@pytest.mark.parametrize("coll", COLLECTIONS)
def test_batch_gt_1_every_collection(coll, data_root, cache):
    ds = _small_dataset(coll, data_root, cache, n=2)
    if len(ds) < 2:
        pytest.skip(f"{coll} has < 2 train files")
    batch = make_loader(ds, batch_size=2, shuffle=False, num_workers=0).__iter__().__next__()
    assert batch.num_graphs == 2
    # shapes
    assert batch.x.shape[1] == 140
    assert batch.edge_index.shape[0] == 2
    assert batch.op.shape[0] == batch.x.shape[0]
    assert batch.y.shape == (2, K)
    # edge bounds: 0 <= idx < n_nodes
    assert int(batch.edge_index.min()) >= 0
    assert int(batch.edge_index.max()) < batch.num_nodes
    # config plumbing per family
    if coll.startswith("tile"):
        assert batch.config_feat.shape == (2, K, 24)
    else:
        assert batch.node_config_feat.shape[1:] == (K, 18)
        assert int(batch.cfg_node_index.max()) < batch.num_nodes
        dense = scatter_node_config_feat(batch)
        assert dense.shape == (batch.num_nodes, K, 18)
        nonconf = ~torch.isin(torch.arange(batch.num_nodes), batch.cfg_node_index)
        assert bool((dense[nonconf] == 0).all())  # non-configurable nodes masked to 0


# ---- T6: finite features after normalisation -------------------------
def test_features_finite_after_norm(data_root, cache):
    coll = "layout:xla:random"
    files = [str(p) for p in list_npz(data_root, coll, "train")[:2]]
    norm = NodeFeatNormalizer.fit(files)
    ds = _small_dataset(coll, data_root, cache, n=2, normalizer=norm)
    batch = make_loader(ds, batch_size=2, num_workers=0).__iter__().__next__()
    assert torch.isfinite(batch.x).all()


# ---- T5: cache round-trip equals cold read ---------------------------
def test_cache_roundtrip(data_root, tmp_path):
    f = str(list_npz(data_root, "layout:nlp:default", "train")[0])
    cold = read_compact(f)
    c = CompactCache(tmp_path / "c", enabled=True)
    warm1 = c.get(f)   # miss -> populate
    warm2 = c.get(f)   # hit
    for k in cold:
        assert np.array_equal(cold[k], warm1[k])
        assert np.array_equal(cold[k], warm2[k])


# ---- T3: streaming reader equals full load ---------------------------
def test_streaming_equals_full(data_root):
    # Pinned to alexnet (54.6 MB full tensor — safe to load as ground truth);
    # do NOT use "first file in dir" here, bigger files would be a RAM bomb.
    cands = [p for p in list_npz(data_root, "layout:xla:random", "train")
             if p.stem == "alexnet_train_batch_32"]
    assert cands, "pinned test file missing"
    f = str(cands[0])
    with np.load(f) as z:
        full = z["node_config_feat"]
        idx = np.array([3, 0, full.shape[0] - 1, 1])
        ref = full[idx]
    got = read_npy_rows(f, "node_config_feat", idx)
    assert np.array_equal(ref, got)
    got8 = read_node_config_feat_rows(f, idx)
    assert got8.dtype == np.int8 and np.array_equal(got8.astype(np.float32), ref)


# ---- guardrail #1: labels exact int64, epoch-aware sampling -----------
def test_labels_exact_int64_and_epoch_sampling(data_root, cache):
    ds = _small_dataset("layout:xla:random", data_root, cache, n=1)
    d0 = ds[0]
    assert d0.y.dtype == torch.int64                      # no float32 collapse
    # values must equal the raw npz runtimes at the sampled indices
    raw = np.load(ds.files[0])["config_runtime"]
    rng = np.random.default_rng((ds.seed, ds.epoch, 0))
    idx = sample_config_indices(raw.shape[0], K, rng)
    assert np.array_equal(d0.y.view(-1).numpy(), raw[idx])
    # epoch-aware: a different epoch draws a different subset (w.h.p.)
    ds.set_epoch(1)
    d1 = ds[0]
    assert not torch.equal(d0.y, d1.y)


# ---- guardrail #2: test labels are placeholders, flagged not fed ------
def test_test_split_is_placeholder(data_root, cache):
    ds = TpugraphsDataset("tile:xla", "test", data_root=str(data_root),
                          k_configs=K, cache=cache, seed=0)
    ds.files = ds.files[:2]
    batch = make_loader(ds, batch_size=2, num_workers=0).__iter__().__next__()
    assert bool(batch.is_placeholder.all())


# ---- T7: grouped-by-graph folds, no leakage --------------------------
def test_graph_folds_no_leakage():
    stems = [f"g{i}" for i in range(23)]
    folds = assign_graph_folds(stems, n_folds=5, seed=1)
    assert set(folds) == set(stems)                 # every graph assigned
    assert set(folds.values()) <= set(range(5))     # valid fold ids
    # determinism
    assert folds == assign_graph_folds(stems, n_folds=5, seed=1)


# ---- spawn-worker picklability ---------------------------------------
def test_dataset_is_picklable(data_root, cache):
    ds = _small_dataset("tile:xla", data_root, cache, n=2)
    ds2 = pickle.loads(pickle.dumps(ds))
    assert len(ds2) == len(ds)
    assert ds2[0].x.shape[1] == 140
