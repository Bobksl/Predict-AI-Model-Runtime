"""PyTorch ``Dataset`` + loader factory tying the pipeline together.

For each graph: read the compact arrays (optionally cached), sample ``k`` configs
lazily (never materialising the giant config tensor), normalise node features
(train-only stats), and build a :class:`~src.data.graph.TPUData`. Batched with
:func:`~src.data.collate.collate` it yields correct PyG ``Batch`` objects at
batch > 1 for every collection.

The dataset is **picklable** (stores paths/config only, no open handles) so it
works with spawn-based DataLoader workers.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .paths import resolve_data_root, list_npz, parse_collection
from .cache import CompactCache, read_bundle
from .configs import sample_config_indices, read_node_config_feat_rows
from .config_shards import find_shard, read_shard_sample
from .graph import build_pyg_data, TPUData
from .collate import collate
from .normalize import NodeFeatNormalizer


class TpugraphsDataset(Dataset):
    def __init__(
        self,
        collection: str,
        split: str,
        data_root=None,
        k_configs: int = 16,
        normalizer: Optional[NodeFeatNormalizer] = None,
        cache: Optional[CompactCache] = None,
        seed: int = 0,
        add_reverse_edges: bool = False,
        add_self_loops: bool = False,
        shard_root: Optional[str] = None,
        shard_pool: int = 2000,
    ):
        self.collection = collection
        self.split = split
        self.family = parse_collection(collection)[0]
        self.is_tile = self.family == "tile"
        self.data_root = str(resolve_data_root(data_root))
        self.files: List[str] = [str(p) for p in list_npz(self.data_root, collection, split)]
        self.k_configs = int(k_configs)
        self.normalizer = normalizer
        self.cache = cache
        self.seed = int(seed)
        self.epoch = 0
        self.add_reverse_edges = add_reverse_edges
        self.add_self_loops = add_self_loops
        # Layout-only config shards (Task C): a capped-pool int8 memmap read path
        # that is faster than streaming when a signature-matching shard exists.
        # ``shard_root=None`` (default) disables shard lookups entirely — every
        # collection (including tile) then behaves exactly as before this task.
        self.shard_root = shard_root
        self.shard_pool = int(shard_pool)

    def set_epoch(self, epoch: int) -> None:
        """Advance the config-sampling RNG stream (call once per training epoch)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.files)

    def _bundle(self, path):
        return self.cache.get(path) if self.cache is not None else read_bundle(path)

    def __getitem__(self, i: int) -> TPUData:
        path = self.files[i]
        stem = Path(path).stem
        bundle = self._bundle(path)             # one file open (cache or original)

        runtimes = bundle["config_runtime"]     # ordering signal only
        n_configs = int(runtimes.shape[0])
        is_placeholder = self.split == "test"   # test labels are placeholder zeros

        node_feat = bundle["node_feat"]
        if self.normalizer is not None:
            node_feat = self.normalizer.transform(node_feat)
        else:
            node_feat = np.ascontiguousarray(node_feat, dtype=np.float32)

        # Epoch-aware sampling: a new k-subset per epoch (call set_epoch each epoch).
        rng = np.random.default_rng((self.seed, self.epoch, i))

        ncf = None  # set on the shard-hit path only; streamed lazily otherwise
        if self.is_tile:
            idx = sample_config_indices(n_configs, self.k_configs, rng)
        else:
            shard = None
            if self.shard_root is not None and self.split != "test":
                shard = find_shard(self.shard_root, self.collection, self.split,
                                   stem, path, self.shard_pool)
            if shard is not None:
                data_f, idx_f = shard
                # ncf's rng draw replaces (not adds to) the streaming draw below —
                # same (seed, epoch, i) stream, so a signature match reproduces the
                # exact ids a streaming read would have chosen when pool==n_configs.
                ncf, idx = read_shard_sample(data_f, idx_f, self.k_configs, rng)
            else:
                # signature mismatch / no shard / test split -> streaming fallback
                idx = sample_config_indices(n_configs, self.k_configs, rng)
        # Keep labels EXACT int64 — runtimes exceed 2^24, so a float32 cast collapses
        # distinct values into spurious ties (guardrail #1). Cast only inside losses.
        # Labels are ALWAYS read from the original bundle here, never from a shard,
        # regardless of which branch produced `idx` (guardrail: no label duplication
        # into shards).
        runtimes_sampled = runtimes[idx]

        kwargs = dict(
            node_feat=node_feat,
            node_opcode=bundle["node_opcode"],
            edge_index=bundle["edge_index"],
            runtimes_sampled=runtimes_sampled,
            is_placeholder=is_placeholder,
            collection=self.collection, split=self.split, stem=stem,
            add_reverse_edges=self.add_reverse_edges,
            add_self_loops=self.add_self_loops,
        )
        if self.is_tile:
            kwargs["config_feat"] = bundle["config_feat"][idx].astype(np.float32)  # [k,24]
        else:
            if ncf is None:
                # the only uncacheable, RAM-bomb array: stream sampled rows from origin
                ncf = read_node_config_feat_rows(path, idx)  # [k, nc, 18] int8
            kwargs["node_config_feat"] = ncf
            kwargs["cfg_node_index"] = bundle["node_config_ids"]
        return build_pyg_data(**kwargs)


def make_loader(dataset: TpugraphsDataset, batch_size: int = 4, shuffle: bool = False,
                num_workers: int = 2, **kwargs) -> DataLoader:
    """Build a DataLoader using the variable-size graph collator."""
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate, pin_memory=torch.cuda.is_available(), **kwargs,
    )
