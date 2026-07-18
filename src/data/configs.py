"""Lazy, memory-safe configuration sampling.

The layout ``node_config_feat`` array has shape ``[n_configs, n_config_nodes, 18]``
and decompresses to **> 1 GB** for large files, so it must never be materialised
in full (guardrail #3). This module reads **only the sampled config rows** by
streaming the compressed ``.npy`` member inside the ``.npz`` zip and copying out
just the requested rows.

Small per-config arrays (``config_runtime`` ``[c]``; tile ``config_feat`` ``[c,24]``
and ``config_runtime_normalizers`` ``[c]``) are tiny and are loaded in full.

Labels (``config_runtime``) are read as an **ordering signal only** — never
clipped, normalised, or reordered (guardrail #1).
"""
from __future__ import annotations
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.lib import format as npformat

_CHUNK = 1 << 20  # 1 MiB streaming buffer


def sample_config_indices(n_configs: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Choose **exactly ``k``** config indices.

    With ``n_configs >= k`` indices are sampled without replacement. With fewer
    than ``k`` configs all of them are used and the remainder is filled by sampling
    with replacement, so every graph yields a fixed-length ``k`` ranking list (which
    is required to stack labels into a ``[B, k]`` batch). Returned in sampled order.
    """
    if n_configs <= 0:
        raise ValueError("n_configs must be positive")
    if k <= 0:
        raise ValueError("k must be positive")
    if n_configs >= k:
        return rng.choice(n_configs, size=k, replace=False).astype(np.int64)
    extra = rng.choice(n_configs, size=k - n_configs, replace=True)
    return np.concatenate([rng.permutation(n_configs), extra]).astype(np.int64)


def _open_member(zf: zipfile.ZipFile, key: str):
    name = key if key.endswith(".npy") else key + ".npy"
    return zf.open(name, "r")


def _read_npy_header(f):
    version = npformat.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = npformat.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran, dtype = npformat.read_array_header_2_0(f)
    else:  # pragma: no cover - numpy only writes 1.0/2.0 in practice
        shape, fortran, dtype = npformat._read_array_header(f, version)
    return shape, fortran, dtype


def _read_exact(f, nbytes: int) -> bytes:
    """Read exactly ``nbytes`` from a (possibly streaming) file object."""
    out = bytearray()
    while nbytes > 0:
        chunk = f.read(min(nbytes, _CHUNK))
        if not chunk:
            raise EOFError("unexpected end of npy stream")
        out.extend(chunk)
        nbytes -= len(chunk)
    return bytes(out)


def _skip(f, nbytes: int) -> None:
    """Advance a streaming file object by ``nbytes`` without keeping the data."""
    while nbytes > 0:
        chunk = f.read(min(nbytes, _CHUNK))
        if not chunk:
            raise EOFError("unexpected end of npy stream while skipping")
        nbytes -= len(chunk)


def read_npy_rows(path, key: str, indices: Sequence[int]) -> np.ndarray:
    """Read ``array[indices]`` along axis 0 from a member of an ``.npz``, streaming.

    Only ``len(indices)`` rows are ever held in memory (plus a 1 MiB buffer). The
    streaming decoder reads forward through the compressed member up to the largest
    requested index, so peak memory is independent of ``n_configs``. Rows are
    returned in the order given by ``indices``.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("indices must be 1-D")
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]

    with zipfile.ZipFile(path) as zf, _open_member(zf, key) as f:
        shape, fortran, dtype = _read_npy_header(f)
        if fortran:  # pragma: no cover - tpugraphs arrays are C-order
            raise NotImplementedError("Fortran-order arrays not supported for row streaming")
        row_shape = tuple(shape[1:])
        row_elems = int(np.prod(row_shape)) if row_shape else 1
        row_bytes = row_elems * dtype.itemsize

        out = np.empty((len(sorted_idx),) + row_shape, dtype=dtype)
        pos = 0  # next un-consumed row index in the data region
        for j, ci in enumerate(sorted_idx):
            if ci < pos or ci >= shape[0]:
                raise IndexError(f"row index {ci} out of range for axis 0 = {shape[0]}")
            _skip(f, int(ci - pos) * row_bytes)
            buf = _read_exact(f, row_bytes)
            out[j] = np.frombuffer(buf, dtype=dtype).reshape(row_shape)
            pos = ci + 1

    # restore the requested (unsorted) order
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return out[inv]


def read_npy_shape(path, key: str):
    """Return the shape/dtype of an ``.npz`` member by reading only its header.

    Does **not** decompress the array body, so it is safe (and fast) even for the
    GB-scale ``node_config_feat``.
    """
    with zipfile.ZipFile(path) as zf, _open_member(zf, key) as f:
        shape, fortran, dtype = _read_npy_header(f)
    return tuple(shape), dtype


def read_runtimes(path, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Read ``config_runtime`` (ordering signal only). Small array — loaded fully."""
    with np.load(path) as d:
        rt = d["config_runtime"]
        if indices is None:
            return np.ascontiguousarray(rt)
        return np.ascontiguousarray(rt[np.asarray(indices, dtype=np.int64)])


def read_tile_config_feat(path, indices: Sequence[int]):
    """Return ``(config_feat[idx] f32 [k,24], normalizers[idx] i64 [k])`` for tile."""
    idx = np.asarray(indices, dtype=np.int64)
    with np.load(path) as d:
        cf = np.ascontiguousarray(d["config_feat"][idx]).astype(np.float32, copy=False)
        nz = np.ascontiguousarray(d["config_runtime_normalizers"][idx])
    return cf, nz


def read_node_config_feat_rows(path, indices: Sequence[int]) -> np.ndarray:
    """Sampled layout config features as **int8** ``[k, n_config_nodes, 18]``.

    Values are in ``{-1,0,1,2,3}`` so int8 is lossless and 4x smaller than f32.
    Streamed row-by-row so the full ``[c,nc,18]`` tensor is never materialised.
    """
    rows = read_npy_rows(path, "node_config_feat", indices)
    return rows.astype(np.int8, copy=False)
