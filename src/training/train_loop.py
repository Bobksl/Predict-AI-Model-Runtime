"""Config-driven training loop (brief T7).

One call = one run: build datasets/loaders from a config dict, train with Adam +
grad clipping, evaluate valid OPA each epoch, early-stop with patience, restore the
best weights, and write ``artifacts/checkpoints/<run_name>/`` containing
``best.pt``, ``metrics.csv`` and the resolved ``config.yaml``.

Smoke mode: set ``data.limit_files`` (e.g. 5) — everything else is identical.
"""
from __future__ import annotations
import copy
import csv
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from src.data import TpugraphsDataset, make_loader, NodeFeatNormalizer
from src.data.cache import CompactCache
from src.models import build_model
from .losses import pairwise_hinge_loss, listmle_loss
from .metrics import opa_from_batch
from .cv import make_fold_files


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataset(cfg: dict, split: str, files_override=None) -> TpugraphsDataset:
    d = cfg["data"]
    norm = NodeFeatNormalizer.load(d["normalizer"]) if d.get("normalizer") else None
    cache = CompactCache(d["cache_dir"]) if d.get("cache_dir") else None
    ds = TpugraphsDataset(
        d["collection"], split,
        k_configs=d.get("k_configs", 16),
        normalizer=norm, cache=cache,
        seed=cfg.get("seed", 0),
        add_reverse_edges=d.get("add_reverse_edges", True),
    )
    if files_override is not None:
        ds.files = list(files_override)
    limit = d.get("limit_files")
    if limit:
        ds.files = ds.files[: int(limit)]
    return ds


def _loss_fn(cfg: dict):
    lc = cfg["loss"]
    if lc["type"] == "pairwise_hinge":
        m = float(lc.get("margin", 0.1))
        return lambda s, y: pairwise_hinge_loss(s, y, margin=m)
    if lc["type"] == "listmle":
        return listmle_loss
    raise KeyError(f"Unknown loss.type {lc['type']!r}")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Mean valid OPA over a loader (skips placeholder-label graphs)."""
    model.eval()
    vals = []
    for batch in loader:
        if bool(batch.is_placeholder.any()):
            continue  # guardrail #2: never score placeholder labels
        batch = batch.to(device)
        v = opa_from_batch(model(batch), batch.y)
        if not torch.isnan(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def train_from_config(cfg: dict, out_root: str = "artifacts/checkpoints") -> dict:
    """Train one run; returns {'best_val_opa', 'best_epoch', 'run_dir', ...}."""
    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cv = cfg.get("cv", {}) or {}
    if cv.get("enabled"):
        base = _make_dataset(cfg, "train")
        fit_files, held_files = make_fold_files(
            base.files, int(cv.get("n_folds", 5)), int(cv["fold"]),
            seed=int(cv.get("seed", 0)))
        train_ds = _make_dataset(cfg, "train", files_override=fit_files)
        val_ds = _make_dataset(cfg, "train", files_override=held_files)
        run_name = f"{cfg['run_name']}_fold{cv['fold']}"
    else:
        train_ds = _make_dataset(cfg, "train")
        val_ds = _make_dataset(cfg, "valid")
        run_name = cfg["run_name"]

    d = cfg["data"]
    train_loader = make_loader(train_ds, batch_size=d.get("batch_size", 64),
                               shuffle=True, num_workers=d.get("num_workers", 0))
    val_loader = make_loader(val_ds, batch_size=d.get("batch_size", 64),
                             shuffle=False, num_workers=d.get("num_workers", 0))

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["optim"].get("lr", 1e-3)),
                           weight_decay=float(cfg["optim"].get("weight_decay", 0.0)))
    clip = float(cfg["optim"].get("clip_norm", 0.5))
    loss_fn = _loss_fn(cfg)

    run_dir = Path(out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    max_epochs = int(cfg["train"].get("max_epochs", 50))
    patience = int(cfg["train"].get("early_stop_patience", 5))
    best_opa, best_epoch, best_state = -1.0, -1, None
    rows = []
    print(f"[{run_name}] device={device} params={n_params:,} "
          f"train_files={len(train_ds)} val_files={len(val_ds)}")

    for epoch in range(max_epochs):
        model.train()
        train_ds.set_epoch(epoch)          # new config subsets each epoch (F2 fix)
        t0, losses = time.perf_counter(), []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.item())
        val_opa = evaluate(model, val_loader, device)
        rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                     "val_opa": val_opa, "seconds": time.perf_counter() - t0})
        print(f"  epoch {epoch:3d}  loss={rows[-1]['train_loss']:.4f}  "
              f"val_opa={val_opa:.4f}  ({rows[-1]['seconds']:.1f}s)")
        if val_opa > best_opa:
            best_opa, best_epoch = val_opa, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            print(f"  early stop at epoch {epoch} (best {best_opa:.4f} @ {best_epoch})")
            break

    assert best_state is not None, "no epoch produced a finite valid OPA"
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": cfg,
                "best_val_opa": best_opa, "best_epoch": best_epoch},
               run_dir / "best.pt")
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    result = {"run_name": run_name, "best_val_opa": best_opa,
              "best_epoch": best_epoch, "n_params": n_params,
              "run_dir": str(run_dir)}
    print(f"[{run_name}] best val OPA {best_opa:.4f} @ epoch {best_epoch} -> {run_dir}")
    return result
