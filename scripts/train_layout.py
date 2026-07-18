#!/usr/bin/env python
"""Train one layout-collection ranker under Graph Segment Training (GST).

Usage:
    python scripts/train_layout.py --config configs/layout_xla_random.yaml
    python scripts/train_layout.py --config configs/layout_xla_random.yaml --smoke
    python scripts/train_layout.py --config configs/layout_xla_random.yaml --limit_files 5
    python scripts/train_layout.py --config configs/layout_xla_random.yaml --cv_fold 2

Extends ``src/training/train_loop.py`` conventions (Adam + grad clip, early stop,
``best.pt`` + ``metrics.csv`` + resolved ``config.yaml``) but trains with Graph
Segment Training (``src/training/gst.py``) — bounded backward memory on layout
graphs up to N=43,615 — and evaluates full-graph / ``no_grad`` (reuses
``train_loop.evaluate`` unchanged: ``LayoutRanker.forward`` defaults to the plain,
non-GST pass, matching the eval guardrail).

``--smoke``: 5 files, ``MAX_KEEP_NODES=512``, ``k=4``, 2 epochs (CPU, minutes).
"""
from __future__ import annotations
import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import TpugraphsDataset, make_loader, NodeFeatNormalizer  # noqa: E402
from src.data.cache import CompactCache                                 # noqa: E402
from src.models import build_model                                      # noqa: E402
from src.training.train_loop import set_seed, evaluate                  # noqa: E402
from src.training.losses import pairwise_hinge_loss, listmle_loss       # noqa: E402
from src.training.cv import make_fold_files                             # noqa: E402


def _make_dataset(cfg: dict, split: str, files_override=None) -> TpugraphsDataset:
    d = cfg["data"]
    norm = NodeFeatNormalizer.load(d["normalizer"]) if d.get("normalizer") else None
    cache = CompactCache(d["cache_dir"]) if d.get("cache_dir") else None
    sh = cfg.get("shards", {}) or {}
    ds = TpugraphsDataset(
        d["collection"], split,
        k_configs=d.get("k_configs", 8),
        normalizer=norm, cache=cache,
        seed=cfg.get("seed", 0),
        add_reverse_edges=d.get("add_reverse_edges", True),
        shard_root=sh.get("root") if sh.get("enabled") else None,
        shard_pool=sh.get("pool", 2000),
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


def train_layout_from_config(cfg: dict, out_root: str = "artifacts/checkpoints") -> dict:
    """Train one layout run; returns {'best_val_opa', 'best_epoch', 'run_dir', ...}."""
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
    train_loader = make_loader(train_ds, batch_size=d.get("batch_size", 4),
                               shuffle=True, num_workers=d.get("num_workers", 0))
    val_loader = make_loader(val_ds, batch_size=d.get("batch_size", 4),
                             shuffle=False, num_workers=d.get("num_workers", 0))

    model = build_model(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["optim"].get("lr", 1e-3)),
                           weight_decay=float(cfg["optim"].get("weight_decay", 0.0)))
    clip = float(cfg["optim"].get("clip_norm", 0.5))
    loss_fn = _loss_fn(cfg)
    gst_cfg = cfg.get("gst", {}) or {}
    max_keep_nodes = int(gst_cfg.get("max_keep_nodes", 1000))
    seed = int(cfg.get("seed", 0))

    run_dir = Path(out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    max_epochs = int(cfg["train"].get("max_epochs", 50))
    patience = int(cfg["train"].get("early_stop_patience", 8))
    best_opa, best_epoch, best_state = -1.0, -1, None
    rows = []
    print(f"[{run_name}] device={device} params={n_params:,} "
          f"train_files={len(train_ds)} val_files={len(val_ds)} "
          f"max_keep_nodes={max_keep_nodes} k={d.get('k_configs', 8)}")

    for epoch in range(max_epochs):
        model.train()
        train_ds.set_epoch(epoch)          # new config subsets + new GST window each epoch
        t0, losses = time.perf_counter(), []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            scores = model(batch, gst=True, max_keep_nodes=max_keep_nodes,
                          seed=seed, epoch=epoch)
            loss = loss_fn(scores, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            losses.append(loss.item())
        val_opa = evaluate(model, val_loader, device)  # full graph, no GST, no_grad
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="CPU smoke: 5 files, MAX_KEEP_NODES=512, k=4, 2 epochs")
    ap.add_argument("--limit_files", type=int, default=None,
                    help="override data.limit_files")
    ap.add_argument("--cv_fold", type=int, default=None,
                    help="override cv.enabled=true, cv.fold=N")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        cfg["data"]["limit_files"] = 5
        cfg["data"]["k_configs"] = 4
        cfg.setdefault("gst", {})
        cfg["gst"]["max_keep_nodes"] = 512
        cfg["train"]["max_epochs"] = 2
        cfg["train"]["early_stop_patience"] = 2
    if args.limit_files is not None:
        cfg["data"]["limit_files"] = args.limit_files
    if args.cv_fold is not None:
        cfg.setdefault("cv", {})
        cfg["cv"]["enabled"] = True
        cfg["cv"]["fold"] = args.cv_fold

    result = train_layout_from_config(cfg)
    print("\nRESULT:", result)


if __name__ == "__main__":
    main()
