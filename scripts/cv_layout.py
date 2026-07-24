#!/usr/bin/env python
"""Grouped-CV evaluation harness for layout configs (Phase-4 P4-0).

Runs K-fold grouped-by-graph CV *within the train split* for a layout config (with
optional overrides), and reports **mean +/- std best-fold OPA** plus the median
best epoch. This de-noises model selection for the tiny xla valid splits (7 files)
and produces the before/after numbers Gate P4 requires — selection is on mean CV
OPA, never the 7-file valid.

Each override lets one lever be swept without editing YAML:
  --max_keep_nodes (GST window)  --k_configs  --loss  --hidden_dim  --max_epochs

Usage:
  python scripts/cv_layout.py --config configs/layout_xla_random.yaml \
      --max_keep_nodes 4000 --k_configs 16 --tag gst4000_k16 \
      --results_csv artifacts/p4_cv_results.csv
"""
from __future__ import annotations
import argparse
import csv
import statistics
import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))      # repo root (for `src`)
sys.path.insert(0, str(_HERE))             # scripts/ (for train_layout)
from train_layout import train_layout_from_config    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--tag", default="base", help="label for this override set")
    ap.add_argument("--results_csv", default="artifacts/p4_cv_results.csv")
    ap.add_argument("--out_root", default="artifacts/cv_runs")
    # overrides (None = keep config value)
    ap.add_argument("--max_keep_nodes", type=int, default=None)
    ap.add_argument("--k_configs", type=int, default=None)
    ap.add_argument("--loss", default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--config_encoder", default=None, choices=[None, "linear", "mlp"])
    ap.add_argument("--use_config_attn", action="store_true")
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--limit_files", type=int, default=None)   # smoke
    ap.add_argument("--shards_root", default=None)
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    collection = cfg["data"]["collection"]
    base_run = cfg["run_name"]

    if args.max_keep_nodes is not None:
        cfg.setdefault("gst", {})["max_keep_nodes"] = args.max_keep_nodes
    if args.k_configs is not None:
        cfg["data"]["k_configs"] = args.k_configs
    if args.loss is not None:
        cfg["loss"]["type"] = args.loss
    if args.hidden_dim is not None:
        cfg["model"]["hidden_dim"] = args.hidden_dim
        cfg["model"]["config_proj_dim"] = args.hidden_dim   # must match hidden_dim
    if args.config_encoder is not None:
        cfg["model"]["config_encoder"] = args.config_encoder
    if args.use_config_attn:
        cfg["model"]["use_config_attn"] = True
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.limit_files is not None:
        cfg["data"]["limit_files"] = args.limit_files
    if args.shards_root is not None:
        cfg.setdefault("shards", {})["root"] = args.shards_root
    if args.cache_dir is not None:
        cfg["data"]["cache_dir"] = args.cache_dir

    cfg.setdefault("cv", {})
    cfg["cv"]["enabled"] = True
    cfg["cv"]["n_folds"] = args.n_folds

    mkn = (cfg.get("gst") or {}).get("max_keep_nodes", 1000)
    k = cfg["data"].get("k_configs", 8)
    print(f"\n=== CV {collection} | tag={args.tag} | max_keep={mkn} k={k} "
          f"loss={cfg['loss']['type']} hidden={cfg['model'].get('hidden_dim',64)} "
          f"| {args.n_folds} folds ===", flush=True)

    opas, epochs = [], []
    for fold in range(args.n_folds):
        cfg["cv"]["fold"] = fold
        cfg["run_name"] = f"{base_run}_{args.tag}"   # train adds _fold{n}
        r = train_layout_from_config(cfg, out_root=args.out_root)
        opas.append(float(r["best_val_opa"]))
        epochs.append(int(r["best_epoch"]))
        print(f"  fold {fold}: OPA={opas[-1]:.4f} @ epoch {epochs[-1]}", flush=True)

    mean = statistics.mean(opas)
    std = statistics.pstdev(opas) if len(opas) > 1 else 0.0
    med_epoch = int(statistics.median(epochs))
    print(f"\n>>> {collection} [{args.tag}] mean CV OPA = {mean:.4f} +/- {std:.4f} "
          f"| per-fold {[round(x,4) for x in opas]} | median best epoch {med_epoch}",
          flush=True)

    out = Path(args.results_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["collection", "tag", "max_keep_nodes", "k_configs", "loss",
                        "hidden_dim", "n_folds", "mean_opa", "std_opa",
                        "per_fold_opa", "median_best_epoch"])
        w.writerow([collection, args.tag, mkn, k, cfg["loss"]["type"],
                    cfg["model"].get("hidden_dim", 64), args.n_folds,
                    f"{mean:.4f}", f"{std:.4f}",
                    ";".join(f"{x:.4f}" for x in opas), med_epoch])
    print(f"appended results -> {out}", flush=True)
    return mean


if __name__ == "__main__":
    main()
