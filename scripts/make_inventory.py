#!/usr/bin/env python
"""Build the per-file dataset inventory parquet.

Usage:
    python scripts/make_inventory.py [--data_root DIR] [--out PATH] [--limit N]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inventory import build_inventory          # noqa: E402
from src.data.paths import writable_dir                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default=None, help="parquet path (default <writable>/inventory.parquet)")
    ap.add_argument("--limit", type=int, default=None, help="max files per collection/split")
    args = ap.parse_args()

    print("Building inventory ...")
    df = build_inventory(args.data_root, limit=args.limit)
    out = Path(args.out) if args.out else (writable_dir() / "inventory.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"\nWrote {out}  ({len(df)} files)")
    print("\nPer-collection/split counts:")
    print(df.groupby(["collection", "split"]).size().unstack(fill_value=0))
    print(f"\nedge_oob files: {int(df['edge_oob'].sum())}")
    print(f"has_nan files : {int(df['has_nan'].sum())}   has_inf files: {int(df['has_inf'].sum())}")
    print(f"global opcode_max: {int(df['opcode_max'].max())}  -> embedding size = {int(df['opcode_max'].max())+1}")
    test = df[df.split == "test"]
    print(f"test files flagged runtime_is_placeholder: "
          f"{int(test['runtime_is_placeholder'].sum())}/{len(test)}")

    # gate assertions
    assert int(df["edge_oob"].sum()) == 0, "edge_oob detected!"
    assert test["runtime_is_placeholder"].all(), "some test files not flagged placeholder!"
    print("\nGate checks: edge_oob == 0  AND  all test runtimes placeholder  -> OK")


if __name__ == "__main__":
    main()
