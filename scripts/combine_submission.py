#!/usr/bin/env python
"""Combine per-collection submission CSVs into one 5-collection submission.

Concatenation = ONE header + all data rows (repeated headers stripped), matching
the starter combine cells (docs/briefs/phase3_brief.md §3 "Combine (cells 44/45)").
Validated against the inventory-derived test ID universe — there is NO
``sample_submission.csv`` on disk (orchestrator-approved correction, brief header);
``artifacts/inventory.parquet``'s test-split rows (``collection``, ``stem``,
``n_configs``) are the ground truth for both the expected ID set and each file's
valid-permutation length.

Usage:
    python scripts/combine_submission.py \
        --inputs artifacts/submissions/submission_layout_xla_random.csv \
                 artifacts/submissions/submission_layout_xla_default.csv \
                 artifacts/submissions/submission_layout_nlp_random.csv \
                 artifacts/submissions/submission_layout_nlp_default.csv \
                 artifacts/submissions/submission_tile.csv \
        --out artifacts/submissions/submission_5col.csv

    # partial inputs (e.g. a smoke run with only some files/collections present):
    # only rows that ARE present are checked for validity; the "every inventory id
    # is covered" completeness check is skipped.
    python scripts/combine_submission.py --inputs ... --out ... --allow_partial
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd   # noqa: E402

HEADER = ["ID", "TopConfigs"]


def combine(inputs, out_path) -> int:
    """Concatenate ``inputs`` into ``out_path`` with a single header. Returns the
    number of data rows written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(HEADER)
        for p in inputs:
            with open(p, "r", newline="", encoding="utf-8") as fin:
                r = csv.reader(fin)
                header = next(r)
                assert header == HEADER, f"{p}: bad header {header}"
                for row in r:
                    w.writerow(row)
                    n_rows += 1
    return n_rows


def validate_combined(path, inventory_path, require_complete: bool = True) -> None:
    """Assert every row is ``<collection>:<stem>`` from the inventory test split
    with ``TopConfigs`` a full permutation of ``range(n_configs)`` for that file,
    with no duplicate ids. When ``require_complete`` (the Gate-P3 full-submission
    check), also assert every inventory test id is present exactly once.
    """
    inv = pd.read_parquet(inventory_path)
    test = inv[inv["split"] == "test"]
    expected = {f"{c}:{s}": int(n) for c, s, n in
               zip(test["collection"], test["stem"], test["n_configs"])}

    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == HEADER, f"bad header: {header}"
        seen = set()
        for row in r:
            assert len(row) == 2, f"malformed row: {row}"
            id_, ranks_str = row
            assert id_ in expected, f"unknown id (not in inventory test universe): {id_}"
            assert id_ not in seen, f"duplicate id: {id_}"
            ranks = [int(x) for x in ranks_str.split(";")]
            n = expected[id_]
            assert sorted(ranks) == list(range(n)), \
                f"{id_}: not a valid permutation of range({n})"
            seen.add(id_)

    if require_complete:
        missing = set(expected) - seen
        assert not missing, f"missing ids: {sorted(missing)[:5]}... ({len(missing)} total)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/submissions/submission_5col.csv")
    ap.add_argument("--inventory", default="artifacts/inventory.parquet")
    ap.add_argument("--skip_validate", action="store_true",
                    help="skip the inventory-universe check entirely")
    ap.add_argument("--allow_partial", action="store_true",
                    help="skip only the completeness check (every present row must "
                        "still be a valid inventory id + permutation)")
    args = ap.parse_args()

    n = combine(args.inputs, args.out)
    print(f"Wrote {args.out}  ({n} rows from {len(args.inputs)} files)")

    if args.skip_validate:
        print("Skipped inventory-universe validation (--skip_validate).")
    else:
        validate_combined(args.out, args.inventory, require_complete=not args.allow_partial)
        print("Validated: header once, valid permutations, IDs match inventory test "
              f"universe{' (partial: completeness not required)' if args.allow_partial else ''}.")


if __name__ == "__main__":
    main()
