"""Assemble and validate the ``ID,TopConfigs`` submission format.

Row: ``<collection>:<stem>,<i0;i1;...>`` — a ``;``-separated, fastest-first
permutation of config indices (ascending predicted score, matching the starter
notebook's ``tf.argsort`` convention — see docs/TEAM.md).
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List


def assemble_submission(rows: Dict[str, List[int]], collection: str,
                        out_path: str) -> None:
    """``rows``: stem -> ranked config-index list. Writes the CSV."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID", "TopConfigs"])
        for stem in sorted(rows):
            ranks = ";".join(str(i) for i in rows[stem])
            w.writerow([f"{collection}:{stem}", ranks])


def validate_submission(path: str, expected_stems: List[str], collection: str,
                        n_configs_by_stem: Dict[str, int]) -> None:
    """Assert the file is well-formed: header, ids, and valid permutations."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == ["ID", "TopConfigs"], f"bad header: {header}"
        seen = set()
        for row in r:
            assert len(row) == 2, f"malformed row: {row}"
            id_, ranks_str = row
            prefix = f"{collection}:"
            assert id_.startswith(prefix), f"bad id prefix: {id_}"
            stem = id_[len(prefix):]
            assert stem in n_configs_by_stem, f"unknown stem: {stem}"
            ranks = [int(x) for x in ranks_str.split(";")]
            n = n_configs_by_stem[stem]
            assert sorted(ranks) == list(range(n)), \
                f"{stem}: not a valid permutation of range({n})"
            seen.add(stem)
    missing = set(expected_stems) - seen
    extra = seen - set(expected_stems)
    assert not missing, f"missing stems: {sorted(missing)[:5]}..."
    assert not extra, f"unexpected stems: {sorted(extra)[:5]}..."
