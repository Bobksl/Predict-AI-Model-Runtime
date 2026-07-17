"""Submission validator accepts good rows / rejects malformed ones (brief T6)."""
import pytest

from src.inference.submission import assemble_submission, validate_submission


def test_validator_accepts_good_file(tmp_path):
    rows = {"a": [0, 1, 2], "b": [1, 0]}
    out = tmp_path / "sub.csv"
    assemble_submission(rows, "tile:xla", str(out))
    validate_submission(str(out), ["a", "b"], "tile:xla",
                        {"a": 3, "b": 2})  # must not raise


def test_validator_rejects_bad_permutation(tmp_path):
    rows = {"a": [0, 0, 2]}  # not a permutation of range(3)
    out = tmp_path / "sub.csv"
    assemble_submission(rows, "tile:xla", str(out))
    with pytest.raises(AssertionError):
        validate_submission(str(out), ["a"], "tile:xla", {"a": 3})


def test_validator_rejects_missing_stem(tmp_path):
    rows = {"a": [0, 1, 2]}
    out = tmp_path / "sub.csv"
    assemble_submission(rows, "tile:xla", str(out))
    with pytest.raises(AssertionError):
        validate_submission(str(out), ["a", "b"], "tile:xla", {"a": 3, "b": 2})
