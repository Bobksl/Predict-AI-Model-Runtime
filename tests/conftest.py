import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.paths import resolve_data_root  # noqa: E402


@pytest.fixture(scope="session")
def data_root():
    try:
        return resolve_data_root()
    except FileNotFoundError:
        pytest.skip("TPUGraphs data root not available")
