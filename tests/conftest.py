from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def small_params() -> dict:
    return {"n_words": 4096, "cols": 64, "block_size": 256, "seed": 7}
