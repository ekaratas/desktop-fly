import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def cfg():
    with open(os.path.join(ROOT, "configs", "btcusdt_1h_mvp.json")) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def bars():
    from trader.data.synthetic import make_synthetic_klines
    return make_synthetic_klines(4000, seed=3, start="2020-01-01")
