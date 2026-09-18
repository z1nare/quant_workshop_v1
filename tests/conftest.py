import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quantsoc.market import MarketParams, generate_market, stress_params  # noqa: E402
from quantsoc.modeling import build_dataset, clean_rows, fit_model        # noqa: E402


@pytest.fixture(scope="session")
def small_params():
    return MarketParams(seed=7, n_sessions=8)


@pytest.fixture(scope="session")
def small_bars(small_params):
    return generate_market(small_params)


@pytest.fixture(scope="session")
def small_dataset(small_bars):
    return build_dataset(small_bars)


@pytest.fixture(scope="session")
def small_model(small_dataset):
    return fit_model(clean_rows(small_dataset, "train"))


@pytest.fixture(scope="session")
def repo_root():
    return ROOT
