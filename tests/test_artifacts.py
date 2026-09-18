import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantsoc.artifacts import check_zip, export_bundle, load_bundle, load_strategy_module, scan_for_secrets, zip_bundle
from quantsoc.market import SYMBOLS
from quantsoc.portfolio import PortfolioConfig
from quantsoc import solutions as S
from quantsoc import reference_strategy as rs


@pytest.fixture
def bundle(tmp_path, small_model, repo_root):
    strat = tmp_path / "strategy.py"
    strat.write_text(S.STRATEGY_PY, encoding="utf-8")
    out = export_bundle(tmp_path / "bundle", strat, small_model, PortfolioConfig(threshold=0.0003), SYMBOLS,
                        strategy_source="student", exercise_status={"1": "passed"}, fund_name="Test Fund", repo_root=repo_root)
    return out


def test_round_trip(bundle, small_model, small_bars):
    b = load_bundle(bundle)
    assert b.config.threshold == 0.0003 and b.symbols == SYMBOLS and b.strategy_source == "student"
    assert b.manifest["fund_name"] == "Test Fund" and b.manifest["exercise_status"] == {"1": "passed"}
    X = np.array([[0.001, 0.002, 0.001], [0.0, 0.0, 0.002]])
    assert np.allclose(b.model.predict(X), small_model.predict(X))
    preds = b.predict_returns(small_bars, b.model)
    pd.testing.assert_series_equal(preds, rs.predict_returns(small_bars, small_model))
    pd.testing.assert_series_equal(b.allocate(preds, b.config).sort_index(), rs.allocate(preds, b.config).sort_index(), check_names=False)


def test_zip_contents_and_no_secrets(bundle, tmp_path):
    (bundle / ".env").write_text("APCA_API_KEY_ID=PKTESTTESTTESTTEST1234\n", encoding="utf-8")     # must be excluded
    (bundle / "state").mkdir(); (bundle / "state" / "orders.json").write_text("{}", encoding="utf-8")
    z = zip_bundle(bundle, tmp_path / "fund.zip")
    rep = check_zip(z)
    assert rep["ok"], rep
    names = zipfile.ZipFile(z).namelist()
    assert not any(n.endswith("/.env") or "/state/" in n for n in names)
    for req in ["strategy.py", "model.joblib", "config.json", "manifest.json", "README_RUN_LOCAL.md", "run_trader.py",
                "dashboard.py", "requirements.txt", "quantsoc/engine.py", "data/workshop_market.csv"]:
        assert any(n.endswith("/" + req) for n in names), req


def test_export_refuses_strategy_with_credentials(tmp_path, small_model, repo_root):
    strat = tmp_path / "strategy.py"
    strat.write_text(S.STRATEGY_PY + '\nAPCA_API_SECRET_KEY = "abcdefghijklmnopqrstuvwxyz123456"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="credentials"):
        export_bundle(tmp_path / "b", strat, small_model, PortfolioConfig(), SYMBOLS, repo_root=repo_root)


def test_scan_for_secrets():
    assert scan_for_secrets("key = 'PKABCDEFGHIJKLMNOPQR'")
    assert not scan_for_secrets("threshold = 0.0002")


def test_reference_label_recorded(tmp_path, small_model, repo_root):
    strat = tmp_path / "strategy.py"; strat.write_text(S.STRATEGY_PY, encoding="utf-8")
    out = export_bundle(tmp_path / "b", strat, small_model, PortfolioConfig(), SYMBOLS, strategy_source="reference", repo_root=repo_root)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["strategy_source"] == "reference" and "NOT the participant" in m["strategy_source_note"]


def test_schema_mismatch_rejected(bundle):
    cfg = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    cfg["feature_names"] = ["mom_3", "ret_1", "vol_12"]
    (bundle / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="feature order"):
        load_bundle(bundle)


def test_exported_app_runs_in_isolated_process(bundle):
    """Import strategy.py + run a replay cycle from a subprocess whose cwd is the bundle (no notebook globals)."""
    code = (
        "import sys; sys.path.insert(0, '.');\n"
        "from quantsoc.artifacts import load_bundle; from quantsoc.market import load_market; from quantsoc.broker import ReplayBroker;\n"
        "from quantsoc.engine import TradingEngine; from quantsoc.state import StateStore\n"
        "b = load_bundle('.'); bars = load_market('data/workshop_market.csv'); br = ReplayBroker(bars, start=bars.timestamp.unique()[100])\n"
        "e = TradingEngine(br, b.predict_returns, b.allocate, b.model, b.config, b.symbols, state=StateStore('state'), mode='replay', sleep=lambda s: None)\n"
        "r = e.run_cycle(); print('STATUS', r.status)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=bundle, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "STATUS" in proc.stdout and "error" not in proc.stdout.lower()


def test_strategy_module_has_no_notebook_dependency(tmp_path):
    p = tmp_path / "strategy.py"; p.write_text(S.STRATEGY_PY, encoding="utf-8")
    proc = subprocess.run([sys.executable, "-c", f"import importlib.util as u; s=u.spec_from_file_location('s', r'{p}'); m=u.module_from_spec(s); s.loader.exec_module(m); print(callable(m.predict_returns))"],
                          capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    assert proc.returncode == 0 and "True" in proc.stdout, proc.stderr
