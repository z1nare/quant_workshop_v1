"""Export and import of a strategy bundle: code + fitted model + configuration + metadata.

Bundle layout (a directory, optionally zipped):

    strategy.py          the participant's actual strategy module (or the labelled reference copy)
    model.joblib         fitted scikit-learn pipeline (StandardScaler + LinearRegression)
    config.json          portfolio configuration, symbols, feature names/order, timeframe, schema version
    manifest.json        provenance: created_at, library versions, strategy_source, exercise status
    README_RUN_LOCAL.md  how to run the exported application locally
    quantsoc/            copy of this package (so strategy.py's imports work)
    run_trader.py, dashboard.py, requirements.txt, .env.example, data/*.csv

Secrets are never written: the exporter refuses files that look like credentials and
scans strategy.py for pasted API keys.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import types
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

from . import SCHEMA_VERSION, __version__
from .features import FEATURE_NAMES, TIMEFRAME
from .portfolio import PortfolioConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_FILES = ["strategy.py", "model.joblib", "config.json", "manifest.json", "README_RUN_LOCAL.md",
                "run_trader.py", "dashboard.py", "requirements.txt", ".env.example"]
EXCLUDED_NAMES = {".env", "state", "logs", "__pycache__", ".ipynb_checkpoints", ".git", ".venv"}
SECRET_PATTERNS = [
    re.compile(r"APCA[_A-Z]*\s*=\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"\b(?:PK|AK)[A-Z0-9]{16,}\b"),                 # Alpaca-style key ids
    re.compile(r"(?i)(secret|api_key|apikey|token)\s*=\s*['\"][A-Za-z0-9/+_\-]{20,}['\"]"),
]


def scan_for_secrets(text: str) -> List[str]:
    hits = []
    for pat in SECRET_PATTERNS:
        for m in pat.finditer(text):
            hits.append(m.group(0)[:12] + "...")
    return hits


def _versions() -> Dict[str, str]:
    import numpy, sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "quantsoc": __version__}


def write_config(path: Path, config: PortfolioConfig, symbols: List[str]) -> dict:
    data = {
        "schema_version": SCHEMA_VERSION,
        "timeframe": TIMEFRAME,
        "feature_names": list(FEATURE_NAMES),
        "symbols": list(symbols),
        "portfolio": config.validate().to_dict(),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def export_bundle(out_dir, strategy_path, model, config: PortfolioConfig, symbols: List[str],
                  strategy_source: str = "student", exercise_status: Optional[dict] = None,
                  fund_name: str = "My First Fund", repo_root: Path = REPO_ROOT,
                  data_files: Optional[List[Path]] = None, extra_notes: str = "") -> Path:
    """Write a complete, runnable bundle directory.  `strategy_source` must be 'student' or 'reference'."""
    if strategy_source not in ("student", "reference"):
        raise ValueError("strategy_source must be 'student' or 'reference'")
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    strategy_path = Path(strategy_path)
    src = strategy_path.read_text(encoding="utf-8")
    hits = scan_for_secrets(src)
    if hits:
        raise ValueError(f"strategy.py appears to contain credentials ({hits}); remove them before exporting")
    if "def predict_returns" not in src or "def allocate" not in src:
        raise ValueError("strategy.py must define predict_returns and allocate")
    (out / "strategy.py").write_text(src, encoding="utf-8")
    joblib.dump(model, out / "model.joblib")
    cfg = write_config(out / "config.json", config, symbols)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fund_name": fund_name,
        "strategy_source": strategy_source,
        "strategy_source_note": ("Reference solution supplied by the workshop (NOT the participant's own code)"
                                 if strategy_source == "reference" else "Participant's own strategy.py"),
        "exercise_status": exercise_status or {},
        "versions": _versions(),
        "feature_names": list(FEATURE_NAMES),
        "timeframe": TIMEFRAME,
        "notes": extra_notes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # application files
    for name in ["run_trader.py", "dashboard.py", "requirements.txt", ".env.example", "generate_data.py"]:
        p = repo_root / name
        if p.exists():
            shutil.copy(p, out / name)
    shutil.copytree(repo_root / "quantsoc", out / "quantsoc",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (out / "data").mkdir()
    for p in (data_files or sorted((repo_root / "data").glob("*.csv"))):
        shutil.copy(p, out / "data" / Path(p).name)
    (out / "README_RUN_LOCAL.md").write_text(_local_readme(fund_name, strategy_source, cfg), encoding="utf-8")
    for bad in EXCLUDED_NAMES:
        if (out / bad).exists():
            shutil.rmtree(out / bad, ignore_errors=True)
    return out


def zip_bundle(bundle_dir, zip_path) -> Path:
    bundle_dir, zip_path = Path(bundle_dir), Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(bundle_dir.rglob("*")):
            rel = p.relative_to(bundle_dir)
            if any(part in EXCLUDED_NAMES for part in rel.parts) or p.suffix == ".pyc":
                continue
            if p.is_file():
                zf.write(p, str(Path(bundle_dir.name) / rel))
    return zip_path


def check_zip(zip_path) -> Dict[str, object]:
    """Report which required files are present and that nothing secret-looking is inside."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        stripped = ["/".join(n.split("/")[1:]) for n in names]
        forbidden = [n for n in stripped if n == ".env" or n.startswith("state/") or n.startswith("logs/")]
        secret_hits = []
        for n in names:
            if n.endswith((".py", ".json", ".md", ".txt", ".example")):
                secret_hits += scan_for_secrets(zf.read(n).decode("utf-8", "ignore"))
    missing = [f for f in BUNDLE_FILES if f not in stripped]
    has_pkg = any(s.startswith("quantsoc/") for s in stripped)
    has_data = any(s.startswith("data/") and s.endswith(".csv") for s in stripped)
    return {"missing": missing, "forbidden_present": forbidden, "secret_hits": secret_hits,
            "package_included": has_pkg, "data_included": has_data,
            "ok": not missing and not forbidden and not secret_hits and has_pkg and has_data}


def _local_readme(fund_name: str, source: str, cfg: dict) -> str:
    label = "your own strategy.py" if source == "student" else "the workshop REFERENCE strategy (not your own code)"
    return f"""# {fund_name} - run your exported strategy locally

This folder was exported from the QuantSoc workshop notebook. It contains {label},
the fitted model (`model.joblib`), your portfolio configuration (`config.json`) and the
supporting application code.

## 1. Install (once)

Windows (PowerShell):

    py -3.12 -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt

macOS / Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## 2. Offline replay (no account needed)

    python run_trader.py --mode replay --bundle . --speed 20

Then, in a second terminal (same virtual environment):

    streamlit run dashboard.py

The dashboard opens at http://localhost:8501 and reads `state/trading_state.json`.

## 3. Optional: Alpaca paper trading

Copy `.env.example` to `.env`, paste your PAPER keys there (never into code), then:

    python run_trader.py --mode paper-preview --bundle .        # shows proposed orders, submits nothing
    python run_trader.py --mode paper --bundle . --i-understand-paper-orders

Configuration used: threshold={cfg['portfolio']['threshold']}, max_positions={cfg['portfolio']['max_positions']},
exposure_budget={cfg['portfolio']['exposure_budget']}, per_asset_cap={cfg['portfolio']['per_asset_cap']},
cost_bps={cfg['portfolio']['cost_bps']}; features={cfg['feature_names']}; timeframe={cfg['timeframe']}.

This is a teaching project. Nothing here is a validated real-market investment strategy.
"""


@dataclass
class StrategyBundle:
    path: Path
    module: types.ModuleType
    model: object
    config: PortfolioConfig
    symbols: List[str]
    manifest: dict
    raw_config: dict

    @property
    def predict_returns(self):
        return self.module.predict_returns

    @property
    def allocate(self):
        return self.module.allocate

    @property
    def strategy_source(self) -> str:
        return self.manifest.get("strategy_source", "unknown")


def load_strategy_module(path, name: str = "exported_strategy") -> types.ModuleType:
    """Import a strategy file by path (transparent, no source extraction from function objects)."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for fn in ("predict_returns", "allocate"):
        if not callable(getattr(module, fn, None)):
            raise ValueError(f"{path} must define a callable `{fn}`")
    return module


def load_bundle(bundle_dir) -> StrategyBundle:
    d = Path(bundle_dir)
    raw = json.loads((d / "config.json").read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"bundle schema {raw.get('schema_version')} != supported {SCHEMA_VERSION}")
    if list(raw.get("feature_names", [])) != list(FEATURE_NAMES):
        raise ValueError(f"bundle feature order {raw.get('feature_names')} != library {FEATURE_NAMES}")
    if raw.get("timeframe") != TIMEFRAME:
        raise ValueError(f"bundle timeframe {raw.get('timeframe')} != {TIMEFRAME}")
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}
    module = load_strategy_module(d / "strategy.py", name=f"strategy_{abs(hash(str(d.resolve())))}")
    model = joblib.load(d / "model.joblib")
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and n_in != len(FEATURE_NAMES):
        raise ValueError(f"model expects {n_in} features, library has {len(FEATURE_NAMES)}")
    return StrategyBundle(path=d, module=module, model=model, config=PortfolioConfig.from_dict(raw["portfolio"]),
                          symbols=list(raw["symbols"]), manifest=manifest, raw_config=raw)
