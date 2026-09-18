"""Build the reference strategy bundle used by `run_trader.py` when no student bundle is given.

The model is fitted once on the TRAINING split of the checked-in workshop data (about a second),
so no pickled model needs to live in the repository.
"""
from __future__ import annotations

from pathlib import Path

from . import solutions as S
from .artifacts import REPO_ROOT, export_bundle
from .market import SYMBOLS, load_market
from .modeling import build_dataset, clean_rows, fit_model


def build_reference_bundle(out_dir=None, data_path=None, repo_root: Path = REPO_ROOT) -> Path:
    out_dir = Path(out_dir or repo_root / "exports" / "reference_bundle")
    data_path = Path(data_path or repo_root / "data" / "workshop_market.csv")
    bars = load_market(data_path)
    dataset = build_dataset(bars)
    model = fit_model(clean_rows(dataset, "train"))
    strategy_path = repo_root / "strategy.py"
    export_bundle(out_dir, strategy_path, model, S.REFERENCE_CONFIG, SYMBOLS, strategy_source="reference",
                  exercise_status={k: "reference" for k in ["1", "2", "3", "4", "5", "6a", "6b"]}, fund_name="Reference Fund",
                  repo_root=repo_root, extra_notes="Built by quantsoc.reference_bundle (instructor reference).")
    return out_dir
