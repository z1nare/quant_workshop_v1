"""Notebook plumbing: environment checks, credentials, truthful mode labels and the export checkpoint.

Credentials are read from (in order) Colab or Kaggle Secrets, environment variables, a git-ignored .env file.
They are never printed and never written into the export.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from .exercises import ExerciseIncomplete

CRED_KEYS = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
DEFAULT_SYMBOL_MAP = {"AURA": "SPY", "BOLT": "QQQ", "CRUX": "IWM", "DUNE": "DIA", "ECHO": "XLK"}


def is_colab() -> bool:
    return "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG") is not None


def is_kaggle() -> bool:
    return not is_colab() and ("KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").is_dir())


def runtime_name() -> str:
    return "Google Colab" if is_colab() else "Kaggle" if is_kaggle() else "local Jupyter"


def setup_ipython() -> None:
    """Show ExerciseIncomplete as a short message instead of a long traceback (IPython/Colab only)."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return

        def handler(shell, etype, evalue, tb, tb_offset=None):
            print(f"\n[stop] {evalue}\n")
            return []
        ip.set_custom_exc((ExerciseIncomplete,), handler)
    except Exception:
        pass


def environment_report() -> pd.DataFrame:
    import numpy, sklearn, matplotlib
    rows = [("python", platform.python_version()), ("numpy", numpy.__version__), ("pandas", pd.__version__),
            ("scikit-learn", sklearn.__version__), ("matplotlib", matplotlib.__version__)]
    try:
        import ipywidgets
        rows.append(("ipywidgets", ipywidgets.__version__))
    except Exception:
        rows.append(("ipywidgets", "NOT AVAILABLE -> static fallbacks will be used"))
    try:
        import importlib.metadata as m
        rows.append(("alpaca-py", m.version("alpaca-py")))
    except Exception:
        rows.append(("alpaca-py", "not installed (only needed for the optional paper demo)"))
    rows.append(("runtime", runtime_name()))
    return pd.DataFrame(rows, columns=["component", "version"]).set_index("component")


def get_alpaca_credentials(dotenv_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str], str]:
    """Return (key, secret, source).  Values are never printed by this function."""
    if is_colab():
        try:
            from google.colab import userdata
            key, secret = userdata.get(CRED_KEYS[0]), userdata.get(CRED_KEYS[1])
            if key and secret:
                return key, secret, "Colab Secrets"
        except Exception:
            pass
    if is_kaggle():
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
            key, secret = client.get_secret(CRED_KEYS[0]), client.get_secret(CRED_KEYS[1])
            if key and secret:
                return key, secret, "Kaggle Secrets"
        except Exception:
            pass
    key, secret = os.environ.get(CRED_KEYS[0]), os.environ.get(CRED_KEYS[1])
    if key and secret:
        return key, secret, "environment variables"
    path = Path(dotenv_path or ".env")
    if path.exists():
        try:
            from dotenv import dotenv_values
            vals = dotenv_values(path)
            key, secret = vals.get(CRED_KEYS[0]), vals.get(CRED_KEYS[1])
            if key and secret:
                return key, secret, f"{path} file"
        except Exception:
            pass
    return None, None, "not found"


def symbol_map_from_env() -> Dict[str, str]:
    raw = os.environ.get("QSW_SYMBOL_MAP", "")
    if not raw:
        return dict(DEFAULT_SYMBOL_MAP)
    out = {}
    for pair in raw.split(","):
        k, v = pair.split(":")
        out[k.strip()] = v.strip().upper()
    return out


def mode_banner(mode: str, detail: str = "") -> None:
    labels = {
        "offline": "OFFLINE SIMULATION - synthetic teaching market, no broker involved",
        "replay": "OFFLINE REPLAY - synthetic data, simulated fills, no broker involved",
        "paper-preview": "ALPACA PAPER PREVIEW - real paper account data, NO orders submitted",
        "paper": "ALPACA PAPER TRADING - orders go to the paper endpoint only (no real money)",
        "mock": "MOCKED BROKER - scripted responses for demonstration, no network",
    }
    line = "=" * 78
    print(f"{line}\n  MODE: {labels.get(mode, mode)}\n  {detail}\n{line}" if detail else f"{line}\n  MODE: {labels.get(mode, mode)}\n{line}")


def runtime_loss_notice() -> None:
    print(f"Reminder: {runtime_name()} sessions are temporary. Files you create here (strategy.py, model.joblib, exports)\n"
          "disappear when the session ends. Saving the notebook does NOT save those files.\n"
          "Use the export checkpoint cell to download a ZIP before you stop.")


def export_checkpoint(bundle_dir, zip_name: str, download: bool = True) -> Path:
    """Zip a bundle and, in Colab, trigger the browser download."""
    from .artifacts import zip_bundle, check_zip
    zip_path = Path(zip_name).with_suffix(".zip")
    zip_bundle(bundle_dir, zip_path)
    report = check_zip(zip_path)
    print(f"ZIP written: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    print(f"contents check: {'OK' if report['ok'] else 'PROBLEM'} {report}")
    if download and is_colab():
        try:
            from google.colab import files
            files.download(str(zip_path))
        except Exception as e:
            print(f"Automatic download failed ({e}); use the Files pane on the left to download {zip_path}.")
    elif download and is_kaggle():
        try:
            where = zip_path.resolve().relative_to("/kaggle/working")
        except ValueError:
            where = zip_path.resolve()
        print(f"Kaggle: download the ZIP now. In the right-hand panel, under Output (/kaggle/working), open {where}\n"
              "and use the file's ⋮ menu → Download. Files in this session are deleted when it ends.")
    elif download:
        print("Not running in Colab: the ZIP is saved next to this notebook.")
    return zip_path


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Locate the repository root from a notebook (works locally and after the Colab bootstrap)."""
    here = Path(start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "quantsoc" / "__init__.py").exists() and (p / "data").exists():
            return p
    raise FileNotFoundError("could not find the repository root (quantsoc/ + data/). Run the setup cell first.")
