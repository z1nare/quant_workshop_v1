"""On-disk trading state: the dashboard READS this; only the runner writes it.

    state/trading_state.json   latest snapshot + short equity history
    state/orders.json          registry of client_order_ids we have submitted (duplicate prevention across restarts)
    state/decisions.log        human-readable, append-only log of decisions, skips and errors
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


class StateStore:
    def __init__(self, directory="state", max_history: int = 2000):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.dir / "trading_state.json"
        self.orders_path = self.dir / "orders.json"
        self.log_path = self.dir / "decisions.log"
        self.max_history = max_history
        self._orders: Dict[str, dict] = self._load_json(self.orders_path, {})
        self._history: List[dict] = self._load_json(self.snapshot_path, {}).get("equity_history", [])

    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return default
        return default

    # -- orders registry
    def has_order(self, client_order_id: str) -> bool:
        return client_order_id in self._orders

    def record_order(self, client_order_id: str, info: dict) -> None:
        self._orders[client_order_id] = {**info, "recorded_at": _now_iso()}
        _atomic_write(self.orders_path, json.dumps(self._orders, indent=2, default=str))

    def orders(self) -> Dict[str, dict]:
        return dict(self._orders)

    # -- log
    def log(self, line: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{_now_iso()}  {line}\n")

    def tail_log(self, n: int = 50) -> List[str]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return lines[-n:]

    # -- snapshot
    def write_snapshot(self, snapshot: dict, equity: Optional[float] = None, sim_time: Optional[str] = None) -> None:
        if equity is not None:
            self._history.append({"time": sim_time or _now_iso(), "equity": float(equity)})
            self._history = self._history[-self.max_history:]
        snapshot = {**snapshot, "written_at": _now_iso(), "equity_history": self._history}
        _atomic_write(self.snapshot_path, json.dumps(snapshot, indent=2, default=str))

    def read_snapshot(self) -> dict:
        return self._load_json(self.snapshot_path, {})

    def reset(self) -> None:
        for p in (self.snapshot_path, self.orders_path, self.log_path):
            if p.exists():
                p.unlink()
        self._orders, self._history = {}, []
