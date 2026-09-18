#!/usr/bin/env python
"""Execute a workshop notebook from a clean kernel (offline) and report per-cell timings.

    python tools/run_notebook.py notebooks/workshop_instructor.ipynb
    QSW_AUTO_REFERENCE=1 python tools/run_notebook.py notebooks/workshop_student.ipynb   # recovery route
    python tools/run_notebook.py notebooks/workshop_student.ipynb --allow-errors          # keep going after errors

Without --allow-errors, execution stops at the first cell whose kernel reply is an error (even when the
notebook's custom exception handler prints a short message instead of a traceback) and the exit code is 1.
Writes an executed copy next to the input as *_executed.ipynb (git-ignored) and prints a summary.
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient


class StopAtFirstErrorClient(NotebookClient):
    """Record the first error reply and skip the remaining cells instead of raising through kernel cleanup."""

    stop_at = None

    async def _check_raise_for_error(self, cell, cell_index, exec_reply):
        if exec_reply is not None and exec_reply.get("content", {}).get("status") == "error":
            c = exec_reply["content"]
            if self.stop_at is None:
                self.stop_at = (cell_index, c.get("ename", "?"), c.get("evalue", ""))
        # never raise: the caller inspects stop_at

    async def async_execute_cell(self, cell, cell_index, *args, **kwargs):
        if self.stop_at is not None and not self.allow_errors:
            return cell
        return await super().async_execute_cell(cell, cell_index, *args, **kwargs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--allow-errors", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=None)
    ap.add_argument("--watchdog", type=int, default=900, help="hard exit after this many seconds (kernel hangs)")
    a = ap.parse_args(argv)
    faulthandler.dump_traceback_later(max(30, a.watchdog - 30), repeat=False, file=sys.stderr)
    if hasattr(signal, "SIGALRM"):
        def _watchdog(signum, frame):
            print(f"WATCHDOG: notebook execution exceeded {a.watchdog}s; exiting", flush=True)
            os._exit(3)
        signal.signal(signal.SIGALRM, _watchdog)
        signal.alarm(a.watchdog)

    path = Path(a.notebook).resolve()
    nb = nbformat.read(path, as_version=4)
    os.environ.setdefault("MPLBACKEND", "Agg")
    client = StopAtFirstErrorClient(nb, timeout=a.timeout, kernel_name="python3", allow_errors=a.allow_errors,
                                    resources={"metadata": {"path": str(path.parent)}}, record_timing=True)
    t0 = time.time()
    client.execute()
    total = time.time() - t0
    out = Path(a.out) if a.out else path.with_name(path.stem + "_executed.ipynb")
    nbformat.write(nb, out)
    print(f"executed {path.name} in {total:.1f}s -> {out.name}")
    slow, errors = [], []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        tm = c.metadata.get("execution", {})
        if "iopub.execute_input" in tm and "shell.execute_reply" in tm:
            dt = (datetime.fromisoformat(tm["shell.execute_reply"].replace("Z", "+00:00"))
                  - datetime.fromisoformat(tm["iopub.execute_input"].replace("Z", "+00:00"))).total_seconds()
            if dt > 3:
                slow.append((i, dt))
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors.append((i, o.get("ename"), (o.get("evalue") or "")[:160]))
    print("cells slower than 3s:", ", ".join(f"#{i} {s:.1f}s" for i, s in slow) or "none")
    if errors:
        print("cells with error outputs:")
        for i, name, val in errors:
            print(f"  #{i}: {name}: {val}")
    if client.stop_at is not None:
        i, name, val = client.stop_at
        print(f"execution stopped at cell #{i}: {name}: {val}")
    rc = 0 if (client.stop_at is None or a.allow_errors) else 1
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    sys.exit(main())
