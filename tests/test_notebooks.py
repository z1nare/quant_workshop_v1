"""Notebook structure, synchronisation and (slow) clean-kernel execution tests."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
STUDENT, INSTRUCTOR = NB / "workshop_student.ipynb", NB / "workshop_instructor.ipynb"
EXPECTED_EXERCISES = ["1", "2", "3", "4", "5", "6a", "6b"]


def _cells(path):
    return nbformat.read(path, as_version=4).cells


def test_notebooks_in_sync_with_source():
    r = subprocess.run([sys.executable, str(NB / "build_notebooks.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_both_notebooks_have_all_exercises_in_order():
    for path in (STUDENT, INSTRUCTOR):
        keys = [c.metadata.get("exercise") for c in _cells(path) if c.metadata.get("exercise")]
        assert keys == EXPECTED_EXERCISES, (path.name, keys)


def test_student_stubs_and_instructor_solutions():
    s = {c.metadata["exercise"]: c.source for c in _cells(STUDENT) if c.metadata.get("exercise")}
    i = {c.metadata["exercise"]: c.source for c in _cells(INSTRUCTOR) if c.metadata.get("exercise")}
    for k in EXPECTED_EXERCISES:
        assert "YOUR CODE" in s[k], k
        assert "YOUR CODE" not in i[k] and "NotImplementedError" not in i[k], k
        assert s[k] != i[k]
    # the student notebook must not carry presenter notes or solutions
    student_text = "\n".join(c.source for c in _cells(STUDENT))
    assert "Presenter notes" not in student_text and "🎤" not in student_text
    assert any(c.metadata.get("instructor_note") for c in _cells(INSTRUCTOR))


def test_cell_count_near_target():
    n = len([c for c in _cells(STUDENT)])
    assert 40 <= n <= 60, n


def test_no_unguarded_order_submission_in_notebooks():
    """Run-all must never submit: submit=True may only appear behind the explicit flag or inside the widget panel."""
    for path in (STUDENT, INSTRUCTOR):
        for c in _cells(path):
            if c.cell_type != "code":
                continue
            for line in c.source.splitlines():
                if "run_cycle(submit=True)" in line:
                    assert "SUBMIT_FOR_REAL" in c.source and "SUBMIT_FOR_REAL = False" in c.source, (path.name, line)
    for path in (STUDENT, INSTRUCTOR):
        text = "\n".join(c.source for c in _cells(path))
        assert "submit_order(" not in text and "MarketOrderRequest" not in text


def test_no_credentials_or_outputs_committed():
    for path in (STUDENT, INSTRUCTOR):
        raw = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(PK|AK)[A-Z0-9]{16,}\b", raw)
        for c in _cells(path):
            if c.cell_type == "code":
                assert not c.get("outputs"), f"{path.name} has committed outputs"


def test_release_settings_centralised():
    from quantsoc import release
    src = (NB / "workshop_source.py").read_text(encoding="utf-8")
    for name in ("OWNER", "REPO", "REVISION"):
        m = re.search(rf'^{name} = "([^"]+)"', src, re.M)
        assert m and m.group(1) == getattr(release, name), name


def test_setup_cell_uses_attached_kaggle_dataset(tmp_path):
    """On Kaggle the setup cell copies the workshop from an attached dataset (no internet needed) and is idempotent."""
    import shutil
    kin, work = tmp_path / "kaggle" / "input", tmp_path / "kaggle" / "working"
    ds = kin / "workshop-dataset" / "quant_workshop_v1-main"            # a dataset made from the GitHub ZIP
    for item in ("quantsoc", "data"):
        shutil.copytree(ROOT / item, ds / item, ignore=shutil.ignore_patterns("__pycache__"))
    for item in ("strategy.py", "requirements-colab.txt"):
        shutil.copy(ROOT / item, ds / item)
    work.mkdir(parents=True)
    setup = next(c.source for c in _cells(STUDENT) if c.cell_type == "code" and "IN_KAGGLE" in c.source)
    assert '"/kaggle/input"' in setup and '"/kaggle/working/workshop"' in setup
    script = tmp_path / "setup_cell.py"
    script.write_text(setup.replace('"/kaggle/input"', repr(kin.as_posix()))
                      .replace('"/kaggle/working/workshop"', repr((work / "workshop").as_posix())), encoding="utf-8")
    env = {**os.environ, "KAGGLE_KERNEL_RUN_TYPE": "Interactive"}
    first = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, cwd=work, timeout=300)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "Using the workshop files attached" in first.stdout and re.search(r"running in:\s+Kaggle", first.stdout)
    assert (work / "workshop" / "quantsoc" / "__init__.py").exists() and (work / "workshop" / "data").is_dir()
    again = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, cwd=work, timeout=300)
    assert again.returncode == 0 and "Using the workshop files" not in again.stdout    # second run changes nothing


def _run(nb_path, env=None, allow_errors=False, out=None):
    import shutil
    shutil.rmtree(ROOT / "workspace", ignore_errors=True)          # each run starts from a clean workspace
    cmd = [sys.executable, str(ROOT / "tools" / "run_notebook.py"), str(nb_path), "--watchdog", "600"]
    if allow_errors:
        cmd.append("--allow-errors")
    if out:
        cmd += ["--out", str(out)]
    e = {**os.environ, "MPLBACKEND": "Agg", **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, env=e, cwd=ROOT, timeout=1800)


@pytest.mark.slow
def test_instructor_notebook_runs_clean(tmp_path):
    r = _run(INSTRUCTOR, out=tmp_path / "ins.ipynb")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    nb = nbformat.read(tmp_path / "ins.ipynb", as_version=4)
    text = "\n".join(str(o.get("text", "")) for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert text.count("[ok]") >= 7 and "[ref]" not in text
    assert "research and exported implementation are equivalent: True" in text
    assert "replay equity == backtest equity" in text and "True" in text
    assert "no Alpaca credentials found" in text or "OFFLINE" in text          # offline in CI


@pytest.mark.slow
def test_student_recovery_route_runs_clean(tmp_path):
    r = _run(STUDENT, env={"QSW_AUTO_REFERENCE": "1"}, out=tmp_path / "stu.ipynb")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    nb = nbformat.read(tmp_path / "stu.ipynb", as_version=4)
    text = "\n".join(str(o.get("text", "")) for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    # 6a's reference writes the complete strategy.py, so 6b then passes on its own: 6 [ref] banners, not 7
    assert text.count("[ref]") >= 6 and "strategy source: REFERENCE" in text


@pytest.mark.slow
def test_untouched_student_notebook_stops_clearly(tmp_path):
    """Run-all must stop at the first checkpoint with ExerciseIncomplete (friendly one-liner, no traceback cascade)."""
    r = _run(STUDENT, out=tmp_path / "raw.ipynb")
    assert r.returncode != 0 and "ExerciseIncomplete" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]
    nb = nbformat.read(tmp_path / "raw.ipynb", as_version=4)
    text = "\n".join(str(o.get("text", "")) for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert "Not started yet" in text and "[stop] Exercise 1" in text
    assert "[ok]" not in text and "ZIP written" not in text                     # never reports completion / exports
    # cell-by-cell route: every later dependent cell must also stop with the friendly message, never a NameError
    r2 = _run(STUDENT, allow_errors=True, out=tmp_path / "raw2.ipynb")
    nb2 = nbformat.read(tmp_path / "raw2.ipynb", as_version=4)
    bad = [(i, o["ename"]) for i, c in enumerate(nb2.cells) if c.cell_type == "code"
           for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert bad == [], bad
    text2 = "\n".join(str(o.get("text", "")) for c in nb2.cells if c.cell_type == "code" for o in c.get("outputs", []))
    assert text2.count("[stop]") >= 10 and "NameError" not in text2
