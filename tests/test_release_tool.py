"""tools/set_release.py rewrites every place the release settings live, on a temporary copy of the repo."""
import re
import shutil
import subprocess
import sys
from pathlib import Path


def test_set_release_updates_all_locations(tmp_path, repo_root):
    for rel in ["quantsoc", "notebooks", "tools", "README.md"]:
        src = repo_root / rel
        (shutil.copytree if src.is_dir() else shutil.copy)(src, tmp_path / rel)
    for p in (tmp_path / "notebooks").glob("*_executed.ipynb"):
        p.unlink()
    r = subprocess.run([sys.executable, str(tmp_path / "tools" / "set_release.py"), "--owner", "quant-soc",
                        "--repo", "first-quant-strategy", "--revision", "v9.9.9"], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    rel = (tmp_path / "quantsoc" / "release.py").read_text(encoding="utf-8")
    assert 'OWNER = "quant-soc"' in rel and 'REVISION = "v9.9.9"' in rel
    src = (tmp_path / "notebooks" / "workshop_source.py").read_text(encoding="utf-8")
    assert 'OWNER = "quant-soc"' in src
    nb = (tmp_path / "notebooks" / "workshop_student.ipynb").read_text(encoding="utf-8")
    assert '\\"quant-soc\\", \\"first-quant-strategy\\", \\"v9.9.9\\"' in nb
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "colab.research.google.com/github/quant-soc/first-quant-strategy/blob/v9.9.9/notebooks/workshop_student.ipynb" in readme
    assert "YOUR-GITHUB-ORG" not in readme
    # the generated notebooks are in sync afterwards
    r2 = subprocess.run([sys.executable, str(tmp_path / "notebooks" / "build_notebooks.py"), "--check"], capture_output=True, text=True)
    assert r2.returncode == 0, r2.stdout
