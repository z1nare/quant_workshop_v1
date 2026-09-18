#!/usr/bin/env python
"""Set the GitHub owner / repository / revision used by the Colab bootstrap everywhere at once.

    python tools/set_release.py --owner quant-soc --repo first-quant-strategy --revision v1.0.0

Updates quantsoc/release.py, the notebook source (notebooks/workshop_source.py), rebuilds both
notebooks and rewrites the Colab badge/link in README.md.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sub_assign(text: str, name: str, value: str) -> str:
    pat = re.compile(rf'^({name}\s*=\s*)"[^"]*"', re.M)
    if not pat.search(text):
        raise SystemExit(f"could not find {name} = \"...\" assignment")
    return pat.sub(rf'\g<1>"{value}"', text, count=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True, help="tag (recommended), branch or commit SHA")
    a = ap.parse_args(argv)
    for rel in ["quantsoc/release.py", "notebooks/workshop_source.py"]:
        p = ROOT / rel
        t = p.read_text(encoding="utf-8")
        t = sub_assign(t, "OWNER", a.owner); t = sub_assign(t, "REPO", a.repo); t = sub_assign(t, "REVISION", a.revision)
        p.write_text(t, encoding="utf-8", newline="\n")
        print(f"updated {rel}")
    readme = ROOT / "README.md"
    t = readme.read_text(encoding="utf-8")
    t = re.sub(r"https://colab\.research\.google\.com/github/[^/]+/[^/]+/blob/[^/]+/notebooks/workshop_student\.ipynb",
               f"https://colab.research.google.com/github/{a.owner}/{a.repo}/blob/{a.revision}/notebooks/workshop_student.ipynb", t)
    t = re.sub(r"https://github\.com/[^/\s)]+/[^/\s)]+(?=[\s)])", f"https://github.com/{a.owner}/{a.repo}", t)
    readme.write_text(t, encoding="utf-8", newline="\n")
    print("updated README.md links")
    subprocess.check_call([sys.executable, str(ROOT / "notebooks" / "build_notebooks.py")])
    print("rebuilt notebooks. Commit, tag the revision, push, then open the Colab link from README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
