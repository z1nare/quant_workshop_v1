"""Central place for the GitHub owner / repository / revision used by the Colab bootstrap.

Nothing else in the repository should hard-code these values.  Use

    python tools/set_release.py --owner ORG --repo REPO --revision v1.0.0

to update this file, the README and the notebook bootstrap cell in one go.
"""

# --- release settings (edited by tools/set_release.py) -----------------------
OWNER = "z1nare"                   # GitHub account that hosts the public repository
REPO = "quant_workshop_v1"         # repository name
REVISION = "v1.0.0"                # tag, branch or commit used for reproducible downloads
# -----------------------------------------------------------------------------

DATA_FILES = ["data/workshop_market.csv", "data/stress_market.csv"]


def is_placeholder() -> bool:
    return OWNER.startswith("YOUR-")


def archive_url() -> str:
    """ZIP archive of the pinned revision (works for tags, branches and commits)."""
    return f"https://github.com/{OWNER}/{REPO}/archive/{REVISION}.zip"


def raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{REVISION}/{path}"


def colab_url(notebook: str = "notebooks/workshop_student.ipynb") -> str:
    return f"https://colab.research.google.com/github/{OWNER}/{REPO}/blob/{REVISION}/{notebook}"
