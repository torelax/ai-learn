"""Shared path helpers for the paper tools in this directory.

Paper folders follow AGENTS.md: ``docs/<domain>/<topic>/<Name>/`` holding
``<Full Title>_<arxiv>.pdf``, ``zh.md`` and ``figs/``. Every tool accepts a
folder (or file) either relative to the current directory or relative to
``docs/`` so both of these work from anywhere in the repo::

    python3 scripts/papers/shrink_pdf.py docs/embedding/DINO/DINOv3
    python3 scripts/papers/shrink_pdf.py embedding/DINO/DINOv3
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

# "..._2401.00368.pdf" / "..._2401.00368v2.pdf" / old-style "..._hep-th0601001.pdf" is not used here
ARXIV_RE = re.compile(r"_(\d{4}\.\d{4,5})(?:v\d+)?\.pdf$", re.I)


def resolve(arg: str) -> Path | None:
    """Return an existing path for ``arg``: as given, else under docs/."""
    for cand in (Path(arg), DOCS_DIR / arg):
        if cand.exists():
            return cand.resolve()
    return None


def pdfs_in(folder: Path) -> list[Path]:
    """PDFs directly inside a paper folder (not in sub-papers)."""
    return sorted(folder.glob("*.pdf"))


def arxiv_id(folder: Path) -> str | None:
    """arXiv id parsed from the paper PDF's file name, if there is one."""
    for pdf in pdfs_in(folder):
        m = ARXIV_RE.search(pdf.name)
        if m:
            return m.group(1)
    return None


def rel(p: Path) -> str:
    """Repo-relative display path."""
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)
