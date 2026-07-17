"""Reproducibility helpers: results manifest + claims verification.

`write_manifest()` hashes every file under `results/` into
`results/MANIFEST.sha256`; `check_manifest()` re-verifies it. Notebook
assertions against CLAIMS.md are added from Stage 2 onward.
"""

from __future__ import annotations

from pathlib import Path

from .data import PROJECT_ROOT, sha256_of

RESULTS_DIR = PROJECT_ROOT / "results"
MANIFEST_PATH = RESULTS_DIR / "MANIFEST.sha256"


def _manifest_files() -> list[Path]:
    return sorted(
        p for p in RESULTS_DIR.rglob("*")
        if p.is_file() and p != MANIFEST_PATH
    )


def write_manifest() -> Path:
    lines = [
        f"{sha256_of(p)}  {p.relative_to(RESULTS_DIR).as_posix()}"
        for p in _manifest_files()
    ]
    MANIFEST_PATH.write_text("\n".join(lines) + "\n" if lines else "")
    return MANIFEST_PATH


def check_manifest() -> bool:
    if not MANIFEST_PATH.exists():
        return False
    recorded = {}
    for line in MANIFEST_PATH.read_text().splitlines():
        digest, _, rel = line.partition("  ")
        recorded[rel] = digest
    actual = {
        p.relative_to(RESULTS_DIR).as_posix(): sha256_of(p)
        for p in _manifest_files()
    }
    return recorded == actual
