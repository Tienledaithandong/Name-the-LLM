"""Reproducibility helpers: seeds, git commit, environment snapshot."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def set_seeds(seed: int) -> None:
    np.random.seed(int(seed))


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_worktree_status(cwd: str | Path | None = None) -> str:
    """Report tracked and untracked non-ignored changes that make HEAD incomplete."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(cwd) if cwd else None, capture_output=True, text=True,
            timeout=10, check=False,
        )
        if out.returncode != 0:
            return "unknown"
        return "dirty" if out.stdout.strip() else "clean"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def env_snapshot() -> dict:
    snapshot = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    for module in ("numpy", "pandas", "scipy", "sklearn", "pyarrow", "yaml", "joblib"):
        try:
            imported = __import__(module)
            snapshot[module] = getattr(imported, "__version__", "unknown")
        except ImportError:
            snapshot[module] = "missing"
    try:
        import psutil

        process = psutil.Process()
        snapshot["cpu_count"] = psutil.cpu_count(logical=True)
        snapshot["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        snapshot["rss_gb"] = round(process.memory_info().rss / 1024**3, 3)
    except Exception:  # pragma: no cover - psutil optional at runtime
        pass
    return snapshot
