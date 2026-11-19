"""Experiment provenance must not present a dirty tree as an exact commit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import repro


@pytest.mark.parametrize("stdout,code,expected", [("", 0, "clean"), (" M src/model.py\n", 0, "dirty"), ("", 1, "unknown")])
def test_git_worktree_status(monkeypatch, stdout: str, code: int, expected: str) -> None:
    monkeypatch.setattr(repro.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=stdout, returncode=code))
    assert repro.git_worktree_status() == expected
