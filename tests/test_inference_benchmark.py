"""The cold-start benchmark must generate an exact-size synthetic frame and fail loudly.

Plan §8 (L208) asks for a full-size inference benchmark with cold start and model load.
These tests pin the frame builder (exact row count, official test schema, unique ids, no
nulls, determinism) and the subprocess wrapper's error propagation, so a broken package
cannot be reported as a successful timing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "inference_benchmark.py"
_spec = importlib.util.spec_from_file_location("inference_benchmark", SCRIPT)
bench = importlib.util.module_from_spec(_spec)
sys.modules["inference_benchmark"] = bench
_spec.loader.exec_module(bench)


def test_frame_has_the_requested_size_and_official_test_schema():
    frame = bench.synthetic_test_frame(250, seed=7)
    assert len(frame) == 250
    assert list(frame.columns) == ["id", "text"]
    assert frame["id"].is_unique
    assert frame["text"].notna().all()
    assert (frame["text"].astype(str).str.len() > 0).all()


def test_frame_is_deterministic_for_a_fixed_seed():
    first = bench.synthetic_test_frame(120, seed=42)
    second = bench.synthetic_test_frame(120, seed=42)
    assert first.equals(second)


def test_odd_small_size_is_still_exact():
    assert len(bench.synthetic_test_frame(37, seed=3)) == 37


def test_non_positive_size_is_rejected():
    with pytest.raises(ValueError, match="n_rows must be positive"):
        bench.synthetic_test_frame(0)


def test_cold_start_failure_propagates_with_the_worker_stderr(tmp_path):
    test_csv = tmp_path / "test.csv"
    bench.synthetic_test_frame(5, seed=1).to_csv(test_csv, index=False)
    empty_package = tmp_path / "no_artifacts"
    empty_package.mkdir()
    with pytest.raises(RuntimeError, match="cold-start worker failed"):
        bench.run_cold_start(empty_package, test_csv, tmp_path / "work")