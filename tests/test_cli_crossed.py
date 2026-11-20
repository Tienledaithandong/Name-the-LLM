"""Withdrawn crossed inference must fail before producing or replacing files."""

from __future__ import annotations

import pytest

from src.cli import main


def test_predict_rejects_crossed_before_fitting_or_writing(tmp_path):
    final_dir = tmp_path / "final"
    out = tmp_path / "submission.parquet"
    with pytest.raises(ValueError, match="crossed inference is withdrawn"):
        main([
            "predict", "--synthetic", "--crossed", "--fit",
            "--final-dir", str(final_dir), "--out", str(out),
        ])
    assert not final_dir.exists()
    assert not out.exists()


def test_package_rejects_crossed_before_replacing_existing_output(tmp_path):
    out = tmp_path / "package"
    out.mkdir()
    sentinel = out / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="crossed inference packaging is withdrawn"):
        main(["package", "--crossed", "--out", str(out)])
    assert sentinel.read_text(encoding="utf-8") == "keep"
