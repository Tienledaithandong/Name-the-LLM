"""Hermetic end-to-end test of the CLI crossed-inference path (hashing encoder)."""

from __future__ import annotations

import pandas as pd

from src.cli import main
from src.constants import LABELS

TINY_CONFIG = """\
seed: 0
models:
  - name: majority
    baseline: majority
  - name: tiny_char
    features: [char]
    classifier: linearsvc
    max_features: 3000
    char_ngram_range: [2, 3]
    min_df: 1
"""


def _write_config(tmp_path):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY_CONFIG, encoding="utf-8")
    return cfg


def test_cli_predict_crossed_writes_valid_submission(tmp_path):
    cfg = _write_config(tmp_path)
    out = tmp_path / "sub.parquet"
    rc = main(
        [
            "predict",
            "--synthetic",
            "--data-dir", str(tmp_path / "data"),
            "--config", str(cfg),
            "--final-dir", str(tmp_path / "final"),
            "--crossed",
            "--encoder-backend", "hashing",
            "--n-clusters", "4",
            "--out", str(out),
        ]
    )
    assert rc == 0
    frame = pd.read_parquet(out)
    assert list(frame.columns) == ["id", "model"]
    assert frame["id"].is_unique
    assert frame["model"].isin(LABELS).all()
    assert not frame.isna().any().any()


def test_cli_predict_crossed_threshold_mode(tmp_path):
    cfg = _write_config(tmp_path)
    out = tmp_path / "sub.parquet"
    rc = main(
        [
            "predict",
            "--synthetic",
            "--data-dir", str(tmp_path / "data"),
            "--config", str(cfg),
            "--final-dir", str(tmp_path / "final"),
            "--crossed",
            "--encoder-backend", "hashing",
            "--distance-threshold", "0.5",
            "--out", str(out),
        ]
    )
    assert rc == 0
    frame = pd.read_parquet(out)
    assert frame["model"].isin(LABELS).all()