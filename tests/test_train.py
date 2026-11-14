from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd

from src.constants import LABELS, N_CLASSES
from src.inference import fit_final, load_final, predict_dataframe
from src.submission import build_submission, validate_submission
from src.train import REGISTRY_COLUMNS, run_cv


def test_cv_run_writes_oof_metrics_and_registry(
    train_df, folds, tiny_char_model, tmp_path
):
    registry = tmp_path / "registry.csv"
    style_model = {
        "name": "style_only",
        "features": [],
        "style": True,
        "classifier": "logreg",
        "C": 1.0,
    }
    results = run_cv(
        train_df,
        folds,
        [{"name": "majority", "baseline": "majority"}, tiny_char_model, style_model],
        tmp_path / "cv",
        seed=0,
        dataset_label="synthetic",
        registry_path=registry,
        split_id="test-split",
    )
    assert len(results) == 3

    oof = np.load(tmp_path / "cv" / "oof" / "tiny_char.npz", allow_pickle=True)
    assert oof["scores"].shape == (len(train_df), N_CLASSES)
    assert oof["y"].shape == (len(train_df),)
    assert [str(v) for v in oof["labels"]] == list(LABELS)
    assert np.isfinite(oof["scores"]).all()
    # every row must have received an out-of-fold prediction
    assert (oof["scores"].max(axis=1) > -1e8).all()

    metrics = json.loads(
        (tmp_path / "cv" / "metrics" / "tiny_char.json").read_text(encoding="utf-8")
    )
    assert metrics["split_id"] == "test-split"
    assert len(metrics["folds"]) == len(folds)
    assert 0.0 <= metrics["pooled"]["balanced_accuracy"] <= 1.0
    assert (tmp_path / "cv" / "summary.md").is_file()

    with registry.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert list(rows[0]) == REGISTRY_COLUMNS
    assert all(row["external_data_id"] == "synthetic-fixture" for row in rows)
    assert all("synthetic" in row["notes"] for row in rows)

    # The fixture encodes style, so the style model must clearly beat chance.
    # (Sparse char n-grams mostly memorise prompt content here, which grouped
    # folds correctly refuse to reward.)
    by_name = {r["name"]: r["pooled"]["balanced_accuracy"] for r in results}
    assert by_name["style_only"] > 0.2
    assert by_name["tiny_char"] > 1.0 / N_CLASSES


def test_fit_final_and_predict_produces_valid_submission(
    train_df, test_df, tiny_char_model, tmp_path
):
    final_dir = tmp_path / "final"
    manifest_path = fit_final(
        [{"name": "majority", "baseline": "majority"}, tiny_char_model],
        train_df,
        final_dir,
        seed=0,
        dataset_label="synthetic",
    )
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["class_order"] == list(LABELS)

    bundles, loaded = load_final(final_dir)
    assert set(bundles) == {"majority", "tiny_char"}
    assert loaded["n_train_rows"] == len(train_df)

    predictions = predict_dataframe(bundles, test_df, weights={"tiny_char": 1.0})
    assert len(predictions) == len(test_df)
    assert set(predictions) <= set(LABELS)

    out = build_submission(
        test_df["id"], predictions, tmp_path / "submission.parquet",
        expected_ids=test_df["id"],
    )
    frame = validate_submission(out, expected_ids=test_df["id"])
    assert isinstance(frame, pd.DataFrame)
    assert frame["id"].tolist() == test_df["id"].tolist()


def test_equal_weight_blend_of_multiple_models(train_df, test_df, tiny_char_model, tmp_path):
    final_dir = tmp_path / "final"
    fit_final(
        [{"name": "majority", "baseline": "majority"}, tiny_char_model],
        train_df,
        final_dir,
        dataset_label="synthetic",
    )
    bundles, _ = load_final(final_dir)
    blended = predict_dataframe(bundles, test_df.head(6))
    assert blended.shape == (6,)
    assert set(blended) <= set(LABELS)
