"""Tests for the offline DeBERTa inference path (``kaggle/deberta/predict.py``).

Pure, network-free parts only: path resolution, label-order resolution, and the
submission-validation wiring. No torch/transformers required.

``kaggle/`` is a notebooks-assets dir (no ``__init__.py``), so the script is
imported by file path rather than as a package member.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import LABELS  # noqa: E402
from src.submission import SubmissionError, build_submission  # noqa: E402

_PRED_MODULE = importlib.util.spec_from_file_location(
    "kaggle_deberta_predict", ROOT / "kaggle" / "deberta" / "predict.py"
)
predict = importlib.util.module_from_spec(_PRED_MODULE)
# predict.py inserts ROOT into sys.path at module level; conftest already has
# ROOT, so the module-level ``src.*`` imports resolve.
_PRED_MODULE.loader.exec_module(predict)
# Rebind the test module's references to the loaded module's functions.
find_parent = predict.find_parent
find_test_dir = predict.find_test_dir
load_label_order = predict.load_label_order
main = predict.main


def test_find_parent_returns_abspath_when_absolute(tmp_path):
    model_file = tmp_path / "x" / "test.csv"
    model_file.parent.mkdir()
    model_file.touch()
    p = find_parent(str(model_file))
    assert p == model_file.parent


def test_find_test_dir_lives_in_repo_data(tmp_path, test_df):
    # The test fixture is in memory; write it into an isolated temporary folder.
    test_df.to_csv(tmp_path / "test.csv", index=False)
    tdir = find_test_dir(str(tmp_path))
    assert (tdir / "test.csv").is_file()


def test_find_test_dir_requires_test_csv(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_test_dir(str(tmp_path))


def test_load_label_order_rejects_missing(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(SubmissionError, match="missing or invalid label order"):
        load_label_order(model_dir)


def test_load_label_order_uses_stored_order(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    stored = list(LABELS)
    (model_dir / "label_order.json").write_text(
        json.dumps(stored), encoding="utf-8"
    )
    assert load_label_order(model_dir) == stored


def test_load_label_order_rejects_malformed(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "label_order.json").write_text("not-a-list", encoding="utf-8")
    with pytest.raises(SubmissionError, match="missing or invalid label order"):
        load_label_order(model_dir)


def test_load_label_order_rejects_permutation(tmp_path):
    stored = list(LABELS)
    stored[0], stored[1] = stored[1], stored[0]
    (tmp_path / "label_order.json").write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(SubmissionError, match="does not match"):
        load_label_order(tmp_path)


def test_inference_manifest_requires_training_configuration(tmp_path):
    manifest = {"schema_version": "deberta-inference/v1", "class_order": list(LABELS),
                "model_name": "local-model", "max_len": 384, "batch_size": 4}
    p = tmp_path / "inference_manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    assert predict.load_inference_manifest(tmp_path)["max_len"] == 384
    del manifest["max_len"]
    p.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SubmissionError, match="max_len"):
        predict.load_inference_manifest(tmp_path)


def test_blend_matches_evaluated_rowwise_formula():
    class Bundle:
        def __init__(self, scores):
            self.scores = scores

        def decision_function(self, texts):
            return self.scores

    deb = np.stack([np.arange(18), np.arange(18)[::-1]]).astype(float)
    char = np.roll(deb, 2, axis=1)
    style = np.roll(deb, 7, axis=1)
    def z(x):
        return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    actual = predict.blend_candidate(deb, {"char_tfidf_logreg": Bundle(char),
                                           "style_only": Bundle(style)}, ["a", "b"])
    np.testing.assert_allclose(actual, .5 * z(deb) + .25 * z(char) + .25 * z(style))
    with pytest.raises(SubmissionError, match="exactly"):
        predict.blend_candidate(deb, {"style_only": Bundle(style)}, ["a", "b"])
    char[0, 0] = np.nan
    with pytest.raises(SubmissionError, match="invalid score"):
        predict.blend_candidate(deb, {"char_tfidf_logreg": Bundle(char),
                                      "style_only": Bundle(style)}, ["a", "b"])


def test_sparse_loader_rejects_sklearn_version_mismatch(tmp_path):
    manifest = {"class_order": list(LABELS), "environment": {"sklearn": "0.0.invalid"},
                "models": [{"name": "char_tfidf_logreg"}, {"name": "style_only"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SubmissionError, match="require scikit-learn"):
        predict.load_sparse_bundles(tmp_path)


def test_main_rejects_sequence_length_drift_before_model_loading(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "label_order.json").write_text(json.dumps(list(LABELS)), encoding="utf-8")
    (model_dir / "inference_manifest.json").write_text(json.dumps({
        "schema_version": "deberta-inference/v1", "class_order": list(LABELS),
        "model_name": "synthetic", "max_len": 384, "batch_size": 4,
    }), encoding="utf-8")
    (tmp_path / "test.csv").write_text("id,text\nsynthetic-id,synthetic text\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="max-len override"):
        main(["--model-dir", str(model_dir), "--test", str(tmp_path), "--max-len", "256"])


def test_main_fails_fast_on_missing_model_dir(tmp_path, test_df, monkeypatch):
    (tmp_path / "test.csv").write_text(test_df.to_csv(index=False), encoding="utf-8")
    monkeypatch.setenv("NTL_MODEL_DIR", str(tmp_path / "missing-model"))
    monkeypatch.setenv("NTL_SUBMISSION", str(tmp_path / "submission.parquet"))

    with pytest.raises(SubmissionError, match="model folder not found"):
        main(["--test", str(tmp_path), "--out", str(tmp_path / "submission.parquet")])


def test_main_fails_fast_on_missing_test_column(tmp_path, test_df, monkeypatch):
    # keep 'id' so the subset check fires on the missing 'text' column
    (tmp_path / "model").mkdir()
    test_df = test_df.drop(columns=["text"])
    (tmp_path / "test.csv").write_text(test_df.to_csv(index=False), encoding="utf-8")
    monkeypatch.setenv("NTL_MODEL_DIR", str(tmp_path / "model"))
    monkeypatch.setenv("NTL_SUBMISSION", str(tmp_path / "submission.parquet"))

    with pytest.raises(SubmissionError, match="missing columns"):
        main(["--test", str(tmp_path), "--out", str(tmp_path / "submission.parquet")])


def test_build_submission_is_wrapped_by_predict(tmp_path, test_df):
    """Verify the project's fail-fast submission checks still apply end-to-end."""
    ids = test_df["id"].tolist()
    predictions = [LABELS[i % len(LABELS)] for i in range(len(ids))]
    out = tmp_path / "submission.parquet"
    path = build_submission(ids, predictions, out, expected_ids=ids)
    assert path.is_file()
    assert path.name == "submission.parquet"
