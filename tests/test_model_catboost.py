"""The optional CatBoost arm must obey the same bundle contract as every other model.

Plan §4 L139 names CatBoost/LightGBM alongside logreg and SVM for the style view.
CatBoost exposes probabilities rather than margins, so these tests pin the
log-probability score path, the 18-column class-order contract, the sentinel for
classes absent from a fold, and the dense-input requirement.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.constants import LABELS, N_CLASSES
from src.data import labels_to_indices
from src.model import (
    DENSE_CLASSIFIERS,
    ModelBundle,
    ModelError,
    ModelSpec,
    build_classifier,
    estimator_from_config,
)

catboost = pytest.importorskip("catboost")

STYLE_CATBOOST = {
    "name": "style_catboost",
    "features": [],
    "style": True,
    "classifier": "catboost",
    "class_weight": "balanced",
    "iterations": 30,
    "depth": 4,
    "learning_rate": 0.1,
}


def _subset(train_df, n_labels: int | None = None, per_label: int = 6):
    frame = train_df.groupby("model", observed=True).head(per_label).reset_index(drop=True)
    if n_labels is not None:
        keep = list(LABELS[:n_labels])
        frame = frame[frame["model"].isin(keep)].reset_index(drop=True)
    return frame


def test_catboost_is_a_registered_dense_classifier() -> None:
    assert "catboost" in DENSE_CLASSIFIERS
    spec = ModelSpec.from_dict(STYLE_CATBOOST)
    assert spec.classifier == "catboost"
    assert spec.params["iterations"] == 30
    assert spec.params["depth"] == 4


def test_catboost_rejects_raw_tfidf_without_svd() -> None:
    with pytest.raises(ModelError, match="needs dense input"):
        ModelSpec.from_dict({**STYLE_CATBOOST, "features": ["char"]})


def test_catboost_accepts_svd_reduced_text() -> None:
    spec = ModelSpec.from_dict({**STYLE_CATBOOST, "features": ["char"], "svd": 32})
    assert spec.svd == 32


def test_factory_builds_a_multiclass_balanced_catboost() -> None:
    clf = build_classifier(ModelSpec.from_dict(STYLE_CATBOOST))
    assert isinstance(clf, catboost.CatBoostClassifier)
    params = clf.get_params()
    assert params["loss_function"] == "MultiClass"
    assert params["auto_class_weights"] == "Balanced"
    assert params["allow_writing_files"] is False
    assert params["verbose"] is False
    assert params["iterations"] == 30
    assert params["depth"] == 4
    assert params["random_seed"] == 42


def test_factory_without_balanced_weights() -> None:
    clf = build_classifier(ModelSpec.from_dict({**STYLE_CATBOOST, "class_weight": None}))
    assert clf.get_params().get("auto_class_weights") is None


def test_estimator_from_config_builds_a_bundle() -> None:
    bundle = estimator_from_config(STYLE_CATBOOST, seed=3)
    assert isinstance(bundle, ModelBundle)
    assert bundle.class_order == LABELS


def test_bundle_scores_use_the_official_class_order(train_df) -> None:
    frame = _subset(train_df)
    bundle = estimator_from_config(STYLE_CATBOOST, seed=0)
    bundle.fit(frame["text"].tolist(), labels_to_indices(frame["model"].tolist()))
    texts = frame["text"].head(9).tolist()
    scores = bundle.decision_function(texts)
    assert scores.shape == (9, N_CLASSES)
    assert np.isfinite(scores).all()
    assert bundle.class_order == LABELS
    assert set(bundle.predict(texts)) <= set(LABELS)


def test_log_probability_scores_are_monotone_in_probability(train_df) -> None:
    frame = _subset(train_df)
    bundle = estimator_from_config(STYLE_CATBOOST, seed=0)
    bundle.fit(frame["text"].tolist(), labels_to_indices(frame["model"].tolist()))
    texts = frame["text"].head(12).tolist()
    scores = bundle.decision_function(texts)
    matrix = bundle.builder.transform(texts)
    proba = bundle.classifier.predict_proba(matrix.toarray() if hasattr(matrix, "toarray") else matrix)
    present = np.asarray(bundle.classifier.classes_, dtype=np.int64)
    assert np.allclose(scores[:, present], np.log(np.clip(proba, 1e-12, 1.0)))
    assert np.array_equal(scores.argmax(axis=1), proba.argmax(axis=1))
    assert (scores <= 0.0).all()


def test_absent_classes_keep_the_sentinel(train_df) -> None:
    frame = _subset(train_df, n_labels=3)
    bundle = estimator_from_config(STYLE_CATBOOST, seed=0)
    bundle.fit(frame["text"].tolist(), labels_to_indices(frame["model"].tolist()))
    scores = bundle.decision_function(frame["text"].head(5).tolist())
    present = np.asarray(bundle.classifier.classes_, dtype=np.int64)
    absent = np.setdiff1d(np.arange(N_CLASSES), present)
    assert present.size == 3
    assert (scores[:, absent] == -1e9).all()
    assert np.isfinite(scores[:, present]).all()


def test_scores_are_deterministic_for_a_fixed_seed(train_df) -> None:
    frame = _subset(train_df)
    texts = frame["text"].head(6).tolist()
    y = labels_to_indices(frame["model"].tolist())
    a = estimator_from_config(STYLE_CATBOOST, seed=5).fit(frame["text"].tolist(), y)
    b = estimator_from_config(STYLE_CATBOOST, seed=5).fit(frame["text"].tolist(), y)
    assert np.allclose(a.decision_function(texts), b.decision_function(texts))


def test_save_load_roundtrip(train_df, tmp_path) -> None:
    frame = _subset(train_df)
    bundle = estimator_from_config(STYLE_CATBOOST, seed=0)
    bundle.fit(frame["text"].tolist(), labels_to_indices(frame["model"].tolist()))
    texts = frame["text"].head(7).tolist()
    scores = bundle.decision_function(texts)
    reloaded = ModelBundle.load(bundle.save(tmp_path / "catboost.joblib"))
    assert reloaded.class_order == LABELS
    assert np.allclose(scores, reloaded.decision_function(texts))


def test_no_catboost_info_directory_is_written(train_df, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    frame = _subset(train_df)
    bundle = estimator_from_config(STYLE_CATBOOST, seed=0)
    bundle.fit(frame["text"].tolist(), labels_to_indices(frame["model"].tolist()))
    bundle.decision_function(frame["text"].head(3).tolist())
    assert not (tmp_path / "catboost_info").exists()