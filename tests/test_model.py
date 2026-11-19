from __future__ import annotations

import numpy as np
import pytest

from src.constants import LABELS, N_CLASSES
from src.data import labels_to_indices
from src.model import ModelBundle, ModelError, ModelSpec, build_classifier, estimator_from_config


def test_spec_validation():
    with pytest.raises(ModelError, match="needs a name"):
        ModelSpec.from_dict({"features": ["char"]})
    with pytest.raises(ModelError, match="unknown classifier"):
        ModelSpec.from_dict({"name": "x", "classifier": "magic"})
    with pytest.raises(ModelError, match="requires an svd"):
        ModelSpec.from_dict({"name": "x", "nuisance_k": 3})
    spec = ModelSpec.from_dict({"name": "x", "features": ["char"], "char_ngram_range": [3, 4]})
    assert spec.features == ("char",)
    assert spec.params["char_ngram_range"] == [3, 4]
    assert spec.to_dict()["name"] == "x"


def test_classifier_factory_rejects_unknown():
    spec = ModelSpec(name="x")
    spec.classifier = "nope"
    with pytest.raises(ModelError, match="unknown classifier"):
        build_classifier(spec)


def test_estimator_from_config_baselines_and_models():
    from src.baselines import MajorityClassifier, StratifiedRandomClassifier

    assert isinstance(estimator_from_config({"name": "m", "baseline": "majority"}), MajorityClassifier)
    assert isinstance(
        estimator_from_config({"name": "r", "baseline": "stratified_random"}),
        StratifiedRandomClassifier,
    )
    assert isinstance(estimator_from_config({"name": "c", "features": ["char"]}), ModelBundle)
    with pytest.raises(ModelError, match="unknown baseline"):
        estimator_from_config({"name": "b", "baseline": "nonsense"})


def test_bundle_scores_align_with_official_class_order(train_df, tiny_char_model, tmp_path):
    subset = train_df.groupby("model", observed=True).head(4).reset_index(drop=True)
    bundle = estimator_from_config(tiny_char_model, seed=0)
    bundle.fit(subset["text"].tolist(), labels_to_indices(subset["model"].tolist()))

    scores = bundle.decision_function(subset["text"].head(7).tolist())
    assert scores.shape == (7, N_CLASSES)
    assert bundle.class_order == LABELS
    assert set(bundle.predict(subset["text"].head(7).tolist())) <= set(LABELS)

    path = bundle.save(tmp_path / "bundle.joblib")
    reloaded = ModelBundle.load(path)
    assert np.allclose(scores, reloaded.decision_function(subset["text"].head(7).tolist()))
    assert reloaded.class_order == LABELS


def test_bundle_rejects_out_of_range_labels(train_df, tiny_char_model):
    bundle = estimator_from_config(tiny_char_model, seed=0)
    with pytest.raises(ModelError, match="class indices"):
        bundle.fit(train_df["text"].head(5).tolist(), np.array([0, 1, 2, 3, 99]))


def test_class_order_mismatch_is_rejected_on_load(train_df, tiny_char_model, tmp_path):
    subset = train_df.groupby("model", observed=True).head(4).reset_index(drop=True)
    bundle = estimator_from_config(tiny_char_model, seed=0)
    bundle.fit(subset["text"].tolist(), labels_to_indices(subset["model"].tolist()))
    bundle.class_order = tuple(reversed(LABELS))
    path = bundle.save(tmp_path / "bad.joblib")
    with pytest.raises(ModelError, match="class order"):
        ModelBundle.load(path)


def test_majority_baseline_predicts_a_valid_label(train_df):
    bundle = estimator_from_config({"name": "maj", "baseline": "majority"})
    bundle.fit(train_df["text"].tolist(), labels_to_indices(train_df["model"].tolist()))
    predictions = set(bundle.predict(["anything at all"]))
    assert len(predictions) == 1
    assert predictions <= set(LABELS)


def test_gbdt_requires_dense_input():
    with pytest.raises(ModelError, match="gbdt needs dense"):
        ModelSpec.from_dict({"name": "g", "classifier": "gbdt", "features": ["char"]})
    spec = ModelSpec.from_dict(
        {"name": "g", "classifier": "gbdt", "features": [], "style": True}
    )
    assert spec.classifier == "gbdt"
    spec2 = ModelSpec.from_dict(
        {"name": "g", "classifier": "gbdt", "features": ["char"], "svd": 8}
    )
    assert spec2.svd == 8


def test_gbdt_classifier_factory():
    from sklearn.ensemble import HistGradientBoostingClassifier

    spec = ModelSpec.from_dict(
        {"name": "g", "classifier": "gbdt", "features": [], "style": True, "max_iter": 15}
    )
    clf = build_classifier(spec)
    assert isinstance(clf, HistGradientBoostingClassifier)
    assert clf.max_iter == 15
    assert clf.class_weight == "balanced"


def _gbdt_style_config():
    return {
        "name": "gbdt_style",
        "features": [],
        "style": True,
        "classifier": "gbdt",
        "max_iter": 15,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 2,
        "learning_rate": 0.2,
    }


def test_gbdt_style_only_bundle(train_df, tmp_path):
    subset = train_df.groupby("model", observed=True).head(6).reset_index(drop=True)
    bundle = estimator_from_config(_gbdt_style_config(), seed=0)
    bundle.fit(subset["text"].tolist(), labels_to_indices(subset["model"].tolist()))
    scores = bundle.decision_function(subset["text"].head(7).tolist())
    assert scores.shape == (7, N_CLASSES)
    assert bundle.class_order == LABELS
    assert set(bundle.predict(subset["text"].head(7).tolist())) <= set(LABELS)
    path = bundle.save(tmp_path / "gbdt.joblib")
    reloaded = ModelBundle.load(path)
    assert np.allclose(scores, reloaded.decision_function(subset["text"].head(7).tolist()))


def test_gbdt_multiview_svd_bundle(train_df):
    subset = train_df.groupby("model", observed=True).head(6).reset_index(drop=True)
    cfg = {
        "name": "gbdt_svd",
        "features": ["char", "word"],
        "style": True,
        "svd": 8,
        "classifier": "gbdt",
        "max_features": 500,
        "min_df": 1,
        "char_ngram_range": [2, 3],
        "max_iter": 15,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 2,
    }
    bundle = estimator_from_config(cfg, seed=0)
    bundle.fit(subset["text"].tolist(), labels_to_indices(subset["model"].tolist()))
    scores = bundle.decision_function(subset["text"].head(5).tolist())
    assert scores.shape == (5, N_CLASSES)
    assert np.isfinite(scores).all()


def test_model_spec_round_trips_residual_fields():
    spec = ModelSpec.from_dict(
        {"name": "r", "features": [], "style": True, "residual_lambda": 0.5}
    )
    assert spec.residual_lambda == 0.5 and spec.residual_alpha == 1.0
    assert ModelSpec.from_dict(spec.to_dict()).residual_lambda == 0.5


def test_model_spec_rejects_sparse_residual():
    with pytest.raises(ModelError, match="residual"):
        ModelSpec.from_dict({"name": "r", "features": ["char"], "residual_lambda": 0.5})
    with pytest.raises(ModelError, match="residual_lambda"):
        ModelSpec.from_dict({"name": "r", "features": [], "style": True, "residual_lambda": 2.0})


def test_residual_bundle_carries_fitted_neutralizer(train_df):
    subset = train_df.groupby("model", observed=True).head(6).reset_index(drop=True)
    cfg = {"name": "resid", "features": [], "style": True, "residual_lambda": 0.5}
    bundle = estimator_from_config(cfg, seed=0)
    bundle.fit(
        subset["text"].tolist(),
        labels_to_indices(subset["model"].tolist()),
        groups=subset["prompt_id"].astype(str).tolist(),
    )
    assert bundle.builder.residual_ is not None
    scores = bundle.decision_function(subset["text"].head(4).tolist())
    assert scores.shape == (4, N_CLASSES) and np.isfinite(scores).all()
    assert tuple(bundle.class_order) == LABELS
