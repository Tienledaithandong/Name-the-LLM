"""Model specification and the fitted bundle that inference loads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from src.constants import LABELS, N_CLASSES

CLASSIFIERS = ("linearsvc", "logreg", "sgd", "gbdt", "catboost")
DENSE_CLASSIFIERS = ("gbdt", "catboost")


class ModelError(ValueError):
    """Raised for an unusable model specification or artifact."""


@dataclass
class ModelSpec:
    name: str
    features: tuple[str, ...] = ("char",)
    style: bool = False
    classifier: str = "linearsvc"
    C: float = 1.0
    max_features: int | None = None
    min_df: int | None = None
    svd: int | None = None
    nuisance_k: int = 0
    embedding: dict[str, Any] | None = None
    multilayer: dict[str, Any] | None = None
    class_weight: str | None = "balanced"
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelSpec":
        if not isinstance(raw, dict) or "name" not in raw:
            raise ModelError(f"model entry needs a name: {raw!r}")
        known = {f for f in cls.__dataclass_fields__ if f != "params"}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}
        if "features" in kwargs:
            kwargs["features"] = tuple(kwargs["features"])
        spec = cls(**kwargs, params=extra)
        if spec.classifier not in CLASSIFIERS:
            raise ModelError(
                f"model {spec.name}: unknown classifier {spec.classifier!r}; "
                f"choose from {CLASSIFIERS}"
            )
        if spec.nuisance_k and not spec.svd:
            raise ModelError(f"model {spec.name}: nuisance_k requires an svd size")
        if spec.classifier in DENSE_CLASSIFIERS and spec.features and not spec.svd:
            raise ModelError(
                f"model {spec.name}: {spec.classifier} needs dense input — set an svd size "
                "or use style-only features (no raw tf-idf kinds)"
            )
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "features": list(self.features),
            "style": self.style,
            "classifier": self.classifier,
            "C": self.C,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "svd": self.svd,
            "nuisance_k": self.nuisance_k,
            "embedding": self.embedding,
            "class_weight": self.class_weight,
            "params": self.params,
        }


class SiblingResidualClassifier:
    """Residual head for the SASCR intra-lab sibling clusters.

    Trained on the multi-layer causal readout (plus style view) to separate the
    hard within-lab version siblings that the 18-class sparse members struggle
    with.  It is a real 18-column ridge/LogReg but is deliberately fit only on
    the sibling labels so its decision scores act as a residual correction
    rather than a stand-alone 18-class model.  Scikit-learn compatible:

    * ``fit(X, y)`` — ``X`` is the per-passage feature matrix (dense), ``y`` the
      full 18-label index vector for the training fold.
    * ``decision_function(X)`` — returns shape ``(n, N_CLASSES)`` margins so it
      plugs straight into the existing row-z blend (D042).
    """

    def __init__(self, C: float = 1.0, random_state: int = 42) -> None:
        self.C = C
        self.random_state = random_state
        self.clf_ = None
        self.scaler_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.scaler_ = StandardScaler(with_mean=False)
        Xs = self.scaler_.fit_transform(X)
        self.clf_ = LogisticRegression(
            C=self.C,
            max_iter=2000,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.clf_.fit(Xs, y)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xs = self.scaler_.transform(X) if self.scaler_ is not None else X
        if self.clf_ is None:
            raise RuntimeError("fit the classifier before calling decision_function")
        margins = np.asarray(self.clf_.decision_function(Xs), dtype=np.float64)
        out = np.full((X.shape[0], 18), -1e9, dtype=np.float32)
        out[:, self.clf_.classes_] = margins
        return out


def build_classifier(spec: ModelSpec):
    if spec.classifier == "linearsvc":
        # liblinear spends most of its time in the last few iterations at 200k
        # features and 18 classes; 2,000 iterations is the measured sweet spot
        # for grouped-CV quality vs. runtime.
        return LinearSVC(
            C=spec.C,
            class_weight=spec.class_weight,
            dual="auto",
            max_iter=2000,
            random_state=spec.seed,
        )
    if spec.classifier == "logreg":
        return LogisticRegression(
            C=spec.C,
            class_weight=spec.class_weight,
            max_iter=2000,
            random_state=spec.seed,
        )
    if spec.classifier == "sgd":
        return SGDClassifier(
            loss="modified_huber",
            alpha=1.0 / max(spec.C, 1e-6) / 1000.0,
            class_weight=spec.class_weight,
            max_iter=2000,
            random_state=spec.seed,
        )
    if spec.classifier == "sibling_residual":
        return SiblingResidualClassifier(C=spec.C, random_state=spec.seed)
    if spec.classifier == "gbdt":
        p = spec.params
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(p.get("learning_rate", 0.1)),
            max_iter=int(p.get("max_iter", 200)),
            max_leaf_nodes=int(p.get("max_leaf_nodes", 31)),
            max_depth=p.get("max_depth"),
            min_samples_leaf=int(p.get("min_samples_leaf", 20)),
            l2_regularization=float(p.get("l2_regularization", 0.0)),
            max_bins=int(p.get("max_bins", 255)),
            class_weight=spec.class_weight,
            early_stopping=False,
            random_state=spec.seed,
        )
    if spec.classifier == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # optional dependency, like sentence-transformers
            raise ModelError(
                "classifier 'catboost' needs the optional catboost package "
                "(python -m pip install catboost)"
            ) from exc
        p = spec.params
        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=int(p.get("iterations", 300)),
            depth=int(p.get("depth", 6)),
            learning_rate=float(p.get("learning_rate", 0.1)),
            l2_leaf_reg=float(p.get("l2_leaf_reg", 3.0)),
            auto_class_weights="Balanced" if spec.class_weight == "balanced" else None,
            random_seed=spec.seed,
            verbose=False,
            allow_writing_files=False,
        )
    raise ModelError(f"unknown classifier {spec.classifier!r}")


class ModelBundle:
    """Feature builder + classifier + fixed class order, saved as one artifact."""

    kind = "sparse_text"

    def __init__(self, spec: ModelSpec, builder=None, classifier=None):
        from src.features_tfidf import TextMatrixBuilder

        self.spec = spec
        self.class_order: tuple[str, ...] = LABELS
        builder_params = {
            k: v
            for k, v in {
                **spec.params,
                "max_features": spec.max_features,
                "min_df": spec.min_df,
            }.items()
            if v is not None
        }
        self.builder = builder or TextMatrixBuilder(
            kinds=spec.features,
            style=spec.style,
            params=builder_params,
            svd=spec.svd,
            nuisance_k=spec.nuisance_k,
            embedding=spec.embedding,
            multilayer=spec.multilayer,
        )
        self.classifier = classifier or build_classifier(spec)

    @property
    def name(self) -> str:
        return self.spec.name

    def fit(self, texts, y_idx, groups=None) -> "ModelBundle":
        y_idx = np.asarray(y_idx, dtype=np.int64)
        if y_idx.size and (y_idx.min() < 0 or y_idx.max() >= N_CLASSES):
            raise ModelError("class indices outside the 18-label range")
        matrix = self.builder.fit_transform(list(texts), groups=groups)
        if self.spec.classifier in DENSE_CLASSIFIERS and sparse.issparse(matrix):
            matrix = matrix.toarray()
        self.classifier.fit(matrix, y_idx)
        return self

    def decision_function(self, texts) -> np.ndarray:
        matrix = self.builder.transform(list(texts))
        if self.spec.classifier in DENSE_CLASSIFIERS and sparse.issparse(matrix):
            matrix = matrix.toarray()
        if hasattr(self.classifier, "decision_function"):
            scores = np.asarray(self.classifier.decision_function(matrix))
        else:
            # CatBoost exposes probabilities only. Log-probability keeps the score
            # on an unbounded additive scale, which is what row-z blending expects;
            # it is a real probability model, unlike a LinearSVC margin.
            proba = np.asarray(self.classifier.predict_proba(matrix), dtype=np.float64)
            scores = np.log(np.clip(proba, 1e-12, 1.0))
        present = np.asarray(self.classifier.classes_, dtype=np.int64)
        n = matrix.shape[0]
        out = np.full((n, N_CLASSES), -1e9, dtype=np.float32)
        if scores.ndim == 1:
            if present.size == 2:
                out[:, present[0]] = -scores
                out[:, present[1]] = scores
            elif present.size == 1:
                out[:, present[0]] = scores
            else:
                raise ModelError(f"unexpected decision_function shape {scores.shape}")
        else:
            out[:, present] = scores
        return out

    def predict(self, texts) -> np.ndarray:
        indices = self.decision_function(texts).argmax(axis=1)
        return np.asarray([self.class_order[i] for i in indices], dtype=object)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        bundle = joblib.load(Path(path))
        if not hasattr(bundle, "decision_function") or not hasattr(bundle, "class_order"):
            raise ModelError(f"{path} is not a fitted ModelBundle")
        if tuple(bundle.class_order) != LABELS:
            raise ModelError(
                f"{path} class order does not match the official labels; retrain"
            )
        return bundle


def estimator_from_config(raw: dict, seed: int = 42):
    """Build an unfitted estimator (model bundle or reference baseline)."""
    from src.baselines import MajorityClassifier, StratifiedRandomClassifier

    baseline = raw.get("baseline")
    if baseline == "majority":
        return MajorityClassifier(seed=seed)
    if baseline == "stratified_random":
        return StratifiedRandomClassifier(seed=seed)
    if baseline:
        raise ModelError(f"unknown baseline {baseline!r}")
    spec = ModelSpec.from_dict({**raw, "seed": seed})
    return ModelBundle(spec)
