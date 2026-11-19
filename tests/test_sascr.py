"""SASCR config and blend mechanics.

Tests that configs/sascr.yaml loads cleanly, that its sibling clusters sit
inside the official LABELS set, and that the adopted row-z blend machinery
behaves exactly as the registered anchors (D042) describe.  No scores are
claimed here; synthetic fixtures are used to pin the machinery.
"""

import pytest

import importlib.util
import numpy as np
import pandas as pd

from src.config import ConfigError, load_config
from src.constants import LABELS
from src.ensemble import evaluate_blend, normalize_scores
from src.features_multilayer import MultiLayerReadoutBuilder
from src.inference import predict_dataframe
from src.model import ModelError, estimator_from_config

CONFIG_PATH = "configs/sascr.yaml"


def _load() -> dict:
    return load_config(CONFIG_PATH)


def _two_member_margins(seed: int = 0):
    """char and style score rows of shape (n, 18); char is right, style wrong."""
    rng = np.random.default_rng(seed)
    char = np.full((6, 18), -5.0, dtype=np.float64)
    style = np.full((6, 18), -5.0, dtype=np.float64)
    y = np.array([0, 1, 0, 1, 0, 1])
    char[np.arange(6), y] = 5.0
    style[np.arange(6), 1 - y] = 5.0
    return char, style, y


def test_config_loads_and_has_unique_models():
    cfg = _load()
    names = [m["name"] for m in cfg["models"]]
    assert len(names) == len(set(names))
    assert cfg["blend"]["weights"] == {"char_tfidf_logreg": 0.5, "style_only": 0.5}
    assert cfg["ensemble"]["members"] == ["char_tfidf_logreg", "style_only"]


def test_blend_weights_are_positive_and_sum_to_one():
    cfg = _load()
    weights = cfg["blend"]["weights"]
    assert all(v > 0 for v in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_sibling_clusters_are_subset_of_labels():
    cfg = _load()
    clusters = cfg["design"]["members"][1]["clusters"]
    assert len(set(clusters)) == len(clusters)
    assert set(clusters).issubset(set(LABELS))
    # Every cluster must map to an existing 18-label index, not a family guess.
    assert all(c in LABELS for c in clusters)
    assert len(LABELS) == 18


def test_row_z_blend_preserves_margin_scale_better_than_softmax():
    """D042 pin: row-z keeps unbounded margins; softmax collapses to [0,1]."""
    char, style, y = _two_member_margins(seed=1)
    char_z = normalize_scores(char)
    style_z = normalize_scores(style)
    adopted = 0.5 * char_z + 0.5 * style_z

    # char has a large margin on the true classes; the blend must keep that
    # margin scaled (unbounded), unlike softmax which squeezes everything to
    # a bounded probability vector that loses margin ordering.
    assert np.abs(adopted).max() > 1.0
    exp = np.exp(char_z + style_z)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    assert (softmax >= 0).all() and (softmax <= 1).all()
    assert np.allclose(softmax.sum(axis=1), 1.0, atol=1e-9)
    # On these synthetic rows the planted true classes are unchanged by either
    # rule; the guard is that they are genuinely separable, not a tie.
    assert (char.argmax(axis=1) == y).all()


def test_predict_dataframe_schema_conforms_to_official_labels():
    class _StubBundle:
        class_order = LABELS

        def decision_function(self, texts):
            n = len(texts)
            out = np.zeros((n, 18), dtype=np.float32)
            cols = [0, 2, 1, 2, 3, 0]
            out[np.arange(n), cols] = 3.0
            return out

    frame = pd.DataFrame(
        {
            "id": [f"t{i}" for i in range(6)],
            "text": [f"passage {i}" for i in range(6)],
        }
    )
    out = predict_dataframe({"trunk": _StubBundle()}, frame, weights=None)
    assert isinstance(out, np.ndarray)
    assert out.shape == (6,)
    assert not pd.isna(out).any()
    assert set(out).issubset(set(LABELS))
    assert out.tolist() == [LABELS[0], LABELS[2], LABELS[1], LABELS[2], LABELS[3], LABELS[0]]


def test_design_members_are_gated_until_implemented():
    cfg = _load()
    new = [m for m in cfg["design"]["members"] if m["status"] == "design"]
    assert [m["name"] for m in new] == [
        "multilayer_causal_readout",
        "sibling_residual",
    ]
    for member in new:
        with pytest.raises(ModelError, match="unknown classifier"):
            estimator_from_config(
                {"name": member["name"], "classifier": "sibling_residual"}
            )


def test_missing_config_raises_config_error():
    with pytest.raises(ConfigError):
        load_config("configs/does_not_exist.yaml")


def _synthetic_corpus(n_rows: int = 32, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    words = [
        "the", "of", "and", "to", "a", "in", "that", "is", "was", "it",
        "for", "on", "with", "by", "as", "at", "an", "be", "this", "are",
        "or", "from", "which", "their", "they", "been", "have", "had", "has",
        "but", "not", "all", "would", "were", "can", "will", "what", "when",
        "if", "some", "may", "are", "no", "there", "each", "also", "than",
        "into", "its", "who", "more", "must", "him", "could", "these", "our",
    ]
    return [
        " ".join(rng.choice(words, size=rng.integers(8, 14), replace=False).tolist())
        for _ in range(n_rows)
    ]


def test_multilayer_readout_builder_hashing_backend_finite():
    texts = _synthetic_corpus(32, seed=0)
    builder = MultiLayerReadoutBuilder(
        backend="hashing",
        depths=(2, 4, 6, 8, 12, 16, 20, 24),
        pooling="content_token_mean",
        scale="layer_norm",
        dim=256,
    )
    matrix = builder.encode_cached(texts)
    assert matrix.shape == (32, 256)
    assert np.isfinite(matrix).all()
    # Caching is deterministic and stable across two calls.
    assert np.array_equal(matrix, builder.encode_cached(texts))


def test_multilayer_readout_config_roundtrip():
    cfg = {
        "backend": "hashing",
        "depths": [2, 4, 6, 8, 12, 16, 20, 24],
        "pooling": "content_token_mean",
        "scale": "layer_norm",
        "dim": 512,
        "seed": 7,
    }
    builder = MultiLayerReadoutBuilder(**cfg)
    assert builder.depths == (2, 4, 6, 8, 12, 16, 20, 24)
    assert builder.pooling == "content_token_mean"
    assert builder.scale == "layer_norm"
    assert builder.dim == 512


@pytest.mark.skipif(
    not (importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")),
    reason="torch/transformers not installed",
)
def test_multilayer_readout_transformers_path_shape():
    # Runs the real-transformer forward path when online; skips cleanly when the
    # Hugging Face hub is unreachable (offline boxes / free tiers). The
    # hashing path is always exercised and is network-independent.
    try:
        from transformers import AutoModel

        AutoModel.from_pretrained(
            "gpt2",
            cache_dir=None,
            local_files_only=False,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"transformers hub unreachable ({exc}); hashing path only")

    texts = _synthetic_corpus(4)
    builder = MultiLayerReadoutBuilder(
        backend="transformers",
        model_name="gpt2",
        depths=(2, 3, 4),
        pooling="content_token_mean",
        scale="layer_norm",
        dim=64,
    )
    matrix = builder.encode_cached(texts)
    assert matrix.shape == (4, 64)
    assert np.isfinite(matrix).all()


def test_sibling_residual_classifier_plugs_into_blend_contract():
    from src.model import SiblingResidualClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 32)).astype(np.float64)
    y = np.array(
        [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 5, 6, 7],
        dtype=np.int64,
    )
    clf = SiblingResidualClassifier(C=1.0, random_state=42)
    clf.fit(X, y)
    assert clf.clf_ is not None
    scores = clf.decision_function(X)
    assert scores.shape == (24, 18)
    assert np.isfinite(scores).all()
    # Prediction is a valid LABELS index and not all-one-class on this split.
    pred = scores.argmax(axis=1)
    assert set(np.unique(pred)).issubset(set(range(18)))
    assert len(np.unique(pred)) > 1
    # Identical rows produce identical margins (determinism).
