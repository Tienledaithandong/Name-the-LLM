from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.features_style import FEATURE_NAMES, StyleFeatureExtractor, extract_style_frame
from src.features_tfidf import FeatureError, TextMatrixBuilder, density


def test_style_feature_contract():
    extractor = StyleFeatureExtractor()
    matrix = extractor.transform(["Hello, world.", "", "a\n\n- item\n"] * 2)
    assert matrix.shape == (6, len(FEATURE_NAMES))
    assert sparse.issparse(matrix) and matrix.dtype == np.float32
    assert np.isfinite(matrix.toarray()).all(), "features must never contain NaN/inf"


def test_style_features_are_deterministic():
    texts = ["The quick brown fox — jumps; over the lazy dog.", "Short."]
    assert np.array_equal(extract_style_frame(texts), extract_style_frame(texts))


def test_style_features_detect_markdown_and_punctuation():
    plain = extract_style_frame(["The answer is a simple sentence about the sea."])
    marked = extract_style_frame(
        ["**Bold** claim:\n\n- first item\n- second item with `code` and an em—dash!"]
    )
    names = list(FEATURE_NAMES)
    assert marked[0, names.index("rate_bold")] > plain[0, names.index("rate_bold")]
    assert marked[0, names.index("rate_bullet")] > 0
    assert marked[0, names.index("rate_dash")] > plain[0, names.index("rate_dash")]
    assert marked[0, names.index("length_paragraphs")] >= 2


def test_style_feature_subset_selection():
    extractor = StyleFeatureExtractor(names=["ttr", "rate_comma"])
    matrix = extractor.transform(["a, b, c"])
    assert matrix.shape == (1, 2)
    assert extractor.get_feature_names_out().tolist() == ["ttr", "rate_comma"]


def test_tfidf_builder_is_fitted_on_train_only():
    train = ["alpha beta gamma" * 5, "delta epsilon zeta" * 4, "eta theta iota"]
    builder = TextMatrixBuilder(
        kinds=("word",), params={"max_features": 1000, "min_df": 1}
    ).fit(train)
    vocabulary = builder._vectorizers[0].get_feature_names_out()
    assert "zeta" in vocabulary
    assert "zzzunseentoken" not in vocabulary

    unseen = builder.transform(["zzzunseentoken appears here"])
    assert unseen.shape == (1, builder.n_output_features_)
    assert unseen.nnz == 0


@pytest.mark.parametrize(
    "kinds, style",
    [(["char"], False), (["word"], False), (["char_wb"], False), ([], True),
     (["char", "word"], False), (["char"], True)],
)
def test_builder_shapes_and_dtype(train_df, kinds, style):
    texts = train_df["text"].head(40).tolist()
    builder = TextMatrixBuilder(
        kinds=tuple(kinds), style=style, params={"max_features": 2000, "min_df": 1}
    ).fit(texts)
    matrix = builder.transform(texts[:5])
    assert matrix.shape[0] == 5
    assert matrix.shape[1] == builder.n_output_features_
    assert (sparse.issparse(matrix) and matrix.dtype == np.float32) or (
        isinstance(matrix, np.ndarray) and matrix.dtype == np.float32
    )
    assert density(matrix) >= 0.0


def test_builder_requires_a_feature_source():
    with pytest.raises(FeatureError):
        TextMatrixBuilder(kinds=(), style=False)
    with pytest.raises(FeatureError):
        TextMatrixBuilder(kinds=("char",), nuisance_k=4)  # needs an SVD
    with pytest.raises(FeatureError):
        TextMatrixBuilder(kinds=("bogus",))


def test_builder_transform_before_fit_fails():
    with pytest.raises(FeatureError, match="before fit"):
        TextMatrixBuilder(kinds=("char",)).transform(["text"])


def test_svd_and_nuisance_builder_outputs_dense():
    texts = ["red green blue " * 3, "cyan magenta yellow " * 3, "black white " * 3] * 6
    groups = ["p1", "p2", "p3"] * 6
    builder = TextMatrixBuilder(
        kinds=("word",),
        params={"max_features": 500, "min_df": 1},
        svd=4,
        nuisance_k=1,
    ).fit(texts, groups=groups)
    matrix = builder.transform(texts[:3])
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 4)
    assert np.isfinite(matrix).all()


def _texts_and_groups():
    texts, groups = [], []
    for prompt in range(6):
        for variant in range(4):
            texts.append(
                f"Prompt {prompt} answer {variant}: "
                + ("step by step we compute the value. " * (variant + 1))
            )
            groups.append(f"p{prompt}")
    return texts, groups


def test_residual_stage_leaves_lambda_zero_untouched():
    texts, groups = _texts_and_groups()
    plain = TextMatrixBuilder(kinds=(), style=True).fit(texts, groups=groups)
    resid = TextMatrixBuilder(kinds=(), style=True, residual_lambda=0.0).fit(texts, groups=groups)
    left, right = plain.transform(texts), resid.transform(texts)
    assert sparse.issparse(left) and sparse.issparse(right)
    assert np.allclose(left.toarray(), right.toarray())
    assert resid.residual_ is None


def test_residual_stage_refuses_sparse_without_svd():
    texts, groups = _texts_and_groups()
    with pytest.raises(FeatureError, match="residual"):
        TextMatrixBuilder(kinds=("char",), residual_lambda=0.5).fit(texts, groups=groups)
    with pytest.raises(FeatureError, match="residual_lambda"):
        TextMatrixBuilder(style=True, residual_lambda=1.5)


def test_residual_stage_needs_groups_to_fit():
    texts, _ = _texts_and_groups()
    with pytest.raises(FeatureError, match="prompt groups"):
        TextMatrixBuilder(kinds=(), style=True, residual_lambda=1.0).fit(texts)


def test_residual_state_is_fitted_on_train_side_only():
    # Transforming a held-out prompt must not depend on which other rows are
    # scored alongside it: the neutralizer is a function of one passage.
    texts, groups = _texts_and_groups()
    builder = TextMatrixBuilder(kinds=(), style=True, residual_lambda=1.0).fit(texts, groups=groups)
    single = builder.transform([texts[0]])
    bigger = builder.transform([texts[0], texts[1]])
    assert np.allclose(single[0], bigger[0])
    assert builder.residual_ is not None and builder.residual_.n_groups_ > 1


def test_residual_stage_on_svd_space_stays_dense_and_ordered():
    texts, groups = _texts_and_groups()
    builder = TextMatrixBuilder(
        kinds=("word",),
        params={"max_features": 500, "min_df": 1},
        svd=8,
        nuisance_k=2,
        residual_lambda=0.5,
    ).fit(texts, groups=groups)
    matrix = builder.transform(texts[:3])
    assert isinstance(matrix, np.ndarray) and matrix.shape == (3, 8)
    assert np.isfinite(matrix).all()
