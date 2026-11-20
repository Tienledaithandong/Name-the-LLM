from __future__ import annotations

import pytest

from experiments.residual_headroom import headroom_table, representation_matrix


def test_headroom_table_reports_every_space():
    texts = [f"passage {i} about topic {i % 3}" for i in range(30)]
    groups = [str(i % 3) for i in range(30)]
    table = headroom_table(texts, groups, spaces=("style",))
    cell = table["style"]
    assert 0.0 <= cell["between_prompt_share"] <= 1.0
    assert cell["n_prompts"] == 3 and cell["n_features"] > 0


def test_identical_prompts_give_zero_share():
    # Style features are a pure function of the text, so identical text has no
    # total variance to explain.
    texts = ["same text here"] * 6
    table = headroom_table(texts, ["p"] * 6, spaces=("style",))
    assert table["style"]["between_prompt_share"] == pytest.approx(0.0, abs=1e-9)


def test_representation_matrix_is_dense_and_finite():
    texts = [f"answer {i} with a short sentence." for i in range(12)]
    matrix = representation_matrix(texts, "style")
    assert matrix.ndim == 2 and matrix.shape[0] == 12
    assert matrix.dtype.name == "float32"


def test_unknown_space_is_rejected():
    with pytest.raises(ValueError, match="unknown space"):
        representation_matrix(["a"], "nope")


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="same length"):
        headroom_table(["a", "b"], ["p1"], spaces=("style",))
