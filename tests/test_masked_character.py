"""Contracts for the fixed character masking experiment."""

from __future__ import annotations

import numpy as np

from src.config import load_config
from src.constants import LABELS, N_CLASSES
from src.masked_character import LENGTH_MARKERS, MaskedCharacterModel, mask_content_words


def test_masking_preserves_function_words_and_layout() -> None:
    short, medium, long = LENGTH_MARKERS
    text = "THE\tcat and café—“contextual,”\n\nA dog's l’esprit. **a-b**"
    assert mask_content_words(text) == (
        f"THE\t{short} and {medium}—“{long},”\n\n"
        f"A {short}'{short} {short}’{medium}. **a-{short}**"
    )


def test_masking_counts_letters_with_combining_marks_once() -> None:
    short, medium, long = LENGTH_MARKERS
    assert mask_content_words("été étés élanée élanées") == f"{short} {medium} {medium} {long}"
    assert mask_content_words("…\n\t❗") == "…\n\t❗"


def test_masked_character_uses_training_only_idf_and_canonical_scores() -> None:
    config = load_config("configs/submission.yaml")["models"][0]
    model = MaskedCharacterModel(config, seed=13).fit(
        ["THE cat at home.", "THE pig at home.", "in a wonderful way!", "in a marvellous way!"],
        np.asarray([17, 17, 2, 2]),
    )
    vectorizer = model.bundle.builder._vectorizers[0]
    vocabulary = dict(vectorizer.vocabulary_)
    idf = vectorizer.idf_.copy()
    assert "THE" in vocabulary
    assert "cat" not in vocabulary
    assert any(LENGTH_MARKERS[0] in ngram for ngram in vocabulary)

    held_out = ["THE dog at home.", "UNSEEN📝 in a ridiculous place!"]
    scores = model.decision_function(held_out)
    assert model.class_order == LABELS
    assert scores.shape == (2, N_CLASSES)
    assert scores[0].argmax() == 17
    assert np.all(scores[:, [i for i in range(N_CLASSES) if i not in {2, 17}]] == -1e9)
    np.testing.assert_allclose(scores[:1], model.decision_function(held_out[:1]))
    assert vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(vectorizer.idf_, idf)
