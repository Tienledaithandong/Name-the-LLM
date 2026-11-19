"""Character TF-IDF with topic words reduced to fixed length markers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
from unicodedata import category

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.constants import RANDOM_STATE
from src.features_style import FUNCTION_WORDS
from src.model import ModelBundle, ModelError, estimator_from_config

LENGTH_BUCKET_MAXIMA = (3, 6)
# Single code points from the private-use area avoid adding punctuation to the
# transformed passage or producing character n-grams from marker names.
LENGTH_MARKERS = ("\ue000", "\ue001", "\ue002")
_FUNCTION_WORDS = frozenset(word.casefold() for word in FUNCTION_WORDS)


def mask_content_words(text: str) -> str:
    """Keep function words and layout, replacing other alphanumeric tokens.

    Unicode combining marks attached to a token belong to that token but do
    not add to its length. Apostrophes and all other punctuation stay literal.
    """
    pieces: list[str] = []
    offset = 0
    while offset < len(text):
        if not text[offset].isalnum():
            pieces.append(text[offset])
            offset += 1
            continue
        start = offset
        length = 0
        while offset < len(text):
            character = text[offset]
            if character.isalnum():
                length += 1
            elif not category(character).startswith("M"):
                break
            offset += 1
        word = text[start:offset]
        if word.casefold() in _FUNCTION_WORDS:
            pieces.append(word)
        else:
            bucket = sum(length > maximum for maximum in LENGTH_BUCKET_MAXIMA)
            pieces.append(LENGTH_MARKERS[bucket])
    return "".join(pieces)


class MaskedCharacterModel:
    """Use the registered raw character model on deterministically masked text."""

    def __init__(
        self, character_config: Mapping[str, Any], *, seed: int = RANDOM_STATE
    ) -> None:
        bundle = estimator_from_config(dict(character_config), seed=seed)
        if not isinstance(bundle, ModelBundle):
            raise ModelError("masking requires the registered character TF-IDF bundle")
        spec = bundle.spec
        if (
            spec.name != "char_tfidf_logreg"
            or spec.features != ("char",)
            or spec.classifier != "logreg"
            or spec.style
            or spec.embedding is not None
            or spec.svd is not None
        ):
            raise ModelError("masking requires the registered char_tfidf_logreg recipe")
        self.bundle = bundle
        self.class_order = bundle.class_order

    def fit(self, texts: Iterable[str], y: ArrayLike) -> MaskedCharacterModel:
        """Fit vocabulary, IDF and the classifier on the supplied training rows."""
        self.bundle.fit([mask_content_words(text) for text in texts], y)
        return self

    def decision_function(self, texts: Iterable[str]) -> NDArray[np.float32]:
        """Return prediction margins in the official label order."""
        return self.bundle.decision_function(mask_content_words(text) for text in texts)
