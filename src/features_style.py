"""Deterministic stylometric features.

The extractor is stateless: every value is a pure function of one passage, so
there is nothing to fit and therefore nothing that can leak across folds.  That
is deliberate — the leakage risk lives in vectoriser vocabularies, not here.

All rates are per 100 word tokens so passages of different lengths are
comparable.  Undefined ratios (e.g. tokens per sentence in an empty string)
fall back to 0.0 rather than NaN, because downstream linear models cannot
consume NaN.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import numpy as np
from scipy import sparse

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
LIST_LINE_RE = re.compile(r"^\s*[-*\u2022]\s+", re.MULTILINE)
NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
CONTRACTION_RE = re.compile(r"\b(?:[A-Za-z]+'(?:s|t|re|ve|ll|d|m))\b", re.IGNORECASE)

FUNCTION_WORDS: tuple[str, ...] = (
    "the", "of", "and", "to", "a", "in", "that", "is", "was",
    "it", "for", "as", "with", "be", "on", "this", "are", "by",
)

HEDGES: tuple[str, ...] = (
    "perhaps", "maybe", "arguably", "it seems", "likely", "possibly",
    "in some sense", "roughly", "somewhat", "tend to", "may", "might",
)

TRANSITIONS: tuple[str, ...] = (
    "however", "moreover", "therefore", "in contrast", "by contrast",
    "that said", "furthermore", "on the other hand", "in practice",
    "for example", "consequently", "meanwhile",
)


def _rate(count: float, n_words: int) -> float:
    return 100.0 * float(count) / max(n_words, 1)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(values)) if len(values) else 0.0


def _extract(text: str) -> dict[str, float]:
    text = text or ""
    lowered = text.lower()
    tokens = WORD_RE.findall(lowered)
    n_words = len(tokens)
    n_chars = len(text)
    stripped = text.strip()
    sentences = [s for s in SENTENCE_SPLIT_RE.split(stripped) if s.strip()]
    paragraphs = [p for p in PARAGRAPH_SPLIT_RE.split(stripped) if p.strip()]
    sentence_words = [len(WORD_RE.findall(s)) for s in sentences]
    word_lengths = [len(t) for t in tokens]

    letters = [c for c in text if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    digits = sum(1 for c in text if c.isdigit())
    non_ascii = sum(1 for c in text if ord(c) > 127)
    emoji = sum(1 for c in text if ord(c) > 0x1F000)

    features: dict[str, float] = {
        "length_words": float(n_words),
        "length_chars": float(n_chars),
        "length_sentences": float(len(sentences)),
        "length_paragraphs": float(len(paragraphs)),
        "mean_sentence_words": float(np.mean(sentence_words)) if sentence_words else 0.0,
        "std_sentence_words": _std(sentence_words),
        "mean_word_len": float(np.mean(word_lengths)) if word_lengths else 0.0,
        "std_word_len": _std(word_lengths),
        "log_length_words": float(np.log1p(n_words)),
        "ttr": _safe_div(len(set(tokens)), n_words),
        "hapax_ratio": _safe_div(sum(1 for t in set(tokens) if tokens.count(t) == 1), n_words),
        "rate_comma": _rate(text.count(","), n_words),
        "rate_semicolon": _rate(text.count(";"), n_words),
        "rate_colon": _rate(text.count(":"), n_words),
        "rate_paren": _rate(text.count("(") + text.count(")"), n_words),
        "rate_dash": _rate(text.count("\u2014") + text.count("\u2013"), n_words),
        "rate_hyphen": _rate(text.count("-"), n_words),
        "rate_dquote": _rate(text.count('"'), n_words),
        "rate_squote": _rate(text.count("'"), n_words),
        "rate_ellipsis": _rate(text.count("\u2026") + text.count("..."), n_words),
        "rate_exclaim": _rate(text.count("!"), n_words),
        "rate_question": _rate(text.count("?"), n_words),
        "rate_newline": _rate(text.count("\n"), n_words),
        "rate_bullet": _rate(len(LIST_LINE_RE.findall(text)), n_words),
        "rate_numbered": _rate(len(NUMBERED_LINE_RE.findall(text)), n_words),
        "rate_bold": _rate(text.count("**") / 2.0, n_words),
        "rate_backtick": _rate(text.count("`"), n_words),
        "rate_digit": _safe_div(digits, n_chars),
        "rate_upper": _safe_div(upper, len(letters)),
        "rate_nonascii": _safe_div(non_ascii, n_chars),
        "rate_emoji": _safe_div(emoji, n_chars),
        "contraction_rate": _rate(len(CONTRACTION_RE.findall(text)), n_words),
        "sentence_start_upper_ratio": _safe_div(
            sum(1 for s in sentences if (m := re.search(r"[A-Za-z]", s)) and m.group(0).isupper()),
            len(sentences),
        ),
        "ends_with_period_ratio": _safe_div(
            sum(1 for s in sentences if s.strip().endswith(".")), len(sentences)
        ),
        "hedge_rate": _rate(sum(lowered.count(h) for h in HEDGES), n_words),
        "transition_rate": _rate(sum(lowered.count(t) for t in TRANSITIONS), n_words),
        "function_word_share": _safe_div(
            sum(1 for t in tokens if t in FUNCTION_WORDS), n_words
        ),
        "unicode_categories_distinct": float(
            len({unicodedata.category(c) for c in text if not c.isspace()})
        ),
    }
    for word in FUNCTION_WORDS:
        features[f"fw_{word}"] = _rate(tokens.count(word), n_words)
    return features


FEATURE_NAMES: tuple[str, ...] = tuple(_extract("").keys())


class StyleFeatureExtractor:
    """Scikit-learn style transformer returning a sparse float32 matrix."""

    def __init__(self, names: Sequence[str] | None = None):
        self.names = tuple(names) if names is not None else FEATURE_NAMES
        self._positions = [FEATURE_NAMES.index(n) for n in self.names]

    def fit(self, texts, y=None):  # noqa: ARG002 - sklearn API
        self.n_features_in_ = len(self.names)
        return self

    def transform(self, texts) -> sparse.csr_matrix:
        rows = []
        for text in texts:
            values = np.fromiter(_extract(str(text)).values(), dtype=np.float32)
            rows.append(values[self._positions])
        if not rows:
            return sparse.csr_matrix((0, len(self.names)), dtype=np.float32)
        return sparse.csr_matrix(np.vstack(rows).astype(np.float32, copy=False))

    def fit_transform(self, texts, y=None) -> sparse.csr_matrix:
        return self.fit(texts, y).transform(texts)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ARG002
        return np.asarray(self.names, dtype=object)


def extract_style_frame(texts: Sequence[str]) -> "np.ndarray":
    """Dense numpy view, used by diagnostics and tests."""
    return StyleFeatureExtractor().transform(texts).toarray()
