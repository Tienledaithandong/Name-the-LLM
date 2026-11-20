"""The near-duplicate audit must separate same-prompt similarity from cross-fold leaks.

Plan §2 L94 asks whether near-duplicate or augmented texts cross folds. These
tests pin ``experiments/near_duplicates.py``: self-pairs are excluded, similar
passages are counted once per ordered pair, and the cross-fold / same-prompt
breakdown is what decides whether a similarity is a leakage risk.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "near_duplicates.py"
_spec = importlib.util.spec_from_file_location("near_duplicates", SCRIPT)
near_duplicates = importlib.util.module_from_spec(_spec)
sys.modules["near_duplicates"] = near_duplicates
_spec.loader.exec_module(near_duplicates)

ALPHA = "alpha beta gamma delta epsilon"
ALPHA_VARIANT = "alpha beta gamma delta zeta"
BETA = "one two three four five six"
GAMMA = "red green blue yellow purple orange"


def test_self_pairs_are_excluded() -> None:
    audit = near_duplicates.pair_audit([ALPHA], np.array([0]), np.array(["p0"]))
    assert audit["n_rows"] == 1
    assert audit["n_pairs_examined"] == 0
    assert audit["thresholds"]["0.3"]["pairs"] == 0
    assert audit["max_offdiagonal_cosine_quantiles"]["1.0"] == 0.0


def test_similar_passages_in_one_prompt_are_not_cross_fold() -> None:
    texts = [ALPHA, ALPHA_VARIANT, BETA, GAMMA]
    folds = np.array([0, 0, 1, 1])
    prompts = np.array(["p0", "p0", "p1", "p2"])
    audit = near_duplicates.pair_audit(texts, folds, prompts)
    entry = audit["thresholds"]["0.3"]
    assert entry["pairs"] >= 1
    assert entry["cross_fold"] == 0
    assert entry["same_prompt"] == entry["pairs"]
    assert entry["cross_fold_different_prompt"] == 0
    assert audit["n_pairs_examined"] == 6


def test_the_same_similarity_across_folds_is_flagged() -> None:
    texts = [ALPHA, ALPHA_VARIANT, BETA, GAMMA]
    folds = np.array([0, 1, 1, 2])
    prompts = np.array(["p0", "p1", "p2", "p3"])
    audit = near_duplicates.pair_audit(texts, folds, prompts)
    entry = audit["thresholds"]["0.3"]
    assert entry["cross_fold"] == entry["pairs"] >= 1
    assert entry["same_prompt"] == 0
    assert entry["cross_fold_different_prompt"] == entry["pairs"]


def test_exact_duplicates_reach_cosine_one() -> None:
    audit = near_duplicates.pair_audit([ALPHA, ALPHA, BETA], np.array([0, 1, 1]), np.array(["p0", "p1", "p2"]))
    assert audit["thresholds"]["0.9"]["pairs"] == 1
    assert audit["thresholds"]["0.9"]["cross_fold"] == 1
    assert audit["top_pairs"][0]["cosine"] == pytest.approx(1.0, abs=1e-9)


def test_counts_are_monotone_in_the_threshold() -> None:
    texts = [ALPHA, ALPHA_VARIANT, BETA, GAMMA, ALPHA + " extra words here"]
    folds = np.array([0, 0, 1, 1, 2])
    prompts = np.array(["p0", "p0", "p1", "p2", "p3"])
    audit = near_duplicates.pair_audit(texts, folds, prompts)
    counts = [audit["thresholds"][str(t)]["pairs"] for t in (0.3, 0.5, 0.7, 0.9)]
    assert counts == sorted(counts, reverse=True)


def test_top_pairs_are_sorted_unique_and_ordered() -> None:
    texts = [ALPHA, ALPHA_VARIANT, ALPHA + " extra", BETA, GAMMA]
    audit = near_duplicates.pair_audit(texts, np.zeros(5, dtype=int), np.array(["p0"] * 5))
    pairs = audit["top_pairs"]
    assert pairs
    assert all(p["row_a"] < p["row_b"] for p in pairs)
    assert len({(p["row_a"], p["row_b"]) for p in pairs}) == len(pairs)
    assert [p["cosine"] for p in pairs] == sorted((p["cosine"] for p in pairs), reverse=True)
    assert len(pairs) <= near_duplicates.TOP_PAIRS