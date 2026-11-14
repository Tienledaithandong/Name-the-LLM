"""Unit tests for prompt-group recovery (src/grouping.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.grouping import GroupingError, cluster_embeddings, grouping_purity


def _separable_embeddings(rng, n_per_group=12, n_groups=3, dim=24):
    centers = np.eye(n_groups, dim) * 10.0
    emb, truth = [], []
    for g in range(n_groups):
        pts = centers[g] + rng.normal(0, 0.1, size=(n_per_group, dim))
        emb.append(pts)
        truth.extend([f"p{g}"] * n_per_group)
    return np.vstack(emb), np.asarray(truth)


def test_recovers_known_clusters_by_count():
    rng = np.random.default_rng(0)
    emb, truth = _separable_embeddings(rng)
    labels = cluster_embeddings(emb, n_clusters=3)
    metrics = grouping_purity(truth, labels)
    assert metrics["adjusted_rand_index"] == pytest.approx(1.0)
    assert metrics["purity"] == pytest.approx(1.0)
    assert metrics["recovered_prompt_fraction"] == pytest.approx(1.0)
    assert metrics["n_clusters"] == 3


def test_recovers_known_clusters_by_threshold():
    rng = np.random.default_rng(1)
    emb, truth = _separable_embeddings(rng)
    labels = cluster_embeddings(emb, distance_threshold=0.5)
    metrics = grouping_purity(truth, labels)
    assert metrics["recovered_prompt_fraction"] == pytest.approx(1.0)


def test_merged_clusters_lower_recovery():
    # Two true prompts forced into one cluster: purity stays high but strict
    # recovery drops because the cluster is not a single prompt.
    truth = np.asarray(["a"] * 5 + ["b"] * 5)
    labels = np.zeros(10, dtype=int)
    metrics = grouping_purity(truth, labels)
    assert metrics["purity"] == pytest.approx(0.5)
    assert metrics["recovered_prompt_fraction"] == pytest.approx(0.0)
    assert metrics["n_clusters"] == 1


def test_split_prompt_lowers_recovery():
    # One true prompt split across two clusters: neither cluster is the whole
    # prompt, so strict recovery is 0 even though purity is 1.
    truth = np.asarray(["a"] * 6)
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    metrics = grouping_purity(truth, labels)
    assert metrics["purity"] == pytest.approx(1.0)
    assert metrics["recovered_prompt_fraction"] == pytest.approx(0.0)


def test_requires_exactly_one_stop_criterion():
    emb = np.random.default_rng(0).normal(size=(6, 8))
    with pytest.raises(GroupingError):
        cluster_embeddings(emb)
    with pytest.raises(GroupingError):
        cluster_embeddings(emb, distance_threshold=0.5, n_clusters=2)


def test_rejects_tiny_input():
    with pytest.raises(GroupingError):
        cluster_embeddings(np.zeros((1, 4)), n_clusters=1)


def test_purity_row_mismatch_raises():
    with pytest.raises(GroupingError):
        grouping_purity(np.asarray(["a", "b"]), np.asarray([0]))