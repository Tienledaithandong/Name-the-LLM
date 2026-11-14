"""Recover prompt groups from passage semantics (Phase B).

``test.csv`` ships no ``prompt_id``, but passages answering the same prompt share
topic and vocabulary, so they cluster together in a semantic embedding space.
This module turns an embedding matrix into group labels and scores how well the
recovered groups match the truth (used to validate on held-out train prompts
before any test-time use).

The clustering is label-free and transductive: it uses only the passages being
grouped, never their model labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)


class GroupingError(ValueError):
    """Raised for an unusable clustering request."""


def cluster_embeddings(
    embeddings,
    distance_threshold: float | None = None,
    n_clusters: int | None = None,
    linkage: str = "average",
) -> np.ndarray:
    """Agglomerative clustering on cosine distance.

    Provide exactly one of ``distance_threshold`` (let the number of groups
    emerge) or ``n_clusters`` (fix it, e.g. ``n_rows / 18`` from the known
    full-factorial design).
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] < 2:
        raise GroupingError("embeddings must be 2-D with at least two rows")
    if (distance_threshold is None) == (n_clusters is None):
        raise GroupingError("set exactly one of distance_threshold or n_clusters")
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(emb)


def grouping_purity(true_groups, pred_labels) -> dict:
    """Agreement between recovered clusters and true prompt groups.

    ``recovered_prompt_fraction`` is the strict, assignment-relevant measure:
    the share of true prompts whose passages land in a single cluster that
    contains *only* that prompt (a perfect bijection for that prompt).
    """
    true_groups = np.asarray(true_groups).astype(str)
    pred_labels = np.asarray(pred_labels)
    if len(true_groups) != len(pred_labels):
        raise GroupingError("true_groups and pred_labels must align row-wise")
    n = len(true_groups)

    # weighted majority purity
    correct = 0
    for cluster in np.unique(pred_labels):
        members = true_groups[pred_labels == cluster]
        _, counts = np.unique(members, return_counts=True)
        correct += int(counts.max())
    purity = correct / n if n else 0.0

    # strict per-prompt recovery
    recovered = 0
    prompts = np.unique(true_groups)
    for prompt in prompts:
        rows = np.flatnonzero(true_groups == prompt)
        labels = pred_labels[rows]
        if labels.size and np.all(labels == labels[0]):
            cluster_rows = np.flatnonzero(pred_labels == labels[0])
            if cluster_rows.size == rows.size:
                recovered += 1
    recovery = recovered / prompts.size if prompts.size else 0.0

    homo, comp, vmeasure = homogeneity_completeness_v_measure(true_groups, pred_labels)
    return {
        "purity": float(purity),
        "adjusted_rand_index": float(adjusted_rand_score(true_groups, pred_labels)),
        "homogeneity": float(homo),
        "completeness": float(comp),
        "v_measure": float(vmeasure),
        "recovered_prompt_fraction": float(recovery),
        "n_true_groups": int(prompts.size),
        "n_clusters": int(np.unique(pred_labels).size),
    }