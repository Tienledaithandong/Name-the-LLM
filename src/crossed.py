"""Inference-time transforms that exploit the crossed prompt x model design.

The training set is a balanced full factorial: every prompt is answered by
17-18 of the 18 models.  The hidden test mirrors this but ships no ``prompt_id``.
If test passages can be grouped by prompt (Phase B), two label-free transforms
become available at inference time:

* :func:`center_scores_within_groups` -- subtract each group's mean score vector
  so passages are scored relative to their prompt-mates (removes prompt
  difficulty / topic offsets).
* :func:`assign_one_per_model` -- within a group, solve a maximum-weight
  assignment so each model is used at most once (the design's mutual-exclusion
  constraint).

Both operate on a score matrix alone (no labels), so they are transductive but
label-free.  They are never applied inside reported grouped-CV *training*
scores; the Phase A oracle uses known train prompt groups only to measure the
ceiling.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


class CrossedError(ValueError):
    """Raised for an unusable score/group pairing."""


def _check(scores: np.ndarray, groups: np.ndarray) -> None:
    if scores.ndim != 2:
        raise CrossedError("scores must be 2-D (rows x classes)")
    if len(groups) != scores.shape[0]:
        raise CrossedError("groups must have one entry per score row")


def _group_indices(groups: np.ndarray):
    groups = np.asarray(groups).astype(str)
    for group in np.unique(groups):
        yield group, np.flatnonzero(groups == group)


def center_scores_within_groups(scores, groups) -> np.ndarray:
    """Subtract the per-group mean score vector from each row in the group."""
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    _check(scores, groups)
    out = scores.copy()
    for _, idx in _group_indices(groups):
        out[idx] -= out[idx].mean(axis=0)
    return out


def assign_one_per_model(scores, groups) -> np.ndarray:
    """Predicted class index per row under a one-passage-per-model constraint.

    Within each group a maximum-weight assignment pairs passages to distinct
    models.  Groups with more rows than classes leave the surplus rows on their
    plain argmax (the constraint cannot cover them).
    """
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups).astype(str)
    _check(scores, groups)
    if not np.isfinite(scores).all():
        raise CrossedError("assignment needs finite scores; normalise sentinels first")
    pred = scores.argmax(axis=1)
    for _, idx in _group_indices(groups):
        block = scores[idx]
        rows, cols = linear_sum_assignment(-block)
        pred[idx[rows]] = cols
    return pred