"""Evaluation metrics.

The official metric is balanced accuracy: the mean, over the 18 candidate
models, of that model's recall.  Classes absent from ``y_true`` have no defined
recall and are skipped (and the fact is reported) instead of being scored as
zero, which would silently inflate or deflate a comparison.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.constants import LABELS


def _as_label_array(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in "iu":
        raise TypeError("metrics take label strings, not integer class indices")
    return arr.astype(str)


def per_class_recall(
    y_true, y_pred, labels: Sequence[str] = LABELS
) -> dict[str, float]:
    """Recall per label; ``nan`` where the label has no support."""
    yt = _as_label_array(y_true)
    yp = _as_label_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    out: dict[str, float] = {}
    for label in labels:
        mask = yt == label
        support = int(mask.sum())
        out[label] = float((yp[mask] == label).sum() / support) if support else float("nan")
    return out


def class_support(y_true, labels: Sequence[str] = LABELS) -> dict[str, int]:
    yt = _as_label_array(y_true)
    return {label: int((yt == label).sum()) for label in labels}


def balanced_accuracy(y_true, y_pred, labels: Sequence[str] = LABELS) -> float:
    """Mean recall over labels that actually occur in ``y_true``."""
    recalls = [
        v for v in per_class_recall(y_true, y_pred, labels).values() if not np.isnan(v)
    ]
    if not recalls:
        raise ValueError("cannot compute balanced accuracy: no supported classes")
    return float(np.mean(recalls))


def macro_f1(y_true, y_pred, labels: Sequence[str] = LABELS) -> float:
    """Unweighted mean F1 over labels with support (diagnostic only)."""
    yt = _as_label_array(y_true)
    yp = _as_label_array(y_pred)
    scores = []
    for label in labels:
        true_pos = int(((yt == label) & (yp == label)).sum())
        support = int((yt == label).sum())
        predicted = int((yp == label).sum())
        if support == 0 or predicted == 0:
            continue
        precision = true_pos / predicted
        recall = true_pos / support
        if precision + recall > 0:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else float("nan")


def confusion_matrix(
    y_true, y_pred, labels: Sequence[str] = LABELS
) -> np.ndarray:
    """Rows are true labels, columns predicted, in ``labels`` order."""
    yt = _as_label_array(y_true)
    yp = _as_label_array(y_pred)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(yt, yp, strict=True):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    return matrix


def evaluate(
    y_true,
    y_pred,
    labels: Sequence[str] = LABELS,
    model: str | None = None,
) -> dict:
    """Full evaluation dict for one set of predictions."""
    yt = _as_label_array(y_true)
    yp = _as_label_array(y_pred)
    support = class_support(yt, labels)
    recalls = per_class_recall(yt, yp, labels)
    result = {
        "model": model,
        "n": int(yt.size),
        "balanced_accuracy": balanced_accuracy(yt, yp, labels),
        "macro_f1": macro_f1(yt, yp, labels),
        "per_class_recall": recalls,
        "support": support,
        "classes_absent": [label for label, n in support.items() if n == 0],
    }
    return result


def aggregate_folds(fold_metrics: Iterable[dict]) -> dict:
    """Mean/std/min/max of per-fold balanced accuracy."""
    scores = [m["balanced_accuracy"] for m in fold_metrics]
    if not scores:
        raise ValueError("no fold metrics to aggregate")
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "n_folds": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "fold_scores": [float(s) for s in arr],
    }


def provider_of(label: str) -> str:
    from src.constants import PROVIDER_OF

    return PROVIDER_OF.get(label, "unknown")


def within_provider_confusions(
    y_true, y_pred, labels: Sequence[str] = LABELS
) -> dict[str, int]:
    """Count how often errors stay inside the same provider (H4 diagnostic)."""
    yt = _as_label_array(y_true)
    yp = _as_label_array(y_pred)
    wrong = yt != yp
    same = np.asarray([provider_of(t) == provider_of(p) for t, p in zip(yt, yp, strict=True)])
    return {
        "errors": int(wrong.sum()),
        "within_provider_errors": int((wrong & same).sum()),
        "cross_provider_errors": int((wrong & ~same).sum()),
    }


def reports_frame(results: Sequence[dict]) -> pd.DataFrame:
    """Tabular view used by reports and the registry."""
    rows = []
    for result in results:
        rows.append(
            {
                "model": result.get("model"),
                "n": result["n"],
                "balanced_accuracy": result["balanced_accuracy"],
                "macro_f1": result["macro_f1"],
            }
        )
    return pd.DataFrame(rows)
