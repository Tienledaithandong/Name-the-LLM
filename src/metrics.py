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
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    scores = []
    for label in labels:
        true_pos = int(((yt == label) & (yp == label)).sum())
        support = int((yt == label).sum())
        predicted = int((yp == label).sum())
        if support == 0:
            continue
        scores.append(2 * true_pos / (support + predicted))
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


def paired_difference(base, other) -> np.ndarray:
    """Elementwise ``other - base`` for paired scores (e.g. method vs argmax).

    Both sequences must have the same length: each pair is measured on the same
    rows/subsample, so the difference removes the shared subsampling noise.
    """
    a = np.asarray(base, dtype=np.float64)
    b = np.asarray(other, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired length mismatch: {a.shape} vs {b.shape}")
    return b - a


def mean_ci(values, confidence: float = 0.95) -> dict:
    """Mean, sample std (ddof=1), n and a Student-t confidence interval.

    Used to summarise a statistic repeated over subsample seeds/folds. With a
    single value the interval collapses to the point (no spread estimate).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        raise ValueError("no values to summarise")
    mean = float(arr.mean())
    if n < 2:
        return {"n": n, "mean": mean, "std": 0.0,
                "ci_low": mean, "ci_high": mean, "ci_half": 0.0}
    std = float(arr.std(ddof=1))
    from scipy.stats import t

    half = float(t.ppf(0.5 + confidence / 2.0, n - 1) * std / np.sqrt(n))
    return {"n": n, "mean": mean, "std": std,
            "ci_low": mean - half, "ci_high": mean + half, "ci_half": half}


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


def length_band_metrics(
    y_true,
    y_pred,
    lengths,
    edges: Sequence[float] = (0, 50, 100, 200, 400, 800),
    labels: Sequence[str] = LABELS,
) -> list[dict]:
    """Balanced accuracy and error mix per passage-length band (plan §9 item 9).

    Bands are ``[edges[i], edges[i+1])`` with the last one open-ended, so a
    length exactly on an edge belongs to the upper band. Each band's balanced
    accuracy averages only the classes it contains; bands with different class
    coverage are therefore not directly comparable, which is why
    ``classes_present`` is reported alongside it. Empty bands yield ``None``.
    """
    yt = _as_label_array(y_true)
    yp = _as_label_array(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    words = np.asarray(lengths, dtype=np.float64)
    if words.shape != yt.shape:
        raise ValueError(f"length shape mismatch: {words.shape} vs {yt.shape}")
    if not np.isfinite(words).all() or (words < 0).any():
        raise ValueError("lengths must be finite and non-negative")
    bounds = [float(e) for e in edges]
    if not bounds or bounds[0] != 0.0:
        raise ValueError("edges must start at 0 so every passage is banded")
    if any(b <= a for a, b in zip(bounds, bounds[1:])):
        raise ValueError("edges must be strictly increasing")

    out: list[dict] = []
    for i, lower in enumerate(bounds):
        upper = bounds[i + 1] if i + 1 < len(bounds) else None
        mask = words >= lower
        if upper is not None:
            mask &= words < upper
        n = int(mask.sum())
        band = {
            "band": f"[{lower:g}, {upper:g})" if upper is not None else f"[{lower:g}, inf)",
            "lower": lower,
            "upper": upper,
            "n": n,
            "classes_present": int(np.unique(yt[mask]).size) if n else 0,
            "balanced_accuracy": None,
            "accuracy": None,
            "errors": 0,
            "within_provider_errors": 0,
            "cross_provider_errors": 0,
            "mean_words": None,
            "median_words": None,
        }
        if n:
            truth, pred = yt[mask], yp[mask]
            band["balanced_accuracy"] = balanced_accuracy(truth, pred, labels)
            band["accuracy"] = float((truth == pred).mean())
            band.update(within_provider_confusions(truth, pred, labels))
            band["mean_words"] = float(words[mask].mean())
            band["median_words"] = float(np.median(words[mask]))
        out.append(band)
    return out


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
