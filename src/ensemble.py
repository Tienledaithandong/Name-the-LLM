"""Score blending and ensemble selection from fold-aligned OOF scores.

LinearSVC margins are not probabilities, so blending happens on row-normalised
margins (per-passage z-scores) rather than on softmax-like values.  Weights are
chosen by greedy selection with replacement (Caruana et al.) on pooled OOF,
which has few effective degrees of freedom and resists overfitting 2,699 rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.constants import LABELS, N_CLASSES
from src.data import indices_to_labels
from src.metrics import balanced_accuracy

SENTINEL = -1e8


def load_oof(artifacts_dir: str | Path) -> dict[str, dict]:
    """Load every saved OOF score matrix under ``<artifacts>/oof``."""
    oof_dir = Path(artifacts_dir) / "oof"
    out: dict[str, dict] = {}
    for path in sorted(oof_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        scores = np.asarray(data["scores"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.int64)
        labels = [str(v) for v in data["labels"]]
        if labels != list(LABELS):
            raise ValueError(f"{path.name}: class order {labels} is not the official order")
        if scores.shape != (y.size, N_CLASSES):
            raise ValueError(f"{path.name}: unexpected score shape {scores.shape}")
        out[path.stem] = {"scores": scores, "y": y}
    if not out:
        raise FileNotFoundError(
            f"no OOF files in {oof_dir}; run the training step first"
        )
    return out


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Row-wise z-scores, ignoring the -1e9 placeholder for absent fold classes."""
    scores = np.asarray(scores, dtype=np.float64).copy()
    scores[scores < SENTINEL] = np.nan
    mean = np.nanmean(scores, axis=1, keepdims=True)
    std = np.nanstd(scores, axis=1, keepdims=True)
    std[~np.isfinite(std) | (std < 1e-9)] = 1.0
    out = (scores - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _score_with_weights(normalized: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = None
    for name, weight in weights.items():
        if weight == 0:
            continue
        block = normalized[name] * weight
        total = block if total is None else total + block
    if total is None:
        raise ValueError("all ensemble weights are zero")
    return total


def evaluate_blend(
    normalized: dict[str, np.ndarray], y: np.ndarray, weights: dict[str, float]
) -> dict:
    if not normalized:
        raise ValueError("no models to blend")
    scores = _score_with_weights(normalized, weights)
    pred = indices_to_labels(scores.argmax(axis=1))
    truth = indices_to_labels(y)
    return {
        "balanced_accuracy": balanced_accuracy(truth, pred),
        "weights": {k: float(v) for k, v in weights.items() if v},
    }


def greedy_blend(
    oof: dict[str, dict],
    max_models: int = 4,
    max_picks: int | None = None,
) -> dict:
    """Caruana greedy selection with replacement, maximising pooled OOF."""
    names = sorted(oof)
    if not names:
        raise ValueError("no models")
    y = oof[names[0]]["y"]
    for name in names:
        if not np.array_equal(oof[name]["y"], y):
            raise ValueError(f"{name}: OOF rows are not fold-aligned with {names[0]}")
    normalized = {name: normalize_scores(oof[name]["scores"]) for name in names}

    singles = {name: balanced_accuracy(*_labels(y, normalized[name])) for name in names}
    best_single = max(singles.values())
    best_name = max(singles, key=singles.get)

    counts: dict[str, int] = {best_name: 1}
    picks_limit = max_picks or max_models * 3
    current = best_single
    history = [{"picks": dict(counts), "bal_acc": current}]

    for _ in range(picks_limit - 1):
        candidate_name, candidate_score = None, current
        for name in names:
            trial = dict(counts)
            trial[name] = trial.get(name, 0) + 1
            weights = {k: v / sum(trial.values()) for k, v in trial.items()}
            score = evaluate_blend(normalized, y, weights)["balanced_accuracy"]
            if score > candidate_score + 1e-9:
                candidate_name, candidate_score = name, score
        if candidate_name is None:
            break
        counts[candidate_name] = counts.get(candidate_name, 0) + 1
        current = candidate_score
        history.append({"picks": dict(counts), "bal_acc": current})

    total = sum(counts.values())
    weights = {k: v / total for k, v in counts.items()}
    blended = evaluate_blend(normalized, y, weights)

    # Keep at most ``max_models`` members, dropping the smallest weights.
    if len(weights) > max_models:
        keep = dict(sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:max_models])
        total_keep = sum(keep.values())
        weights = {k: v / total_keep for k, v in keep.items()}
        blended = evaluate_blend(normalized, y, weights)

    return {
        "method": "greedy",
        "weights": weights,
        "pooled_oof_balanced_accuracy": blended["balanced_accuracy"],
        "best_single": {"model": best_name, "balanced_accuracy": best_single},
        "gain_over_best_single": blended["balanced_accuracy"] - best_single,
        "singles": singles,
        "history": history,
        "max_models": max_models,
    }


def _labels(y: np.ndarray, scores: np.ndarray):
    pred = indices_to_labels(scores.argmax(axis=1))
    return indices_to_labels(y), pred


def run_ensemble(
    artifacts_dir: str | Path,
    out_path: str | Path | None = None,
    max_models: int = 4,
) -> dict:
    artifacts_dir = Path(artifacts_dir)
    oof = load_oof(artifacts_dir)
    result = greedy_blend(oof, max_models=max_models)
    out = Path(out_path) if out_path else artifacts_dir / "ensemble" / "weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
