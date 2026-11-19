from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from src.constants import LABELS, N_CLASSES
from src.data import indices_to_labels
from src.metrics import balanced_accuracy

SENTINEL = -1e8


def load_weights(path: str | Path | None) -> dict[str, float] | None:
    """Load ensemble weights from a JSON artifact."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("weights document must be a mapping")
    weight = payload.get("weights", payload)
    if not isinstance(weight, dict):
        raise ValueError("weights must be a mapping")
    weight = {str(k): float(v) for k, v in weight.items()}
    if any(not np.isfinite(v) or v < 0 for v in weight.values()):
        raise ValueError("weights must be finite and nonnegative")
    weight = {k: v for k, v in weight.items() if v > 0}
    return weight if weight else None


def load_oof(artifacts_dir: str | Path, members: list[str] | None = None) -> dict[str, dict]:
    """Load OOF matrices, optionally restricting to explicitly named members."""
    oof_dir = Path(artifacts_dir) / "oof"
    out: dict[str, dict] = {}
    if members is not None and (not members or len(set(members)) != len(members)):
        raise ValueError("ensemble members must be a nonempty list without duplicates")
    for path in sorted(oof_dir.glob("*.npz")):
        if members is not None and path.stem not in members:
            continue
        with np.load(path, allow_pickle=True) as data:
            required = {"scores", "y", "labels", "ordered_dataset_sha256", "validation_fold_ids"}
            if required - set(data.files):
                raise ValueError(f"{path.name}: OOF identity metadata missing; rerun training")
            scores = np.asarray(data["scores"], dtype=np.float64)
            y = np.asarray(data["y"], dtype=np.int64)
            labels = [str(v) for v in data["labels"]]
            fingerprint = str(np.asarray(data["ordered_dataset_sha256"]).item())
            folds = np.asarray(data["validation_fold_ids"], dtype=np.int64)
        if labels != list(LABELS):
            raise ValueError(f"{path.name}: class order {labels} is not the official order")
        if y.ndim != 1 or scores.shape != (y.size, N_CLASSES) or folds.shape != y.shape:
            raise ValueError(f"{path.name}: unexpected score shape {scores.shape}")
        if not re.fullmatch(r"[0-9a-f]{64}", fingerprint) or not np.isfinite(scores).all() or (folds < 0).any():
            raise ValueError(f"{path.name}: invalid OOF fingerprint, scores or fold assignments")
        out[path.stem] = {"scores": scores, "y": y,
                          "ordered_dataset_sha256": fingerprint, "validation_fold_ids": folds}
    if not out:
        raise FileNotFoundError(
            f"no OOF files in {oof_dir}; run the training step first"
        )
    if members is not None and set(out) != set(members):
        raise FileNotFoundError(f"missing OOF members: {sorted(set(members) - set(out))}")
    names = sorted(out)
    reference = out[names[0]]
    for name in names[1:]:
        item = out[name]
        if (item["ordered_dataset_sha256"] != reference["ordered_dataset_sha256"] or
                not np.array_equal(item["validation_fold_ids"], reference["validation_fold_ids"]) or
                not np.array_equal(item["y"], reference["y"])):
            raise ValueError(f"{name}: OOF rows or folds are not aligned with {names[0]}")
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
    reference_hash = oof[names[0]].get("ordered_dataset_sha256")
    reference_folds = oof[names[0]].get("validation_fold_ids")
    if not isinstance(reference_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", reference_hash) or reference_folds is None:
        raise ValueError("OOF identity metadata missing; rerun training")
    for name in names:
        if (not np.array_equal(oof[name]["y"], y) or
                oof[name].get("ordered_dataset_sha256") != reference_hash or
                not np.array_equal(oof[name].get("validation_fold_ids"), reference_folds)):
            raise ValueError(f"{name}: OOF rows or folds are not aligned with {names[0]}")
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
    members: list[str] | None = None,
) -> dict:
    artifacts_dir = Path(artifacts_dir)
    oof = load_oof(artifacts_dir, members=members)
    result = greedy_blend(oof, max_models=max_models)
    out = Path(out_path) if out_path else artifacts_dir / "ensemble" / "weights.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result
