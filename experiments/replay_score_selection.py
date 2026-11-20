"""Evaluate the CPU-first recipe selection procedure on frozen outer prompt folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from experiments.run_modernbert_probe import pilot_passes
from experiments.run_score_improvement import (
    BASE_MEMBERS,
    complete_scores,
    load_official_train,
    masked_arms,
    nested_outer_folds,
    refit_control,
    report_scores,
)
from src.config import load_config
from src.constants import LABELS, N_CLASSES, RANDOM_STATE
from src.data import indices_to_labels, labels_to_indices, ordered_dataset_fingerprint
from src.ensemble import normalize_scores
from src.frozen_modernbert import FrozenModernBert
from src.masked_character import MaskedCharacterModel
from src.metrics import balanced_accuracy
from src.oof_identity import fold_assignments
from src.provider_specialist import ProviderSpecialist
from src.repro import env_snapshot, git_commit, git_worktree_status
from src.validation import make_splits


CPU_ARMS = ("provider_specialist", "masked_blend")
ENCODER_ARM = "modernbert_blend"
INNER_FOLDS = 5
MIN_GAIN = 0.010
MIN_POSITIVE_FOLDS = 4
PILOT_FOLDS = (0, 1)
MODEL_SEED = RANDOM_STATE

# Shared by inner selection and the single refit. This takes no labels on the
# scoring side, even when the caller holds the complete official frame.
Encode = Callable[[Sequence[str], float], np.ndarray]


def _fold_accuracy(y: np.ndarray, scores: np.ndarray) -> float:
    return balanced_accuracy(indices_to_labels(y), indices_to_labels(scores.argmax(axis=1)))


def screen_summary(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    assignments: np.ndarray,
) -> dict:
    """Apply the preregistered gain and stability checks to complete inner OOF."""
    baseline = complete_scores(baseline, len(y))
    candidate = complete_scores(candidate, len(y))
    if assignments.shape != y.shape or not np.array_equal(np.unique(assignments), np.arange(INNER_FOLDS)):
        raise ValueError("selection requires five complete inner validation folds")
    control_ba = _fold_accuracy(y, baseline)
    candidate_ba = _fold_accuracy(y, candidate)
    differences = []
    for index in range(INNER_FOLDS):
        mask = assignments == index
        delta = _fold_accuracy(y[mask], candidate[mask]) - _fold_accuracy(y[mask], baseline[mask])
        differences.append(0.0 if abs(delta) < 1e-12 else delta)
    positive = sum(delta > 0 for delta in differences)
    gain = candidate_ba - control_ba
    return {
        "baseline_balanced_accuracy": control_ba,
        "candidate_balanced_accuracy": candidate_ba,
        "delta": gain,
        "fold_deltas": differences,
        "positive_folds": positive,
        "qualifies": gain + 1e-12 >= MIN_GAIN and positive >= MIN_POSITIVE_FOLDS,
    }


def _fit_cpu(
    arm: str,
    configs: dict,
    train_texts: list[str],
    valid_texts: list[str],
    y: np.ndarray,
    raw_char: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    if arm == "provider_specialist":
        model = ProviderSpecialist(alpha=0.2, seed=MODEL_SEED, max_features=100_000)
        result, _diagnostic = model.fit(train_texts, y).compare(valid_texts, baseline)
        return result
    if arm == "masked_blend":
        model = MaskedCharacterModel(configs["char_tfidf_logreg"], seed=MODEL_SEED)
        margins = model.fit(train_texts, y).decision_function(valid_texts)
        return masked_arms(baseline, raw_char, margins)[arm]
    raise ValueError(f"unknown CPU arm: {arm}")


def _fit_encoder(
    train: np.ndarray,
    valid: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    head = LogisticRegression(C=1.0, class_weight="balanced", random_state=MODEL_SEED, max_iter=500)
    head.fit(train, y)
    if not np.array_equal(head.classes_, np.arange(N_CLASSES)):
        raise ValueError("encoder head did not fit all canonical labels")
    raw = complete_scores(head.decision_function(valid), len(valid))
    return raw, 0.75 * baseline + 0.25 * normalize_scores(raw)


def _check_time(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError("nested encoder replay exceeded its phase budget")


def choose_arm(
    outer_train: pd.DataFrame,
    *,
    outer_fold: int,
    configs: dict,
    encode: Encode,
    pilot_seconds: int,
    full_seconds: int,
) -> tuple[dict, np.ndarray | None]:
    """Select a fixed recipe from inner OOF without receiving any outer holdout rows."""
    inner_seed = 212 + outer_fold
    folds, method = make_splits(outer_train, n_splits=INNER_FOLDS, seed=inner_seed)
    assignments = fold_assignments(len(outer_train), folds)
    texts = outer_train["text"].tolist()
    y = labels_to_indices(outer_train["model"])
    matrices = {name: np.full((len(y), N_CLASSES), np.nan) for name in ("baseline", *CPU_ARMS)}
    for tr, va in folds:
        train_texts = [texts[i] for i in tr]
        valid_texts = [texts[i] for i in va]
        char, baseline = refit_control(configs, train_texts, valid_texts, y[tr], MODEL_SEED)
        matrices["baseline"][va] = baseline
        for arm in CPU_ARMS:
            result = _fit_cpu(arm, configs, train_texts, valid_texts, y[tr], char, baseline)
            matrices[arm][va] = complete_scores(result, len(va))

    baseline = matrices["baseline"]
    cpu = {arm: screen_summary(y, baseline, matrices[arm], assignments) for arm in CPU_ARMS}
    decision = {
        "outer_fold": outer_fold,
        "inner_seed": inner_seed,
        "inner_method": method,
        "inner_folds": len(folds),
        "ordered_outer_training_sha256": ordered_dataset_fingerprint(outer_train),
        "inner_validation_assignments_sha256": hashlib.sha256(assignments.astype("<i4").tobytes()).hexdigest(),
        "cpu_screens": cpu,
        "encoder_pilot": None,
        "encoder_full_screen": None,
        "selected_arm": "baseline",
    }
    for arm in CPU_ARMS:
        if cpu[arm]["qualifies"]:
            decision["selected_arm"] = arm
            return decision, None

    deadline = time.perf_counter() + pilot_seconds
    embeddings = encode(texts, deadline)
    _check_time(deadline)
    blended = np.full_like(baseline, np.nan)
    pilot = []
    for index in PILOT_FOLDS:
        tr, va = folds[index]
        raw, blended[va] = _fit_encoder(embeddings[tr], embeddings[va], y[tr], baseline[va])
        pilot.append({
            "fold": index,
            "baseline": _fold_accuracy(y[va], baseline[va]),
            "encoder": _fold_accuracy(y[va], raw),
            "blend": _fold_accuracy(y[va], blended[va]),
        })
        _check_time(deadline)
    passed = pilot_passes(pilot, list(PILOT_FOLDS), MIN_GAIN)
    decision["encoder_pilot"] = {"folds": pilot, "passed": passed}
    if not passed:
        return decision, None

    deadline = time.perf_counter() + full_seconds
    # Reuse the exact first two predictions instead of re-fitting those heads.
    for tr, va in folds[len(PILOT_FOLDS):]:
        _raw, blended[va] = _fit_encoder(embeddings[tr], embeddings[va], y[tr], baseline[va])
        _check_time(deadline)
    full = screen_summary(y, baseline, blended, assignments)
    decision["encoder_full_screen"] = full
    if full["qualifies"]:
        decision["selected_arm"] = ENCODER_ARM
        return decision, embeddings
    return decision, None


def _load_recipes() -> tuple[dict, dict, dict, dict]:
    baseline = {model["name"]: model for model in load_config("configs/submission.yaml")["models"]}
    provider = load_config("configs/provider_specialist.yaml")
    masked = load_config("configs/masked_character.yaml")
    modernbert = load_config("configs/frozen_modernbert_probe.yaml")
    if set(baseline) != set(BASE_MEMBERS):
        raise ValueError("expected the two registered baseline members")
    if (provider["alpha"], provider["max_features"], provider["solver_basis"]) != (
        0.2, 100_000, "centered_training_row_span",
    ):
        raise ValueError("expected the registered specialist configuration")
    if (masked["weight"], masked["character_member"], masked["length_bucket_maxima"]) != (
        0.25, "char_tfidf_logreg", [3, 6],
    ):
        raise ValueError("expected the registered masked-character configuration")
    probe = modernbert["probe"]
    if (probe["pilot_folds"], probe["pilot_min_gain"], probe["blend_weight"], probe["classifier_c"]) != (
        list(PILOT_FOLDS), MIN_GAIN, 0.25, 1.0,
    ):
        raise ValueError("expected the registered ModernBERT probe configuration")
    return baseline, provider, masked, modernbert


def run_replay(args: argparse.Namespace) -> dict:
    """Replay inside each outer train split and score selected models once outside it."""
    if args.out.exists():
        raise FileExistsError(f"refusing to replace experiment directory: {args.out}")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    frame = load_official_train(args.data_dir)
    fingerprint = ordered_dataset_fingerprint(frame)
    folds, manifest_split_id = nested_outer_folds(frame, args.nested_plan, fingerprint)
    assignments = fold_assignments(len(frame), folds)
    configs, provider, masked, modernbert = _load_recipes()
    probe = modernbert["probe"]
    matrices = {name: np.full((len(frame), N_CLASSES), np.nan) for name in ("baseline", "selection_procedure")}
    decisions = []
    started = time.perf_counter()
    encoder = None

    def encode(texts: Sequence[str], deadline: float) -> np.ndarray:
        nonlocal encoder
        if encoder is None:
            if args.checkpoint is None:
                raise ValueError("supply the pinned ModernBERT checkpoint for the fallback branch")
            encoder = FrozenModernBert(args.checkpoint, args.cache_dir, **modernbert["encoder"])
            encoder.load()
        return encoder.encode_cached(texts, deadline=deadline)

    with threadpool_limits(limits=args.threads):
        for outer_fold, (tr, va) in enumerate(folds):
            outer_train = frame.iloc[tr].reset_index(drop=True)
            decision, train_embeddings = choose_arm(
                outer_train, outer_fold=outer_fold, configs=configs, encode=encode,
                pilot_seconds=probe["pilot_max_seconds"], full_seconds=probe["full_max_seconds"],
            )
            selected = decision["selected_arm"]
            decisions.append(decision)
            print(f"[outer {outer_fold}] selected from inner OOF: {selected}", flush=True)

            train_texts = outer_train["text"].tolist()
            valid_texts = frame.iloc[va]["text"].tolist()
            training_y = labels_to_indices(outer_train["model"])
            char, baseline = refit_control(configs, train_texts, valid_texts, training_y, MODEL_SEED)
            if selected in CPU_ARMS:
                result = _fit_cpu(selected, configs, train_texts, valid_texts, training_y, char, baseline)
            elif selected == ENCODER_ARM:
                if train_embeddings is None:
                    raise RuntimeError("selected encoder lacks the outer training embeddings")
                deadline = time.perf_counter() + probe["full_max_seconds"]
                valid_embeddings = encode(valid_texts, deadline)
                _raw, result = _fit_encoder(train_embeddings, valid_embeddings, training_y, baseline)
                _check_time(deadline)
            elif selected == "baseline":
                result = baseline
            else:
                raise RuntimeError(f"unexpected selected arm: {selected}")
            matrices["baseline"][va] = complete_scores(baseline, len(va))
            matrices["selection_procedure"][va] = complete_scores(result, len(va))

    y = labels_to_indices(frame["model"])
    groups = frame["prompt_id"].astype(str).to_numpy()
    lengths = np.array([len(text.split()) for text in frame["text"]])
    results = report_scores(y, matrices, assignments, groups, lengths)
    payload = {
        "schema_version": "official-selection-replay/v1",
        "dataset": "official",
        "ordered_dataset_sha256": fingerprint,
        "labels": list(LABELS),
        "outer_manifest_split_id": manifest_split_id,
        "split_id": "nested-grouped-outer3-inner5-cpu-first",
        "validation_assignments_sha256": hashlib.sha256(assignments.astype("<i4").tobytes()).hexdigest(),
        "outer_decisions": decisions,
        "policy": {"cpu_priority": list(CPU_ARMS), "minimum_gain": MIN_GAIN,
                   "minimum_positive_folds": MIN_POSITIVE_FOLDS, "encoder_pilot_folds": list(PILOT_FOLDS)},
        "recipes": {"baseline": configs, "provider": provider, "masked": masked, "modernbert": modernbert},
        "model_seed": MODEL_SEED,
        "environment": env_snapshot(),
        "git_commit": git_commit(),
        "worktree": git_worktree_status(),
        "total_elapsed_seconds": time.perf_counter() - started,
        "inner_selection_performed": True,
        "previously_explored_prompts": True,
        "test_rows_read": 0,
        "transductive": False,
        "note": ("Evaluation of the fixed recipe-selection procedure on previously explored prompts. "
                 "The score is neither a Kaggle result nor the score of any single recipe. "
                 "Selection-procedure OOF values rank classes within rows, not across arms."),
        "results": results,
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    args.out.mkdir(parents=True, exist_ok=False)
    oof_dir = args.out / "oof"
    oof_dir.mkdir()
    for name, matrix in matrices.items():
        np.savez_compressed(
            oof_dir / f"{name}.npz", scores=matrix, y=y, labels=np.asarray(LABELS),
            ordered_dataset_sha256=fingerprint, validation_fold_ids=assignments,
        )
    (args.out / "results.json").write_text(serialized, encoding="utf-8")
    print(f"selection procedure: pooled BA {results['selection_procedure']['balanced_accuracy']:.6f} "
          f"(matched baseline delta {results['selection_procedure']['delta']:+.6f})")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--nested-plan", type=Path, default=Path("artifacts/splits/nested_grouped.json"))
    parser.add_argument("--checkpoint", type=Path, help="pinned local checkpoint for the CUDA fallback")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/score_improvement/encoder_cache"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    run_replay(parser.parse_args())


if __name__ == "__main__":
    main()
