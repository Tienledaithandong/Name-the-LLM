"""Cross-validation runner.

One model per config entry, five fixed prompt-grouped folds, OOF decision
scores saved for every model so ensembles are built from identical fold
assignments.  All learned state (TF-IDF vocabulary, IDF, SVD, nuisance
projection, classifier) is fitted on the training side of each fold only.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import DEFAULT_REGISTRY, LABELS, N_CLASSES, RANDOM_STATE
from src.data import (
    dataset_fingerprint,
    indices_to_labels,
    labels_to_indices,
    ordered_dataset_fingerprint,
)
from src.validation import validate_splits
from src.metrics import aggregate_folds, evaluate
from src.model import estimator_from_config
from src.oof_identity import fold_assignments
from src.repro import env_snapshot, git_commit, git_worktree_status

REGISTRY_COLUMNS = [
    "experiment_id",
    "date",
    "git_commit",
    "method",
    "features",
    "seed",
    "split_id",
    "fold_scores",
    "pooled_oof_balanced_accuracy",
    "macro_f1",
    "fit_seconds",
    "infer_seconds",
    "peak_ram_gb",
    "external_data_id",
    "artifact_path",
    "public_lb_score",
    "notes",
    "status",
]


def rss_gb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / 1024**3)
    except Exception:  # pragma: no cover - psutil optional
        return float("nan")


def describe(raw: dict) -> str:
    if raw.get("baseline"):
        return f"baseline:{raw['baseline']}"
    features = "+".join(raw.get("features", ["char"]))
    bits = [features]
    if raw.get("style"):
        bits.append("style")
    if raw.get("svd"):
        bits.append(f"svd{raw['svd']}")
    if raw.get("nuisance_k"):
        bits.append(f"nuisance{raw['nuisance_k']}")
    if raw.get("embedding"):
        bits.append(f"emb:{raw['embedding'].get('backend', 'hashing')}")
    bits.append(raw.get("classifier", "linearsvc"))
    return "|".join(bits)


def run_cv(
    train_df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_configs: list[dict],
    out_dir: str | Path,
    seed: int = RANDOM_STATE,
    dataset_label: str = "official",
    registry_path: str | Path | None = DEFAULT_REGISTRY,
    split_id: str = "auto",
    save_fold_models: bool = False,
) -> list[dict]:
    """Fit every model on every fold; persist OOF scores and metrics."""
    validate_splits(train_df, folds)
    out_dir = Path(out_dir)
    (out_dir / "oof").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    texts = train_df["text"].tolist()
    groups = train_df["prompt_id"].astype(str).to_numpy()
    y_idx = labels_to_indices(train_df["model"].tolist())
    y_labels = indices_to_labels(y_idx)
    ordered_fingerprint = ordered_dataset_fingerprint(train_df)
    validation_folds = fold_assignments(len(train_df), folds)

    fold_ids = np.empty(len(train_df), dtype=np.int64)
    for fold, (_, valid_idx) in enumerate(folds):
        fold_ids[valid_idx] = fold

    results: list[dict] = []
    for raw in model_configs:
        name = str(raw["name"])
        scores = np.full((len(train_df), N_CLASSES), -1e9, dtype=np.float32)
        fold_metrics: list[dict] = []
        fit_seconds = 0.0
        infer_seconds = 0.0
        peak_ram = rss_gb()

        for fold, (train_idx, valid_idx) in enumerate(folds):
            estimator = estimator_from_config(raw, seed=seed)
            start = time.perf_counter()
            estimator.fit(
                [texts[i] for i in train_idx],
                y_idx[train_idx],
                groups=groups[train_idx],
            )
            fit_seconds += time.perf_counter() - start
            peak_ram = max(peak_ram, rss_gb())

            start = time.perf_counter()
            fold_scores = estimator.decision_function([texts[i] for i in valid_idx])
            infer_seconds += time.perf_counter() - start

            scores[valid_idx] = fold_scores
            pred = indices_to_labels(fold_scores.argmax(axis=1))
            fold_metrics.append(
                evaluate(y_labels[valid_idx], pred, model=f"{name}#fold{fold}")
            )
            if save_fold_models:
                estimator.save(
                    out_dir / "fold_models" / name / f"fold{fold}.joblib"
                )

        oof_pred = indices_to_labels(scores.argmax(axis=1))
        pooled = evaluate(y_labels, oof_pred, model=name)
        aggregate = aggregate_folds(fold_metrics)

        oof_path = out_dir / "oof" / f"{name}.npz"
        np.savez_compressed(
            oof_path,
            scores=scores,
            y=y_idx,
            labels=np.asarray(LABELS, dtype=str),
            ordered_dataset_sha256=np.asarray(ordered_fingerprint),
            validation_fold_ids=validation_folds,
            row_ids=train_df["id"].astype(str).to_numpy(dtype=str),
            fold_ids=fold_ids,
            dataset_fingerprint=dataset_fingerprint(train_df),
            config=json.dumps(raw, sort_keys=True),
        )
        payload = {
            "model": name,
            "config": raw,
            "dataset": dataset_label,
            "seed": int(seed),
            "split_id": split_id,
            "class_order": list(LABELS),
            "pooled": pooled,
            "fold_aggregate": aggregate,
            "folds": fold_metrics,
            "fit_seconds": round(fit_seconds, 3),
            "infer_seconds": round(infer_seconds, 3),
            "peak_ram_gb": round(peak_ram, 3),
            "environment": env_snapshot(),
            "git_commit": git_commit(Path(__file__).resolve().parents[1]),
            "git_worktree_status": git_worktree_status(Path(__file__).resolve().parents[1]),
        }
        metrics_path = out_dir / "metrics" / f"{name}.json"
        metrics_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

        result = {
            "name": name,
            "config": raw,
            "pooled": pooled,
            "fold_aggregate": aggregate,
            "fit_seconds": payload["fit_seconds"],
            "infer_seconds": payload["infer_seconds"],
            "peak_ram_gb": payload["peak_ram_gb"],
            "metrics_path": str(metrics_path),
            "oof_path": str(oof_path),
            "dataset": dataset_label,
        }
        results.append(result)

        if registry_path is not None:
            append_registry(
                registry_path,
                {
                    "experiment_id": (
                        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-"
                        f"{dataset_label}-{name}-s{seed}"
                    ),
                    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "git_commit": payload["git_commit"],
                    "method": name,
                    "features": describe(raw),
                    "seed": seed,
                    "split_id": split_id,
                    "fold_scores": " ".join(
                        f"{s:.4f}" for s in aggregate["fold_scores"]
                    ),
                    "pooled_oof_balanced_accuracy": round(
                        pooled["balanced_accuracy"], 6
                    ),
                    "macro_f1": round(float(pooled["macro_f1"]), 6),
                    "fit_seconds": payload["fit_seconds"],
                    "infer_seconds": payload["infer_seconds"],
                    "peak_ram_gb": payload["peak_ram_gb"],
                    "external_data_id": raw.get("external_data_id")
                    or ("synthetic-fixture" if dataset_label == "synthetic" else "none"),
                    "artifact_path": str(oof_path),
                    "public_lb_score": "",
                    "notes": "; ".join(filter(None, [
                        "synthetic smoke run - not a competition result" if dataset_label == "synthetic" else "",
                        f"worktree {payload['git_worktree_status']} at recording"
                        if payload["git_worktree_status"] != "clean" else "",
                    ])),
                    "status": "done",
                },
            )

    write_summary(out_dir, results, dataset_label)
    return results


def append_registry(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_summary(
    out_dir: str | Path, results: list[dict], dataset_label: str
) -> Path:
    out_dir = Path(out_dir)
    summary_path = out_dir / "summary.md"
    rows = sorted(results, key=lambda r: r["pooled"]["balanced_accuracy"], reverse=True)
    lines = [
        "# Cross-validation summary",
        "",
        f"- dataset: **{dataset_label}**",
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- metric: pooled out-of-fold balanced accuracy (primary), "
        f"mean±std over folds (stability)",
        "",
    ]
    if dataset_label == "synthetic":
        lines += [
            "> **Synthetic fixture.** These numbers validate that the pipeline runs; "
            "they say nothing about competition performance.",
            "",
        ]
    lines += [
        "| model | pooled OOF bal. acc | fold mean ± std | macro F1 | fit s | infer s | peak RSS GB |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        agg = r["fold_aggregate"]
        lines.append(
            f"| `{r['name']}` | {r['pooled']['balanced_accuracy']:.4f} | "
            f"{agg['mean']:.4f} ± {agg['std']:.4f} | {float(r['pooled']['macro_f1']):.4f} | "
            f"{r['fit_seconds']:.1f} | {r['infer_seconds']:.2f} | {r['peak_ram_gb']:.2f} |"
        )
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def best_result(results: list[dict]) -> dict:
    if not results:
        raise ValueError("no results")
    return max(results, key=lambda r: r["pooled"]["balanced_accuracy"])
