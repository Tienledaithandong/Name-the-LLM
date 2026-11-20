"""Final refit on all labelled data and test-time prediction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import LABELS, N_CLASSES, RANDOM_STATE
from src.data import dataset_fingerprint, labels_to_indices
from src.ensemble import normalize_scores
from src.model import ModelError, estimator_from_config
from src.repro import env_snapshot, git_commit


def fit_final(
    model_configs,
    train_df,
    out_dir,
    seed=RANDOM_STATE,
    dataset_label="official",
    pool_df=None,
    pool_models=(),
    cos_threshold=0.3,
    pool_cap=12000,
):
    """Refit every model on all labelled rows and write a loadable artifact dir.

    Members named in ``pool_models`` (D028) are additionally trained on the
    globally-decontaminated external pool: the pool is screened **once** against the
    complete official training set (the only reference, so eligibility never depends on
    any holdout) and class-proportionally capped by ``pool_cap`` (a compute budget).
    Other members train on official rows only. ``match_weight``/``style_score`` are
    never used as features. The pool provenance and decontamination stats are recorded
    in the manifest for audit; the inference blend is unchanged because it already
    row-normalises each bundle (D035).
    """
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    texts = train_df["text"].tolist()
    groups = train_df["prompt_id"].astype(str).to_numpy()
    y_idx = labels_to_indices(train_df["model"].tolist())

    pool_models = set(pool_models or ())
    pool_meta = None
    fit_pool_texts = fit_pool_y = fit_pool_groups = None
    if pool_models:
        if pool_df is None or len(pool_df) == 0:
            raise ValueError("pool_models given without a non-empty pool_df")
        from src.pool import cap_pool_stratified, decontaminate_global

        keep, gstats = decontaminate_global(pool_df, train_df, cos_threshold=cos_threshold)
        clean = pool_df.loc[keep].reset_index(drop=True)
        capped = cap_pool_stratified(clean, pool_cap, seed)
        fit_pool_texts = capped["text"].tolist()
        fit_pool_y = labels_to_indices(capped["model"].tolist())
        fit_pool_groups = capped["prompt_canon"].astype(str).to_numpy()
        pool_meta = {
            "external_data_id": "open-pool-v15",
            "pool_models": sorted(pool_models),
            "cos_threshold": float(cos_threshold),
            "pool_cap": int(pool_cap),
            "decontamination": gstats,
            "capped_rows": int(len(capped)),
            "reference": "full_official_train",
        }

    fitted = []
    for raw in model_configs:
        name = str(raw["name"])
        estimator = estimator_from_config(raw, seed=seed)
        if name in pool_models:
            estimator.fit(texts + fit_pool_texts, np.concatenate([y_idx, fit_pool_y]),
                          groups=np.concatenate([groups, fit_pool_groups]))
            fit_mode = f"train+global_pool(capped={len(fit_pool_texts)})"
        else:
            estimator.fit(texts, y_idx, groups=groups)
            fit_mode = "train_only"
        estimator.save(out_dir / "models" / f"{name}.joblib")
        fitted.append({"name": name, "config": raw, "path": f"models/{name}.joblib",
                       "fit_mode": fit_mode})

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "dataset": dataset_label,
        "dataset_fingerprint": dataset_fingerprint(train_df),
        "n_train_rows": int(len(train_df)),
        "seed": int(seed),
        "class_order": list(LABELS),
        "n_classes": N_CLASSES,
        "models": fitted,
        "pool": pool_meta,
        "environment": env_snapshot(),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def load_final(out_dir):
    """Load fitted bundles and the manifest from a final artifact directory."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no manifest in {out_dir}; run `python -m src.cli predict --fit` first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("class_order") != list(LABELS):
        raise ModelError("manifest class order does not match the official labels")
    bundles = {}
    for entry in manifest["models"]:
        bundle = estimator_from_config(entry["config"], seed=manifest["seed"])
        loaded = type(bundle).load(out_dir / entry["path"])
        bundles[entry["name"]] = loaded
    return bundles, manifest


def load_weights(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"weights file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: weights document must be a mapping")
    weights = payload.get("weights", payload)
    if not isinstance(weights, dict):
        raise ValueError(f"{path}: weights must be a mapping")
    weights = {str(k): float(v) for k, v in weights.items()}
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError(f"{path}: weights must be finite and nonnegative")
    weights = {k: v for k, v in weights.items() if v > 0}
    if not weights:
        raise ValueError(f"{path}: no positive weights")
    return weights


def validate_artifact_provenance(manifest: dict, train_df, model_configs: list[dict], seed: int) -> None:
    if manifest.get("dataset_fingerprint") != dataset_fingerprint(train_df):
        raise ModelError("final artifact dataset fingerprint differs; refit with --fit")
    if manifest.get("seed") != seed or [m["config"] for m in manifest.get("models", [])] != model_configs:
        raise ModelError("final artifact config or seed differs; refit with --fit")


def validate_weight_provenance(path, manifest: dict, bundles: dict) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = load_weights(path)
    missing = set(weights) - set(bundles)
    if missing:
        raise ValueError(f"weights reference unknown models: {sorted(missing)}")
    if payload.get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
        raise ValueError("weights and final models have different or missing dataset provenance")
    configs = {entry["name"]: entry["config"] for entry in manifest["models"]}
    if payload.get("model_configs") != {name: configs[name] for name in weights}:
        raise ValueError("weights and final models have different or missing model configs")
    return weights


def blend_score_matrix(
    bundles: dict,
    texts: list[str],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Row-normalised blend margins (n_rows x N_CLASSES); not probabilities."""
    if not bundles:
        raise ValueError("no fitted models")
    if weights is not None:
        if not isinstance(weights, dict) or not weights or any(not np.isfinite(float(w)) or float(w) < 0 for w in weights.values()):
            raise ValueError("weights must be a nonempty mapping of finite nonnegative values")
        if not any(float(w) > 0 for w in weights.values()):
            raise ValueError("at least one model weight must be positive")
        missing = set(weights) - set(bundles)
        if missing:
            raise ValueError(f"weights reference unknown models: {sorted(missing)}")
        active = {name: w for name, w in weights.items()}
    elif len(bundles) == 1:
        active = {next(iter(bundles)): 1.0}
    else:
        active = {name: 1.0 for name in sorted(bundles)}

    total = None
    for name, weight in active.items():
        scores = normalize_scores(bundles[name].decision_function(texts))
        block = scores * weight
        total = block if total is None else total + block
    return np.asarray(total)


def predict_dataframe(bundles, test_df, weights=None):
    """Predict labels for test_df; blends normalised margins if >1 model."""
    texts = test_df["text"].astype(str).tolist()
    indices = blend_score_matrix(bundles, texts, weights).argmax(axis=1)
    return np.asarray([LABELS[int(i)] for i in indices], dtype=object)


def predict_dataframe_crossed(
    bundles,
    test_df,
    weights=None,
    encoder=None,
    distance_threshold=0.4,
    n_clusters=None,
    models_per_prompt=18,
):
    """Reject the withdrawn transductive inference path (D024)."""
    raise ValueError("crossed inference is withdrawn under D024")


def predict_file(out_dir, test_df, weights_path=None):
    """Convenient wrapper to load a final artifact dir and run prediction."""
    bundles, manifest = load_final(out_dir)
    weights = load_weights(weights_path)
    return predict_dataframe(bundles, test_df, weights), manifest
