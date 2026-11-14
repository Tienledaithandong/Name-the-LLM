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
    model_configs: list[dict],
    train_df: pd.DataFrame,
    out_dir: str | Path,
    seed: int = RANDOM_STATE,
    dataset_label: str = "official",
) -> Path:
    """Refit every model on all labelled rows and write a loadable artifact dir."""
    out_dir = Path(out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    texts = train_df["text"].tolist()
    groups = train_df["prompt_id"].astype(str).to_numpy()
    y_idx = labels_to_indices(train_df["model"].tolist())

    fitted = []
    for raw in model_configs:
        name = str(raw["name"])
        estimator = estimator_from_config(raw, seed=seed)
        estimator.fit(texts, y_idx, groups=groups)
        estimator.save(out_dir / "models" / f"{name}.joblib")
        fitted.append({"name": name, "config": raw, "path": f"models/{name}.joblib"})

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
        "environment": env_snapshot(),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def load_final(out_dir: str | Path) -> tuple[dict, dict]:
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


def load_weights(path: str | Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"weights file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    weights = payload.get("weights", payload)
    weights = {str(k): float(v) for k, v in weights.items() if float(v) > 0}
    if not weights:
        raise ValueError(f"{path}: no positive weights")
    return weights


def predict_dataframe(
    bundles: dict,
    test_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Predict labels for ``test_df``; blends normalised margins if >1 model."""
    if not bundles:
        raise ValueError("no fitted models")
    texts = test_df["text"].astype(str).tolist()

    if weights:
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
    indices = np.asarray(total).argmax(axis=1)
    return np.asarray([LABELS[int(i)] for i in indices], dtype=object)


def predict_file(
    out_dir: str | Path,
    test_df: pd.DataFrame,
    weights_path: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    bundles, manifest = load_final(out_dir)
    weights = load_weights(weights_path)
    return predict_dataframe(bundles, test_df, weights), manifest
