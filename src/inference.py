"""Final refit on all labelled data and test-time prediction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import LABELS, N_CLASSES, RANDOM_STATE
from src.crossed import assign_one_per_model
from src.data import dataset_fingerprint, labels_to_indices
from src.ensemble import normalize_scores
from src.grouping import cluster_embeddings
from src.model import ModelError, estimator_from_config
from src.repro import env_snapshot, git_commit


def fit_final(
    model_configs,
    train_df,
    out_dir,
    seed=RANDOM_STATE,
    dataset_label="official",
):
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
    weights = payload.get("weights", payload)
    weights = {str(k): float(v) for k, v in weights.items() if float(v) > 0}
    if not weights:
        raise ValueError(f"{path}: no positive weights")
    return weights


def blend_score_matrix(bundles, texts, weights=None):
    """Row-normalised blend margins (n_rows x N_CLASSES); not probabilities."""
    if not bundles:
        raise ValueError("no fitted models")
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
    """Crossed-design prediction: cluster passages into prompt groups, then
    enforce one-passage-per-model assignment within each recovered group.

    Label-free and transductive (uses only test text). encoder embeds the
    passages for clustering; provide exactly one of n_clusters or
    distance_threshold (default cosine 0.4).  When neither is given, the
    group count is estimated as n_rows / models_per_prompt.
    """
    if encoder is None:
        raise ValueError("crossed prediction requires an encoder for grouping")

    texts = test_df["text"].astype(str).tolist()
    embeddings = encoder.encode(texts)

    if n_clusters is not None:
        cluster_labels = cluster_embeddings(embeddings, n_clusters=n_clusters)
    elif distance_threshold is not None:
        cluster_labels = cluster_embeddings(embeddings, distance_threshold=distance_threshold)
    else:
        n_rows = len(test_df)
        n_clusters = max(1, n_rows // models_per_prompt)
        cluster_labels = cluster_embeddings(embeddings, n_clusters=n_clusters)

    model_names = list(bundles.keys())
    n_models = len(model_names)

    predictions = np.full(len(test_df), None, dtype=object)

    for group_id in np.unique(cluster_labels):
        group_indices = np.where(cluster_labels == group_id)[0]
        group_texts = [texts[j] for j in group_indices]
        gscores = blend_score_matrix(bundles, group_texts)
        if len(group_indices) <= n_models:
            choice = assign_one_per_model(gscores, group_indices)
        else:
            choice = np.empty(len(group_indices), dtype=int)
            ranked = np.argsort(-gscores.max(axis=1))
            used_models = []
            pos = 0
            for rank in ranked:
                if pos < n_models:
                    ordered = np.argsort(-gscores[rank])
                    chosen = next(m for m in ordered if m not in used_models)
                    choice[rank] = chosen
                    used_models.append(chosen)
                    pos += 1
                else:
                    choice[rank] = gscores[rank].argmax()
        for k, idx in enumerate(group_indices):
            predictions[idx] = LABELS[choice[k]]

    if np.any(predictions is None):
        raise ValueError(
            "Some predictions are missing; check clustering and assignment logic"
        )

    return predictions


def predict_file(out_dir, test_df, weights_path=None):
    """Convenient wrapper to load a final artifact dir and run prediction."""
    bundles, manifest = load_final(out_dir)
    weights = load_weights(weights_path)
    return predict_dataframe(bundles, test_df, weights), manifest
