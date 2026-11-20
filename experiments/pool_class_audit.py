"""Aggregate class-imbalance and confusion diagnostics for the external pool.

Only aggregate counts and recalls are written; text, identifiers and individual
predictions are never exported. These diagnostics do not establish causation or
validate the pool for training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.constants import LABELS  # noqa: E402


def summarize_pool(pool: pd.DataFrame, labels: list[str]) -> dict:
    """Return only per-label row counts and route metadata coverage."""
    if "model" not in pool.columns:
        raise ValueError("pool requires a model column")
    result = {}
    for label in labels:
        subset = pool.loc[pool["model"] == label]
        info: dict = {"rows": len(subset)}
        for col in ("gen_route", "gen_served_model"):
            if col in subset.columns:
                values = subset[col].fillna("").astype(str).str.strip()
                info[f"{col}_missing"] = int(values.eq("").sum())
        result[label] = info
    return result


def summarize_confusions(y: np.ndarray, scores: np.ndarray, labels: list[str]) -> dict:
    """Return per-class support, recall and aggregate predicted-class counts."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    if scores.shape != (len(y), len(labels)) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite with one row and one column per label")
    if np.any(y < 0) or np.any(y >= len(labels)):
        raise ValueError("class indices must match label order")
    predictions = scores.argmax(axis=1)
    result = {}
    for idx, label in enumerate(labels):
        selected = predictions[y == idx]
        counts = np.bincount(selected, minlength=len(labels))
        result[label] = {
            "support": int(len(selected)),
            "recall": float(counts[idx] / len(selected)) if len(selected) else None,
            "predicted_counts": {labels[i]: int(n) for i, n in enumerate(counts) if n},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("artifacts/pool_audit/ntl_open_pool.parquet"))
    parser.add_argument("--oof", type=Path, default=Path("artifacts/deberta_v4_output/deberta_cv/oof/official_pool.npz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/diagnostics/pool_class_audit.json"))
    args = parser.parse_args()
    pool = pd.read_parquet(args.pool, columns=["model", "gen_route", "gen_served_model"])
    with np.load(args.oof, allow_pickle=True) as arr:
        stored_labels = [str(value) for value in arr["labels"]]
        if stored_labels != list(LABELS):
            raise ValueError("OOF class order does not match the official order")
        confusion = summarize_confusions(arr["y"], arr["scores"], stored_labels)
    result = {
        "pool_class_summary": summarize_pool(pool, list(LABELS)),
        "oof_class_summary": confusion,
        "caveat": "Descriptive only; OOF identity and external-prompt isolation are not established.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aggregate diagnostics to {args.out}")


if __name__ == "__main__":
    main()
