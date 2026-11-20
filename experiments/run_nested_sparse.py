"""Run a prospective nested grouped-CV check of the sparse baseline.

Only the sparse members are evaluated. This does not validate DeBERTa or the
external pool, and prior development on these prompts limits independence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data import indices_to_labels, labels_to_indices, load_competition_data, ordered_dataset_fingerprint  # noqa: E402
from src.ensemble import normalize_scores  # noqa: E402
from src.model import estimator_from_config  # noqa: E402
from src.metrics import evaluate, within_provider_confusions  # noqa: E402
from src.nested_validation import make_nested_splits  # noqa: E402


def choose_char_weight(y: np.ndarray, char: np.ndarray, style: np.ndarray) -> tuple[float, dict]:
    """Choose from a fixed grid using inner predictions only; ties prefer 0.5."""
    candidates = (0.5, 0.0, 1.0)
    scores = {weight: float(balanced_accuracy_score(y, (weight * char + (1 - weight) * style).argmax(1)))
              for weight in candidates}
    best = max(candidates, key=lambda weight: scores[weight])
    return best, {str(weight): score for weight, score in scores.items()}


def _scores(df, train_idx, valid_idx, configs, seed: int) -> dict:
    result = {}
    y = labels_to_indices(df["model"].tolist())
    texts = df["text"].tolist()
    groups = df["prompt_id"].astype(str).to_numpy()
    for cfg in configs:
        model = estimator_from_config(cfg, seed=seed)
        model.fit([texts[i] for i in train_idx], y[train_idx], groups=groups[train_idx])
        result[cfg["name"]] = normalize_scores(model.decision_function([texts[i] for i in valid_idx]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config", default="configs/submission.yaml")
    parser.add_argument("--plan", type=Path, default=Path("artifacts/splits/nested_grouped.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/nested_sparse/results.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.out}")
    df, _ = load_competition_data(args.data_dir)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "nested-grouped-splits/v1" or plan.get("ordered_dataset_sha256") != ordered_dataset_fingerprint(df):
        raise SystemExit("nested split plan does not match ordered data")
    saved = plan["folds"]
    plans = make_nested_splits(df, outer_splits=len(saved), inner_splits=len(saved[0]["inner"]),
                               outer_seed=plan["outer_seed"], inner_seed=plan["inner_seed"])
    for p, stored in zip(plans, saved):
        if (not np.array_equal(p["train"], stored["train"]) or not np.array_equal(p["valid"], stored["valid"])
                or any(not np.array_equal(tr, z["train"]) or not np.array_equal(va, z["valid"])
                       for (tr, va), z in zip(p["inner"], stored["inner"]))):
            raise SystemExit("nested split plan differs from regenerated grouped splits")
    configs = load_config(args.config)["models"]
    if [c["name"] for c in configs] != ["char_tfidf_logreg", "style_only"]:
        raise SystemExit("config must contain the two registered sparse members in order")
    y = labels_to_indices(df["model"].tolist())
    fixed = np.full(len(df), -1, dtype=int)
    selected = np.full(len(df), -1, dtype=int)
    reports = []
    started = time.perf_counter()
    for outer in plans:
        train_idx, valid_idx = outer["train"], outer["valid"]
        inner_char = np.zeros((len(df), 18))
        inner_style = np.zeros((len(df), 18))
        for inner_index, (tr, va) in enumerate(outer["inner"]):
            print(f"[outer {outer['outer_fold']}] inner fold {inner_index}", flush=True)
            scores = _scores(df, tr, va, configs, seed=42)
            inner_char[va] = scores["char_tfidf_logreg"]
            inner_style[va] = scores["style_only"]
        weight, inner_scores = choose_char_weight(y[train_idx], inner_char[train_idx], inner_style[train_idx])
        print(f"[outer {outer['outer_fold']}] refitting selected and fixed candidates", flush=True)
        scores = _scores(df, train_idx, valid_idx, configs, seed=42)
        char, style = scores["char_tfidf_logreg"], scores["style_only"]
        fixed[valid_idx] = (0.5 * char + 0.5 * style).argmax(1)
        selected[valid_idx] = (weight * char + (1 - weight) * style).argmax(1)
        reports.append({"outer_fold": outer["outer_fold"], "selected_char_weight": weight,
                        "inner_balanced_accuracy": inner_scores,
                        "fixed_outer_balanced_accuracy": float(balanced_accuracy_score(y[valid_idx], fixed[valid_idx])),
                        "selected_outer_balanced_accuracy": float(balanced_accuracy_score(y[valid_idx], selected[valid_idx]))})
    if (fixed < 0).any() or (selected < 0).any():
        raise RuntimeError("outer predictions do not cover every row")
    result = {"dataset": "official", "method": "nested grouped sparse only",
              "ordered_dataset_sha256": plan["ordered_dataset_sha256"],
              "fixed_char_weight": 0.5, "inner_weight_grid": [0.0, 0.5, 1.0],
              "fixed_pooled_balanced_accuracy": float(balanced_accuracy_score(y, fixed)),
              "selected_pooled_balanced_accuracy": float(balanced_accuracy_score(y, selected)),
              "fixed_metrics": evaluate(indices_to_labels(y), indices_to_labels(fixed), model="fixed_sparse"),
              "fixed_within_provider": within_provider_confusions(indices_to_labels(y), indices_to_labels(fixed)),
              "selected_metrics": evaluate(indices_to_labels(y), indices_to_labels(selected), model="selected_sparse"),
              "selected_within_provider": within_provider_confusions(indices_to_labels(y), indices_to_labels(selected)),
              "folds": reports, "elapsed_seconds": round(time.perf_counter() - started, 2),
              "caveat": "Prior development used all official prompts; no DeBERTa or external pool was evaluated."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[result] fixed={result['fixed_pooled_balanced_accuracy']:.6f} "
          f"selected={result['selected_pooled_balanced_accuracy']:.6f}; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
