"""Per-feature style analysis (plan §4 L142) — descriptive, not causal.

Refits the registered ``style_only`` member on each frozen training fold and
first proves the refit reproduces the saved out-of-fold scores, so the reported
coefficients provably belong to the models that produced the recorded score.
Coefficients are on per-fold standardised style columns, which makes magnitudes
comparable across features.

    python experiments/style_features.py --data-dir data

Writes ``artifacts/style_features/results.json`` and ``feature_table.md``.
Nothing here is a causal claim: a coefficient is an association inside one
linear model on grouped folds, and correlated style columns share credit.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.constants import LABELS, N_CLASSES  # noqa: E402
from src.data import (  # noqa: E402
    indices_to_labels,
    labels_to_indices,
    load_competition_data,
    ordered_dataset_fingerprint,
)
from src.ensemble import load_oof  # noqa: E402
from src.metrics import per_class_recall  # noqa: E402
from src.model import estimator_from_config  # noqa: E402
from src.oof_identity import fold_assignments  # noqa: E402
from src.repro import env_snapshot, git_commit  # noqa: E402

TOP_OVERALL = 20
TOP_PER_CLASS = 5


def aggregate(coefficients: np.ndarray) -> dict[str, np.ndarray]:
    """Fold-wise statistics per class and feature: each array is (classes, features)."""
    mean = coefficients.mean(axis=0)
    return {
        "mean": mean,
        "mean_abs": np.abs(coefficients).mean(axis=0),
        "std": coefficients.std(axis=0, ddof=1) if coefficients.shape[0] > 1 else np.zeros_like(mean),
        "sign_agreement": np.maximum((coefficients > 0).mean(axis=0), (coefficients < 0).mean(axis=0)),
    }


def slice_stats(stats: dict[str, np.ndarray], class_index: int | None) -> dict[str, np.ndarray]:
    """One class's statistics, or the class-averaged profile when ``None``."""
    if class_index is None:
        return {key: value.mean(axis=0) for key, value in stats.items()}
    return {key: value[class_index] for key, value in stats.items()}


def feature_rows(names: list[str], stats: dict[str, np.ndarray], order: np.ndarray, limit: int) -> list[dict]:
    return [
        {
            "feature": names[int(j)],
            "mean_coefficient": float(stats["mean"][j]),
            "mean_abs_coefficient": float(stats["mean_abs"][j]),
            "fold_std": float(stats["std"][j]),
            "sign_agreement": float(stats["sign_agreement"][j]),
        }
        for j in order[:limit]
    ]


def markdown_table(rows: list[dict], *, signed: bool = True) -> str:
    head = "| feature | mean coef | mean abs coef | fold std | sign agreement |" if signed else \
        "| feature | mean abs coef | fold std | sign agreement |"
    lines = [head, "|---|---|---|---|---|" if signed else "|---|---|---|---|"]
    for row in rows:
        cells = [f"`{row['feature']}`"]
        if signed:
            cells.append(f"{row['mean_coefficient']:+.4f}")
        cells += [f"{row['mean_abs_coefficient']:.4f}", f"{row['fold_std']:.4f}", f"{row['sign_agreement']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def per_class_markdown(per_class: dict[str, dict], limit: int = 3) -> str:
    lines = [
        "| label | style-only recall | strongest positive features | strongest negative features |",
        "|---|---|---|---|",
    ]
    for label, payload in per_class.items():
        def cell(rows: list[dict]) -> str:
            return ", ".join(
                f"`{r['feature'].removeprefix('style:')}` ({r['mean_coefficient']:+.3f})" for r in rows[:limit]
            )
        recall = payload["style_only_recall"]
        shown = "—" if recall != recall else f"{recall:.3f}"
        lines.append(f"| {label} | {shown} | {cell(payload['top_positive'])} | {cell(payload['top_negative'])} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--config", default="configs/submission.yaml")
    ap.add_argument("--splits", default="artifacts/splits/splits_official_verified_s42.json")
    ap.add_argument("--oof", default="artifacts/cv_identity/official")
    ap.add_argument("--out", default="artifacts/style_features")
    args = ap.parse_args()

    df, _test = load_competition_data(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    fingerprint = ordered_dataset_fingerprint(df)
    if splits.get("ordered_dataset_sha256") != fingerprint:
        raise SystemExit("[abort] the local train frame does not match the split fingerprint")
    oof = load_oof(args.oof, ["style_only"])["style_only"]
    if oof["ordered_dataset_sha256"] != fingerprint:
        raise SystemExit("[abort] OOF fingerprint differs from the splits")
    folds = [(np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int)) for f in splits["folds"]]
    if not np.array_equal(oof["validation_fold_ids"], fold_assignments(len(df), folds)):
        raise SystemExit("[abort] OOF fold assignments differ from the splits")
    if not np.array_equal(oof["y"], labels_to_indices(df["model"].tolist())):
        raise SystemExit("[abort] OOF labels differ from the local train frame")

    config = load_config(args.config)
    style_cfg = next((c for c in config["models"] if c["name"] == "style_only"), None)
    if style_cfg is None:
        raise SystemExit("[abort] config has no style_only member")
    model_seed = int(config.get("seed", 42))
    texts = df["text"].tolist()
    y = labels_to_indices(df["model"].tolist())
    groups = df["prompt_id"].astype(str).to_numpy()

    names: list[str] | None = None
    coefficients = []
    max_drift = 0.0
    for index, (train_idx, valid_idx) in enumerate(folds):
        model = estimator_from_config(style_cfg, seed=model_seed)
        model.fit([texts[i] for i in train_idx], y[train_idx], groups=groups[train_idx])
        reproduced = np.asarray(model.decision_function([texts[i] for i in valid_idx]), dtype=np.float64)
        saved = oof["scores"][valid_idx]
        drift = float(np.abs(reproduced - saved).max())
        max_drift = max(max_drift, drift)
        # the saved OOF is float32; anything beyond that rounding is a different model
        if drift > 1e-4:
            raise SystemExit(
                f"[abort] fold {index}: refit does not reproduce the saved OOF (max drift {drift:.3g}); "
                "the coefficients would not belong to the recorded run"
            )
        fold_names = [str(n) for n in model.builder.get_feature_names_out()]
        if names is None:
            names = fold_names
        elif names != fold_names:
            raise SystemExit(f"[abort] fold {index}: style feature names differ across folds")
        coef = np.asarray(model.classifier.coef_, dtype=np.float64)
        if coef.shape != (N_CLASSES, len(names)):
            raise SystemExit(f"[abort] fold {index}: unexpected coefficient shape {coef.shape}")
        coefficients.append(coef)
        print(f"[fold {index}] refit reproduces saved OOF (max drift {drift:.2g}), {len(names)} style features", flush=True)

    stack = np.stack(coefficients)
    stats = aggregate(stack)
    overall_stats = slice_stats(stats, None)
    order = np.argsort(-overall_stats["mean_abs"], kind="stable")
    overall = feature_rows(names, overall_stats, order, TOP_OVERALL)

    truth = indices_to_labels(y)
    style_pred = indices_to_labels(oof["scores"].argmax(axis=1))
    recall = per_class_recall(truth, style_pred)
    per_class = {}
    for c, label in enumerate(LABELS):
        class_stats = slice_stats(stats, c)
        positive = np.argsort(-class_stats["mean"], kind="stable")
        negative = np.argsort(class_stats["mean"], kind="stable")
        per_class[label] = {
            "style_only_recall": recall[label],
            "top_positive": feature_rows(names, class_stats, positive, TOP_PER_CLASS),
            "top_negative": feature_rows(names, class_stats, negative, TOP_PER_CLASS),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment": env_snapshot(),
        "analysis": "per_feature_style_coefficients",
        "plan_item": "§4 L142",
        "transductive": False,
        "test_rows_read": 0,
        "ordered_dataset_sha256": fingerprint,
        "split_id": f"{splits['method']}-s{splits['seed']}-n{splits['n_splits']}",
        "member": "style_only",
        "config": args.config,
        "model_seed": model_seed,
        "n_folds": len(folds),
        "n_features": len(names),
        "feature_names": names,
        "refit_reproduces_saved_oof": True,
        "max_oof_drift": max_drift,
        "feature_scaling": "StandardScaler(with_mean=False) fit inside each training fold",
        "overall_ranking": "mean absolute coefficient averaged over folds and the 18 classes",
        "top_features_overall": overall,
        "per_class": per_class,
        "interpretation": (
            "Associational only. Coefficients are per-fold-standardised logistic-regression weights "
            "averaged over the frozen grouped folds; correlated style columns share credit, so a large "
            "weight is not evidence that the feature causes a model's style, nor that removing it would "
            "change the score. No causal attribution is claimed."
        ),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    table = markdown_table(overall, signed=False)
    class_table = per_class_markdown(per_class)
    (out / "feature_table.md").write_text(
        "# Style-feature associations (descriptive, not causal)\n\n"
        "Class-averaged magnitude across the five frozen grouped folds. The signed mean is\n"
        "omitted here because multinomial coefficients are centred across the 18 classes, so\n"
        "averaging them cancels; per-class signed weights are in the second table.\n\n"
        + table + "\n\n## Per class\n\n" + class_table + "\n",
        encoding="utf-8",
    )
    print()
    print(table)
    print()
    print(class_table)
    print(f"\nwrote {out / 'results.json'}")
    print(f"wrote {out / 'feature_table.md'}")


if __name__ == "__main__":
    main()