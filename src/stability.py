"""Section 7 robustness: alternative-seed CV and the secondary prompt holdout.

The blend weights were selected on the seed-42 out-of-fold predictions, so the
gain is mildly optimistic.  These checks re-measure it on *different* prompt
partitions without re-tuning anything:

* ``seed_stability`` re-runs grouped CV for the blend members under alternative
  fold seeds and re-selects the greedy blend on each, reporting the gain.
* ``holdout_check`` fits the members once on the training side of the untouched
  secondary prompt-held-out split and scores the fixed adopted weights there.

Neither is used to pick the winner; they only confirm or refute the seed-42 gain.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.data import indices_to_labels, labels_to_indices
from src.ensemble import normalize_scores, run_ensemble
from src.metrics import balanced_accuracy, evaluate
from src.model import estimator_from_config
from src.repro import git_commit
from src.train import append_registry, run_cv
from src.validation import make_holdout, make_splits, save_splits


def seed_stability(
    train_df,
    member_configs: list[dict],
    seeds: list[int],
    out_root: str | Path,
    dataset_label: str = "official",
    registry_path=None,
    n_splits: int = 5,
) -> list[dict]:
    out_root = Path(out_root)
    rows: list[dict] = []
    for seed in seeds:
        folds, method = make_splits(train_df, n_splits=n_splits, seed=seed)
        run_dir = out_root / f"seed_{seed}"
        save_splits(run_dir / "splits.json", train_df, folds, method, seed)
        run_cv(
            train_df,
            folds,
            member_configs,
            run_dir,
            seed=seed,
            dataset_label=dataset_label,
            registry_path=registry_path,
            split_id=f"{method}-s{seed}-n{len(folds)}",
        )
        ens = run_ensemble(run_dir, max_models=4)
        rows.append(
            {
                "seed": int(seed),
                "method": method,
                "weights": ens["weights"],
                "blend_pooled_oof": ens["pooled_oof_balanced_accuracy"],
                "best_single": ens["best_single"],
                "gain_over_best_single": ens["gain_over_best_single"],
            }
        )
    return rows


def holdout_check(
    train_df,
    member_configs: list[dict],
    weights: dict[str, float],
    seed: int = 7,
    n_groups: int = 25,
    fit_seed: int = 42,
) -> dict:
    train_idx, valid_idx = make_holdout(train_df, n_groups=n_groups, seed=seed)
    texts = train_df["text"].tolist()
    groups = train_df["prompt_id"].astype(str).to_numpy()
    y = labels_to_indices(train_df["model"].tolist())
    truth = indices_to_labels(y[valid_idx])

    normalized: dict[str, object] = {}
    singles: dict[str, float] = {}
    for raw in member_configs:
        estimator = estimator_from_config(raw, seed=fit_seed)
        estimator.fit(
            [texts[i] for i in train_idx], y[train_idx], groups=groups[train_idx]
        )
        scores = estimator.decision_function([texts[i] for i in valid_idx])
        normalized[raw["name"]] = normalize_scores(scores)
        singles[raw["name"]] = balanced_accuracy(
            truth, indices_to_labels(scores.argmax(axis=1))
        )

    used = {k: float(v) for k, v in weights.items() if k in normalized and v > 0}
    total = sum(used.values())
    used = {k: v / total for k, v in used.items()}
    blended = sum(normalized[k] * v for k, v in used.items())
    result = evaluate(truth, indices_to_labels(blended.argmax(axis=1)), model="holdout_blend")
    return {
        "seed": int(seed),
        "n_groups": int(n_groups),
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "weights": used,
        "singles": singles,
        "blend_balanced_accuracy": result["balanced_accuracy"],
        "gain_over_best_single": result["balanced_accuracy"] - max(singles.values()),
    }


def run_stability(
    train_df,
    member_configs: list[dict],
    weights: dict[str, float],
    seeds: list[int],
    out_root: str | Path,
    dataset_label: str = "official",
    registry_path=None,
    holdout_seed: int = 7,
    holdout_groups: int = 25,
) -> dict:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    seed_rows = seed_stability(
        train_df, member_configs, seeds, out_root,
        dataset_label=dataset_label, registry_path=registry_path,
    )
    holdout = holdout_check(
        train_df, member_configs, weights, seed=holdout_seed, n_groups=holdout_groups
    )
    summary = {"seed_stability": seed_rows, "holdout": holdout}
    (out_root / "stability.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    _write_report(out_root / "stability.md", summary, dataset_label)

    if registry_path is not None:
        append_registry(
            registry_path,
            {
                "experiment_id": (
                    f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{dataset_label}-"
                    f"holdout_blend-s{holdout_seed}"
                ),
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": git_commit(Path(__file__).resolve().parents[1]),
                "method": "holdout_blend",
                "features": "blend",
                "seed": holdout_seed,
                "split_id": f"holdout-s{holdout_seed}-g{holdout_groups}",
                "fold_scores": "",
                "pooled_oof_balanced_accuracy": round(holdout["blend_balanced_accuracy"], 6),
                "macro_f1": "",
                "fit_seconds": "",
                "infer_seconds": "",
                "peak_ram_gb": "",
                "external_data_id": "synthetic-fixture" if dataset_label == "synthetic" else "none",
                "artifact_path": str(out_root / "stability.json"),
                "public_lb_score": "",
                "notes": "secondary prompt-held-out robustness check; not used for selection",
                "status": "done",
            },
        )
    return summary


def _write_report(path: Path, summary: dict, dataset_label: str) -> None:
    lines = [
        "# Blend stability check (Section 7)",
        "",
        f"- dataset: **{dataset_label}**",
        f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- purpose: confirm the seed-42 blend gain on other prompt partitions;",
        "  these runs never re-tune the weights or select the winner.",
        "",
        "## Alternative fold seeds (greedy blend re-selected per seed)",
        "",
        "| seed | blend pooled OOF | best single | gain | weights |",
        "|---|---|---|---|---|",
    ]
    for row in summary["seed_stability"]:
        weights = ", ".join(f"{k}={v:.2f}" for k, v in row["weights"].items())
        lines.append(
            f"| {row['seed']} | {row['blend_pooled_oof']:.4f} | "
            f"{row['best_single']['balanced_accuracy']:.4f} ({row['best_single']['model']}) | "
            f"{row['gain_over_best_single']:+.4f} | {weights} |"
        )
    holdout = summary["holdout"]
    lines += [
        "",
        "## Secondary prompt-held-out split (fixed adopted weights)",
        "",
        f"- held-out prompts: {holdout['n_groups']} (seed {holdout['seed']}), "
        f"{holdout['n_valid']} validation rows / {holdout['n_train']} training rows",
        f"- blend balanced accuracy: **{holdout['blend_balanced_accuracy']:.4f}**",
        f"- gain over best single on this holdout: {holdout['gain_over_best_single']:+.4f}",
        "",
        "| member | holdout balanced accuracy |",
        "|---|---|",
    ]
    for name, acc in holdout["singles"].items():
        lines.append(f"| {name} | {acc:.4f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")