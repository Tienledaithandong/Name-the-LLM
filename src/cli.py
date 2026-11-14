"""Command-line interface.

Examples
--------
    python -m src.cli audit  --data-dir data
    python -m src.cli splits --data-dir data
    python -m src.cli train  --config configs/baseline.yaml --data-dir data
    python -m src.cli ensemble
    python -m src.cli predict --config configs/baseline.yaml --data-dir data
    python -m src.cli package
    python -m src.cli demo            # synthetic end-to-end smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit import run_audit
from src.config import load_config
from src.constants import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DEFAULT_REGISTRY,
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    LABELS,
    RANDOM_STATE,
    SUBMISSION_FILENAME,
)
from src.data import (
    dataset_fingerprint,
    labels_to_indices,
    load_competition_data,
    validate_test_frame,
    validate_train_frame,
)
from src.ensemble import run_ensemble
from src.inference import fit_final, load_final, predict_dataframe
from src.metrics import evaluate
from src.repro import env_snapshot
from src.submission import build_submission, validate_submission
from src.train import run_cv
from src.validation import (
    fold_groups,
    load_splits,
    make_splits,
    save_splits,
)


def load_data(data_dir, synthetic: bool):
    """Return ``(train, test, truth, dataset_label)``."""
    if synthetic:
        from src.synthetic import write_synthetic_dataset

        base = Path(data_dir) if data_dir else DATA_DIR / "synthetic"
        if not (base / "train.csv").is_file():
            write_synthetic_dataset(base)
            print(f"[synthetic] generated fixture in {base}")
        train = validate_train_frame(pd.read_csv(base / "train.csv"))
        test = validate_test_frame(pd.read_csv(base / "test.csv"))
        truth_path = base / "test_truth.csv"
        truth = pd.read_csv(truth_path) if truth_path.is_file() else None
        return train, test, truth, "synthetic"
    train, test = load_competition_data(data_dir)
    return train, test, None, "official"


def _split_path(label: str, explicit: str | None) -> Path:
    return Path(explicit) if explicit else ARTIFACTS_DIR / "splits" / f"splits_{label}.json"


def ensure_splits(train_df, args, label: str):
    path = _split_path(label, getattr(args, "splits", None))
    seed = int(getattr(args, "seed", RANDOM_STATE))
    if path.is_file():
        folds, payload = load_splits(path, train_df)
        split_id = f"{payload['method']}-s{payload['seed']}-n{payload['n_splits']}"
        print(f"[splits] loaded {path} ({split_id})")
    else:
        folds, method = make_splits(
            train_df,
            n_splits=int(getattr(args, "n_splits", 5)),
            seed=seed,
            method="auto",
        )
        save_splits(path, train_df, folds, method, seed)
        split_id = f"{method}-s{seed}-n{len(folds)}"
        print(f"[splits] created {path} ({split_id})")
    return folds, split_id


# -- commands --------------------------------------------------------------
def cmd_audit(args) -> int:
    train, test, _truth, label = load_data(args.data_dir, args.synthetic)
    out_dir = Path(args.out) if args.out else ARTIFACTS_DIR / "audit" / label
    figures = FIGURES_DIR if label == "official" else out_dir / "figures"
    summary = run_audit(train, test, out_dir, dataset_label=label, figures_dir=figures)
    print(json.dumps({k: summary[k] for k in ("rows", "n_prompts", "n_labels_present")}, indent=1))
    print(f"[audit] wrote {out_dir / 'data_audit.md'}")
    for path in summary.get("figures", []):
        print(f"[audit] figure {path}")
    return 0


def cmd_splits(args) -> int:
    train, _test, _truth, label = load_data(args.data_dir, args.synthetic)
    folds, method = make_splits(
        train, n_splits=args.n_splits, seed=args.seed, method=args.method
    )
    path = _split_path(label, args.out)
    save_splits(path, train, folds, method, args.seed)
    for row in fold_groups(train, folds):
        print(
            f"[fold {row['fold']}] train {row['train_rows']} rows/"
            f"{row['train_prompts']} prompts | valid {row['valid_rows']} rows/"
            f"{row['valid_prompts']} prompts"
        )
    print(f"[splits] {method} -> {path}")
    return 0


def cmd_train(args) -> int:
    cfg = load_config(args.config)
    train, _test, _truth, label = load_data(args.data_dir, args.synthetic)
    folds, split_id = ensure_splits(train, args, label)
    out_dir = Path(args.out) if args.out else ARTIFACTS_DIR / "cv" / label
    registry = None if args.no_registry else Path(args.registry)
    results = run_cv(
        train,
        folds,
        cfg["models"],
        out_dir,
        seed=int(args.seed),
        dataset_label=label,
        registry_path=registry,
        split_id=split_id,
        save_fold_models=args.save_fold_models,
    )
    print(f"\n{'model':38s} {'pooled OOF':>10s} {'fold mean':>10s} {'fold std':>9s}")
    for r in sorted(results, key=lambda x: -x["pooled"]["balanced_accuracy"]):
        agg = r["fold_aggregate"]
        print(
            f"{r['name']:38s} {r['pooled']['balanced_accuracy']:10.4f} "
            f"{agg['mean']:10.4f} {agg['std']:9.4f}"
        )
    print(f"\n[train] artifacts: {out_dir}")
    if label == "synthetic":
        print("[train] SYNTHETIC fixture — these numbers are a pipeline check, not a score.")
    return 0


def cmd_ensemble(args) -> int:
    cfg = load_config(args.config)
    artifacts = Path(args.artifacts) if args.artifacts else ARTIFACTS_DIR / "cv" / args.dataset
    max_models = int(cfg.get("ensemble", {}).get("max_models", 4))
    result = run_ensemble(
        artifacts,
        out_path=args.out or artifacts / "ensemble" / "weights.json",
        max_models=max_models,
    )
    print("[ensemble] weights:", json.dumps(result["weights"], indent=1))
    print(
        f"[ensemble] pooled OOF {result['pooled_oof_balanced_accuracy']:.4f} "
        f"vs best single {result['best_single']['balanced_accuracy']:.4f} "
        f"({result['best_single']['model']}), gain "
        f"{result['gain_over_best_single']:+.4f}"
    )
    if result["gain_over_best_single"] <= 0:
        print("[ensemble] no validated gain; prefer the single best model for submission.")
    return 0


def cmd_predict(args) -> int:
    cfg = load_config(args.config)
    train, test, truth, label = load_data(args.data_dir, args.synthetic)
    final_dir = (
        Path(args.final_dir)
        if args.final_dir
        else ARTIFACTS_DIR / "final" / label
    )

    manifest_path = final_dir / "manifest.json"
    if args.fit or not manifest_path.is_file():
        manifest = fit_final(
            cfg["models"], train, final_dir, seed=int(args.seed), dataset_label=label
        )
        print(f"[predict] refit on {len(train)} rows -> {manifest}")
    else:
        print(f"[predict] using existing artifacts in {final_dir}")

    weights_path = Path(args.weights) if args.weights else None
    default_weights = Path(args.artifacts or ARTIFACTS_DIR / "cv" / label) / "ensemble" / "weights.json"
    if weights_path is None and default_weights.is_file():
        weights_path = default_weights
        print(f"[predict] blending with {weights_path}")
    elif weights_path is None:
        print("[predict] no ensemble weights; averaging available models equally")

    bundles, manifest = load_final(final_dir)
    weights = None
    if weights_path is not None:
        from src.inference import load_weights

        weights = load_weights(weights_path)
        missing = sorted(set(weights) - set(bundles))
        if missing:
            print(f"[predict] weights reference models absent from {final_dir}: {missing}")
            weights = {k: v for k, v in weights.items() if k in bundles}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

    predictions = predict_dataframe(bundles, test, weights)
    out = Path(args.out) if args.out else ARTIFACTS_DIR / "submissions" / label / SUBMISSION_FILENAME
    build_submission(test["id"], predictions, out, expected_ids=test["id"])
    print(f"[predict] wrote {out} ({len(predictions)} rows, {len(set(predictions))} distinct labels)")

    if label == "synthetic" and truth is not None:
        truth_map = dict(zip(truth["id"].astype(str), truth["model"].astype(str), strict=True))
        missing = [i for i in test["id"].astype(str) if i not in truth_map]
        if not missing:
            y_true = [truth_map[i] for i in test["id"].astype(str)]
            result = evaluate(y_true, predictions)
            print(
                f"[synthetic] held-out prompts balanced accuracy "
                f"{result['balanced_accuracy']:.4f} (fixture only)"
            )
        print("[synthetic] NOT a Kaggle submission — delete before packaging.")
    return 0


def cmd_package(args) -> int:
    """Copy a final artifact dir into an upload-ready Kaggle dataset folder."""
    import shutil

    final_dir = Path(args.final_dir) if args.final_dir else ARTIFACTS_DIR / "final" / args.dataset
    if not (final_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"{final_dir} has no manifest; run predict --fit first")
    out = Path(args.out) if args.out else ARTIFACTS_DIR / "kaggle_dataset"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(final_dir, out)

    weights = final_dir.parent / "cv" / args.dataset / "ensemble" / "weights.json"
    if weights.is_file():
        shutil.copy2(weights, out / "weights.json")

    # The pickled bundles reference src.model, so ship the source alongside.
    src_out = out / "src"
    src_out.mkdir(exist_ok=True)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path, src_out / path.name)

    metadata = {
        "title": f"name-that-llm artifacts ({args.dataset})",
        "id": f"{args.kaggle_user}/name-that-llm-artifacts",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    (out / "README.md").write_text(
        "# Name That LLM prediction artifacts\n\n"
        "Attach this dataset to `kaggle/submission.ipynb` as a notebook input.\n"
        "Contents: fitted models (`models/`), `manifest.json` (class order, run\n"
        "metadata), optional `weights.json` for blending, and `src/` needed to\n"
        "unpickle the bundles. Generated locally; no competition data included.\n",
        encoding="utf-8",
    )
    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024**2
    print(f"[package] {out} ({size_mb:.1f} MB)")
    print(f"[package] upload with: kaggle datasets create -p {out}")
    return 0


def cmd_demo(args) -> int:
    """Synthetic end-to-end smoke run: audit -> splits -> CV -> ensemble -> submission."""
    args.synthetic = True
    args.data_dir = str(Path(args.data_dir) if args.data_dir else DATA_DIR / "synthetic")
    label = "synthetic"
    print("=== SYNTHETIC SMOKE RUN (no competition data touched) ===")

    train, test, truth, label = load_data(args.data_dir, True)
    print(f"[data] dense fixture: {len(train)} train rows, {len(test)} test rows")

    run_audit(train, test, ARTIFACTS_DIR / "demo" / "audit", dataset_label=label,
              figures_dir=ARTIFACTS_DIR / "demo" / "audit" / "figures")

    cfg = load_config(args.config)
    folds, method = make_splits(train, n_splits=args.n_splits, seed=args.seed)
    save_splits(ARTIFACTS_DIR / "demo" / "splits.json", train, folds, method, args.seed)
    print(f"[splits] {method}, {len(folds)} folds")

    results = run_cv(
        train,
        folds,
        cfg["models"],
        ARTIFACTS_DIR / "demo" / "cv",
        seed=args.seed,
        dataset_label=label,
        registry_path=None,
        split_id=f"{method}-s{args.seed}-n{len(folds)}",
    )
    for r in sorted(results, key=lambda x: -x["pooled"]["balanced_accuracy"]):
        print(f"  {r['name']:34s} pooled OOF {r['pooled']['balanced_accuracy']:.4f}")

    run_ensemble(ARTIFACTS_DIR / "demo" / "cv", max_models=3)

    fit_final(cfg["models"], train, ARTIFACTS_DIR / "demo" / "final",
              seed=args.seed, dataset_label=label)
    bundles, _manifest = load_final(ARTIFACTS_DIR / "demo" / "final")
    from src.inference import load_weights

    weights_path = ARTIFACTS_DIR / "demo" / "cv" / "ensemble" / "weights.json"
    weights = load_weights(weights_path) if weights_path.is_file() else None
    predictions = predict_dataframe(bundles, test, weights)
    out = ARTIFACTS_DIR / "demo" / SUBMISSION_FILENAME
    build_submission(test["id"], predictions, out, expected_ids=test["id"])
    validate_submission(out, expected_ids=test["id"])

    truth_map = dict(zip(truth["id"].astype(str), truth["model"].astype(str), strict=True))
    y_true = [truth_map[i] for i in test["id"].astype(str)]
    result = evaluate(y_true, predictions)
    print(
        f"[demo] submission validated: {out} | synthetic held-out balanced accuracy "
        f"{result['balanced_accuracy']:.4f} (fixture only)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Name That LLM: CPU-first attribution pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, config: bool = True):
        p.add_argument("--data-dir", default=None, help="folder containing train.csv/test.csv")
        p.add_argument("--synthetic", action="store_true", help="use the synthetic fixture")
        p.add_argument("--seed", type=int, default=RANDOM_STATE)
        if config:
            p.add_argument("--config", default="configs/baseline.yaml")

    p_audit = sub.add_parser("audit", help="dataset audit and figures")
    add_common(p_audit, config=False)
    p_audit.add_argument("--out", default=None)
    p_audit.set_defaults(func=cmd_audit)

    p_splits = sub.add_parser("splits", help="build prompt-grouped CV folds")
    add_common(p_splits, config=False)
    p_splits.add_argument("--n-splits", type=int, default=5)
    p_splits.add_argument("--method", default="auto", choices=["auto", "group", "stratified_group"])
    p_splits.add_argument("--out", default=None)
    p_splits.set_defaults(func=cmd_splits)

    p_train = sub.add_parser("train", help="cross-validate every model in a config")
    add_common(p_train)
    p_train.add_argument("--out", default=None)
    p_train.add_argument("--splits", default=None)
    p_train.add_argument("--n-splits", type=int, default=5)
    p_train.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p_train.add_argument("--no-registry", action="store_true")
    p_train.add_argument("--save-fold-models", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_ens = sub.add_parser("ensemble", help="greedy blend over saved OOF scores")
    add_common(p_ens)
    p_ens.add_argument("--artifacts", default=None)
    p_ens.add_argument("--dataset", default="official", choices=["official", "synthetic"])
    p_ens.add_argument("--out", default=None)
    p_ens.set_defaults(func=cmd_ensemble)

    p_pred = sub.add_parser("predict", help="refit on all data and write a submission")
    add_common(p_pred)
    p_pred.add_argument("--final-dir", default=None)
    p_pred.add_argument("--weights", default=None)
    p_pred.add_argument("--artifacts", default=None)
    p_pred.add_argument("--fit", action="store_true", help="force a refit on all train rows")
    p_pred.add_argument("--out", default=None)
    p_pred.set_defaults(func=cmd_predict)

    p_pkg = sub.add_parser("package", help="build an upload-ready Kaggle dataset folder")
    p_pkg.add_argument("--final-dir", default=None)
    p_pkg.add_argument("--dataset", default="official", choices=["official", "synthetic"])
    p_pkg.add_argument("--out", default=None)
    p_pkg.add_argument("--kaggle-user", default="your-kaggle-username")
    p_pkg.set_defaults(func=cmd_package)

    p_demo = sub.add_parser("demo", help="synthetic end-to-end smoke run")
    p_demo.add_argument("--data-dir", default=None)
    p_demo.add_argument("--config", default="configs/baseline.yaml")
    p_demo.add_argument("--seed", type=int, default=RANDOM_STATE)
    p_demo.add_argument("--n-splits", type=int, default=5)
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    print(f"[env] {json.dumps(env_snapshot())[:240]}")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
