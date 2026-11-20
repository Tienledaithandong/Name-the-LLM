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
from src.config import load_config, resolved_models
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
    indices_to_labels,
    labels_to_indices,
    load_competition_data,
    ordered_dataset_fingerprint,
    validate_test_frame,
    validate_train_frame,
)
from src.ensemble import load_oof, normalize_scores, run_ensemble
from src.inference import fit_final, load_final, predict_dataframe
from src.metrics import (
    aggregate_folds,
    confusion_matrix,
    evaluate,
    length_band_metrics,
    mean_ci,
    paired_difference,
    within_provider_confusions,
)
from src.oof_identity import fold_assignments
from src.repro import env_snapshot, git_commit
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


def ensure_splits(train_df, args, label: str, cfg=None):
    path = _split_path(label, getattr(args, "splits", None))
    cfg = cfg or {}
    seed = int(getattr(args, "seed", None) if getattr(args, "seed", None) is not None else cfg.get("seed", RANDOM_STATE))
    if path.is_file():
        folds, payload = load_splits(path, train_df)
        expected_n = int(getattr(args, "n_splits", None) or cfg.get("cv", {}).get("n_splits", 5))
        if payload["seed"] != seed or payload["n_splits"] != expected_n:
            raise ValueError("saved splits do not match requested seed or fold count; regenerate splits")
        if payload.get("shuffle") != cfg.get("cv", {}).get("shuffle", True):
            raise ValueError("saved splits do not match requested shuffle; regenerate splits")
        requested_method = cfg.get("cv", {}).get("method", "auto")
        if requested_method != "auto" and payload["method"] != requested_method:
            raise ValueError("saved splits do not match requested method; regenerate splits")
        split_id = f"{payload['method']}-s{payload['seed']}-n{payload['n_splits']}"
        print(f"[splits] loaded {path} ({split_id})")
    else:
        folds, method = make_splits(
            train_df,
            n_splits=int(getattr(args, "n_splits", None) or cfg.get("cv", {}).get("n_splits", 5)),
            seed=seed,
            method=cfg.get("cv", {}).get("method", "auto"),
            shuffle=cfg.get("cv", {}).get("shuffle", True),
        )
        save_splits(path, train_df, folds, method, seed, shuffle=cfg.get("cv", {}).get("shuffle", True))
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
        train, n_splits=args.n_splits, seed=args.seed if args.seed is not None else RANDOM_STATE, method=args.method
    )
    path = _split_path(label, args.out)
    save_splits(path, train, folds, method, args.seed if args.seed is not None else RANDOM_STATE)
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
    folds, split_id = ensure_splits(train, args, label, cfg)
    out_dir = Path(args.out) if args.out else ARTIFACTS_DIR / "cv" / label
    registry = None if args.no_registry else Path(args.registry)
    results = run_cv(
        train,
        folds,
        resolved_models(cfg),
        out_dir,
        seed=int(args.seed if args.seed is not None else cfg["seed"]),
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
    members = cfg.get("ensemble", {}).get("members")
    if not isinstance(members, list) or not members or any(not isinstance(x, str) or not x for x in members):
        raise ValueError("ensemble config must explicitly list the candidate members")
    min_gain = float(cfg.get("ensemble", {}).get("min_gain", 0.0))
    if not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError("ensemble min_gain must be a finite nonnegative number")
    candidate_path = Path(args.out) if args.out else artifacts / "ensemble" / "candidate_weights.json"
    result = run_ensemble(
        artifacts,
        out_path=candidate_path,
        max_models=max_models,
        members=members,
        min_gain=min_gain,
    )
    result["min_gain"] = min_gain
    result["meets_min_gain_on_selection_oof"] = result["gain_over_best_single"] >= min_gain
    candidate_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("[ensemble] weights:", json.dumps(result["weights"], indent=1))
    print(
        f"[ensemble] pooled OOF {result['pooled_oof_balanced_accuracy']:.4f} "
        f"vs best single {result['best_single']['balanced_accuracy']:.4f} "
        f"({result['best_single']['model']}), gain "
        f"{result['gain_over_best_single']:+.4f}"
    )
    print(f"[ensemble] exploratory candidate saved to {candidate_path}; independent confirmation is required.")
    if not result["meets_min_gain_on_selection_oof"]:
        print(f"[ensemble] selection OOF gain does not meet min_gain={min_gain:.4f}.")
    return 0


# D041 bands: the official corpus spans 36-274 words, so the top band stays open-ended.
LENGTH_BAND_EDGES = (0, 50, 100, 150, 200)


def _variant_metrics(truth, pred, folds, name: str, words=None) -> dict:
    entry = evaluate(truth, pred, model=name)
    entry["within_provider"] = within_provider_confusions(truth, pred)
    entry["confusion_matrix"] = np.asarray(confusion_matrix(truth, pred)).tolist()
    entry["folds"] = aggregate_folds(
        evaluate(truth[folds == k], pred[folds == k], model=name) for k in np.unique(folds)
    )
    entry["fold_ci"] = mean_ci(entry["folds"]["fold_scores"])
    if words is not None:
        entry["length_bands"] = length_band_metrics(truth, pred, words, edges=LENGTH_BAND_EDGES)
    return entry


def cmd_evaluate(args) -> int:
    """Score saved identity-verified OOF; fits nothing and scores no test row."""
    artifacts = Path(args.artifacts) if args.artifacts else ARTIFACTS_DIR / "cv" / args.dataset
    members = list(args.members)
    oof = load_oof(artifacts, members)
    splits = json.loads(_split_path(args.dataset, args.splits).read_text(encoding="utf-8"))
    fingerprint = splits.get("ordered_dataset_sha256")
    reference = oof[members[0]]
    n_rows = int(reference["y"].size)
    folds_full = [
        (np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int))
        for f in splits["folds"]
    ]
    expected_folds = fold_assignments(n_rows, folds_full)
    for name in members:
        item = oof[name]
        if item["ordered_dataset_sha256"] != fingerprint:
            raise ValueError(f"{name}: OOF fingerprint differs from the frozen splits")
        if not np.array_equal(item["validation_fold_ids"], expected_folds):
            raise ValueError(f"{name}: OOF fold assignments differ from the frozen splits")

    truth = indices_to_labels(reference["y"])
    folds = reference["validation_fold_ids"]
    words = None
    if args.length_bands:
        train, _test = load_competition_data(args.data_dir)
        if ordered_dataset_fingerprint(train) != fingerprint:
            raise ValueError("the local train frame does not match the split fingerprint")
        if len(train) != n_rows:
            raise ValueError(f"OOF has {n_rows} rows but train has {len(train)}")
        words = train["text"].astype(str).str.split().str.len().to_numpy(dtype=np.int64)

    results = {}
    for name in members:
        pred = indices_to_labels(oof[name]["scores"].argmax(axis=1))
        results[name] = _variant_metrics(truth, pred, folds, name, words)

    if args.weights:
        weights = [float(w) for w in args.weights]
        if len(weights) != len(members):
            raise ValueError("--weights must match --members one-to-one")
        if any(not np.isfinite(w) or w < 0 for w in weights) or not any(w > 0 for w in weights):
            raise ValueError("--weights must be finite, nonnegative and not all zero")
        matrix = sum(
            normalize_scores(oof[name]["scores"]) * w for name, w in zip(members, weights)
        )
        pred = indices_to_labels(matrix.argmax(axis=1))
        blend = _variant_metrics(truth, pred, folds, "blend", words)
        blend["weights"] = dict(zip(members, weights))
        blend["blend_rule"] = "row_z_mean (src.ensemble.normalize_scores); margins are not probabilities"
        blend["paired_vs_members"] = {
            name: mean_ci(
                paired_difference(results[name]["folds"]["fold_scores"], blend["folds"]["fold_scores"])
            )
            for name in members
        }
        results["blend"] = blend

    split_id = f"{splits['method']}-s{splits['seed']}-n{splits['n_splits']}"
    payload = {
        "dataset": args.dataset,
        "artifacts": str(artifacts),
        "members": members,
        "split_id": split_id,
        "ordered_dataset_sha256": fingerprint,
        "n_rows": n_rows,
        "git_commit": git_commit(),
        "environment": env_snapshot(),
        "fits_performed": 0,
        "transductive": False,
        "timing_note": (
            "Post-hoc scoring of saved OOF: fit/infer seconds and peak RAM are properties of "
            "the training run and are recorded by `cli train` in experiments/registry.csv, "
            "not re-measured here."
        ),
        "results": results,
    }
    out_path = (
        Path(args.out) if args.out else ARTIFACTS_DIR / "evaluate" / args.dataset / "metrics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"[evaluate] {split_id} | {n_rows} rows | fingerprint {str(fingerprint)[:16]}")
    for name, entry in results.items():
        ci = entry["fold_ci"]
        print(
            f"[evaluate] {name:22s} pooled BA {entry['balanced_accuracy']:.4f} | "
            f"fold mean {entry['folds']['mean']:.4f} +/- {entry['folds']['std']:.4f} | "
            f"95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] | macro F1 {entry['macro_f1']:.4f}"
        )
        if "paired_vs_members" in entry:
            for other, diff in entry["paired_vs_members"].items():
                print(
                    f"[evaluate]   blend - {other}: {diff['mean']:+.4f} "
                    f"[{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]"
                )
    print(f"[evaluate] wrote {out_path}")
    return 0


def cmd_predict(args) -> int:
    if getattr(args, "crossed", False):
        raise ValueError("crossed inference is withdrawn under D024; omit --crossed")
    cfg = load_config(args.config)
    train, test, truth, label = load_data(args.data_dir, args.synthetic)
    final_dir = (
        Path(args.final_dir)
        if args.final_dir
        else ARTIFACTS_DIR / "final" / label
    )

    manifest_path = final_dir / "manifest.json"
    if args.fit or not manifest_path.is_file():
        pool_df = None
        pool_models = list(getattr(args, "pool_models", []) or [])
        if pool_models:
            if not getattr(args, "pool", None):
                raise ValueError("--pool-models requires --pool")
            from src.pool import load_pool

            pool_df = load_pool(args.pool)
            known = {str(m["name"]) for m in cfg["models"]}
            if not set(pool_models) <= known:
                raise ValueError(f"--pool-models not in config: {sorted(set(pool_models) - known)}")
        manifest = fit_final(
            resolved_models(cfg), train, final_dir,
            seed=int(args.seed if args.seed is not None else cfg["seed"]),
            dataset_label=label,
            pool_df=pool_df, pool_models=pool_models,
            cos_threshold=float(getattr(args, "cos_threshold", 0.3)),
            pool_cap=int(getattr(args, "pool_cap", 12000)),
        )
        tag = f" (pool for {pool_models}, cap {args.pool_cap})" if pool_df is not None else ""
        print(f"[predict] refit on {len(train)} rows{tag} -> {manifest}")
    else:
        print(f"[predict] using existing artifacts in {final_dir}")

    bundles, manifest = load_final(final_dir)
    from src.inference import (
        load_weights,
        validate_artifact_provenance,
        validate_weight_provenance,
    )

    weights_path = Path(args.weights) if args.weights else None
    # A custom fitted-model directory may describe a different experiment. Do
    # not silently attach global weights unless the caller also chose artifacts.
    if args.final_dir is not None and args.artifacts is None:
        default_weights = final_dir / "weights.json"
    else:
        default_weights = Path(args.artifacts or ARTIFACTS_DIR / "cv" / label) / "ensemble" / "weights.json"
    if weights_path is None and default_weights.is_file():
        weights_path = default_weights
        print(f"[predict] blending with {weights_path}")
    elif weights_path is None:
        if not args.synthetic:
            raise FileNotFoundError("official prediction requires an explicit validated weights file")
        if len(bundles) > 1 and not args.allow_equal_weights:
            raise ValueError("no verified weights; supply --weights or explicitly use --allow-equal-weights")
        print("[predict] no ensemble weights; averaging synthetic models equally")

    weights = None
    if weights_path is not None:
        declared = load_weights(weights_path)
        missing = sorted(set(declared) - set(bundles))
        if missing:
            raise ValueError(f"weights reference models absent from {final_dir}: {missing}")
        weights = validate_weight_provenance(weights_path, manifest, bundles)

    validate_artifact_provenance(
        manifest, train, resolved_models(cfg), int(args.seed if args.seed is not None else cfg["seed"])
    )

    if args.crossed:
        from src.features_embedding import FrozenEncoder
        from src.inference import predict_dataframe_crossed

        model_name = args.encoder_path or "sentence-transformers/all-MiniLM-L6-v2"
        encoder = FrozenEncoder(backend=args.encoder_backend, model_name=model_name)
        predictions = predict_dataframe_crossed(
            bundles,
            test,
            weights,
            encoder=encoder,
            distance_threshold=args.distance_threshold,
            n_clusters=args.n_clusters,
        )
        grouping = (
            f"n_clusters={args.n_clusters}"
            if args.n_clusters is not None
            else f"cosine thr={args.distance_threshold}"
        )
        print(f"[predict] crossed inference: {args.encoder_backend} grouping ({grouping})")
    else:
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


def _align_packaged_weights(weights_path: Path, manifest_path: Path) -> None:
    """Drop CV fields measured for a different fit than the bundles being shipped.

    The ensemble weights file is written by the official-only CV run, so its
    recorded score describes that fit. When a member was refit with the external
    pool the numbers no longer apply to this package (D037/D040).
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fit_modes = {
        str(m["name"]): str(m.get("fit_mode", "train_only"))
        for m in manifest.get("models", [])
    }
    pooled = sorted(n for n, mode in fit_modes.items() if "pool" in mode)
    if not pooled:
        return
    doc = json.loads(weights_path.read_text(encoding="utf-8"))
    for key in ("pooled_oof_balanced_accuracy", "best_single", "gain_over_best_single", "singles"):
        doc.pop(key, None)
    doc["fit_modes"] = fit_modes
    doc["cv_score_of_record"] = "experiments/registry.csv"
    doc["note"] = (
        f"{', '.join(pooled)} refit on official train + globally decontaminated external "
        "pool (D028/D037); the official-only CV fields were removed because they describe "
        "a different fit. Blend rule and weights are unchanged."
    )
    weights_path.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def _package_licence(manifest_path: Path) -> tuple[str, str]:
    """Licence name plus attribution block for the shipped weights (D050).

    The weights are a derived adaptation of CC BY-SA 4.0 text — the official
    train always, the external pool for any member fitted with it — so
    share-alike travels with them. A synthetic-fixture package is locally
    generated and carries no competition attribution.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("dataset", "")) == "synthetic":
        return "CC0-1.0", (
            "\n## Licence\n\nSynthetic fixture artifacts generated locally for a smoke\n"
            "run; no competition text was used. CC0-1.0.\n"
        )
    pool = manifest.get("pool") or {}
    pooled = sorted(
        str(m["name"]) for m in manifest.get("models", [])
        if "pool" in str(m.get("fit_mode", ""))
    )
    lines = [
        "\n## Licence and attribution\n",
        "These fitted weights are a derived adaptation of CC BY-SA 4.0 text and are",
        "redistributed under **CC BY-SA 4.0** — share-alike applies:",
        "https://creativecommons.org/licenses/by-sa/4.0/\n",
        "- Official training passages: Kaggle **Name That LLM**",
        "  (https://www.kaggle.com/competitions/name-that-llm), data CC BY-SA 4.0.",
    ]
    if pooled:
        lines += [
            "- External training passages: **Open Pool v15**",
            "  (https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool),",
            f"  CC BY-SA 4.0, used for {', '.join(pooled)} only",
            f"  (`external_data_id: {pool.get('external_data_id', 'open-pool-v15')}`).",
            "  The pool parquet is not shipped — only the fitted weights are.",
        ]
    lines += [
        "\nNo raw competition rows, test text or labels are included; the bundles were",
        "fitted locally and the training text is not redistributed here.\n",
    ]
    return "CC-BY-SA-4.0", "\n".join(lines)


def cmd_package(args) -> int:
    """Copy a final artifact dir into an upload-ready Kaggle dataset folder."""
    if getattr(args, "crossed", False):
        raise ValueError("crossed inference packaging is withdrawn under D024; omit --crossed")
    import shutil

    final_dir = Path(args.final_dir) if args.final_dir else ARTIFACTS_DIR / "final" / args.dataset
    if not (final_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"{final_dir} has no manifest; run predict --fit first")
    out = Path(args.out) if args.out else ARTIFACTS_DIR / "kaggle_dataset"
    final_resolved, out_resolved = final_dir.resolve(), out.resolve()
    if out_resolved == final_resolved or out_resolved in final_resolved.parents or final_resolved in out_resolved.parents:
        raise ValueError("package output and final artifacts must not overlap")
    if out.exists():
        if not getattr(args, "overwrite", False):
            raise FileExistsError(f"{out} exists; use --overwrite to replace it")
        shutil.rmtree(out)
    shutil.copytree(final_dir, out)

    weights = ARTIFACTS_DIR / "cv" / args.dataset / "ensemble" / "weights.json"
    if weights.is_file():
        shutil.copy2(weights, out / "weights.json")
        _align_packaged_weights(out / "weights.json", out / "manifest.json")

    # The pickled bundles reference src.model, so ship the source alongside.
    src_out = out / "src"
    src_out.mkdir(exist_ok=True)
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path, src_out / path.name)

    crossed = getattr(args, "crossed", False)
    encoder_note = ""
    if crossed:
        encoder_src = getattr(args, "encoder_path", None)
        if encoder_src:
            encoder_dir = Path(encoder_src)
            if not encoder_dir.is_dir():
                raise FileNotFoundError(f"encoder path not found: {encoder_dir}")
        else:
            from huggingface_hub import snapshot_download

            encoder_dir = Path(
                snapshot_download(
                    "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
                )
            )
        shutil.copytree(encoder_dir, out / "encoder" / "minilm")
        inference = {
            "crossed": True,
            "encoder_backend": "sentence-transformers",
            "encoder_path": "encoder/minilm",
            "distance_threshold": 0.4,
            "n_clusters": None,
        }
        (out / "inference.json").write_text(json.dumps(inference, indent=1), encoding="utf-8")
        encoder_note = (
            "\nCrossed inference (D016): this package also ships the MiniLM grouping\n"
            "encoder (`encoder/minilm`) and `inference.json`. The notebook clusters\n"
            "test passages into prompt groups and enforces one-per-model assignment.\n"
            "The encoder is loaded from the attached folder; no download at runtime.\n"
        )

    licence, attribution = _package_licence(out / "manifest.json")
    metadata = {
        "title": f"name-that-llm artifacts ({args.dataset})",
        "id": f"{args.kaggle_user}/name-that-llm-artifacts",
        "licenses": [{"name": licence}],
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    (out / "README.md").write_text(
        "# Name That LLM prediction artifacts\n\n"
        "Attach this dataset to `kaggle/submission.ipynb` as a notebook input.\n"
        "Contents: fitted models (`models/`), `manifest.json` (class order, run\n"
        "metadata), optional `weights.json` for blending, and `src/` needed to\n"
        "unpickle the bundles. Generated locally; no raw competition rows included.\n"
        + encoder_note
        + attribution,
        encoding="utf-8",
    )
    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1024**2
    print(f"[package] {out} ({size_mb:.1f} MB)")
    print(f"[package] upload with: kaggle datasets create -p {out}")
    return 0


def cmd_stability(args) -> int:
    """Section 7: re-check the blend gain on alternative seeds + secondary holdout."""
    from src.stability import run_stability

    cfg = load_config(args.config)
    train, _test, _truth, label = load_data(args.data_dir, args.synthetic)
    weights_path = Path(args.weights) if args.weights else (
        ARTIFACTS_DIR / "cv" / label / "ensemble" / "weights.json"
    )
    if not weights_path.is_file():
        raise FileNotFoundError(f"{weights_path} missing; run the ensemble step first")
    weights = json.loads(weights_path.read_text(encoding="utf-8")).get("weights", {})
    out_root = Path(args.out) if args.out else ARTIFACTS_DIR / "stability" / label
    registry = None if args.no_registry else Path(args.registry)
    summary = run_stability(
        train,
        resolved_models(cfg),
        weights,
        seeds=[int(s) for s in args.seeds],
        out_root=out_root,
        dataset_label=label,
        registry_path=registry,
        holdout_seed=args.holdout_seed,
        holdout_groups=args.holdout_groups,
    )
    for row in summary["seed_stability"]:
        print(
            f"[stability] seed {row['seed']}: blend {row['blend_pooled_oof']:.4f} "
            f"vs best single {row['best_single']['balanced_accuracy']:.4f} "
            f"gain {row['gain_over_best_single']:+.4f}"
        )
    h = summary["holdout"]
    print(
        f"[stability] holdout(seed {h['seed']}, {h['n_groups']} prompts): "
        f"blend {h['blend_balanced_accuracy']:.4f} gain {h['gain_over_best_single']:+.4f}"
    )
    print(f"[stability] report: {out_root / 'stability.md'}")
    return 0



def cmd_reserve_holdout(args) -> int:
    """Create selection and reserved prompt partitions before any model selection."""
    from src.validation import make_holdout
    train, _, _, _ = load_data(args.data_dir, args.synthetic)
    selection, holdout = make_holdout(train, n_groups=args.holdout_groups, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, indices in (("selection.csv", selection), ("holdout.csv", holdout)):
        path = out / name
        if path.exists():
            raise FileExistsError(f"{path} already exists; choose a fresh output directory")
    train.iloc[selection].to_csv(out / "selection.csv", index=False)
    train.iloc[holdout].to_csv(out / "holdout.csv", index=False)
    print(f"[holdout] reserved {len(holdout)} rows; keep holdout.csv out of selection")
    return 0


def cmd_score_reserved(args) -> int:
    """Score a previously reserved holdout with selection-only model artifacts."""
    from src.inference import validate_artifact_provenance, validate_weight_provenance
    from src.validation import LeakageError
    cfg = load_config(args.config)
    selection = validate_train_frame(pd.read_csv(args.selection))
    holdout = validate_train_frame(pd.read_csv(args.holdout))
    if set(selection["prompt_id"]) & set(holdout["prompt_id"]) or set(selection["id"]) & set(holdout["id"]):
        raise LeakageError("selection and reserved holdout overlap")
    bundles, manifest = load_final(args.final_dir)
    validate_artifact_provenance(manifest, selection, resolved_models(cfg), args.seed if args.seed is not None else cfg["seed"])
    weights = validate_weight_provenance(args.weights, manifest, bundles)
    predictions = predict_dataframe(bundles, holdout, weights)
    result = evaluate(holdout["model"].tolist(), predictions, model="reserved_holdout")
    print(json.dumps(result, indent=2))
    return 0

def cmd_figures(args) -> int:
    """Render the Section-9 report figures from saved CV artifacts."""
    from src.figures import make_report_figures

    artifacts = Path(args.artifacts) if args.artifacts else ARTIFACTS_DIR / "cv" / args.dataset
    weights_path = Path(args.weights) if args.weights else artifacts / "ensemble" / "weights.json"
    weights = None
    if weights_path.is_file():
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        weights = payload.get("weights", payload)
        print(f"[figures] blend weights from {weights_path}: {weights}")
    else:
        print("[figures] no weights.json; using the best single model")
    out_dir = Path(args.out) if args.out else (
        FIGURES_DIR if args.dataset == "official" else artifacts / "figures"
    )
    for path in make_report_figures(artifacts, out_dir, weights=weights, dataset_label=args.dataset):
        print(f"[figures] {path}")
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
        resolved_models(cfg),
        ARTIFACTS_DIR / "demo" / "cv",
        seed=args.seed,
        dataset_label=label,
        registry_path=None,
        split_id=f"{method}-s{args.seed}-n{len(folds)}",
    )
    for r in sorted(results, key=lambda x: -x["pooled"]["balanced_accuracy"]):
        print(f"  {r['name']:34s} pooled OOF {r['pooled']['balanced_accuracy']:.4f}")

    run_ensemble(ARTIFACTS_DIR / "demo" / "cv", max_models=3)

    fit_final(resolved_models(cfg), train, ARTIFACTS_DIR / "demo" / "final",
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
        p.add_argument("--seed", type=int, default=None)
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
    p_train.add_argument("--n-splits", type=int, default=None)
    p_train.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p_train.add_argument("--no-registry", action="store_true")
    p_train.add_argument("--save-fold-models", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_ens = sub.add_parser("ensemble", help="greedy blend over saved OOF scores")
    add_common(p_ens)
    p_ens.set_defaults(config="configs/ensemble.yaml")
    p_ens.add_argument("--artifacts", default=None)
    p_ens.add_argument("--dataset", default="official", choices=["official", "synthetic"])
    p_ens.add_argument("--out", default=None)
    p_ens.set_defaults(func=cmd_ensemble, config="configs/ensemble.yaml")

    p_eval = sub.add_parser("evaluate", help="score saved identity-verified OOF (no fitting)")
    add_common(p_eval, config=False)
    p_eval.add_argument("--artifacts", default=None)
    p_eval.add_argument("--dataset", default="official", choices=["official", "synthetic"])
    p_eval.add_argument("--members", nargs="+", required=True, help="explicit OOF member names")
    p_eval.add_argument("--weights", nargs="+", default=None, help="fixed blend weights, one per member")
    p_eval.add_argument("--splits", default=None)
    p_eval.add_argument("--length-bands", action="store_true", help="also score passage-length bands (reads train.csv)")
    p_eval.add_argument("--out", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="refit on all data and write a submission")
    add_common(p_pred)
    p_pred.add_argument("--final-dir", default=None)
    p_pred.add_argument("--weights", default=None)
    p_pred.add_argument("--artifacts", default=None)
    p_pred.add_argument("--allow-equal-weights", action="store_true", help="explicitly allow unvalidated equal weighting")
    p_pred.add_argument("--fit", action="store_true", help="force a refit on all train rows")
    p_pred.add_argument("--out", default=None)
    p_pred.add_argument(
        "--pool", default=None,
        help="external pool parquet; with --pool-models fits those members on train+clean pool (D028)",
    )
    p_pred.add_argument(
        "--pool-models", nargs="*", default=[],
        help="model names trained with the globally-decontaminated pool (e.g. char_tfidf_logreg)",
    )
    p_pred.add_argument("--pool-cap", type=int, default=12000,
                        help="stratified cap on pool rows for the final fit (0 = full clean pool)")
    p_pred.add_argument("--cos-threshold", type=float, default=0.3,
                        help="global decontamination cosine threshold (D028)")
    p_pred.add_argument(
        "--crossed", action="store_true",
        help="withdrawn under D024; this option fails closed",
    )
    p_pred.add_argument(
        "--distance-threshold", type=float, default=0.4,
        help="cosine distance threshold for prompt-group clustering (crossed; default 0.4)",
    )
    p_pred.add_argument(
        "--n-clusters", type=int, default=None,
        help="fixed number of prompt groups; overrides --distance-threshold",
    )
    p_pred.add_argument(
        "--encoder-backend", default="sentence-transformers",
        choices=["sentence-transformers", "hashing"],
        help="grouping encoder backend (hashing is hermetic, for smoke tests)",
    )
    p_pred.add_argument(
        "--encoder-path", default=None,
        help="local MiniLM checkpoint dir for offline grouping (crossed)",
    )
    p_pred.set_defaults(func=cmd_predict)

    p_pkg = sub.add_parser("package", help="build an upload-ready Kaggle dataset folder")
    p_pkg.add_argument("--final-dir", default=None)
    p_pkg.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    p_pkg.add_argument("--dataset", default="official", choices=["official", "synthetic"])
    p_pkg.add_argument("--out", default=None)
    p_pkg.add_argument("--kaggle-user", default="your-kaggle-username")
    p_pkg.add_argument(
        "--crossed", action="store_true",
        help="withdrawn under D024; this option fails closed",
    )
    p_pkg.add_argument(
        "--encoder-path", default=None,
        help="local MiniLM checkpoint dir to ship (default: resolve the HF cache offline)",
    )
    p_pkg.set_defaults(func=cmd_package)

    p_stab = sub.add_parser("stability", help="blend robustness: alternative seeds + secondary holdout")
    add_common(p_stab, config=False)
    p_stab.add_argument("--config", default="configs/submission.yaml")
    p_stab.add_argument("--seeds", nargs="+", type=int, default=[7, 13])
    p_stab.add_argument("--weights", default=None)
    p_stab.add_argument("--out", default=None)
    p_stab.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    p_stab.add_argument("--no-registry", action="store_true")
    p_stab.add_argument("--holdout-seed", type=int, default=7)
    p_stab.add_argument("--holdout-groups", type=int, default=25)
    p_stab.set_defaults(func=cmd_stability)

    p_reserve = sub.add_parser("reserve-holdout", help="reserve prompts before model selection")
    p_reserve.add_argument("--data-dir", default=None)
    p_reserve.add_argument("--synthetic", action="store_true")
    p_reserve.add_argument("--seed", type=int, default=7)
    p_reserve.add_argument("--holdout-groups", type=int, default=25)
    p_reserve.add_argument("--out", required=True, help="private directory for selection.csv and holdout.csv")
    p_reserve.set_defaults(func=cmd_reserve_holdout)

    p_score = sub.add_parser("score-reserved", help="evaluate selection-only artifacts on reserved prompts")
    p_score.add_argument("--selection", required=True)
    p_score.add_argument("--holdout", required=True)
    p_score.add_argument("--final-dir", required=True)
    p_score.add_argument("--weights", required=True)
    p_score.add_argument("--config", default="configs/submission.yaml")
    p_score.add_argument("--seed", type=int, default=None)
    p_score.set_defaults(func=cmd_score_reserved)

    p_fig = sub.add_parser("figures", help="render report figures from saved CV artifacts")
    p_fig.add_argument("--artifacts", default=None)
    p_fig.add_argument("--dataset", default="official", choices=["official", "synthetic"])
    p_fig.add_argument("--weights", default=None)
    p_fig.add_argument("--out", default=None)
    p_fig.set_defaults(func=cmd_figures)

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
