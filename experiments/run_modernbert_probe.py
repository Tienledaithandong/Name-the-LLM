"""Budgeted frozen-encoder pilot and fixed official-data screens."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from experiments.run_score_improvement import (
    EXPECTED_BASELINE,
    complete_scores,
    load_official_train,
    report_scores,
    verified_control,
)
from src.config import load_config
from src.constants import LABELS, N_CLASSES
from src.data import indices_to_labels, labels_to_indices, ordered_dataset_fingerprint
from src.ensemble import load_oof, normalize_scores
from src.frozen_modernbert import FrozenModernBert
from src.metrics import balanced_accuracy
from src.oof_identity import fold_assignments
from src.repro import env_snapshot, git_commit, git_worktree_status
from src.train import append_registry, rss_gb
from src.validation import load_splits


def default_splits(seed: int) -> Path:
    return Path(f"artifacts/splits/splits_official_verified_s{seed}.json")


def default_baseline(seed: int) -> Path:
    suffix = "" if seed == 42 else f"_s{seed}"
    return Path(f"artifacts/cv_identity/official{suffix}")


def pilot_passes(rows: list[dict], expected_folds: list[int], minimum: float) -> bool:
    """Expand only if both preregistered folds clear standalone and blend checks."""
    return [row["fold"] for row in rows] == expected_folds and all(
        all(0 <= row[key] <= 1 for key in ("encoder", "blend", "baseline"))
        and row["encoder"] > 1 / N_CLASSES and row["blend"] - row["baseline"] >= minimum
        for row in rows
    )


def require_passing_full(
    path: Path, df, y: np.ndarray, fingerprint: str, config: dict, runtime: dict,
) -> dict:
    """Verify the five-fold seed-42 run and gate the fixed seed-13 comparison."""
    prior = json.loads(path.read_text(encoding="utf-8"))
    spec = config["probe"]
    environment = prior.get("environment", {})
    if (
        prior.get("stage") != "full" or prior.get("dataset") != "official"
        or prior.get("seed") != 42 or prior.get("n_train_rows") != len(y)
        or prior.get("ordered_dataset_sha256") != fingerprint
        or prior.get("labels") != list(LABELS) or prior.get("runtime") != runtime
        or prior.get("encoder") != config["encoder"] or prior.get("probe") != spec
        or prior.get("budget_seconds") != spec["full_max_seconds"]
        or prior.get("test_rows_read") != 0 or prior.get("transductive") is not False
        or environment.get("sklearn") != version("scikit-learn")
        or environment.get("numpy") != np.__version__
        or environment.get("scipy") != version("scipy")
    ):
        raise ValueError("robustness screen requires a full seed-42 run on the identical recipe and runtime")
    folds, meta = load_splits(default_splits(42), df)
    if meta["seed"] != 42 or len(folds) != 5:
        raise ValueError("expected the five frozen seed-42 reference folds")
    assignments = fold_assignments(len(df), folds)
    _, control = verified_control(default_baseline(42), fingerprint, assignments, y)
    if (
        prior.get("validation_assignments_sha256")
        != hashlib.sha256(assignments.astype("<i4").tobytes()).hexdigest()
        or prior.get("normalized_control_sha256")
        != hashlib.sha256(control.astype("<f8").tobytes()).hexdigest()
    ):
        raise ValueError("full reference does not match the frozen seed-42 folds and control")
    members = load_oof(path.parent, ["baseline", "modernbert_head", "modernbert_blend"])
    for member in members.values():
        if (member["ordered_dataset_sha256"] != fingerprint
                or not np.array_equal(member["validation_fold_ids"], assignments)
                or not np.array_equal(member["y"], y)):
            raise ValueError("full reference OOF has different rows or folds")
    raw = complete_scores(members["modernbert_head"]["scores"], len(y))
    blend = complete_scores(members["modernbert_blend"]["scores"], len(y))
    weight = spec["blend_weight"]
    if (not np.array_equal(members["baseline"]["scores"], control)
            or not np.allclose(blend, (1 - weight) * control + weight * normalize_scores(raw),
                               atol=1e-12, rtol=0)):
        raise ValueError("full reference OOF does not reproduce the fixed blend")
    truth = indices_to_labels(y)
    base_pred = indices_to_labels(control.argmax(1))
    blend_pred = indices_to_labels(blend.argmax(1))
    baseline_ba = balanced_accuracy(truth, base_pred)
    gain = balanced_accuracy(truth, blend_pred) - baseline_ba
    positive = sum(
        balanced_accuracy(truth[va], blend_pred[va])
        - balanced_accuracy(truth[va], base_pred[va]) > 1e-12
        for _, va in folds
    )
    reference_result = prior.get("results", {}).get("modernbert_blend", {})
    if (
        not np.isclose(baseline_ba, EXPECTED_BASELINE[42], atol=1e-10, rtol=0)
        or not np.isclose(gain, reference_result.get("delta", np.nan), atol=1e-12, rtol=0)
        or reference_result.get("positive_folds") != positive
        or gain < 0.010 or positive < 4
    ):
        raise ValueError("full seed-42 reference must reproduce a gain of 0.010 and four positive folds")
    return {"path": str(path), "seed": 42, "pooled_gain": gain, "positive_folds": positive}


def run_probe(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the pilot requires a CUDA-enabled Python environment")
    if args.out.exists():
        raise FileExistsError(f"refusing to replace experiment directory: {args.out}")
    config = load_config("configs/frozen_modernbert_probe.yaml")
    spec = config["probe"]
    weight = spec["blend_weight"]
    if spec["pilot_folds"] != [0, 1] or weight != 0.25 or spec["pilot_min_gain"] != 0.01:
        raise ValueError("unexpected preregistered pilot settings")
    seed = 13 if args.reference_full else 42
    stage = "robustness" if args.reference_full else ("full" if args.pilot else "pilot")
    df = load_official_train(args.data_dir)
    folds, meta = load_splits(args.splits or default_splits(seed), df)
    if meta["seed"] != seed or len(folds) != 5:
        raise ValueError(f"expected the five frozen seed-{seed} folds")
    groups = df["prompt_id"].astype(str).to_numpy()
    all_texts = df["text"].tolist()
    y = labels_to_indices(df["model"])
    fingerprint = ordered_dataset_fingerprint(df)
    assignments = fold_assignments(len(df), folds)
    fold_digest = hashlib.sha256(assignments.astype("<i4").tobytes()).hexdigest()
    _, control = verified_control(args.baseline or default_baseline(seed), fingerprint, assignments, y)
    control_digest = hashlib.sha256(control.astype("<f8").tobytes()).hexdigest()
    runtime = {"torch": torch.__version__, "transformers": version("transformers"),
               "tokenizers": version("tokenizers"), "gpu": torch.cuda.get_device_name(0)}
    ba_control = balanced_accuracy(df["model"], indices_to_labels(control.argmax(1)))
    if not np.isclose(ba_control, EXPECTED_BASELINE[seed], atol=1e-10, rtol=0):
        raise ValueError("baseline control did not reproduce the registered score")
    if args.pilot:
        prior = json.loads(args.pilot.read_text(encoding="utf-8"))
        if (prior.get("stage") != "pilot" or prior.get("ordered_dataset_sha256") != fingerprint
                or prior.get("validation_assignments_sha256") != fold_digest
                or prior.get("normalized_control_sha256") != control_digest or prior.get("runtime") != runtime
                or prior.get("encoder") != config["encoder"] or prior.get("probe") != spec
                or not pilot_passes(prior["folds"], spec["pilot_folds"], spec["pilot_min_gain"])):
            raise ValueError("full GPU screen requires a passing pilot on the identical recipe and data")
    reference = (
        require_passing_full(args.reference_full, df, y, fingerprint, config, runtime)
        if args.reference_full else None
    )
    run_folds = spec["pilot_folds"] if stage == "pilot" else list(range(len(folds)))
    budget = spec["pilot_max_seconds"] if stage == "pilot" else spec["full_max_seconds"]
    started = time.perf_counter()
    deadline = started + budget
    torch.cuda.reset_peak_memory_stats()
    encoder = FrozenModernBert(args.checkpoint, args.cache_dir, **config["encoder"])
    encoder.load()
    benchmark_texts = [all_texts[j] for j in folds[0][0][:64]]
    start_benchmark = time.perf_counter()
    encoder.encode(benchmark_texts, deadline=deadline)
    torch.cuda.synchronize()
    benchmark_seconds = time.perf_counter() - start_benchmark
    estimate = 1.5 * benchmark_seconds * len(df) / len(benchmark_texts)
    print(f"[benchmark] {len(benchmark_texts)} training rows in {benchmark_seconds:.2f}s; "
          f"conservative full extraction estimate {estimate:.1f}s", flush=True)
    if time.perf_counter() - started + estimate >= budget:
        raise TimeoutError("training-only benchmark predicts the extraction will exceed its budget")
    # The frozen encoder processes passages individually. No fitting, comparisons or
    # statistics across validation rows are used to construct these representations.
    start_extract = time.perf_counter()
    embeddings = encoder.encode_cached(all_texts, deadline=deadline)
    torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - start_extract
    gpu_gib = torch.cuda.max_memory_allocated() / 1024**3
    rss = rss_gb()
    print(f"[extract] {len(df)} official training passages in {extraction_seconds:.1f}s; "
          f"peak allocated VRAM {gpu_gib:.2f} GiB", flush=True)
    if time.perf_counter() - started >= budget:
        raise TimeoutError("extraction exceeded its time budget")
    truth = indices_to_labels(y)
    fold_report = []
    head_fit_seconds = 0.0
    base_scores = np.full_like(control, np.nan)
    blend_scores = np.full_like(control, np.nan)
    with threadpool_limits(limits=2):
        for i in run_folds:
            tr, va = folds[i]
            head = LogisticRegression(C=spec["classifier_c"], class_weight="balanced",
                                      random_state=42, max_iter=500)
            head_start = time.perf_counter()
            head.fit(embeddings[tr], y[tr])
            head_fit_seconds += time.perf_counter() - head_start
            if not np.array_equal(head.classes_, np.arange(N_CLASSES)):
                raise ValueError("fold encoder head did not fit all canonical labels")
            raw = complete_scores(head.decision_function(embeddings[va]), len(va))
            combined = (1 - weight) * control[va] + weight * normalize_scores(raw)
            base_scores[va], blend_scores[va] = raw, combined
            row = {
                "fold": i,
                "baseline": balanced_accuracy(truth[va], indices_to_labels(control[va].argmax(1))),
                "encoder": balanced_accuracy(truth[va], indices_to_labels(raw.argmax(1))),
                "blend": balanced_accuracy(truth[va], indices_to_labels(combined.argmax(1))),
                "head_n_iter": head.n_iter_.tolist(),
                "head_hit_iteration_limit": bool(np.any(head.n_iter_ >= head.max_iter)),
            }
            fold_report.append(row)
            print(f"[fold {i}] baseline {row['baseline']:.6f} encoder {row['encoder']:.6f} "
                  f"blend {row['blend']:.6f} delta {row['blend'] - row['baseline']:+.6f}; "
                  f"head iterations {row['head_n_iter']}, cap reached {row['head_hit_iteration_limit']}",
                  flush=True)
            if time.perf_counter() - started >= budget:
                raise TimeoutError("screen exceeded its time budget")
    git = git_commit()
    payload = {
        "stage": stage, "dataset": "official", "seed": seed, "n_train_rows": len(df),
        "ordered_dataset_sha256": fingerprint, "labels": list(LABELS), "encoder": config["encoder"],
        "validation_assignments_sha256": fold_digest, "normalized_control_sha256": control_digest,
        "probe": spec, "environment": env_snapshot(), "runtime": runtime,
        "peak_gpu_allocated_gib": gpu_gib, "rss_after_extract_gb": rss,
        "benchmark_train_rows": len(benchmark_texts), "benchmark_seconds": benchmark_seconds,
        "extraction_seconds": extraction_seconds, "head_fit_seconds": head_fit_seconds,
        "total_elapsed_seconds": time.perf_counter() - started, "budget_seconds": budget,
        "folds": fold_report, "pilot_passed": stage == "pilot" and
        pilot_passes(fold_report, spec["pilot_folds"], spec["pilot_min_gain"]),
        "git_commit": git, "worktree": git_worktree_status(), "test_rows_read": 0, "transductive": False,
    }
    if reference is not None:
        payload["reference_full"] = reference
        payload["interpretation"] = "Descriptive robustness on previously explored official prompts."
    args.out.mkdir(parents=True, exist_ok=False)
    if stage != "pilot":
        matrices = {"baseline": control, "modernbert_head": base_scores, "modernbert_blend": blend_scores}
        payload["results"] = report_scores(y, matrices, assignments, groups, np.array([len(t.split()) for t in all_texts]))
        directory = args.out / "oof"
        directory.mkdir()
        for name, matrix in matrices.items():
            np.savez_compressed(directory / f"{name}.npz", scores=matrix, y=y, labels=np.array(LABELS),
                                ordered_dataset_sha256=fingerprint, validation_fold_ids=assignments)
    path = args.out / "results.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    now = datetime.now(timezone.utc)
    result = payload.get("results", {}).get("modernbert_blend", {})
    append_registry(args.registry, {
        "experiment_id": f"{now:%Y%m%dT%H%M%S}-official-modernbert-{stage}-s{seed}",
        "date": now.isoformat(timespec="seconds"), "git_commit": git,
        "method": f"ModernBERT frozen {stage} blend", "seed": seed, "features": "frozen mean-pooled encoder, logistic head",
        "split_id": "stratified_group-s42-folds0-1-pilot" if stage == "pilot" else f"stratified_group-s{seed}-n5",
        "fold_scores": " ".join(f"{row['blend']:.6f}" for row in fold_report),
        "pooled_oof_balanced_accuracy": result.get("balanced_accuracy"), "macro_f1": result.get("macro_f1"),
        "fit_seconds": head_fit_seconds, "external_data_id": "modernbert-large Apache-2.0 pretrained weights",
        "artifact_path": str(path), "status": "done",
        "notes": (f"Official labeled data only; {payload['worktree']} worktree; "
                  f"{stage} fold details and resource measurements in results.json. "
                  "Exploratory; no Kaggle submission."),
    })
    print(f"[{stage}] wrote {path}; expands={payload['pilot_passed']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/score_improvement/encoder_cache"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--splits", type=Path, help="frozen splits; defaults to the stage's registered split")
    parser.add_argument("--baseline", type=Path, help="matched OOF root; defaults to the stage's registered control")
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--pilot", type=Path, help="passing pilot results required to run the full seed-42 screen")
    reference.add_argument("--reference-full", type=Path,
                           help="passing full seed-42 results required for the descriptive seed-13 screen")
    run_probe(parser.parse_args())


if __name__ == "__main__":
    main()
