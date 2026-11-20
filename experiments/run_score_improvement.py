"""Score fixed, official-data experiments against matched sparse controls."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import load_config
from src.constants import LABELS, N_CLASSES
from src.data import (
    find_competition_file,
    indices_to_labels,
    labels_to_indices,
    ordered_dataset_fingerprint,
    validate_train_frame,
)
from src.ensemble import SENTINEL, load_oof, normalize_scores
from src.metrics import balanced_accuracy, evaluate, length_band_metrics, within_provider_confusions
from src.model import estimator_from_config
from src.nested_validation import make_nested_splits
from src.oof_identity import fold_assignments
from src.repro import env_snapshot, git_commit, git_worktree_status
from src.train import append_registry, rss_gb
from src.validation import load_splits, validate_splits


BASE_MEMBERS = ("char_tfidf_logreg", "style_only")
EXPECTED_BASELINE = {42: 0.3759706686552324, 13: 0.35819040517027095}


def load_official_train(data_dir: Path) -> pd.DataFrame:
    """Read the labeled competition file without opening the test file."""
    frame = validate_train_frame(pd.read_csv(find_competition_file("train.csv", data_dir)))
    if set(frame["model"]) != set(LABELS):
        raise ValueError("expected support for all 18 official labels")
    return frame


def complete_scores(scores: np.ndarray, n_rows: int) -> np.ndarray:
    """Reject missing classes, unfilled rows and invalid scores before evaluation."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (n_rows, N_CLASSES) or not np.isfinite(scores).all():
        raise ValueError("model did not produce one finite score per official class and row")
    if np.any(scores < SENTINEL):
        raise ValueError("model scores contain absent classes")
    return scores


def verified_control(
    directory: Path, fingerprint: str, assignments: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Load registered controls only if exact row, fold and class identity matches."""
    oof = load_oof(directory, list(BASE_MEMBERS))
    for name, member in oof.items():
        if (member["ordered_dataset_sha256"] != fingerprint
                or not np.array_equal(member["validation_fold_ids"], assignments)
                or not np.array_equal(member["y"], y)):
            raise ValueError(f"{name}: registered control does not match this dataset and split")
    char, style = (
        normalize_scores(complete_scores(oof[name]["scores"], len(y))) for name in BASE_MEMBERS
    )
    return char, 0.5 * char + 0.5 * style


def masked_arms(baseline: np.ndarray, raw_char: np.ndarray, masked: np.ndarray) -> dict[str, np.ndarray]:
    """Pair the fixed masked blend with a reweighted raw-character control."""
    return {
        "masked_character": masked,
        "masked_blend": 0.75 * baseline + 0.25 * normalize_scores(masked),
        "raw_char_reweighted": 0.75 * baseline + 0.25 * raw_char,
    }


def prompt_bootstrap(
    y: np.ndarray, control: np.ndarray, candidate: np.ndarray, groups: np.ndarray,
    *, draws: int = 4000, seed: int = 20261118,
) -> dict:
    """Sample paired whole prompts and recompute pooled balanced-accuracy gains."""
    prompts, inverse = np.unique(groups, return_inverse=True)
    if len(prompts) < 2 or draws < 1 or not (y.shape == control.shape == candidate.shape == groups.shape):
        raise ValueError("prompt bootstrap requires aligned rows, draws and at least two prompts")
    shape = (len(prompts), N_CLASSES)
    support = np.zeros(shape, dtype=np.int64)
    hits = np.zeros(shape, dtype=np.int64)
    np.add.at(support, (inverse, y), 1)
    paired_hits = (candidate == y).astype(np.int64) - (control == y).astype(np.int64)
    np.add.at(hits, (inverse, y), paired_hits)
    rng = np.random.default_rng(seed)
    sample_weights = rng.multinomial(len(prompts), np.full(len(prompts), 1 / len(prompts)), size=draws)
    sampled_support = sample_weights @ support
    sampled_hits = sample_weights @ hits
    present = sampled_support > 0
    recalls = np.divide(sampled_hits, sampled_support, where=present, out=np.zeros_like(sampled_hits, dtype=float))
    deltas = recalls.sum(axis=1) / present.sum(axis=1)
    lo, hi = np.quantile(deltas, (0.025, 0.975))
    return {"low": float(lo), "high": float(hi), "draws": draws, "seed": seed,
            "n_prompts": int(len(prompts)), "method": "paired prompt percentile bootstrap"}


def report_scores(y: np.ndarray, matrices: dict[str, np.ndarray], assignments: np.ndarray,
                  groups: np.ndarray, lengths: np.ndarray) -> dict:
    """Aggregate errors, paired fold gains and prompt uncertainty without passage content."""
    truth = indices_to_labels(y)
    control = matrices["baseline"].argmax(axis=1)
    out = {}
    for name, scores in matrices.items():
        pred_idx = complete_scores(scores, len(y)).argmax(axis=1)
        predicted = indices_to_labels(pred_idx)
        result = evaluate(truth, predicted, model=name)
        result["fold_scores"] = [
            balanced_accuracy(truth[mask], predicted[mask])
            for fold in np.unique(assignments) for mask in (assignments == fold,)
        ]
        result["errors_by_provider"] = within_provider_confusions(truth, predicted)
        result["length_bands"] = length_band_metrics(truth, predicted, lengths)
        if name != "baseline":
            baseline = out["baseline"]
            result["delta"] = result["balanced_accuracy"] - baseline["balanced_accuracy"]
            result["fold_deltas"] = [
                a - b for a, b in zip(result["fold_scores"], baseline["fold_scores"], strict=True)
            ]
            result["fold_deltas"] = [
                0.0 if abs(delta) < 1e-12 else delta for delta in result["fold_deltas"]
            ]
            result["positive_folds"] = sum(delta > 0 for delta in result["fold_deltas"])
            result["paired_prompt_ci_95"] = prompt_bootstrap(y, control, pred_idx, groups)
        out[name] = result
    return out


def nested_outer_folds(df: pd.DataFrame, path: Path, fingerprint: str) -> tuple[list, str]:
    """Verify the frozen inner/outer manifest before a fixed-recipe confirmation."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema_version") != "nested-grouped-splits/v1" or saved.get("ordered_dataset_sha256") != fingerprint:
        raise ValueError("nested split is not bound to this ordered training dataset")
    if (saved.get("outer_seed") != 121 or saved.get("inner_seed") != 212
            or len(saved["folds"]) != 3 or any(len(fold["inner"]) != 3 for fold in saved["folds"])):
        raise ValueError("expected the registered 3x3 nested split with seeds 121 and 212")
    plans = make_nested_splits(
        df, outer_splits=3, inner_splits=3, outer_seed=121, inner_seed=212,
    )
    for new, old in zip(plans, saved["folds"], strict=True):
        if (not np.array_equal(new["train"], old["train"]) or not np.array_equal(new["valid"], old["valid"])
                or any(not np.array_equal(tr, fold["train"]) or not np.array_equal(va, fold["valid"])
                       for (tr, va), fold in zip(new["inner"], old["inner"], strict=True))):
            raise ValueError("nested manifest differs from regenerated grouped folds")
    return [(fold["train"], fold["valid"]) for fold in plans], "nested-grouped-outer3-inner3"


def refit_control(configs: dict, train_texts: list[str], valid_texts: list[str], y: np.ndarray,
                  seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit both fixed sparse controls on the same outer training partition."""
    normalized = []
    for name in BASE_MEMBERS:
        model = estimator_from_config(configs[name], seed=seed)
        model.fit(train_texts, y)
        normalized.append(normalize_scores(complete_scores(model.decision_function(valid_texts), len(valid_texts))))
    return normalized[0], 0.5 * normalized[0] + 0.5 * normalized[1]


def run_experiment(args: argparse.Namespace) -> dict:
    """Run the declared recipe; write OOF only after each row has a valid prediction."""
    if args.nested_plan and args.experiment != "provider":
        raise ValueError("only the predeclared provider comparison is eligible for fixed nested confirmation")
    if args.out.exists():
        raise FileExistsError(f"refusing to replace experiment directory: {args.out}")
    df = load_official_train(args.data_dir)
    texts = df["text"].tolist()
    groups = df["prompt_id"].astype(str).to_numpy()
    lengths = np.asarray([len(t.split()) for t in texts])
    y = labels_to_indices(df["model"])
    fingerprint = ordered_dataset_fingerprint(df)
    if args.nested_plan:
        folds, split_id = nested_outer_folds(df, args.nested_plan, fingerprint)
    else:
        folds, meta = load_splits(args.splits, df)
        if meta["seed"] != args.seed or len(folds) != 5:
            raise ValueError("expected the chosen seed's five registered folds")
        split_id = f"{meta['method']}-s{meta['seed']}-n{len(folds)}"
    validate_splits(df, folds)
    assignments = fold_assignments(len(df), folds)
    saved_control = None
    if not args.nested_plan:
        saved_control = verified_control(args.baseline, fingerprint, assignments, y)
        score = balanced_accuracy(df["model"], indices_to_labels(saved_control[1].argmax(axis=1)))
        if not np.isclose(score, EXPECTED_BASELINE[args.seed], atol=1e-10, rtol=0):
            raise ValueError("baseline control did not reproduce its registered score")
    configs = {member["name"]: member for member in load_config("configs/submission.yaml")["models"]}
    config_path = Path("configs/provider_specialist.yaml" if args.experiment == "provider"
                       else "configs/masked_character.yaml")
    config = load_config(config_path)
    if args.experiment == "provider":
        from src.provider_specialist import ProviderSpecialist
        if config["alpha"] != 0.2 or config["solver_basis"] != "centered_training_row_span":
            raise ValueError("expected the preregistered 0.2 specialist weight")
        names = ("provider_specialist", "flat_diagnostic")
        make_model = lambda: ProviderSpecialist(alpha=config["alpha"], seed=args.seed,
                                                 max_features=config["max_features"])
    else:
        from src.masked_character import MaskedCharacterModel
        if config["weight"] != 0.25 or config["length_bucket_maxima"] != [3, 6]:
            raise ValueError("expected the preregistered mask and blend settings")
        names = ("masked_blend", "masked_character", "raw_char_reweighted")
        make_model = lambda: MaskedCharacterModel(configs[config["character_member"]], seed=args.seed)
    scores = {name: np.full((len(df), N_CLASSES), np.nan) for name in ("baseline", *names)}
    fit_seconds = infer_seconds = 0.0
    peak_rss = rss_gb()
    start = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        for i, (tr, va) in enumerate(folds):
            train_texts, valid_texts = ([texts[j] for j in rows] for rows in (tr, va))
            if saved_control is None:
                char, baseline = refit_control(configs, train_texts, valid_texts, y[tr], args.seed)
            else:
                char, baseline = (matrix[va] for matrix in saved_control)
            t = time.perf_counter()
            model = make_model()
            model.fit(train_texts, y[tr])
            fit_seconds += time.perf_counter() - t
            t = time.perf_counter()
            if args.experiment == "provider":
                arm_scores = dict(zip(names, model.compare(valid_texts, baseline), strict=True))
            else:
                arm_scores = masked_arms(baseline, char, model.decision_function(valid_texts))
            infer_seconds += time.perf_counter() - t
            peak_rss = max(peak_rss, rss_gb())
            scores["baseline"][va] = baseline
            for name, result in arm_scores.items():
                scores[name][va] = complete_scores(result, len(va))
            print(f"[{args.experiment}] fold {i + 1}/{len(folds)} finished", flush=True)
            if time.perf_counter() - start > 7200:
                raise TimeoutError("CPU screen exceeded the two-hour budget")
    results = report_scores(y, scores, assignments, groups, lengths)
    now = datetime.now(timezone.utc)
    git = git_commit()
    payload = {
        "experiment": args.experiment, "dataset": "official", "split_id": split_id, "seed": args.seed,
        "labels": list(LABELS), "n_train_rows": len(df), "ordered_dataset_sha256": fingerprint,
        "recipe": config, "baseline_recipe": configs, "git_commit": git, "worktree": git_worktree_status(),
        "environment": env_snapshot(), "candidate_fit_seconds": fit_seconds,
        "candidate_infer_seconds": infer_seconds, "rss_at_fold_boundaries_gb": peak_rss,
        "total_elapsed_seconds": time.perf_counter() - start,
        "inner_selection_performed": False, "test_rows_read": 0, "transductive": False, "results": results,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    oof_dir = args.out / "oof"
    oof_dir.mkdir()
    for name, matrix in scores.items():
        np.savez_compressed(oof_dir / f"{name}.npz", scores=matrix, y=y, labels=np.asarray(LABELS),
                            ordered_dataset_sha256=fingerprint, validation_fold_ids=assignments)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, result in results.items():
        print(f"{name}: BA {result['balanced_accuracy']:.6f} (delta {result.get('delta', 0):+.6f})")
        if name == "baseline":
            continue
        append_registry(args.registry, {
            "experiment_id": f"{now:%Y%m%dT%H%M%S}-official-{name}-{split_id}",
            "date": now.isoformat(timespec="seconds"), "git_commit": git,
            "method": name, "features": "fixed official sparse blend experiment", "seed": args.seed,
            "split_id": split_id, "fold_scores": " ".join(f"{s:.6f}" for s in result["fold_scores"]),
            "pooled_oof_balanced_accuracy": result["balanced_accuracy"], "macro_f1": result["macro_f1"],
            "external_data_id": "none", "artifact_path": str(args.out / "results.json"),
            "notes": (f"Matched baseline {results['baseline']['balanced_accuracy']:.6f}; "
                      f"paired delta {result['delta']:+.6f}; {result['positive_folds']} positive folds; "
                      f"paired prompt bootstrap {result['paired_prompt_ci_95']['low']:+.6f} to "
                      f"{result['paired_prompt_ci_95']['high']:+.6f}; {payload['worktree']} worktree. "
                      "Previously explored prompts. Candidate timings in results.json. No Kaggle score."),
            "status": "done",
        })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("provider", "masked"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, choices=(42, 13), default=42)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--nested-plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("experiments/registry.csv"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.splits = args.splits or Path(f"artifacts/splits/splits_official_verified_s{args.seed}.json")
    args.baseline = args.baseline or Path(f"artifacts/cv_identity/official{'_s13' if args.seed == 13 else ''}")
    run_experiment(args)


if __name__ == "__main__":
    main()
