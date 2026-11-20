"""Honest nested gate for the D010/SVD third member (D059 follow-up).

D059 swept (svd, k) and read the best cell off the *reporting* OOF
(svd128, k=8 -> 0.3856, +0.0096 over the adopted 0.375971), which is selection
on validation rows and therefore not adoptable.  This module closes that hole:

* outer folds are the frozen prompt-grouped split files (s42 and s13);
* inside each outer fold the (svd, k) grid is scored by grouped inner CV built
  from **outer-train rows only**, and one cell is selected there;
* the cell is refit on all outer-train rows and the pre-specified equal-thirds
  blend ``(z_char + z_style + z_candidate) / 3`` is scored **once** on the
  untouched outer holdout.

The selection metric is the candidate's own pooled balanced accuracy on the
inner OOF.  It cannot be the blended inner score: the adopted members' saved OOF
for an outer-train row comes from a model that trained on the current outer
holdout, so blending on inner rows would leak the holdout back into the choice.

The grid, the equal thirds recipe and the tie-break (smallest ``svd``, then
smallest ``k``) are fixed before the run; no weight and no cell is fitted to the
outer holdout.  Reported deltas compare the nested blend against the adopted
two-member blend on the same holdout rows, with a paired prompt bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import (  # noqa: E402
    DEFAULT_REGISTRY,
    N_CLASSES,
    RANDOM_STATE,
)
from src.ensemble import load_oof, normalize_scores  # noqa: E402
from src.features_tfidf import TextMatrixBuilder  # noqa: E402
from src.metrics import mean_ci, paired_difference  # noqa: E402
from src.prompt_invariance import NuisanceSuppressor  # noqa: E402
from src.repro import git_commit  # noqa: E402
from src.train import append_registry  # noqa: E402
from src.validation import load_splits  # noqa: E402

from experiments.run_score_improvement import prompt_bootstrap  # noqa: E402
from experiments.sweep_d010 import (  # noqa: E402
    ADOPTED_BA,
    MEMBERS,
    NUISANCE_KS,
    SVD_WIDTHS,
    _expand,
    _fold_scores,
    _pooled,
    blend_with_third,
    load_official_train,
)

SPLIT_FILES = {
    42: "artifacts/splits/splits_official_verified_s42.json",
    13: "artifacts/splits/splits_official_verified_s13.json",
}
OOF_DIRS = {
    42: "artifacts/cv_identity/official",
    13: "artifacts/cv_identity/official_s13",
}
SENTINEL = -1e9
INNER_SPLITS = 3
INNER_SEED = 71


def inner_grouped_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    """Prompt-grouped folds over outer-train rows, with no prompt shared."""
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    placeholder = np.zeros((len(y), 1))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        raw = list(splitter.split(placeholder, y, groups))
    except ValueError:
        raw = list(GroupKFold(n_splits=n_splits).split(placeholder, y, groups))
    folds = [(np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64)) for tr, va in raw]
    for train_idx, valid_idx in folds:
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise ValueError("inner fold shares a prompt between train and valid")
    return folds


def cell_matrices(texts, groups, train_idx, eval_idx, svd: int):
    """Char TF-IDF -> SVD fitted on ``train_idx`` only, applied to both sides."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    eval_idx = np.asarray(eval_idx, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    builder = TextMatrixBuilder(kinds=("char",), svd=int(svd)).fit(
        [texts[i] for i in train_idx], groups=[groups[i] for i in train_idx]
    )
    train_matrix = np.asarray(builder.transform([texts[i] for i in train_idx]), dtype=np.float32)
    eval_matrix = np.asarray(builder.transform([texts[i] for i in eval_idx]), dtype=np.float32)
    return train_matrix, eval_matrix


def score_cell(
    train_matrix,
    y_train: np.ndarray,
    train_groups,
    eval_matrix,
    nuisance_k: int,
    c: float = 1.0,
) -> np.ndarray:
    """D010 projection (fit on train) -> balanced logreg -> 18-column margins."""
    if nuisance_k:
        suppressor = NuisanceSuppressor(
            n_components=int(nuisance_k), min_group_size=4
        ).fit(train_matrix, np.asarray(train_groups).astype(str))
        train_matrix = suppressor.transform(train_matrix)
        eval_matrix = suppressor.transform(eval_matrix)
    classifier = LogisticRegression(
        C=c, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
    ).fit(train_matrix, np.asarray(y_train, dtype=np.int64))
    return _expand(classifier.decision_function(eval_matrix), classifier.classes_)


def select_cell(inner_balanced_accuracy: dict[str, float]) -> str:
    """Best inner pooled BA; ties keep the smallest (svd, k) in grid order."""
    if not inner_balanced_accuracy:
        raise ValueError("no inner cells to select from")

    def order(key: str):
        svd, k = key.split("_k")
        return (int(svd[3:]), int(k))

    best, best_score = None, -np.inf
    for key in sorted(inner_balanced_accuracy, key=order):
        if inner_balanced_accuracy[key] > best_score:
            best, best_score = key, inner_balanced_accuracy[key]
    assert best is not None
    return best


def run_nested(
    texts,
    y,
    groups,
    folds,
    adopted_scores: np.ndarray,
    svd_widths=SVD_WIDTHS,
    nuisance_ks=NUISANCE_KS,
    inner_splits: int = INNER_SPLITS,
    inner_seed: int = INNER_SEED,
    bootstrap_draws: int = 4000,
    c: float = 1.0,
) -> dict:
    """Select (svd, k) inside every outer fold and score the blend on the holdout."""
    texts = list(texts)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    adopted = np.asarray(adopted_scores, dtype=np.float64)
    grid = [(int(s), int(k)) for s in svd_widths for k in nuisance_ks]
    if not grid:
        raise ValueError("the (svd, k) grid must not be empty")

    third = np.full((len(texts), N_CLASSES), SENTINEL, dtype=np.float32)
    reports = []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        started = time.perf_counter()
        train_idx = np.asarray(train_idx, dtype=np.int64)
        valid_idx = np.asarray(valid_idx, dtype=np.int64)
        train_y = y[train_idx]
        train_groups = groups[train_idx]

        inner_folds = inner_grouped_folds(train_y, train_groups, inner_splits, inner_seed)
        inner_scores = {
            f"svd{svd}_k{k}": np.full((len(train_idx), N_CLASSES), SENTINEL, dtype=np.float32)
            for svd, k in grid
        }
        for inner_train, inner_valid in inner_folds:
            for svd in sorted({s for s, _ in grid}):
                train_matrix, eval_matrix = cell_matrices(
                    texts, groups, train_idx[inner_train], train_idx[inner_valid], svd
                )
                for _, k in [g for g in grid if g[0] == svd]:
                    inner_scores[f"svd{svd}_k{k}"][inner_valid] = score_cell(
                        train_matrix,
                        train_y[inner_train],
                        train_groups[inner_train],
                        eval_matrix,
                        k,
                        c,
                    )
        inner_ba = {key: float(_pooled(train_y, scores)) for key, scores in inner_scores.items()}
        chosen = select_cell(inner_ba)
        chosen_svd, chosen_k = (int(chosen[3:].split("_k")[0]), int(chosen.split("_k")[1]))

        train_matrix, eval_matrix = cell_matrices(texts, groups, train_idx, valid_idx, chosen_svd)
        third[valid_idx] = score_cell(
            train_matrix, train_y, train_groups, eval_matrix, chosen_k, c
        )

        candidate_fold_ba = float(_pooled(y[valid_idx], third[valid_idx]))
        reports.append(
            {
                "outer_fold": fold,
                "selected_cell": chosen,
                "inner_balanced_accuracy": round(inner_ba[chosen], 6),
                "inner_cells": {key: round(value, 6) for key, value in inner_ba.items()},
                "candidate_holdout_balanced_accuracy": round(candidate_fold_ba, 6),
                "inner_folds": [
                    {"train_rows": int(len(tr)), "valid_rows": int(len(va))}
                    for tr, va in inner_folds
                ],
                "fit_seconds": round(time.perf_counter() - started, 2),
            }
        )

    if (third == SENTINEL).all(axis=1).any():
        raise RuntimeError("outer predictions do not cover every row")

    blended = blend_with_third(adopted, normalize_scores(third))
    control_pred = adopted.argmax(axis=1)
    blended_pred = np.asarray(blended).argmax(axis=1)

    control_folds = [float(v) for v in _fold_scores(y, adopted, folds)]
    blend_folds = [float(v) for v in _fold_scores(y, blended, folds)]
    for report, control_ba, blend_ba in zip(reports, control_folds, blend_folds):
        report["adopted_holdout_balanced_accuracy"] = round(control_ba, 6)
        report["blend_holdout_balanced_accuracy"] = round(blend_ba, 6)
        report["holdout_delta"] = round(blend_ba - control_ba, 6)

    return {
        "control_pooled": float(_pooled(y, adopted)),
        "blend_pooled": float(_pooled(y, blended)),
        "candidate_pooled": float(_pooled(y, third)),
        "delta": float(_pooled(y, blended) - _pooled(y, adopted)),
        "control_fold_scores": control_folds,
        "blend_fold_scores": blend_folds,
        "paired_fold_delta": mean_ci(paired_difference(control_folds, blend_folds)),
        "bootstrap": prompt_bootstrap(
            np.asarray(y),
            control_pred,
            blended_pred,
            np.asarray(groups).astype(str),
            draws=bootstrap_draws,
        ),
        "selection_metric": "candidate pooled balanced accuracy on inner grouped OOF",
        "recipe": "equal thirds (no weight search)",
        "grid": [f"svd{svd}_k{k}" for svd, k in grid],
        "inner_splits": int(inner_splits),
        "inner_seed": int(inner_seed),
        "folds": reports,
    }


def adopted_matrix(oof_dir: Path, y: np.ndarray, folds) -> np.ndarray:
    """Rebuild the adopted 0.5/0.5 blend and verify it is the fold-aligned OOF."""
    oof = load_oof(oof_dir, list(MEMBERS))
    reference = oof[MEMBERS[0]]
    if not np.array_equal(reference["y"], y):
        raise ValueError("saved OOF labels are not aligned with the training frame")
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    for index, (_, valid_idx) in enumerate(folds):
        fold_ids[np.asarray(valid_idx, dtype=np.int64)] = index
    if not np.array_equal(np.asarray(reference["validation_fold_ids"], dtype=np.int64), fold_ids):
        raise ValueError("saved OOF fold ids do not match the split file")
    normalized = {name: normalize_scores(oof[name]["scores"]) for name in MEMBERS}
    return 0.5 * normalized[MEMBERS[0]] + 0.5 * normalized[MEMBERS[1]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 13])
    parser.add_argument("--out-dir", default="artifacts/residual")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--no-registry", action="store_true")
    parser.add_argument("--svd-widths", nargs="+", type=int, default=list(SVD_WIDTHS))
    parser.add_argument("--ks", nargs="+", type=int, default=list(NUISANCE_KS))
    parser.add_argument("--inner-splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--inner-seed", type=int, default=INNER_SEED)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    args = parser.parse_args(argv)

    frame = load_official_train(args.data_dir)
    texts = frame["text"].tolist()
    groups = frame["prompt_id"].astype(str).to_numpy()
    from src.data import labels_to_indices

    y = labels_to_indices(frame["model"].tolist())
    commit = git_commit(Path(__file__).resolve().parents[1])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for seed in args.seeds:
        if seed not in SPLIT_FILES:
            raise SystemExit(f"no frozen split file registered for seed {seed}")
        folds, payload = load_splits(SPLIT_FILES[seed], frame)
        split_id = f"{payload['method']}-s{payload['seed']}-n{payload['n_splits']}"
        adopted = adopted_matrix(Path(OOF_DIRS[seed]), y, folds)
        control_ba = _pooled(y, adopted)
        if seed == 42 and abs(control_ba - ADOPTED_BA) > 5e-7:
            raise SystemExit(
                f"adopted blend scored {control_ba:.6f} but the registered value is "
                f"{ADOPTED_BA:.6f}; refusing to report deltas against a different control"
            )

        started = time.perf_counter()
        result = run_nested(
            texts,
            y,
            groups,
            folds,
            adopted,
            svd_widths=tuple(args.svd_widths),
            nuisance_ks=tuple(args.ks),
            inner_splits=args.inner_splits,
            inner_seed=args.inner_seed,
            bootstrap_draws=args.bootstrap_draws,
        )
        elapsed = round(time.perf_counter() - started, 2)

        payload_out = {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": "official",
            "seed": seed,
            "split_id": split_id,
            "dataset_fingerprint": payload["dataset_fingerprint"],
            "n_rows": int(len(frame)),
            "n_prompts": int(frame["prompt_id"].nunique()),
            "adopted": {
                "members": list(MEMBERS),
                "weights": [0.5, 0.5],
                "pooled": float(control_ba),
            },
            "status": (
                "nested gate: inner-CV (svd, k) selection per outer fold; the blend is "
                "scored once on outer holdouts"
            ),
            "seconds": elapsed,
            "command": "python -m experiments.nested_d010 " + " ".join(sys.argv[1:]),
            **result,
        }
        out = out_dir / f"nested_d010_s{seed}.json"
        out.write_text(json.dumps(payload_out, indent=1), encoding="utf-8")
        written.append(out)
        ci = result["bootstrap"]

        if not args.no_registry:
            append_registry(
                args.registry,
                {
                    "experiment_id": (
                        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-official-"
                        f"nested-d010-th-rds-s{seed}"
                    ),
                    "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "git_commit": commit,
                    "method": "nested_thirds_char+style+d010",
                    "features": (
                        "0.5 char_tfidf_logreg + 0.5 style_only + candidate "
                        "char|svd(inner-selected)|nuisance(inner-selected)|logreg "
                        "(equal thirds)"
                    ),
                    "seed": seed,
                    "split_id": split_id,
                    "fold_scores": " ".join(f"{v:.4f}" for v in result["blend_fold_scores"]),
                    "pooled_oof_balanced_accuracy": round(result["blend_pooled"], 6),
                    "macro_f1": "",
                    "fit_seconds": elapsed,
                    "infer_seconds": "",
                    "peak_ram_gb": "",
                    "external_data_id": "none",
                    "artifact_path": str(out),
                    "public_lb_score": "",
                    "notes": (
                        f"Nested gate, not adopted: control {result['control_pooled']:.6f}; "
                        f"delta {result['delta']:+.6f}; paired prompt CI "
                        f"[{ci['low']:+.4f}, {ci['high']:+.4f}]; selected cells "
                        + ", ".join(f"f{f['outer_fold']}={f['selected_cell']}" for f in result["folds"])
                        + f"; command: {payload_out['command']}"
                    ),
                    "status": "exploratory",
                },
            )

        print(
            f"[nested] seed {seed} | {split_id} | control {result['control_pooled']:.4f} "
            f"-> blend {result['blend_pooled']:.4f} ({result['delta']:+.4f}) | "
            f"CI [{ci['low']:+.4f}, {ci['high']:+.4f}] | {elapsed}s",
            flush=True,
        )
        for report in result["folds"]:
            print(
                f"  fold {report['outer_fold']}: {report['selected_cell']:12s} "
                f"inner {report['inner_balanced_accuracy']:.4f} | "
                f"holdout single {report['candidate_holdout_balanced_accuracy']:.4f} | "
                f"control {report['adopted_holdout_balanced_accuracy']:.4f} -> "
                f"blend {report['blend_holdout_balanced_accuracy']:.4f} "
                f"({report['holdout_delta']:+.4f})",
                flush=True,
            )

    print(f"[nested] wrote {', '.join(str(p) for p in written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
