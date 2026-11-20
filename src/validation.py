"""Prompt-grouped validation harness.

Test passages come from prompts that never appear in ``train.csv``, so random
row-wise folds would overstate generalisation.  Every fold therefore holds out
whole prompts, and the harness refuses to run if a prompt crosses folds.

Splits are computed once, persisted with the dataset fingerprint, and reused for
every model so comparisons stay like-for-like.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from src.data import dataset_fingerprint, labels_to_indices, ordered_dataset_fingerprint


class LeakageError(RuntimeError):
    """Raised when a split mixing prompt groups (or missing rows) is detected."""


def _n_groups_per_class(df: pd.DataFrame) -> int:
    return int(df.groupby("model", observed=True)["prompt_id"].nunique().min())


def make_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    method: str = "auto",
    shuffle: bool = True,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Build prompt-grouped folds.

    ``method`` is one of ``auto`` (StratifiedGroupKFold, falling back to
    GroupKFold when stratification is impossible), ``stratified_group`` or
    ``group``.  Returns the folds and the method actually used.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    y = labels_to_indices(df["model"].tolist())
    groups = df["prompt_id"].astype(str).to_numpy()
    n_groups = int(pd.unique(groups).size)
    if n_groups < n_splits:
        raise ValueError(
            f"only {n_groups} distinct prompts for {n_splits} folds; reduce n_splits"
        )

    chosen = method
    if method == "auto":
        chosen = "stratified_group" if _n_groups_per_class(df) >= n_splits else "group"

    if chosen == "stratified_group":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None
        )
        try:
            folds = [
                (np.asarray(tr), np.asarray(va))
                for tr, va in splitter.split(np.zeros((len(df), 1)), y, groups)
            ]
        except ValueError:
            chosen = "group"
    if chosen == "group":
        splitter = GroupKFold(n_splits=n_splits)
        folds = [
            (np.asarray(tr), np.asarray(va))
            for tr, va in splitter.split(np.zeros((len(df), 1)), y, groups)
        ]
    elif chosen not in {"stratified_group"}:
        raise ValueError(f"unknown split method: {method}")

    folds = [(tr, va) for tr, va in folds if len(tr) and len(va)]
    validate_splits(df, folds)
    return folds, chosen


def validate_splits(
    df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    """Raise ``LeakageError`` on prompt overlap, empty folds or uncovered rows."""
    if not folds:
        raise LeakageError("no folds")
    prompts = df["prompt_id"].astype(str).to_numpy()
    covered = np.zeros(len(df), dtype=bool)
    for i, (train_idx, valid_idx) in enumerate(folds):
        for side in (train_idx, valid_idx):
            if np.asarray(side).ndim != 1 or not np.issubdtype(np.asarray(side).dtype, np.integer):
                raise LeakageError(f"fold {i} indices must be one-dimensional integers")
            if np.any(side < 0) or np.any(side >= len(df)) or len(np.unique(side)) != len(side):
                raise LeakageError(f"fold {i} has invalid or duplicate indices")
        if len(train_idx) + len(valid_idx) != len(df):
            raise LeakageError(f"fold {i} does not partition all rows")
        if len(train_idx) == 0 or len(valid_idx) == 0:
            raise LeakageError(f"fold {i} has an empty side")
        if len(np.intersect1d(train_idx, valid_idx)):
            raise LeakageError(f"fold {i} shares row indices between train and valid")
        overlap = set(prompts[train_idx]) & set(prompts[valid_idx])
        if overlap:
            raise LeakageError(
                f"fold {i} leaks {len(overlap)} prompt(s), e.g. {sorted(overlap)[:3]}"
            )
        if covered[valid_idx].any():
            raise LeakageError(f"fold {i} reuses validation rows from an earlier fold")
        covered[valid_idx] = True
    if not covered.all():
        raise LeakageError(
            f"{int((~covered).sum())} row(s) never appear in a validation fold"
        )


def fold_class_support(df: pd.DataFrame, folds) -> list[dict[str, int]]:
    """Per-fold validation counts, including classes with zero support."""
    from src.constants import LABELS

    rows = []
    for train_idx, valid_idx in folds:
        counts = df.iloc[valid_idx]["model"].value_counts()
        rows.append({label: int(counts.get(label, 0)) for label in LABELS})
    return rows


def fold_groups(df: pd.DataFrame, folds) -> list[dict]:
    """Describe prompt counts per side, useful in the protocol report."""
    prompts = df["prompt_id"].astype(str).to_numpy()
    out = []
    for i, (train_idx, valid_idx) in enumerate(folds):
        out.append(
            {
                "fold": i,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "train_prompts": int(pd.unique(prompts[train_idx]).size),
                "valid_prompts": int(pd.unique(prompts[valid_idx]).size),
            }
        )
    return out


def make_holdout(
    df: pd.DataFrame, n_groups: int = 25, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Secondary prompt-held-out split for robustness checks, not for selection.

    The default seed-7 holdout has been examined (D015-D017). The seed-2027
    subset has not been scored separately, but its prompts participated in prior
    full-OOF development and are not a fully untouched evaluation (D024).
    """
    prompts = np.asarray(sorted(pd.unique(df["prompt_id"].astype(str))))
    if n_groups < 1:
        raise ValueError("holdout must contain at least one prompt")
    if n_groups >= len(prompts):
        raise ValueError("holdout would consume every prompt")
    rng = np.random.default_rng(seed)
    held = rng.choice(prompts, size=n_groups, replace=False)
    mask = df["prompt_id"].astype(str).isin(set(held.tolist())).to_numpy()
    if not mask.any() or mask.all():
        raise ValueError("holdout selection produced an empty side")
    return np.where(~mask)[0], np.where(mask)[0]


def save_splits(
    path: str | Path,
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    method: str,
    seed: int,
    shuffle: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_fingerprint": dataset_fingerprint(df),
        "ordered_dataset_sha256": ordered_dataset_fingerprint(df),
        "n_rows": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "n_splits": len(folds),
        "method": method,
        "seed": int(seed),
        "shuffle": shuffle,
        "folds": [
            {
                "fold": i,
                "train": [int(v) for v in tr],
                "valid": [int(v) for v in va],
            }
            for i, (tr, va) in enumerate(folds)
        ],
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_splits(
    path: str | Path, df: pd.DataFrame | None = None
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """Load persisted folds only when they carry a verifiable ordered fingerprint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"split file not found: {path}; run `python -m src.cli splits` first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    ordered = payload.get("ordered_dataset_sha256")
    if (not isinstance(ordered, str) or len(ordered) != 64 or
            any(c not in "0123456789abcdef" for c in ordered)):
        raise LeakageError(
            "split ordered dataset fingerprint is missing or invalid; "
            "regenerate splits rather than reusing unverifiable row indices"
        )
    if df is not None:
        if ordered != ordered_dataset_fingerprint(df):
            raise LeakageError("split ordered dataset fingerprint does not match data")
        expected = payload.get("dataset_fingerprint")
        actual = dataset_fingerprint(df)
        if expected != actual:
            raise LeakageError(
                f"split fingerprint {expected} does not match data {actual}; "
                "re-run the split step instead of reusing stale indices"
            )
    folds = [
        (
            np.asarray(f["train"], dtype=np.int64),
            np.asarray(f["valid"], dtype=np.int64),
        )
        for f in payload["folds"]
    ]
    if df is not None:
        # Fingerprinting binds indices to rows but cannot prove that a changed
        # split file keeps prompt groups separated or covers each row once.
        validate_splits(df, folds)
    return folds, payload
