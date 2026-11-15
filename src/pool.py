"""External-pool loading, audit and per-fold decontamination.

The Open Pool (``crtproduction/name-that-llm-open-pool``, CC BY-SA 4.0) is public
community data — a candidate to investigate, not automatically trusted. This
module verifies its schema and flags against the documented claims and, for each
validation fold, removes any external passage that duplicates, nearly duplicates
or shares the held-out prompt. The pool's own global decontamination
(``max_cos_train <= 0.3``, train-prompt answers removed, routing-risk rows moved
out) is *verified*, not assumed; per-fold masks are recomputed independently with
a per-fold fit so no held-out information leaks into the external training side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.constants import LABELS

REQUIRED_COLUMNS = ("id", "text", "model", "prompt_canon", "max_cos_train")
FLAG_COLUMNS = (
    "flag_gemini37_routing_risk",
    "flag_near_train_prompt",
    "flag_train_prompt_regen",
    "flag_kb_proxy",
    "flag_kb_proxy_after_ruling",
)


class PoolError(ValueError):
    """Raised when the pool schema or labels fail validation."""


def load_pool(path: str | Path, *, strict: bool = False) -> pd.DataFrame:
    """Load and validate the pool parquet.

    ``strict`` additionally drops ``flag_kb_proxy_after_ruling`` rows (the hosts
    allow training on public datasets made via the proxy, so this is optional).
    """
    pool = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in pool.columns]
    if missing:
        raise PoolError(f"pool is missing required columns: {missing}")
    unknown = set(pool["model"].astype(str)) - set(LABELS)
    if unknown:
        raise PoolError(f"pool has labels outside the official 18: {sorted(unknown)}")
    if pool["text"].isna().any():
        raise PoolError("pool has null text")
    if strict and "flag_kb_proxy_after_ruling" in pool.columns:
        pool = pool[~pool["flag_kb_proxy_after_ruling"].astype(bool)].reset_index(drop=True)
    return pool


def audit_pool(pool: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    """Verify the pool's documented claims against the actual rows."""
    train_texts = set(train_df["text"].astype(str))
    flags = {}
    for col in FLAG_COLUMNS:
        if col in pool.columns:
            flags[col] = {str(k): int(v) for k, v in pool[col].value_counts().items()}
    counts = {label: int((pool["model"] == label).sum()) for label in LABELS}
    return {
        "n_rows": int(len(pool)),
        "n_columns": int(pool.shape[1]),
        "labels_complete": set(pool["model"].astype(str)) == set(LABELS),
        "label_counts": counts,
        "flags": flags,
        "max_cos_train_max": float(pool["max_cos_train"].max()),
        "max_cos_train_mean": float(pool["max_cos_train"].mean()),
        "max_sim_train_prompt_max": (
            float(pool["max_sim_train_prompt"].max())
            if "max_sim_train_prompt" in pool.columns else None
        ),
        "exact_text_duplicates_vs_train": int(pool["text"].astype(str).isin(train_texts).sum()),
        "n_words_min": int(pool["n_words"].min()) if "n_words" in pool.columns else None,
        "n_words_max": int(pool["n_words"].max()) if "n_words" in pool.columns else None,
        "null_text": int(pool["text"].isna().sum()),
    }


def decontaminate_fold(
    pool: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    cos_threshold: float = 0.3,
    max_features: int = 30000,
    chunk_size: int = 20000,
) -> tuple[np.ndarray, dict]:
    """Return a keep-mask excluding pool rows that contaminate this fold.

    A pool row is dropped if it (a) exactly duplicates a held-out passage,
    (b) shares a held-out ``prompt_id`` via ``prompt_canon``, or (c) has word
    TF-IDF cosine > ``cos_threshold`` to any held-out passage. The vectorizer is
    fit on the held-out texts only (per-fold fit); cosine is computed in chunks.
    """
    hold_texts = holdout_df["text"].astype(str).tolist()
    hold_prompts = set(holdout_df["prompt_id"].astype(str))
    pool_texts = pool["text"].astype(str)

    exact = pool_texts.isin(set(hold_texts)).to_numpy()
    shared = pool["prompt_canon"].astype(str).isin(hold_prompts).to_numpy()

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=max_features,
        sublinear_tf=True, norm="l2", dtype=np.float64,
    )
    hold_mat = vectorizer.fit_transform(hold_texts)
    near = np.zeros(len(pool), dtype=bool)
    max_cos = np.zeros(len(pool), dtype=np.float64)
    for start in range(0, len(pool), chunk_size):
        chunk = pool_texts.iloc[start:start + chunk_size].tolist()
        sims = (vectorizer.transform(chunk) @ hold_mat.T).toarray()
        mc = sims.max(axis=1) if sims.shape[1] else np.zeros(len(chunk))
        max_cos[start:start + len(chunk)] = mc
        near[start:start + len(chunk)] = mc > cos_threshold

    keep = ~(exact | shared | near)
    stats = {
        "n_pool": int(len(pool)),
        "n_holdout": int(len(holdout_df)),
        "exact_dup": int(exact.sum()),
        "shared_prompt": int(shared.sum()),
        "near_dup": int(near.sum()),
        "removed": int((~keep).sum()),
        "kept": int(keep.sum()),
        "max_cos_holdout_max": float(max_cos.max()),
        "cos_threshold": float(cos_threshold),
    }
    return keep, stats


def per_fold_decontamination(
    pool: pd.DataFrame,
    train_df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    cos_threshold: float = 0.3,
) -> dict:
    """Decontaminate the pool against every fold's held-out side.

    Returns per-fold stats and the union of removed pool ids (a row removed for
    any fold is excluded from external training to keep the comparison clean).
    """
    removed_ids: set[str] = set()
    per_fold = []
    for i, (_, valid_idx) in enumerate(folds):
        keep, stats = decontaminate_fold(
            pool, train_df.iloc[valid_idx], cos_threshold=cos_threshold
        )
        ids = pool.loc[~keep, "id"].astype(str).tolist()
        removed_ids.update(ids)
        per_fold.append({"fold": i, **stats, "removed_ids_sample": ids[:10]})
    return {
        "cos_threshold": float(cos_threshold),
        "n_folds": len(folds),
        "per_fold": per_fold,
        "union_removed": int(len(removed_ids)),
        "union_removed_ids": sorted(removed_ids),
        "clean_pool_rows": int(len(pool) - len(removed_ids)),
    }