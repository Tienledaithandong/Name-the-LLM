"""External-pool loading, audit and per-fold decontamination.

The Open Pool (``crtproduction/name-that-llm-open-pool``, CC BY-SA 4.0) is public
community data — a candidate to investigate, not automatically trusted. Prompt
identifiers from different datasets must not be assumed to share a namespace.
Per-fold screening requires a verified mapping and a similarity vectorizer fitted
only on fold-training text. Similarity is a heuristic, not proof of no overlap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
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


def load_prompt_mapping(path: str | Path) -> tuple[dict[str, list[str]], str]:
    """Read an evidence-backed official-ID to pool-canonical-ID crosswalk.

    JSON shape: ``{"provenance": "...", "mapping": {"official-id":
    ["pool-canonical-id"]}}``. An empty list declares a verified absence of
    pool matches. The file's digest is included in the returned provenance so
    experiment reports identify the exact mapping used. This loader validates
    structure, not the truth or completeness of the source evidence.
    """
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise PoolError(f"cannot read prompt mapping: {exc}") from exc
    if not isinstance(value, dict):
        raise PoolError("prompt mapping document must be an object")
    mapping = value.get("mapping")
    provenance = value.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise PoolError("prompt mapping must record nonempty evidence provenance")
    if not isinstance(mapping, dict) or not mapping:
        raise PoolError("prompt mapping must contain a nonempty mapping object")
    for key, ids in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise PoolError("official prompt IDs must be nonempty strings")
        if not isinstance(ids, list) or any(not isinstance(x, str) or not x.strip() for x in ids):
            raise PoolError("each prompt mapping value must be a list of nonempty canonical IDs")
    digest = hashlib.sha256(raw).hexdigest()
    return mapping, f"{provenance.strip()} (sha256:{digest})"


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


def cap_pool_stratified(pool: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Subsample the pool to <=cap rows preserving class proportions.

    A validation-independent compute budget (random by class share), so the same
    clean set is capped identically regardless of any held-out text or label.
    """
    if cap <= 0 or len(pool) <= cap:
        return pool
    rng = np.random.default_rng(seed)
    counts = pool["model"].value_counts()
    parts = []
    for label, n in counts.items():
        take = max(1, int(round(cap * n / len(pool))))
        idx = pool.index[pool["model"] == label].to_numpy()
        parts.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    return pool.loc[np.sort(np.concatenate(parts))].reset_index(drop=True)


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
    fit_texts: Sequence[str] | None = None,
    prompt_mapping: Mapping[str, Sequence[str]] | None = None,
    mapping_provenance: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Screen exact, mapped-prompt and near-duplicate external passages.

    ``fit_texts`` must contain fold-training text only. ``prompt_mapping`` maps
    every held-out official prompt ID to verified pool canonical IDs for that
    prompt; an empty sequence asserts the prompt was checked and has no match.
    ``mapping_provenance`` identifies the evidence supporting the complete
    mapping. The caller must validate that evidence. Similarity is a heuristic,
    not proof that all overlap was found. Pool rows with missing or empty
    canonical prompt IDs are excluded because they cannot be crosswalk-checked.
    """
    if fit_texts is None or len(fit_texts) == 0:
        raise PoolError("fold-training fit_texts are required for decontamination")
    if prompt_mapping is None or not mapping_provenance or not mapping_provenance.strip():
        raise PoolError("a verified prompt_mapping and mapping_provenance are required; "
                        "official prompt IDs and pool canonical IDs may differ")
    if not 0 <= cos_threshold <= 1:
        raise PoolError("cos_threshold must be between 0 and 1")
    if chunk_size <= 0:
        raise PoolError("chunk_size must be positive")
    if holdout_df.empty:
        raise PoolError("held-out data must not be empty")
    if holdout_df["prompt_id"].isna().any() or holdout_df["prompt_id"].astype(str).str.strip().eq("").any():
        raise PoolError("held-out prompt IDs must not be missing or empty")
    if holdout_df["text"].isna().any():
        raise PoolError("held-out text must not be missing")
    hold_texts = holdout_df["text"].astype(str).tolist()
    hold_prompts = set(holdout_df["prompt_id"].astype(str))
    if set(hold_texts).intersection(map(str, fit_texts)):
        raise PoolError("fold-training fit_texts overlap held-out text")
    missing = hold_prompts - set(prompt_mapping)
    if missing:
        raise PoolError(f"verified prompt_mapping lacks {len(missing)} held-out prompt(s)")
    canonical_ids: set[str] = set()
    for prompt_id in hold_prompts:
        values = prompt_mapping[prompt_id]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise PoolError("each prompt_mapping value must be a sequence of canonical IDs")
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise PoolError("mapped canonical IDs must be nonempty strings")
        canonical_ids.update(v.strip() for v in values)
    pool_texts = pool["text"].astype(str)

    exact = pool_texts.isin(set(hold_texts)).to_numpy()
    pool_prompts = pool["prompt_canon"].astype("string").str.strip()
    unmapped = (pool_prompts.isna() | pool_prompts.eq("").fillna(True)).to_numpy()
    shared = pool_prompts.isin(canonical_ids).to_numpy()

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=max_features,
        sublinear_tf=True, norm="l2", dtype=np.float64,
    )
    try:
        vectorizer.fit(list(map(str, fit_texts)))
    except ValueError as exc:
        raise PoolError(f"cannot fit similarity vectorizer on fold-training text: {exc}") from exc
    hold_mat = vectorizer.transform(hold_texts)
    near = np.zeros(len(pool), dtype=bool)
    max_cos = np.zeros(len(pool), dtype=np.float64)
    for start in range(0, len(pool), chunk_size):
        chunk = pool_texts.iloc[start:start + chunk_size].tolist()
        sims = (vectorizer.transform(chunk) @ hold_mat.T).toarray()
        mc = sims.max(axis=1) if sims.shape[1] else np.zeros(len(chunk))
        max_cos[start:start + len(chunk)] = mc
        near[start:start + len(chunk)] = mc > cos_threshold

    keep = ~(exact | shared | near | unmapped)
    stats = {
        "n_pool": int(len(pool)),
        "n_holdout": int(len(holdout_df)),
        "exact_dup": int(exact.sum()),
        "shared_prompt": int(shared.sum()),
        "unmapped_pool_prompt": int(unmapped.sum()),
        "near_dup": int(near.sum()),
        "removed": int((~keep).sum()),
        "kept": int(keep.sum()),
        "max_cos_holdout_max": float(max_cos.max()) if len(max_cos) else 0.0,
        "cos_threshold": float(cos_threshold),
        "prompt_mapping_provenance": mapping_provenance,
        "similarity_fit_scope": "fold_training_only",
    }
    return keep, stats


def decontaminate_global(
    pool: pd.DataFrame,
    train_df: pd.DataFrame,
    *,
    cos_threshold: float = 0.3,
    max_features: int = 30000,
    chunk_size: int = 20000,
) -> tuple[np.ndarray, dict]:
    """Screen the pool ONCE against the full official training set (D028).

    Non-transductive by construction: the reference corpus is the complete set of
    known official training passages, so the returned keep-mask is identical for
    every CV fold and never depends on which rows a fold holds out. No fold's
    validation text influences which pool rows enter training, which is the exact
    leak D026 disabled. Only text-level exact and near-duplicate checks are used
    because the pool's ``prompt_canon`` namespace is not proven to match official
    ``prompt_id`` (D019/D024); prompt crosswalks are left for a separately verified
    protocol. Similarity is a heuristic, not proof that all overlap was found.
    """
    if cos_threshold < 0 or cos_threshold > 1:
        raise PoolError("cos_threshold must be between 0 and 1")
    if chunk_size <= 0:
        raise PoolError("chunk_size must be positive")
    if train_df.empty:
        raise PoolError("official training set must not be empty")
    if train_df["text"].isna().any():
        raise PoolError("official training text must not be missing")

    reference_texts = train_df["text"].astype(str).tolist()
    reference_set = set(reference_texts)
    pool_texts = pool["text"].astype(str)

    exact = pool_texts.isin(reference_set).to_numpy()

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=max_features,
        sublinear_tf=True, norm="l2", dtype=np.float64,
    )
    try:
        vectorizer.fit(reference_texts)
    except ValueError as exc:
        raise PoolError(f"cannot fit similarity vectorizer on official training text: {exc}") from exc
    reference_mat = vectorizer.transform(reference_texts)
    near = np.zeros(len(pool), dtype=bool)
    max_cos = np.zeros(len(pool), dtype=np.float64)
    for start in range(0, len(pool), chunk_size):
        chunk = pool_texts.iloc[start:start + chunk_size].tolist()
        sims = (vectorizer.transform(chunk) @ reference_mat.T).toarray()
        mc = sims.max(axis=1) if sims.shape[1] else np.zeros(len(chunk))
        max_cos[start:start + len(chunk)] = mc
        near[start:start + len(chunk)] = mc > cos_threshold

    keep = ~(exact | near)
    stats = {
        "n_pool": int(len(pool)),
        "n_reference": int(len(train_df)),
        "exact_dup": int(exact.sum()),
        "near_dup": int(near.sum()),
        "removed": int((~keep).sum()),
        "kept": int(keep.sum()),
        "max_cos_reference_max": float(max_cos.max()) if len(max_cos) else 0.0,
        "cos_threshold": float(cos_threshold),
        "decontamination_scope": "global_full_official_train",
        "validation_independent": True,
    }
    return keep, stats


def per_fold_decontamination(
    pool: pd.DataFrame,
    train_df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    cos_threshold: float = 0.3,
    prompt_mapping: Mapping[str, Sequence[str]] | None = None,
    mapping_provenance: str | None = None,
) -> dict:
    """DEPRECATED diagnostic: decontaminate the pool against every fold's held-out side.

    Transductive and disabled for reported CV by D026 — retained only to reproduce
    the historical union-removed diagnostic. Use :func:`decontaminate_global` (D028)
    for any score that will be reported or submitted.
    """
    removed_ids: set[str] = set()
    per_fold = []
    for i, (train_idx, valid_idx) in enumerate(folds):
        train_groups = set(train_df.iloc[train_idx]["prompt_id"].astype(str))
        valid_groups = set(train_df.iloc[valid_idx]["prompt_id"].astype(str))
        if not train_groups.isdisjoint(valid_groups):
            raise PoolError(f"fold {i} contains a prompt on both sides")
        keep, stats = decontaminate_fold(
            pool, train_df.iloc[valid_idx], cos_threshold=cos_threshold,
            fit_texts=train_df.iloc[train_idx]["text"].astype(str).tolist(),
            prompt_mapping=prompt_mapping, mapping_provenance=mapping_provenance,
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
