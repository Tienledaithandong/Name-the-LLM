"""DeBERTa-v3 attribution: official-only vs official+pool grouped CV (Kaggle GPU).

Research TRAINING kernel — NOT the submission. Runs on Kaggle GPU with internet ON
to fetch the public DeBERTa checkpoint; saves OOF + metrics + final weights for a
separate OFFLINE inference notebook (internet OFF, weights attached).

Protocol: D018 (full prompt-grouped out-of-fold balanced accuracy is the PRIMARY
metric for this independent-passage model) + D019 (audited Open Pool v15; per-fold
decontamination against the held-out prompts; the upstream match_weight/style_score
columns are NEVER used). Every learned transform is fit on the training side only;
pool rows never enter validation; no transductive step.

Budget: <=9h. A benchmark cell measures throughput/peak memory and extrapolates the
full runtime BEFORE the long runs, and caps the pool per fold to stay in budget.

Push (owner account, GPU quota):
    kaggle kernels push -p kaggle/deberta
    kaggle kernels status tiennamlee/name-that-llm-deberta-train
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import time
from pathlib import Path

# Reduce CUDA fragmentation so the long pool run does not OOM after the official
# run (T4 14.5 GB; DeBERTa-v3-large disentangled attention is memory-heavy).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

LABELS = (
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
)
PROVIDER_OF = {
    "gpt-5.5": "OpenAI", "gpt-5.6-luna": "OpenAI", "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI", "gpt-6-astra": "OpenAI",
    "gemini-3.1-pro-preview": "Google", "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google", "gemini-3.8-flash": "Google",
    "claude-fable-5": "Anthropic", "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic", "claude-opus-5": "Anthropic",
    "claude-sonnet-5": "Anthropic", "qwen3.8-max": "Qwen",
    "glm-5.3": "ZAI", "glm-5.3-flash": "ZAI", "kimi-k3": "Moonshot AI",
}
N_CLASSES = len(LABELS)
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


# -- data ------------------------------------------------------------------
def find_data_dir() -> Path:
    for pat in ("/kaggle/input/**/train.csv", "data/train.csv"):
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return Path(hits[0]).parent
    raise FileNotFoundError("train.csv not found under /kaggle/input or data/")


def find_pool() -> Path | None:
    hits = sorted(glob.glob("/kaggle/input/**/ntl_open_pool.parquet", recursive=True))
    return Path(hits[0]) if hits else None


def load_official(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "train.csv")
    need = {"id", "prompt_id", "text", "model"}
    if not need.issubset(df.columns):
        raise ValueError(f"train.csv missing columns: {need - set(df.columns)}")
    unknown = set(df["model"].astype(str)) - set(LABELS)
    if unknown:
        raise ValueError(f"train has labels outside the official 18: {sorted(unknown)}")
    if df["text"].isna().any():
        raise ValueError("train has null text")
    return df.reset_index(drop=True)


def load_pool(path: Path, *, strict: bool = False) -> pd.DataFrame:
    pool = pd.read_parquet(path)
    unknown = set(pool["model"].astype(str)) - set(LABELS)
    if unknown:
        raise ValueError(f"pool has labels outside the official 18: {sorted(unknown)}")
    if strict and "flag_kb_proxy_after_ruling" in pool.columns:
        pool = pool[~pool["flag_kb_proxy_after_ruling"].astype(bool)]
    # Only text/model/prompt_canon are used downstream (D019 leakage guard).
    return pool.reset_index(drop=True)


def make_grouped_folds(df: pd.DataFrame, n_splits: int, seed: int):
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    y = df["model"].map(LABEL_TO_INDEX).to_numpy()
    groups = df["prompt_id"].astype(str).to_numpy()
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.zeros(len(df)), y, groups))
        method = "stratified_group"
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(np.zeros(len(df)), y, groups))
        method = "group"
    return [(np.asarray(tr), np.asarray(va)) for tr, va in folds], method


def load_shipped_folds(path, df):
    """Load the exact local cv_s42 folds shipped with the kernel.

    Avoids the sklearn train/Kaggle version mismatch shifting fold assignments;
    guarantees the OOF is fold-aligned with the local sparse OOF for blending.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload["n_rows"]) != len(df):
        raise ValueError(f"shipped folds n_rows {payload['n_rows']} != data {len(df)}")
    folds = [(np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int))
             for f in payload["folds"]]
    for tr, va in folds:
        if not set(df.iloc[tr]["prompt_id"]).isdisjoint(set(df.iloc[va]["prompt_id"])):
            raise ValueError("shipped folds leak a prompt across train/valid")
    return folds, str(payload["method"]) + "-shipped"


# Exact local cv_s42 fold assignment, embedded because Kaggle script kernels do
# NOT upload companion files (folds.json was dropped on v3). One fold id per
# official train row (0..4); validated against the local splits before shipping.
EMBEDDED_FOLD_ID_STR = "1,1,2,1,0,3,3,3,4,0,1,1,0,0,1,2,4,1,4,4,2,3,4,2,0,3,2,4,4,4,3,0,2,3,4,2,1,0,3,2,2,3,2,3,2,4,1,1,0,2,4,0,4,2,3,3,3,0,0,1,3,1,3,2,0,3,1,3,4,3,1,0,1,0,3,1,0,1,3,3,4,4,0,2,2,0,1,4,2,0,1,3,1,0,3,0,4,3,2,4,0,3,4,3,1,4,2,2,2,4,1,1,3,1,1,0,0,3,0,0,1,0,0,2,4,2,3,3,4,1,3,0,1,2,1,3,3,3,0,4,2,3,1,2,2,3,3,4,2,1,2,0,3,0,2,1,3,2,1,4,4,3,1,2,1,3,1,4,2,0,3,2,2,0,0,0,2,3,3,2,4,1,0,1,0,0,3,0,4,2,2,1,0,2,2,1,4,2,0,1,3,4,2,4,4,4,4,3,1,0,4,4,4,4,4,1,3,2,2,3,2,0,2,4,3,4,1,3,1,2,4,4,0,3,4,0,2,4,3,3,1,4,4,1,1,2,0,3,4,0,4,2,4,4,3,3,1,1,4,4,2,3,3,4,4,4,2,3,0,2,3,2,4,4,1,4,3,3,3,4,2,4,0,2,4,3,4,3,0,0,2,4,0,3,1,1,3,0,1,2,1,3,3,3,3,2,2,1,3,1,2,4,4,1,3,2,4,4,4,1,3,2,1,4,1,1,4,2,4,2,4,4,2,2,0,1,3,0,3,2,0,4,1,3,3,3,2,1,4,4,1,1,0,3,4,3,1,2,4,1,3,1,1,0,2,0,3,2,4,3,4,4,2,3,3,0,4,3,4,3,3,4,1,4,4,3,0,4,1,1,4,4,1,1,3,4,3,2,0,1,0,4,3,2,2,1,0,2,3,1,1,2,2,1,3,2,2,1,2,0,2,2,4,4,2,0,4,1,2,2,2,4,0,3,0,1,2,0,1,0,3,2,3,3,2,1,1,0,4,0,4,1,3,2,3,3,3,4,3,1,2,4,3,3,3,1,4,3,3,1,3,4,0,0,3,4,4,2,0,0,0,2,3,0,2,4,3,1,1,0,0,0,4,2,2,4,0,4,2,3,1,0,3,3,3,2,4,4,1,0,0,3,3,4,2,3,3,3,3,4,0,4,3,1,1,4,4,4,1,1,4,0,0,2,2,1,3,1,1,3,0,2,3,1,1,3,2,4,0,3,3,0,4,0,1,2,0,4,1,4,0,1,3,3,2,0,3,2,3,1,2,1,0,0,3,0,4,3,1,0,1,4,1,0,3,3,3,1,2,3,1,0,3,3,1,4,0,3,1,0,0,2,1,1,4,1,3,1,4,2,3,0,1,2,1,3,1,2,4,3,3,2,4,4,3,3,4,0,2,4,0,0,0,0,4,1,4,2,0,4,2,1,2,0,3,1,2,2,4,3,3,3,0,2,3,0,2,0,4,1,4,0,2,4,1,1,1,3,1,1,0,0,3,0,0,0,2,1,3,1,4,4,0,4,0,3,3,1,4,2,1,4,4,3,3,4,0,3,2,4,0,4,0,0,0,4,2,2,0,3,1,0,1,2,3,3,4,0,0,2,4,3,0,1,3,1,4,2,2,3,2,2,2,0,0,0,0,0,0,1,4,1,0,1,4,2,3,0,0,2,2,4,3,4,3,2,2,3,1,2,1,4,3,1,4,4,3,2,1,0,2,1,4,3,1,0,2,2,1,1,2,4,4,2,1,3,4,2,4,4,4,4,4,0,2,1,1,3,1,4,3,4,4,3,0,0,3,4,2,4,3,1,1,1,0,0,1,1,1,2,4,2,4,2,3,4,2,3,4,4,1,4,4,0,1,3,0,0,1,3,2,2,4,3,2,3,1,4,0,1,0,3,0,1,3,3,1,1,4,0,0,1,4,4,4,0,1,2,1,4,2,1,0,4,0,0,1,4,0,1,2,4,2,2,3,0,2,0,0,1,2,3,4,1,0,0,0,0,1,0,0,0,0,4,0,1,0,0,3,3,4,0,2,2,1,4,0,4,0,2,2,1,2,4,4,2,2,0,2,4,3,0,2,0,3,2,1,2,4,2,3,1,0,3,1,3,3,4,1,3,2,2,4,2,3,4,0,3,1,0,2,4,0,4,4,1,1,0,4,2,0,1,2,4,2,1,0,3,4,1,3,2,2,4,3,0,3,4,3,0,2,0,1,0,1,1,2,4,1,0,0,1,4,1,2,4,2,1,3,1,3,4,1,3,0,4,4,2,4,2,4,4,2,4,3,2,2,2,3,1,3,4,2,2,1,0,2,2,1,4,4,1,4,3,4,2,2,4,2,2,1,2,4,4,1,1,1,0,1,1,3,3,2,3,2,4,0,4,2,1,0,4,4,3,4,2,4,4,1,1,2,2,2,2,4,0,0,0,2,0,4,3,0,3,0,4,4,0,4,1,2,1,4,4,4,4,0,0,1,2,4,0,2,1,4,1,4,1,3,3,0,1,2,0,3,2,3,1,2,0,3,3,4,4,1,1,0,0,4,3,0,2,0,1,1,0,4,0,1,2,0,3,1,4,1,2,1,0,2,3,0,1,1,3,1,0,3,2,2,3,0,0,3,4,1,4,4,0,2,3,3,0,1,0,4,3,4,0,3,4,2,2,3,0,4,1,0,3,0,2,4,4,0,4,1,2,1,3,4,3,1,1,2,2,0,3,0,2,2,3,0,3,4,4,3,2,2,2,3,2,4,3,2,3,2,3,0,1,1,2,2,0,3,4,3,3,4,3,2,3,4,2,4,4,0,0,0,1,4,3,3,2,0,3,2,3,2,3,2,3,4,4,4,4,3,0,3,1,2,2,0,2,1,4,0,0,3,0,4,0,4,0,4,3,0,3,1,4,1,4,2,4,4,1,3,3,0,0,3,3,3,0,2,1,3,1,3,0,3,0,0,4,2,4,0,4,3,1,3,0,1,4,2,1,4,1,3,4,0,3,1,2,3,0,2,0,3,2,0,2,4,0,1,1,1,0,1,4,3,2,3,2,4,4,0,0,1,4,2,4,1,2,0,0,1,2,0,0,3,1,0,3,0,0,0,4,3,3,1,4,0,3,1,4,0,4,4,4,4,3,4,1,1,4,1,2,1,4,1,3,1,2,4,2,3,0,1,4,1,2,0,1,3,1,2,1,3,3,4,1,0,0,2,4,3,2,1,2,1,4,3,2,3,2,0,4,1,1,4,0,4,1,1,2,1,4,3,4,3,4,0,2,1,2,1,2,3,0,3,2,1,2,2,0,1,3,4,3,2,0,2,2,3,4,1,2,0,4,2,2,2,2,2,3,4,2,3,4,1,2,3,0,2,0,4,2,0,1,1,2,4,4,2,3,1,3,2,1,2,3,2,1,3,0,3,2,0,2,2,3,1,0,2,2,3,1,0,1,0,0,1,1,4,3,0,3,0,0,3,4,4,2,4,3,3,3,2,3,1,4,0,3,2,1,2,1,0,0,3,4,4,4,0,0,4,3,2,2,4,3,3,2,4,4,0,2,4,2,4,0,2,1,3,2,2,1,2,1,0,0,4,0,4,1,0,1,1,0,4,4,0,1,2,0,3,1,3,3,3,0,3,2,1,3,3,1,1,4,4,2,1,1,4,0,2,3,3,1,3,3,4,1,3,3,4,0,3,2,3,1,1,3,0,4,1,3,2,2,3,3,2,3,3,3,2,1,4,4,0,4,4,0,4,1,0,2,4,4,2,0,2,0,3,0,1,3,1,4,3,1,4,0,2,1,3,3,2,1,2,3,2,2,0,2,1,3,0,0,2,1,1,2,3,4,3,4,3,4,3,0,4,0,0,3,2,1,1,1,4,3,3,3,2,4,3,1,4,0,0,0,0,3,1,0,3,4,2,4,0,1,2,0,0,4,4,0,2,0,0,1,2,0,1,2,2,0,0,0,1,3,3,2,3,1,1,4,1,1,4,0,4,3,1,1,3,1,0,1,0,2,2,0,0,4,3,1,4,2,1,1,2,2,0,2,2,1,4,0,2,3,1,3,2,1,1,3,1,3,2,4,0,0,2,4,0,0,4,3,3,0,3,3,2,1,1,0,1,3,0,1,0,1,0,2,1,0,3,0,4,0,4,2,2,1,4,0,3,3,0,0,1,1,4,1,1,0,3,1,2,1,1,1,4,2,3,4,3,4,1,0,2,0,2,2,1,0,2,1,2,3,4,2,1,3,2,1,4,0,0,3,3,2,3,0,0,2,0,1,2,1,2,2,1,1,1,0,1,0,3,4,2,3,0,1,1,1,3,2,4,3,4,1,2,3,2,1,1,1,0,2,2,0,1,3,1,2,4,0,3,3,2,4,2,3,1,4,0,0,0,4,3,0,0,2,3,2,0,1,0,1,4,1,3,2,3,0,0,0,1,4,0,2,0,2,1,0,2,2,1,3,4,1,0,0,4,2,2,4,1,0,2,1,0,1,1,2,0,4,4,0,1,1,4,4,4,1,0,0,3,0,1,3,3,4,2,4,1,0,3,3,0,2,2,1,3,2,2,4,2,2,1,2,1,2,4,4,4,4,3,4,3,1,2,3,2,2,0,0,1,2,0,2,0,2,2,0,4,2,1,2,4,1,3,4,0,4,4,0,0,0,3,0,4,1,2,1,3,0,2,2,2,1,2,0,4,0,0,4,2,0,4,4,2,3,1,2,0,2,2,4,2,4,0,2,0,2,4,0,3,1,4,2,0,0,0,2,1,1,1,4,2,2,1,2,3,4,0,4,2,1,4,1,0,2,2,3,2,1,0,4,3,4,4,3,4,0,2,3,1,0,4,0,0,4,3,2,3,3,4,0,3,2,3,0,1,3,1,3,0,2,3,0,4,4,0,1,3,0,4,0,1,3,4,0,2,4,1,1,0,3,2,0,0,0,1,3,1,4,1,4,1,2,1,4,0,1,0,3,2,2,1,0,3,4,0,4,3,2,0,2,1,3,2,2,4,4,0,0,1,0,3,0,0,1,4,2,4,2,1,3,2,4,3,4,4,3,3,1,3,1,4,3,0,1,4,3,3,3,0,3,0,4,1,0,1,3,4,1,2,0,2,1,1,3,4,4,2,3,0,0,0,3,4,2,3,1,2,2,2,0,3,1,4,2,3,0,1,0,2,4,3,1,2,2,2,1,0,1,1,3,4,4,1,2,1,4,0,2,4,3,2,0,0,1,2,3,1,2,1,2,0,1,4,4,0,4,1,0,4,3,0,3,4,4,3,0,4,2,0,4,1,2,3,2,0,3,0,2,1,4,1,2,3,3,4,1,0,4,3,2,1,2,1,1,1,1,3,0,0,4,2,3,0,3,0,4,4,0,0,3,0,3,1,2,2,0,1,1,3,2,4,0,1,0,0,2,4,0,3,0,3,3,3,0,4,3,1,3,2,3,3,2,2,3,1,0,0,4,2,2,0,4,1,3,4,0,4,4,0,3,1,2,2,3,4,2,3,0,3,2,1,1,0,0,2,2,0,1,0,4,2,1,3,3,3,2,4,0,2,2,0,0,3,2,2,3,2,2,2,0,1,0,1,4,1,0,3,4,2,3,4,0,0,3,2,0,2,0,4,0,2,2,1,4,0,1,0,2,3,3,4,0,4,3,3,3,4,2,2,1,3,1,3,4,0,1,1,2,2,4,1,4,1,1,3,4,2,0,3,4,4,3,1,0,3,3,0,1,2,0,3,3,2,0,4,4,4,1,3,3,4,1,3,2,1,2,4,4,2,4,4,1,0,1,1,1,2,1,0,0,1,2,0,4,1,4,0,4,4,3,2,4,4,4,2,3,3,1,3,4,0,2,4,1,2,1,0,3,1,1,2,1,4,0,1,1,0,3,4,4,0,1,2,4,2,1,2,2,2,1,3,3,0,3,1,4,2,1,1,0,0,3,1,1,4,3,3,2,1,3,3,2,4,2,0,3,4,3,3,3,4,2,4,1,4,3,3,4,1,2,1,1,3,1,4,1,2,1,2,1,1,1,2,4,4,3,1,1,4"


def embedded_folds(df):
    """Rebuild the local cv_s42 folds from the embedded per-row fold ids."""
    ids = np.fromstring(EMBEDDED_FOLD_ID_STR, dtype=int, sep=",")
    if ids.size != len(df):
        raise ValueError(f"embedded folds {ids.size} != data {len(df)}")
    folds = []
    for f in range(int(ids.max()) + 1):
        va = np.where(ids == f)[0]
        tr = np.where(ids != f)[0]
        folds.append((tr, va))
    for tr, va in folds:
        if not set(df.iloc[tr]["prompt_id"]).isdisjoint(set(df.iloc[va]["prompt_id"])):
            raise ValueError("embedded folds leak a prompt across train/valid")
    return folds, "stratified_group-embedded"


def decontaminate_fold(pool: pd.DataFrame, holdout: pd.DataFrame, *, cos_threshold=0.3,
                       max_features=30000, chunk_size=20000):
    """Mirror of src.pool.decontaminate_fold: per-fold fit on the holdout only."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    hold_texts = holdout["text"].astype(str).tolist()
    hold_prompts = set(holdout["prompt_id"].astype(str))
    pool_texts = pool["text"].astype(str)
    exact = pool_texts.isin(set(hold_texts)).to_numpy()
    shared = pool["prompt_canon"].astype(str).isin(hold_prompts).to_numpy()
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=max_features,
                          sublinear_tf=True, norm="l2", dtype=np.float64)
    hold_mat = vec.fit_transform(hold_texts)
    near = np.zeros(len(pool), dtype=bool)
    for start in range(0, len(pool), chunk_size):
        chunk = pool_texts.iloc[start:start + chunk_size].tolist()
        sims = (vec.transform(chunk) @ hold_mat.T).toarray()
        mc = sims.max(axis=1) if sims.shape[1] else np.zeros(len(chunk))
        near[start:start + len(chunk)] = mc > cos_threshold
    keep = ~(exact | shared | near)
    stats = {"n_pool": int(len(pool)), "n_holdout": int(len(holdout)),
             "exact_dup": int(exact.sum()), "shared_prompt": int(shared.sum()),
             "near_dup": int(near.sum()), "removed": int((~keep).sum()), "kept": int(keep.sum())}
    return keep, stats


def cap_pool_stratified(pool: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Subsample the pool to <=cap rows preserving class proportions (budget guard)."""
    if cap <= 0 or len(pool) <= cap:
        return pool
    rng = np.random.default_rng(seed)
    parts = []
    counts = pool["model"].value_counts()
    for label, n in counts.items():
        take = max(1, int(round(cap * n / len(pool))))
        idx = pool.index[pool["model"] == label].to_numpy()
        parts.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    keep = np.concatenate(parts)
    return pool.loc[np.sort(keep)].reset_index(drop=True)


# -- metrics ---------------------------------------------------------------
def per_class_recall(y_true, y_pred):
    yt = np.asarray(y_true, dtype=str)
    yp = np.asarray(y_pred, dtype=str)
    out = {}
    for label in LABELS:
        mask = yt == label
        s = int(mask.sum())
        out[label] = float((yp[mask] == label).sum() / s) if s else float("nan")
    return out


def balanced_accuracy(y_true, y_pred):
    recalls = [v for v in per_class_recall(y_true, y_pred).values() if not np.isnan(v)]
    return float(np.mean(recalls)) if recalls else float("nan")


def within_provider(y_true, y_pred):
    yt = np.asarray(y_true, dtype=str)
    yp = np.asarray(y_pred, dtype=str)
    wrong = yt != yp
    same = np.array([PROVIDER_OF.get(t) == PROVIDER_OF.get(p) for t, p in zip(yt, yp)])
    return {"errors": int(wrong.sum()), "within_provider_errors": int((wrong & same).sum()),
            "cross_provider_errors": int((wrong & ~same).sum())}


def mean_ci(values, confidence=0.95):
    from scipy.stats import t
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean())
    if n < 2:
        return {"n": n, "mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean, "ci_half": 0.0}
    std = float(arr.std(ddof=1))
    half = float(t.ppf(0.5 + confidence / 2.0, n - 1) * std / np.sqrt(n))
    return {"n": n, "mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half, "ci_half": half}


# -- model -----------------------------------------------------------------
def ensure_deps():
    import importlib
    for mod, pip in (("sentencepiece", "sentencepiece"), ("transformers", "transformers"),
                     ("torch", "torch"), ("scipy", "scipy")):
        try:
            importlib.import_module(mod)
        except ImportError:
            import subprocess
            subprocess.check_call([os.sys.executable, "-m", "pip", "install", "-q", pip])


class TextDataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len,
                             padding=False, return_tensors=None)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def collate(batch, tokenizer):
    import torch
    from transformers import DataCollatorWithPadding
    coll = DataCollatorWithPadding(tokenizer, padding="longest")
    return coll(batch)


def build_model(model_name, device, *, gradient_checkpointing=True):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=N_CLASSES,
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id=LABEL_TO_INDEX,
    )
    # DeBERTa-v3 ships fp16 weights; upcast to fp32 master weights so AMP autocast
    # produces fp32 gradients (GradScaler cannot unscale fp16 gradients).
    model = model.float()
    if gradient_checkpointing:
        # Trade ~30% compute for a large activation-memory saving so deberta-v3-large
        # at len 384 fits the T4 (v3 OOM'd at 14.5 GB without it).
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device)


def train_fold(train_texts, train_y, valid_texts, *, model_name, tokenizer, device,
               epochs, batch_size, max_len, lr, seed, valid_batch_size=64, log_every=200):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(model_name, device)
    ds = TextDataset(train_texts, train_y, tokenizer, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tokenizer), drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(dl) * epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    model.train()
    t0 = time.perf_counter()
    seen = 0
    for ep in range(epochs):
        for step, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            seen += batch["labels"].shape[0]
            if log_every and (step + 1) % log_every == 0:
                el = time.perf_counter() - t0
                print(f"    ep{ep} step{step+1}/{len(dl)} loss{loss.item():.3f} "
                      f"{seen/el:.1f} seq/s", flush=True)
    train_seconds = time.perf_counter() - t0

    model.eval()
    vds = TextDataset(valid_texts, [0] * len(valid_texts), tokenizer, max_len)
    vdl = DataLoader(vds, batch_size=valid_batch_size, shuffle=False,
                     collate_fn=lambda b: collate(b, tokenizer))
    probs = []
    t1 = time.perf_counter()
    with torch.no_grad():
        for batch in vdl:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(**batch).logits
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    infer_seconds = time.perf_counter() - t1
    proba = np.concatenate(probs, axis=0)
    peak_gb = (torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0
    stats = {"train_seconds": round(train_seconds, 2), "infer_seconds": round(infer_seconds, 2),
             "peak_gpu_gb": round(peak_gb, 3), "seq_per_s": round(len(train_texts) * epochs / max(train_seconds, 1e-6), 2),
             "n_train": len(train_texts), "epochs": epochs}
    return proba, model, stats


def to_scores(proba):
    """Log-probabilities as decision scores (argmax-equivalent; fold-aligned)."""
    return np.log(np.clip(proba, 1e-9, 1.0)).astype(np.float32)


# -- orchestration ---------------------------------------------------------
def run_cv(df, folds, pool, *, cfg, device, tokenizer, augment: bool):
    texts = df["text"].tolist()
    y = df["model"].map(LABEL_TO_INDEX).to_numpy()
    labels = df["model"].astype(str).to_numpy()
    scores = np.full((len(df), N_CLASSES), -1e9, dtype=np.float32)
    fold_metrics, decon, fold_stats = [], [], []
    for fold, (tr, va) in enumerate(folds):
        fit_texts = [texts[i] for i in tr]
        fit_y = y[tr].tolist()
        if augment and pool is not None:
            keep, stats = decontaminate_fold(pool, df.iloc[va], cos_threshold=cfg["cos_threshold"])
            decon.append({"fold": fold, **stats})
            pool_keep = pool.loc[keep]
            pool_keep = cap_pool_stratified(pool_keep, cfg["pool_cap_per_fold"], cfg["seed"] + fold)
            fit_texts = fit_texts + pool_keep["text"].tolist()
            fit_y = fit_y + pool_keep["model"].map(LABEL_TO_INDEX).tolist()
            print(f"  [fold {fold}] +pool kept {stats['kept']} -> capped {len(pool_keep)} "
                  f"| fit rows {len(fit_texts)}", flush=True)
        else:
            print(f"  [fold {fold}] official-only fit rows {len(fit_texts)}", flush=True)
        proba, _model, stats = train_fold(
            fit_texts, fit_y, [texts[i] for i in va],
            model_name=cfg["model_name"], tokenizer=tokenizer, device=device,
            epochs=cfg["epochs_pool"] if augment else cfg["epochs_official"],
            batch_size=cfg["batch_size"], max_len=cfg["max_len"], lr=cfg["lr"],
            seed=cfg["seed"] + fold,
        )
        fold_stats.append(stats)
        scores[va] = to_scores(proba)
        pred = np.array(LABELS)[proba.argmax(axis=1)]
        ba = balanced_accuracy(labels[va], pred)
        fold_metrics.append({"fold": fold, "balanced_accuracy": ba,
                             "per_class_recall": per_class_recall(labels[va], pred),
                             "within_provider": within_provider(labels[va], pred)})
        print(f"  [fold {fold}] BA {ba:.4f} | {stats['seq_per_s']} seq/s | "
              f"peak {stats['peak_gpu_gb']} GB | train {stats['train_seconds']}s", flush=True)
        del _model
        gc.collect()
        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    pooled_pred = np.array(LABELS)[np.nanargmax(np.where(scores < -1e8, np.nan, scores), axis=1)]
    pooled = {"balanced_accuracy": balanced_accuracy(labels, pooled_pred),
              "per_class_recall": per_class_recall(labels, pooled_pred),
              "within_provider": within_provider(labels, pooled_pred)}
    return scores, pooled, fold_metrics, decon, fold_stats


def benchmark(df, folds, *, cfg, device, tokenizer):
    """Measure throughput on a small slice; extrapolate full runtime vs 9h budget."""
    tr, va = folds[0]
    n = min(512, len(tr))
    t0 = time.perf_counter()
    _proba, model, stats = train_fold(
        [df["text"].iloc[i] for i in tr[:n]], df["model"].map(LABEL_TO_INDEX).to_numpy()[tr[:n]].tolist(),
        [df["text"].iloc[i] for i in va[:64]],
        model_name=cfg["model_name"], tokenizer=tokenizer, device=device,
        epochs=1, batch_size=cfg["batch_size"], max_len=cfg["max_len"], lr=cfg["lr"], seed=0,
    )
    del model
    if device.type == "cuda":
        import torch
        torch.cuda.empty_cache()
    sps = max(stats["seq_per_s"], 1e-6)
    off_rows = sum(len(tr) for tr, _ in folds) * cfg["epochs_official"]
    pool_rows = sum(len(tr) for tr, _ in folds) * cfg["epochs_pool"] + \
        cfg["pool_cap_per_fold"] * len(folds) * cfg["epochs_pool"]
    est_off_h = off_rows / sps / 3600
    est_pool_h = pool_rows / sps / 3600
    report = {"seq_per_s": sps, "peak_gpu_gb": stats["peak_gpu_gb"],
              "est_official_hours": round(est_off_h, 3), "est_pool_hours": round(est_pool_h, 3),
              "est_total_hours": round(est_off_h + est_pool_h, 3), "budget_hours": 9.0,
              "within_budget": (est_off_h + est_pool_h) < 8.5}
    print("[benchmark]", json.dumps(report), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default="microsoft/deberta-v3-large")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs-official", type=int, default=3)
    ap.add_argument("--epochs-pool", type=int, default=2)
    ap.add_argument("--pool-cap-per-fold", type=int, default=12000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cos-threshold", type=float, default=0.3)
    ap.add_argument("--strict-pool", action="store_true")
    ap.add_argument("--no-pool", action="store_true", help="official-only ablation")
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--out", default="/kaggle/working/deberta_cv")
    ap.add_argument("--save-final", action="store_true", help="also fit a final model on all official+pool")
    args = ap.parse_args()

    ensure_deps()
    import torch
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'none'}", flush=True)
    if device.type == "cpu":
        print("[warn] no GPU — this kernel is intended for Kaggle GPU; CPU will be very slow.", flush=True)

    cfg = {k: getattr(args, k) for k in
           ("model_name", "max_len", "batch_size", "epochs_official", "epochs_pool",
            "pool_cap_per_fold", "lr", "seed", "cos_threshold")}
    out_dir = Path(args.out)
    (out_dir / "oof").mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir()
    df = load_official(data_dir)
    shipped = Path(__file__).resolve().parent / "folds.json"
    if EMBEDDED_FOLD_ID_STR:
        folds, method = embedded_folds(df)
        print(f"[data] using embedded local folds ({method}) — aligned to local cv_s42", flush=True)
    elif shipped.is_file():
        folds, method = load_shipped_folds(shipped, df)
        print(f"[data] using shipped local folds ({method}) — aligned to local cv_s42", flush=True)
    else:
        folds, method = make_grouped_folds(df, args.n_folds, args.seed)
        print("[warn] no embedded/shipped folds; recomputed with Kaggle sklearn "
              "(may differ from local cv_s42)", flush=True)
    print(f"[data] official {len(df)} rows | {method} folds={len(folds)} | data_dir={data_dir}", flush=True)

    pool = None
    pool_path = find_pool()
    if pool_path is not None and not args.no_pool:
        pool = load_pool(pool_path, strict=args.strict_pool)
        print(f"[data] pool {len(pool)} rows from {pool_path} (strict={args.strict_pool})", flush=True)
    else:
        print("[data] pool not attached or --no-pool: official-only ablation", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    bench = None
    if not args.no_benchmark:
        bench = benchmark(df, folds, cfg=cfg, device=device, tokenizer=tokenizer)
        (out_dir / "benchmark.json").write_text(json.dumps(bench, indent=1), encoding="utf-8")
        if not bench["within_budget"]:
            (out_dir / "benchmark.json").write_text(json.dumps(bench, indent=1), encoding="utf-8")
            raise SystemExit(
                f"[abort] estimated {bench['est_total_hours']}h exceeds the 9h budget; "
                "reduce --pool-cap-per-fold / --epochs-pool / --max-len or use a smaller model."
            )

    results = {"config": cfg, "split_method": method, "n_official": len(df),
               "n_pool": int(len(pool)) if pool is not None else 0, "benchmark": bench}

    print("\n=== official-only CV ===", flush=True)
    off_scores, off_pooled, off_folds, _decon, off_stats = run_cv(
        df, folds, None, cfg=cfg, device=device, tokenizer=tokenizer, augment=False)
    np.savez_compressed(out_dir / "oof" / "official_only.npz", scores=off_scores,
                        y=df["model"].map(LABEL_TO_INDEX).to_numpy(),
                        labels=np.array(LABELS, dtype=object))
    results["official_only"] = {"pooled": off_pooled, "folds": off_folds, "stats": off_stats}
    print(f"[official-only] pooled BA {off_pooled['balanced_accuracy']:.4f}", flush=True)

    if pool is not None:
        print("\n=== official+pool CV ===", flush=True)
        pool_scores, pool_pooled, pool_folds, decon, pool_stats = run_cv(
            df, folds, pool, cfg=cfg, device=device, tokenizer=tokenizer, augment=True)
        np.savez_compressed(out_dir / "oof" / "official_pool.npz", scores=pool_scores,
                            y=df["model"].map(LABEL_TO_INDEX).to_numpy(),
                            labels=np.array(LABELS, dtype=object))
        off_fold_ba = [f["balanced_accuracy"] for f in off_folds]
        pool_fold_ba = [f["balanced_accuracy"] for f in pool_folds]
        diffs = np.asarray(pool_fold_ba) - np.asarray(off_fold_ba)
        ci = mean_ci(diffs)
        recall_delta = {k: (None if np.isnan(off_pooled["per_class_recall"][k]) or
                            np.isnan(pool_pooled["per_class_recall"][k])
                            else round(pool_pooled["per_class_recall"][k] - off_pooled["per_class_recall"][k], 4))
                        for k in LABELS}
        results["official_pool"] = {"pooled": pool_pooled, "folds": pool_folds, "stats": pool_stats,
                                    "decontamination": decon}
        results["comparison"] = {
            "official_only_ba": off_pooled["balanced_accuracy"],
            "official_pool_ba": pool_pooled["balanced_accuracy"],
            "ba_delta": pool_pooled["balanced_accuracy"] - off_pooled["balanced_accuracy"],
            "paired_fold_diff": diffs.tolist(), "paired_diff_ci": ci,
            "per_class_recall_delta": recall_delta,
        }
        print(f"[+pool] pooled BA {pool_pooled['balanced_accuracy']:.4f} "
              f"(delta {results['comparison']['ba_delta']:+.4f}) | paired CI "
              f"[{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}]", flush=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out_dir/'results.json'}", flush=True)

    if args.save_final:
        print("\n=== final fit (all official" + ("+pool" if pool is not None else "") + ") ===", flush=True)
        fit_texts = df["text"].tolist()
        fit_y = df["model"].map(LABEL_TO_INDEX).to_numpy().tolist()
        if pool is not None:
            pool_cap = cap_pool_stratified(pool, args.pool_cap_per_fold * args.n_folds, args.seed)
            fit_texts += pool_cap["text"].tolist()
            fit_y += pool_cap["model"].map(LABEL_TO_INDEX).tolist()
        _proba, model, stats = train_fold(
            fit_texts, fit_y, df["text"].head(8).tolist(),
            model_name=args.model_name, tokenizer=tokenizer, device=device,
            epochs=args.epochs_pool if pool is not None else args.epochs_official,
            batch_size=args.batch_size, max_len=args.max_len, lr=args.lr, seed=args.seed)
        final_dir = out_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        (final_dir / "label_order.json").write_text(json.dumps(list(LABELS)), encoding="utf-8")
        print(f"[final] saved weights to {final_dir} | {stats}", flush=True)


if __name__ == "__main__":
    main()