"""Encoder attribution: official-only vs official+pool grouped CV (Kaggle GPU).

Research TRAINING kernel — NOT the submission. Runs on Kaggle GPU with internet ON
to fetch the public checkpoint; saves OOF + metrics + final weights for a
separate OFFLINE inference notebook (internet OFF, weights attached).

Backbone: DeBERTa-v3-large by default; set the module-level BACKBONE constant to
"modernbert-large" for the H8 probe (D052). A script kernel passes no CLI args, so
that constant — not --model-name — is what a pushed run uses.

Protocol: full prompt-grouped out-of-fold balanced accuracy is the PRIMARY metric.
External pool CV uses the NON-TRANSDUCTIVE global screen (D028): the pool is screened
ONCE against the complete official training set, so every fold reuses an identical
keep-mask and no held-out passage can influence which pool rows get trained on. The
transductive per-fold screen is disabled (D026) and only remains as --pool-scope fold
for reproducing historical diagnostics. Upstream match_weight/style_score columns are never used.
Every learned transform is fit on the training side only; pool rows never enter
validation; no transductive step.

Budget: <=9h. A benchmark cell measures throughput/peak memory and extrapolates the
full runtime BEFORE the long runs, and caps the pool per fold to stay in budget.

Push (owner account, GPU quota):
    kaggle kernels push -p kaggle/deberta
    kaggle kernels status tiennamlee/name-that-llm-deberta-official-vs-pool-cv

ModernBERT needs transformers >= 4.48; see requirements_gpu.txt.
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
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


def ordered_dataset_fingerprint(df):
    """Bind fold indices to ordered ids, groups, labels, and input text."""
    digest = hashlib.sha256()
    for row in df.loc[:, ["id", "prompt_id", "text", "model"]].itertuples(index=False, name=None):
        digest.update(json.dumps([str(v) for v in row], ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fold_assignments(n_rows, folds):
    """Build exact validation-fold IDs and reject incomplete or overlapping folds."""
    if n_rows <= 0 or not folds:
        raise ValueError("OOF metadata requires rows and folds")
    result = np.full(n_rows, -1, dtype=np.int32)
    for i, (tr, va) in enumerate(folds):
        tr, va = np.asarray(tr, dtype=int), np.asarray(va, dtype=int)
        if tr.ndim != 1 or va.ndim != 1 or not tr.size or not va.size:
            raise ValueError("fold indices must be nonempty vectors")
        if np.any(tr < 0) or np.any(tr >= n_rows) or np.any(va < 0) or np.any(va >= n_rows):
            raise ValueError("fold index out of range")
        if np.unique(tr).size != tr.size or np.unique(va).size != va.size:
            raise ValueError("fold indices contain duplicates")
        if np.intersect1d(tr, va).size or tr.size + va.size != n_rows:
            raise ValueError("training indices must complement validation indices")
        if np.any(result[va] != -1):
            raise ValueError("row appears in multiple validation folds")
        result[va] = i
    if np.any(result < 0):
        raise ValueError("some rows have no validation fold")
    return result


def load_shipped_folds(path, df):
    """Load the exact local cv_s42 folds shipped with the kernel.

    Avoids the sklearn train/Kaggle version mismatch shifting fold assignments;
    guarantees the OOF is fold-aligned with the local sparse OOF for blending.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.get("ordered_dataset_sha256")
    if not expected or expected != ordered_dataset_fingerprint(df):
        raise ValueError("shipped folds lack a matching ordered dataset fingerprint")
    if int(payload["n_rows"]) != len(df):
        raise ValueError(f"shipped folds n_rows {payload['n_rows']} != data {len(df)}")
    folds = [(np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int))
             for f in payload["folds"]]
    fold_assignments(len(df), folds)
    for tr, va in folds:
        if not set(df.iloc[tr]["prompt_id"]).isdisjoint(set(df.iloc[va]["prompt_id"])):
            raise ValueError("shipped folds leak a prompt across train/valid")
    return folds, str(payload["method"]) + "-shipped"


# Exact local grouped-CV fold assignments, embedded because Kaggle script kernels do
# NOT upload companion files (folds.json was dropped on v3). One fold id per official
# train row (0..4) per seed; each verified identical to its local splits before push.
EMBEDDED_FOLD_IDS = {
    "s42": "1,1,2,1,0,3,3,3,4,0,1,1,0,0,1,2,4,1,4,4,2,3,4,2,0,3,2,4,4,4,3,0,2,3,4,2,1,0,3,2,2,3,2,3,2,4,1,1,0,2,4,0,4,2,3,3,3,0,0,1,3,1,3,2,0,3,1,3,4,3,1,0,1,0,3,1,0,1,3,3,4,4,0,2,2,0,1,4,2,0,1,3,1,0,3,0,4,3,2,4,0,3,4,3,1,4,2,2,2,4,1,1,3,1,1,0,0,3,0,0,1,0,0,2,4,2,3,3,4,1,3,0,1,2,1,3,3,3,0,4,2,3,1,2,2,3,3,4,2,1,2,0,3,0,2,1,3,2,1,4,4,3,1,2,1,3,1,4,2,0,3,2,2,0,0,0,2,3,3,2,4,1,0,1,0,0,3,0,4,2,2,1,0,2,2,1,4,2,0,1,3,4,2,4,4,4,4,3,1,0,4,4,4,4,4,1,3,2,2,3,2,0,2,4,3,4,1,3,1,2,4,4,0,3,4,0,2,4,3,3,1,4,4,1,1,2,0,3,4,0,4,2,4,4,3,3,1,1,4,4,2,3,3,4,4,4,2,3,0,2,3,2,4,4,1,4,3,3,3,4,2,4,0,2,4,3,4,3,0,0,2,4,0,3,1,1,3,0,1,2,1,3,3,3,3,2,2,1,3,1,2,4,4,1,3,2,4,4,4,1,3,2,1,4,1,1,4,2,4,2,4,4,2,2,0,1,3,0,3,2,0,4,1,3,3,3,2,1,4,4,1,1,0,3,4,3,1,2,4,1,3,1,1,0,2,0,3,2,4,3,4,4,2,3,3,0,4,3,4,3,3,4,1,4,4,3,0,4,1,1,4,4,1,1,3,4,3,2,0,1,0,4,3,2,2,1,0,2,3,1,1,2,2,1,3,2,2,1,2,0,2,2,4,4,2,0,4,1,2,2,2,4,0,3,0,1,2,0,1,0,3,2,3,3,2,1,1,0,4,0,4,1,3,2,3,3,3,4,3,1,2,4,3,3,3,1,4,3,3,1,3,4,0,0,3,4,4,2,0,0,0,2,3,0,2,4,3,1,1,0,0,0,4,2,2,4,0,4,2,3,1,0,3,3,3,2,4,4,1,0,0,3,3,4,2,3,3,3,3,4,0,4,3,1,1,4,4,4,1,1,4,0,0,2,2,1,3,1,1,3,0,2,3,1,1,3,2,4,0,3,3,0,4,0,1,2,0,4,1,4,0,1,3,3,2,0,3,2,3,1,2,1,0,0,3,0,4,3,1,0,1,4,1,0,3,3,3,1,2,3,1,0,3,3,1,4,0,3,1,0,0,2,1,1,4,1,3,1,4,2,3,0,1,2,1,3,1,2,4,3,3,2,4,4,3,3,4,0,2,4,0,0,0,0,4,1,4,2,0,4,2,1,2,0,3,1,2,2,4,3,3,3,0,2,3,0,2,0,4,1,4,0,2,4,1,1,1,3,1,1,0,0,3,0,0,0,2,1,3,1,4,4,0,4,0,3,3,1,4,2,1,4,4,3,3,4,0,3,2,4,0,4,0,0,0,4,2,2,0,3,1,0,1,2,3,3,4,0,0,2,4,3,0,1,3,1,4,2,2,3,2,2,2,0,0,0,0,0,0,1,4,1,0,1,4,2,3,0,0,2,2,4,3,4,3,2,2,3,1,2,1,4,3,1,4,4,3,2,1,0,2,1,4,3,1,0,2,2,1,1,2,4,4,2,1,3,4,2,4,4,4,4,4,0,2,1,1,3,1,4,3,4,4,3,0,0,3,4,2,4,3,1,1,1,0,0,1,1,1,2,4,2,4,2,3,4,2,3,4,4,1,4,4,0,1,3,0,0,1,3,2,2,4,3,2,3,1,4,0,1,0,3,0,1,3,3,1,1,4,0,0,1,4,4,4,0,1,2,1,4,2,1,0,4,0,0,1,4,0,1,2,4,2,2,3,0,2,0,0,1,2,3,4,1,0,0,0,0,1,0,0,0,0,4,0,1,0,0,3,3,4,0,2,2,1,4,0,4,0,2,2,1,2,4,4,2,2,0,2,4,3,0,2,0,3,2,1,2,4,2,3,1,0,3,1,3,3,4,1,3,2,2,4,2,3,4,0,3,1,0,2,4,0,4,4,1,1,0,4,2,0,1,2,4,2,1,0,3,4,1,3,2,2,4,3,0,3,4,3,0,2,0,1,0,1,1,2,4,1,0,0,1,4,1,2,4,2,1,3,1,3,4,1,3,0,4,4,2,4,2,4,4,2,4,3,2,2,2,3,1,3,4,2,2,1,0,2,2,1,4,4,1,4,3,4,2,2,4,2,2,1,2,4,4,1,1,1,0,1,1,3,3,2,3,2,4,0,4,2,1,0,4,4,3,4,2,4,4,1,1,2,2,2,2,4,0,0,0,2,0,4,3,0,3,0,4,4,0,4,1,2,1,4,4,4,4,0,0,1,2,4,0,2,1,4,1,4,1,3,3,0,1,2,0,3,2,3,1,2,0,3,3,4,4,1,1,0,0,4,3,0,2,0,1,1,0,4,0,1,2,0,3,1,4,1,2,1,0,2,3,0,1,1,3,1,0,3,2,2,3,0,0,3,4,1,4,4,0,2,3,3,0,1,0,4,3,4,0,3,4,2,2,3,0,4,1,0,3,0,2,4,4,0,4,1,2,1,3,4,3,1,1,2,2,0,3,0,2,2,3,0,3,4,4,3,2,2,2,3,2,4,3,2,3,2,3,0,1,1,2,2,0,3,4,3,3,4,3,2,3,4,2,4,4,0,0,0,1,4,3,3,2,0,3,2,3,2,3,2,3,4,4,4,4,3,0,3,1,2,2,0,2,1,4,0,0,3,0,4,0,4,0,4,3,0,3,1,4,1,4,2,4,4,1,3,3,0,0,3,3,3,0,2,1,3,1,3,0,3,0,0,4,2,4,0,4,3,1,3,0,1,4,2,1,4,1,3,4,0,3,1,2,3,0,2,0,3,2,0,2,4,0,1,1,1,0,1,4,3,2,3,2,4,4,0,0,1,4,2,4,1,2,0,0,1,2,0,0,3,1,0,3,0,0,0,4,3,3,1,4,0,3,1,4,0,4,4,4,4,3,4,1,1,4,1,2,1,4,1,3,1,2,4,2,3,0,1,4,1,2,0,1,3,1,2,1,3,3,4,1,0,0,2,4,3,2,1,2,1,4,3,2,3,2,0,4,1,1,4,0,4,1,1,2,1,4,3,4,3,4,0,2,1,2,1,2,3,0,3,2,1,2,2,0,1,3,4,3,2,0,2,2,3,4,1,2,0,4,2,2,2,2,2,3,4,2,3,4,1,2,3,0,2,0,4,2,0,1,1,2,4,4,2,3,1,3,2,1,2,3,2,1,3,0,3,2,0,2,2,3,1,0,2,2,3,1,0,1,0,0,1,1,4,3,0,3,0,0,3,4,4,2,4,3,3,3,2,3,1,4,0,3,2,1,2,1,0,0,3,4,4,4,0,0,4,3,2,2,4,3,3,2,4,4,0,2,4,2,4,0,2,1,3,2,2,1,2,1,0,0,4,0,4,1,0,1,1,0,4,4,0,1,2,0,3,1,3,3,3,0,3,2,1,3,3,1,1,4,4,2,1,1,4,0,2,3,3,1,3,3,4,1,3,3,4,0,3,2,3,1,1,3,0,4,1,3,2,2,3,3,2,3,3,3,2,1,4,4,0,4,4,0,4,1,0,2,4,4,2,0,2,0,3,0,1,3,1,4,3,1,4,0,2,1,3,3,2,1,2,3,2,2,0,2,1,3,0,0,2,1,1,2,3,4,3,4,3,4,3,0,4,0,0,3,2,1,1,1,4,3,3,3,2,4,3,1,4,0,0,0,0,3,1,0,3,4,2,4,0,1,2,0,0,4,4,0,2,0,0,1,2,0,1,2,2,0,0,0,1,3,3,2,3,1,1,4,1,1,4,0,4,3,1,1,3,1,0,1,0,2,2,0,0,4,3,1,4,2,1,1,2,2,0,2,2,1,4,0,2,3,1,3,2,1,1,3,1,3,2,4,0,0,2,4,0,0,4,3,3,0,3,3,2,1,1,0,1,3,0,1,0,1,0,2,1,0,3,0,4,0,4,2,2,1,4,0,3,3,0,0,1,1,4,1,1,0,3,1,2,1,1,1,4,2,3,4,3,4,1,0,2,0,2,2,1,0,2,1,2,3,4,2,1,3,2,1,4,0,0,3,3,2,3,0,0,2,0,1,2,1,2,2,1,1,1,0,1,0,3,4,2,3,0,1,1,1,3,2,4,3,4,1,2,3,2,1,1,1,0,2,2,0,1,3,1,2,4,0,3,3,2,4,2,3,1,4,0,0,0,4,3,0,0,2,3,2,0,1,0,1,4,1,3,2,3,0,0,0,1,4,0,2,0,2,1,0,2,2,1,3,4,1,0,0,4,2,2,4,1,0,2,1,0,1,1,2,0,4,4,0,1,1,4,4,4,1,0,0,3,0,1,3,3,4,2,4,1,0,3,3,0,2,2,1,3,2,2,4,2,2,1,2,1,2,4,4,4,4,3,4,3,1,2,3,2,2,0,0,1,2,0,2,0,2,2,0,4,2,1,2,4,1,3,4,0,4,4,0,0,0,3,0,4,1,2,1,3,0,2,2,2,1,2,0,4,0,0,4,2,0,4,4,2,3,1,2,0,2,2,4,2,4,0,2,0,2,4,0,3,1,4,2,0,0,0,2,1,1,1,4,2,2,1,2,3,4,0,4,2,1,4,1,0,2,2,3,2,1,0,4,3,4,4,3,4,0,2,3,1,0,4,0,0,4,3,2,3,3,4,0,3,2,3,0,1,3,1,3,0,2,3,0,4,4,0,1,3,0,4,0,1,3,4,0,2,4,1,1,0,3,2,0,0,0,1,3,1,4,1,4,1,2,1,4,0,1,0,3,2,2,1,0,3,4,0,4,3,2,0,2,1,3,2,2,4,4,0,0,1,0,3,0,0,1,4,2,4,2,1,3,2,4,3,4,4,3,3,1,3,1,4,3,0,1,4,3,3,3,0,3,0,4,1,0,1,3,4,1,2,0,2,1,1,3,4,4,2,3,0,0,0,3,4,2,3,1,2,2,2,0,3,1,4,2,3,0,1,0,2,4,3,1,2,2,2,1,0,1,1,3,4,4,1,2,1,4,0,2,4,3,2,0,0,1,2,3,1,2,1,2,0,1,4,4,0,4,1,0,4,3,0,3,4,4,3,0,4,2,0,4,1,2,3,2,0,3,0,2,1,4,1,2,3,3,4,1,0,4,3,2,1,2,1,1,1,1,3,0,0,4,2,3,0,3,0,4,4,0,0,3,0,3,1,2,2,0,1,1,3,2,4,0,1,0,0,2,4,0,3,0,3,3,3,0,4,3,1,3,2,3,3,2,2,3,1,0,0,4,2,2,0,4,1,3,4,0,4,4,0,3,1,2,2,3,4,2,3,0,3,2,1,1,0,0,2,2,0,1,0,4,2,1,3,3,3,2,4,0,2,2,0,0,3,2,2,3,2,2,2,0,1,0,1,4,1,0,3,4,2,3,4,0,0,3,2,0,2,0,4,0,2,2,1,4,0,1,0,2,3,3,4,0,4,3,3,3,4,2,2,1,3,1,3,4,0,1,1,2,2,4,1,4,1,1,3,4,2,0,3,4,4,3,1,0,3,3,0,1,2,0,3,3,2,0,4,4,4,1,3,3,4,1,3,2,1,2,4,4,2,4,4,1,0,1,1,1,2,1,0,0,1,2,0,4,1,4,0,4,4,3,2,4,4,4,2,3,3,1,3,4,0,2,4,1,2,1,0,3,1,1,2,1,4,0,1,1,0,3,4,4,0,1,2,4,2,1,2,2,2,1,3,3,0,3,1,4,2,1,1,0,0,3,1,1,4,3,3,2,1,3,3,2,4,2,0,3,4,3,3,3,4,2,4,1,4,3,3,4,1,2,1,1,3,1,4,1,2,1,2,1,1,1,2,4,4,3,1,1,4",
    "s13": "0,3,4,3,1,0,0,1,1,0,3,1,4,2,4,2,0,2,4,4,1,4,2,3,4,0,2,2,4,2,4,4,4,4,3,0,4,4,4,2,2,2,4,3,2,3,0,4,0,0,0,4,3,0,0,4,1,1,4,4,0,2,1,4,0,1,0,4,1,1,4,3,2,3,0,0,0,0,1,3,2,4,4,0,2,3,2,0,2,2,4,0,0,1,4,1,0,4,1,3,0,0,2,1,4,4,1,0,1,0,2,2,4,0,0,1,3,4,4,4,0,1,0,3,1,3,1,0,1,4,3,0,3,1,0,4,3,3,4,2,3,3,3,2,2,1,1,3,3,0,2,4,4,4,1,0,4,0,0,2,4,1,3,1,3,3,1,0,2,0,0,2,1,3,4,2,2,3,1,4,2,3,4,1,4,3,0,0,2,3,2,3,4,0,1,4,4,2,1,0,3,4,0,2,2,1,1,1,4,0,2,3,2,2,2,0,1,4,4,2,2,4,1,0,0,3,3,4,4,1,1,4,4,3,1,4,3,4,0,0,4,0,3,3,3,0,3,1,1,2,2,2,2,2,0,0,0,0,1,3,0,3,3,2,2,2,2,3,1,0,1,3,3,1,0,2,0,3,4,2,4,2,0,3,1,1,3,1,4,3,1,0,4,1,4,4,4,1,2,2,2,0,4,4,0,4,4,0,0,4,0,3,4,3,4,0,4,4,2,2,3,2,3,2,0,2,0,3,2,2,4,4,0,2,1,3,2,3,4,4,4,3,2,4,4,3,3,3,3,4,3,4,1,1,0,0,2,4,2,2,0,0,1,1,4,3,3,2,0,0,4,2,1,1,0,1,1,0,2,4,4,2,0,1,3,4,0,0,4,1,4,1,2,1,0,2,4,0,4,0,0,3,2,0,2,0,4,3,1,1,2,0,2,0,3,2,0,0,2,1,2,0,3,1,0,1,2,4,2,1,4,4,1,1,1,1,4,0,4,2,3,1,4,1,2,1,1,3,2,1,2,0,3,0,4,1,3,1,3,3,0,1,1,0,4,3,2,0,3,1,4,1,4,4,0,2,2,4,0,4,1,2,4,4,3,3,4,3,1,4,2,3,3,3,0,1,2,2,1,2,1,2,1,1,0,2,2,3,2,0,2,1,4,1,2,1,3,0,4,3,1,1,3,4,1,2,3,0,3,2,2,3,0,1,3,2,0,3,2,3,0,4,1,0,1,0,2,2,4,4,1,1,1,1,3,2,3,4,1,3,0,1,3,3,1,3,1,4,1,0,3,4,1,0,2,0,0,3,3,1,2,2,0,1,0,0,4,4,4,4,1,2,0,2,0,1,4,4,0,1,2,2,3,2,2,4,4,1,2,2,0,4,3,3,3,1,3,4,4,4,0,4,2,4,0,0,1,2,1,2,1,4,0,4,2,3,1,2,4,4,4,4,4,1,0,3,2,0,0,1,3,2,3,0,1,1,3,2,3,3,4,1,3,1,0,0,3,0,2,0,2,1,1,4,3,0,0,2,4,2,0,3,1,2,4,2,4,4,1,1,4,3,4,4,1,2,3,2,1,2,2,4,4,4,1,0,3,4,1,3,4,2,2,0,3,4,0,3,3,4,2,3,1,4,1,0,4,2,4,3,4,4,3,2,0,1,1,1,0,0,2,2,1,4,1,2,4,2,0,2,4,4,4,3,3,3,4,4,3,1,4,2,0,3,2,4,3,2,4,4,0,0,4,1,4,2,4,4,2,1,0,2,0,1,3,0,4,4,1,4,3,1,4,1,4,1,1,1,0,1,0,2,2,0,2,0,0,4,1,1,3,2,2,3,1,4,3,3,2,1,3,4,0,1,0,3,1,0,2,4,2,4,2,0,1,3,1,2,3,3,4,0,2,3,1,0,1,1,0,0,0,3,0,2,1,0,4,3,1,3,4,2,0,4,3,0,3,3,3,2,0,3,0,0,0,1,4,2,3,3,3,1,2,4,3,0,4,0,4,2,2,0,2,4,0,4,1,1,0,4,4,4,4,1,2,0,3,4,3,4,0,2,3,4,3,0,3,1,3,3,3,2,4,0,2,1,0,1,4,1,4,0,4,3,0,1,2,0,4,0,3,2,0,3,0,2,3,4,3,4,2,0,0,1,3,1,0,3,3,2,4,0,2,2,3,4,4,2,2,4,0,1,0,3,2,1,4,4,2,3,1,3,2,1,3,0,2,2,0,3,4,0,1,1,0,3,4,1,3,3,1,1,2,3,3,4,0,0,3,1,4,2,0,0,3,4,2,2,2,2,1,3,1,1,3,4,4,3,1,0,1,4,3,1,0,3,0,0,1,3,4,2,3,3,1,1,3,0,3,3,3,2,4,0,1,3,3,0,2,3,1,4,0,1,0,4,4,4,2,3,0,0,4,2,2,3,4,1,3,4,2,4,2,1,3,4,3,0,4,0,3,1,4,0,1,1,3,0,3,3,0,0,0,2,3,4,0,2,3,1,2,1,2,0,4,3,2,3,0,0,2,3,3,1,3,4,2,3,2,0,0,4,4,2,2,2,0,4,1,4,1,2,4,2,0,2,2,0,1,3,4,4,0,1,3,0,2,2,0,4,3,1,4,0,3,0,1,4,0,0,0,0,0,2,3,3,1,4,2,0,3,4,4,0,1,0,2,4,1,1,1,4,4,0,2,3,1,3,2,2,1,4,4,2,2,0,4,0,1,1,0,0,4,4,4,3,1,0,0,1,3,0,2,2,4,1,3,2,3,3,2,4,4,0,4,1,4,2,1,2,1,2,1,2,0,0,3,2,2,1,3,4,2,2,0,3,0,1,3,0,1,3,0,3,3,0,3,0,4,3,4,2,1,2,3,1,1,3,1,4,3,3,0,2,1,2,4,1,2,4,2,4,2,1,3,1,4,2,2,3,2,2,2,0,3,0,1,1,3,1,0,0,4,4,1,4,3,4,3,0,1,4,2,2,2,3,3,0,4,1,3,4,2,2,1,4,0,4,0,1,4,2,4,0,3,1,0,2,4,4,1,4,2,1,1,3,2,2,2,2,0,3,2,3,2,2,1,1,1,3,1,3,1,4,0,2,1,2,2,0,0,3,0,0,2,1,1,0,3,3,1,4,2,2,2,4,2,3,1,4,2,3,4,2,2,3,0,3,3,4,2,2,3,2,0,1,0,0,0,2,0,1,0,2,1,3,4,4,4,2,4,2,2,1,0,3,3,1,0,1,0,1,3,2,2,1,2,2,3,2,1,0,2,3,4,2,0,1,1,1,0,1,3,3,2,2,4,4,3,2,0,3,3,0,4,2,1,1,2,3,2,3,0,1,0,3,0,2,0,4,1,2,4,3,3,1,1,3,3,2,1,3,4,3,3,4,0,3,3,2,0,2,0,0,0,4,0,3,4,0,3,3,4,4,3,4,1,1,2,3,1,4,3,2,3,0,4,1,1,4,1,0,3,1,1,3,4,1,0,4,2,1,3,1,0,0,0,4,0,2,4,3,3,2,4,4,1,0,3,0,0,2,4,1,2,1,2,4,4,1,3,0,4,0,0,1,1,0,4,1,4,3,0,1,3,0,2,2,4,2,0,0,1,3,3,3,4,1,1,3,4,3,1,2,1,0,0,1,0,1,4,2,3,0,3,3,4,4,1,4,1,4,2,1,4,0,0,1,3,0,2,3,0,3,1,4,2,3,0,3,3,3,2,1,3,4,2,4,1,3,2,1,4,0,0,0,1,1,4,2,4,0,2,2,4,4,1,1,2,4,0,1,4,4,0,4,1,3,1,0,1,1,2,2,3,1,2,3,2,0,0,0,4,0,4,3,1,3,0,1,3,1,1,1,4,4,4,3,1,1,2,4,1,4,2,1,0,1,1,1,4,4,1,2,4,2,2,4,4,1,0,3,3,4,1,2,3,2,4,4,2,4,4,2,0,4,2,1,3,3,4,1,3,2,2,1,1,0,4,2,1,1,0,2,3,2,2,1,2,3,0,3,0,0,3,1,2,4,4,2,3,0,2,2,2,1,1,4,4,3,0,3,2,4,2,3,4,3,0,0,4,0,3,0,3,3,1,0,2,1,3,4,1,3,4,4,1,0,4,3,3,0,1,0,3,0,0,3,4,3,3,3,0,0,2,3,2,0,2,1,2,4,1,0,1,1,0,1,3,3,3,4,1,2,4,1,0,0,0,3,2,2,3,1,4,1,3,1,0,3,1,3,4,3,0,4,3,3,4,4,4,2,2,2,4,4,0,1,1,1,3,2,1,0,1,4,3,4,0,0,3,2,1,3,0,1,0,2,4,0,3,3,4,1,3,4,0,3,2,2,1,1,0,2,0,2,0,3,1,1,3,3,4,4,2,3,2,0,1,3,4,4,3,1,3,1,4,4,0,2,4,2,0,0,1,2,4,0,2,2,2,2,0,3,4,3,0,1,0,2,3,2,2,0,4,0,1,3,1,0,1,1,4,3,3,0,0,0,2,0,3,3,1,2,3,2,4,1,1,2,3,0,3,0,4,3,1,4,0,2,0,0,1,1,3,2,3,1,3,1,4,1,2,1,1,3,0,4,0,4,2,3,4,2,4,3,4,4,1,3,0,4,1,4,0,4,1,2,2,2,3,1,4,1,2,0,1,3,3,2,2,3,1,4,3,2,3,1,4,2,4,4,4,0,1,3,4,4,3,2,1,0,3,4,2,1,4,1,4,2,1,2,4,1,4,1,4,0,4,2,1,2,2,2,0,0,2,4,3,2,1,0,0,4,1,4,3,3,1,2,0,2,3,1,3,1,1,2,4,0,1,2,3,4,1,4,0,0,4,2,1,2,2,1,4,0,3,4,2,4,4,4,2,1,3,2,4,4,2,3,0,2,0,4,4,1,2,3,0,4,2,1,2,4,1,3,4,4,3,2,2,3,1,4,0,2,2,0,4,0,2,4,0,4,3,2,2,0,2,1,2,2,0,3,4,2,0,3,2,3,0,2,3,0,3,2,4,2,3,2,4,1,3,4,1,0,4,0,3,2,3,4,0,2,3,3,4,0,0,1,2,1,0,2,3,4,0,4,4,0,1,0,0,2,2,3,0,1,0,1,2,3,0,3,1,3,1,2,3,4,2,4,4,1,3,1,4,3,4,1,0,2,2,1,0,1,2,4,0,0,0,4,2,2,3,4,1,1,4,2,3,4,3,3,3,1,0,4,0,3,3,2,3,1,1,1,1,3,0,3,2,2,3,0,3,0,3,0,2,3,4,0,2,4,0,2,2,3,3,4,1,1,4,4,3,0,4,4,3,0,4,3,0,1,1,1,1,1,4,1,2,1,3,1,3,2,0,2,0,0,2,4,3,2,0,3,0,2,0,1,3,0,1,0,0,0,3,0,0,4,1,1,2,4,3,4,3,0,0,3,3,0,4,1,0,3,0,0,1,4,3,0,1,1,0,3,4,1,4,1,0,2,2,1,3,0,0,2,2,0,1,1,1,3,3,4,2,2,1,0,0,3,0,4,2,2,3,0,0,3,2,2,4,4,3,3,0,4,3,3,2,3,1,1,2,3,3,4,2,4,3,1,4,1,4,0,1,3,3,0,4,1,1,1,1,4,3,2,3,1,3,2,4,1,2,0,1,1,1,2,1,3,0,1,1,3,0,4,3,0,2,1,1,3,1,1,2,3,4,3,4,2,1,0,0,0,0,2,0,2,0,2,0,0,2,1,3,2,1,4,2,4,3,0,3,3,2,0,4,4,0,1,2,4,1,2,4,4,3,0,1,3,2,1,2,0,1,2,2,4,3,3,3,0,4,0,1,0,0,4,2,1,1,1,4,2,0,2,1,2,2,4,1,1,2,1,0,3,1,2,3,4,4,3,1,0,1,1,4,3,2,4,1,1,1,3,2,1,2,3,4,0,2,3,1,2,0,4,2,2,3,0,1,3,3,4,0,3,4,2,2,0,1,2,0,1,0,3,0,0,3,1,2,1,0,4,4,0,3,2,1,1,2,2,2,2,3,3,3,2,0,1,2,0,3,3,2,0,1,0,1,4,2,3,2,2,2,3,2",
}
# Which embedded fold seed this pushed run trains on. s42 matches the primary
# protocol of the validated cap-40k pool-MIX candidate (D037), so the DeBERTa-pool
# arm is directly comparable; use "s13" for the alternative-seed confirmation.
FOLD_SEED = "s42"

# Backbone presets. A Kaggle SCRIPT kernel runs `python deberta_train.py` with no
# arguments, so --model-name is unreachable there (the same constraint that forced
# the folds to be embedded, D023). BACKBONE below is therefore the operative switch
# for a pushed run: edit this one line, push, and the choice is recorded in
# results.json config and in the inference manifest so a run cannot be misattributed.
# H8 (D052): two participants self-report ModernBERT-large ~0.55 against their own
# DeBERTa-v3-large ~0.50, which would make D027's encoder negative backbone-specific.
# max_len/batch_size/lr are deliberately NOT preset per backbone: the first H8 probe
# must change only the backbone, keeping the measured D038 budget (384/4/1e-5) so the
# comparison is one-variable. ModernBERT's long-context advantage is a later probe.
BACKBONES = {
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "modernbert-large": "answerdotai/ModernBERT-large",
}
BACKBONE = "deberta-v3-large"


EXPECTED_ORDERED_DATASET_SHA256 = "dc60875fd8b536cb8db67bdac8119132c6d0c0cf667fbca262d5ffc47f3e6e5a"


def embedded_folds(df, seed_label=FOLD_SEED):
    """Rebuild the named local grouped-CV folds from the embedded per-row fold ids."""
    if seed_label not in EMBEDDED_FOLD_IDS:
        raise ValueError(f"no embedded folds for seed {seed_label!r}")
    ids = np.fromstring(EMBEDDED_FOLD_IDS[seed_label], dtype=int, sep=",")
    if ids.size != len(df):
        raise ValueError(f"embedded folds {ids.size} != data {len(df)}")
    if ordered_dataset_fingerprint(df) != EXPECTED_ORDERED_DATASET_SHA256:
        raise ValueError("embedded folds do not match ordered dataset fingerprint")
    folds = []
    for f in range(int(ids.max()) + 1):
        va = np.where(ids == f)[0]
        tr = np.where(ids != f)[0]
        folds.append((tr, va))
    fold_assignments(len(df), folds)
    for tr, va in folds:
        if not set(df.iloc[tr]["prompt_id"]).isdisjoint(set(df.iloc[va]["prompt_id"])):
            raise ValueError("embedded folds leak a prompt across train/valid")
    return folds, f"stratified_group-{seed_label}-embedded"


def load_prompt_mapping(path: str | Path):
    """Read an independently reviewed official-prompt to pool-ID crosswalk."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("prompt mapping document must be an object")
    mapping, evidence = payload.get("mapping"), payload.get("provenance")
    if not isinstance(mapping, dict) or not mapping or not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("prompt mapping requires a nonempty mapping and evidence provenance")
    for key, values in mapping.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(values, list):
            raise ValueError("prompt mapping keys must be nonempty strings and values lists")
        if any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("mapped pool prompt IDs must be nonempty strings")
    return mapping, f"{evidence.strip()} (sha256:{hashlib.sha256(raw).hexdigest()})"


def decontaminate_fold(pool: pd.DataFrame, holdout: pd.DataFrame, *, fit_texts=None,
                       prompt_mapping=None, mapping_provenance=None, cos_threshold=0.3,
                       max_features=30000, chunk_size=20000):
    """Screen overlap with a verified crosswalk and a fold-training-only vocabulary."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if fit_texts is None or len(fit_texts) == 0:
        raise ValueError("fold-training fit_texts are required for decontamination")
    if not isinstance(prompt_mapping, dict) or not mapping_provenance or not mapping_provenance.strip():
        raise ValueError("verified prompt_mapping and mapping_provenance are required")
    if not 0 <= cos_threshold <= 1 or chunk_size <= 0 or holdout.empty:
        raise ValueError("invalid decontamination inputs")
    hold_texts = holdout["text"].astype(str).tolist()
    hold_prompts = set(holdout["prompt_id"].astype(str))
    if set(hold_texts).intersection(map(str, fit_texts)):
        raise ValueError("fold-training fit_texts overlap held-out text")
    missing = hold_prompts - set(prompt_mapping)
    if missing:
        raise ValueError(f"verified prompt_mapping lacks {len(missing)} held-out prompt(s)")
    canonical_ids = set()
    for prompt_id in hold_prompts:
        values = prompt_mapping[prompt_id]
        if not isinstance(values, (tuple, list)) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("prompt_mapping values must be lists of nonempty canonical IDs")
        canonical_ids.update(values)
    pool_texts = pool["text"].astype(str)
    exact = pool_texts.isin(set(hold_texts)).to_numpy()
    pool_prompts = pool["prompt_canon"].fillna("").astype(str).str.strip()
    unmapped_pool_prompt = pool_prompts.eq("").to_numpy()
    shared = pool_prompts.isin(canonical_ids).to_numpy()
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=max_features,
                          sublinear_tf=True, norm="l2", dtype=np.float64)
    vec.fit(list(map(str, fit_texts)))
    hold_mat = vec.transform(hold_texts)
    near = np.zeros(len(pool), dtype=bool)
    for start in range(0, len(pool), chunk_size):
        chunk = pool_texts.iloc[start:start + chunk_size].tolist()
        sims = (vec.transform(chunk) @ hold_mat.T).toarray()
        mc = sims.max(axis=1) if sims.shape[1] else np.zeros(len(chunk))
        near[start:start + len(chunk)] = mc > cos_threshold
    keep = ~(exact | shared | near | unmapped_pool_prompt)
    stats = {"n_pool": int(len(pool)), "n_holdout": int(len(holdout)),
             "exact_dup": int(exact.sum()), "shared_prompt": int(shared.sum()),
             "unmapped_pool_prompt": int(unmapped_pool_prompt.sum()),
             "near_dup": int(near.sum()), "removed": int((~keep).sum()), "kept": int(keep.sum()),
             "prompt_mapping_provenance": mapping_provenance, "similarity_fit_scope": "fold_training_only"}
    return keep, stats


def decontaminate_global(pool: pd.DataFrame, train_df: pd.DataFrame, *,
                         cos_threshold=0.3, max_features=30000, chunk_size=20000):
    """Screen the pool ONCE against the full official training set (D028).

    Non-transductive by construction: the reference corpus is the complete set of
    known official training passages, so the returned keep-mask is identical for
    every CV fold and never depends on which rows a fold holds out. Only text-level
    exact and near-duplicate checks are used because the pool's ``prompt_canon``
    namespace is not proven to match official ``prompt_id`` (D019/D024). Similarity
    is a heuristic, not proof that all overlap was found.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not 0 <= cos_threshold <= 1 or chunk_size <= 0:
        raise ValueError("invalid decontamination inputs")
    if train_df.empty:
        raise ValueError("official training set must not be empty")
    if train_df["text"].isna().any():
        raise ValueError("official training text must not be missing")

    reference_texts = train_df["text"].astype(str).tolist()
    reference_set = set(reference_texts)
    pool_texts = pool["text"].astype(str)
    exact = pool_texts.isin(reference_set).to_numpy()

    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=max_features,
                          sublinear_tf=True, norm="l2", dtype=np.float64)
    vec.fit(reference_texts)
    reference_mat = vec.transform(reference_texts)
    near = np.zeros(len(pool), dtype=bool)
    max_cos = np.zeros(len(pool), dtype=np.float64)
    for start in range(0, len(pool), chunk_size):
        chunk = pool_texts.iloc[start:start + chunk_size].tolist()
        sims = (vec.transform(chunk) @ reference_mat.T).toarray()
        mc = sims.max(axis=1) if sims.shape[1] else np.zeros(len(chunk))
        max_cos[start:start + len(chunk)] = mc
        near[start:start + len(chunk)] = mc > cos_threshold

    keep = ~(exact | near)
    stats = {"n_pool": int(len(pool)), "n_reference": int(len(train_df)),
             "exact_dup": int(exact.sum()), "near_dup": int(near.sum()),
             "removed": int((~keep).sum()), "kept": int(keep.sum()),
             "max_cos_reference_max": float(max_cos.max()) if len(max_cos) else 0.0,
             "cos_threshold": float(cos_threshold),
             "decontamination_scope": "global_full_official_train",
             "validation_independent": True}
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


def cap_pool_balanced(pool: pd.DataFrame, per_class_cap: int, seed: int) -> pd.DataFrame:
    """Cap every class to <=per_class_cap rows (keep rare classes whole).

    The Open Pool v15 is class-imbalanced (gemini-3.7-flash 3,541 vs 3.8-flash
    23,474, D019/D023), so proportional caps leave the rare siblings undertrained.
    A per-class floor equalises exposure: rare classes are kept in full, frequent
    classes downsampled, which directly targets the gemini-3.7-flash blend
    regression. Falls back to the full pool when per_class_cap <= 0.
    """
    if per_class_cap <= 0:
        return pool
    rng = np.random.default_rng(seed)
    parts = []
    for label in pool["model"].unique():
        idx = pool.index[pool["model"] == label].to_numpy()
        take = min(per_class_cap, len(idx))
        parts.append(rng.choice(idx, size=take, replace=False) if take < len(idx) else idx)
    keep = np.concatenate(parts)
    return pool.loc[np.sort(keep)].reset_index(drop=True)


def cap_pool(pool: pd.DataFrame, cfg: dict, seed: int) -> pd.DataFrame:
    """Apply the configured pool cap: balanced per-class if set, else proportional."""
    if cfg.get("pool_balance", 0) > 0:
        return cap_pool_balanced(pool, int(cfg["pool_balance"]), seed)
    return cap_pool_stratified(pool, int(cfg["pool_cap_per_fold"]), seed)


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
    # produces fp32 gradients (GradScaler cannot unscale fp16 gradients). A no-op for
    # checkpoints that already ship fp32, so this is safe across backbones.
    model = model.float()
    if gradient_checkpointing:
        # Trade ~30% compute for a large activation-memory saving so deberta-v3-large
        # at len 384 fits the T4 (v3 OOM'd at 14.5 GB without it).
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.to(device)


def training_loss_warning(epoch_losses, n_classes=N_CLASSES):
    """Flag near-uniform, non-improving training loss without using validation data.

    This is a diagnostic rather than a proof of collapse or a tuning criterion;
    class imbalance and non-cross-entropy objectives can change the reference.
    """
    values = np.asarray(epoch_losses, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or n_classes < 2:
        raise ValueError("epoch losses and class count must be finite and valid")
    return bool(values[-1] >= np.log(n_classes) - 0.05 and values[0] - values[-1] < 0.1)


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
    epoch_losses = []
    for ep in range(epochs):
        loss_sum = torch.zeros((), device=device)
        epoch_rows = 0
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
            batch_rows = batch["labels"].shape[0]
            loss_sum += loss.detach().float() * batch_rows
            epoch_rows += batch_rows
            seen += batch_rows
            if log_every and (step + 1) % log_every == 0:
                el = time.perf_counter() - t0
                print(f"    ep{ep} step{step+1}/{len(dl)} loss{loss.item():.3f} "
                      f"{seen/el:.1f} seq/s", flush=True)
        if not epoch_rows:
            raise ValueError("training epoch contained no examples")
        epoch_losses.append(float((loss_sum / epoch_rows).item()))
        print(f"    ep{ep} mean training loss {epoch_losses[-1]:.4f}", flush=True)
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
             "n_train": len(train_texts), "epochs": epochs, "epoch_mean_loss": epoch_losses,
             "training_loss_near_uniform_warning": training_loss_warning(epoch_losses)}
    return proba, model, stats


def to_scores(proba):
    """Log-probabilities as decision scores (argmax-equivalent; fold-aligned)."""
    return np.log(np.clip(proba, 1e-9, 1.0)).astype(np.float32)


# -- orchestration ---------------------------------------------------------
def run_cv(df, folds, pool, *, cfg, device, tokenizer, augment: bool,
           pool_scope: str = "global", prompt_mapping=None, mapping_provenance=None):
    if augment:
        if pool_scope == "fold":
            raise ValueError("transductive per-fold pool CV is disabled (D026): "
                             "held-out-text filtering changes training data; use pool_scope='global' (D028)")
        if pool is None:
            raise ValueError("augment=True requires a globally-decontaminated pool")
    texts = df["text"].tolist()
    y = df["model"].map(LABEL_TO_INDEX).to_numpy()
    labels = df["model"].astype(str).to_numpy()
    scores = np.full((len(df), N_CLASSES), -1e9, dtype=np.float32)
    fold_metrics, decon, fold_stats = [], [], []
    for fold, (tr, va) in enumerate(folds):
        fit_texts = [texts[i] for i in tr]
        fit_y = y[tr].tolist()
        if augment:
            # ``pool`` was screened ONCE against the full official train (D028), so the
            # eligible rows are identical for every fold; the cap only varies which of
            # those eligible rows enter the compute budget, never what is held out.
            pool_keep = cap_pool(pool, cfg, cfg["seed"] + fold)
            decon.append({"fold": fold, "n_pool_clean": int(len(pool)),
                          "n_capped": int(len(pool_keep)),
                          "pool_scope": "global_full_official_train",
                          "validation_independent": True})
            fit_texts = fit_texts + pool_keep["text"].tolist()
            fit_y = fit_y + pool_keep["model"].map(LABEL_TO_INDEX).tolist()
            print(f"  [fold {fold}] +pool(global clean) capped {len(pool_keep)} "
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
        cfg["max_pool_rows_per_fold"] * len(folds) * cfg["epochs_pool"]
    est_off_h = off_rows / sps / 3600
    est_pool_h = pool_rows / sps / 3600
    report = {"seq_per_s": sps, "peak_gpu_gb": stats["peak_gpu_gb"],
              "est_official_hours": round(est_off_h, 3), "est_pool_hours": round(est_pool_h, 3),
              "est_total_hours": round(est_off_h + est_pool_h, 3), "budget_hours": 9.0,
              "within_budget": (est_off_h + est_pool_h) < 8.5}
    print("[benchmark]", json.dumps(report), flush=True)
    return report


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=sorted(BACKBONES), default=BACKBONE,
                    help="preset checkpoint; on a Kaggle script kernel edit the module-level "
                    "BACKBONE constant instead, since no CLI args are passed")
    ap.add_argument("--model-name", default=None,
                    help="explicit HF checkpoint id; overrides --backbone")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs-official", type=int, default=3)
    ap.add_argument("--epochs-pool", type=int, default=2)
    ap.add_argument("--pool-cap-per-fold", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cos-threshold", type=float, default=0.3)
    ap.add_argument("--strict-pool", action="store_true")
    ap.add_argument("--prompt-mapping", default=None,
                    help="reviewed JSON crosswalk with mapping and evidence provenance; "
                    "only required for the transductive --pool-scope fold diagnostics (D026)")
    ap.add_argument("--pool-balance", type=int, default=0,
                    help="cap every class to <=N pool rows (0 = proportional cap). "
                    "Targets the gemini-3.7-flash imbalance (D023).")
    ap.add_argument("--pool-scope", choices=("global", "fold"), default="global",
                    help="global = screen the pool once against the full official train "
                    "(non-transductive, D028, the default); fold = per-fold held-out "
                    "screening (transductive, DISABLED under D026, diagnostics only).")
    ap.add_argument("--no-pool", action="store_true", help="official-only ablation")
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--out", default="/kaggle/working/deberta_cv")
    ap.add_argument("--save-final", action="store_true", help="also fit a final model on all official+pool")
    return ap


def resolve_model_name(backbone: str, model_name: str | None = None) -> str:
    """Explicit checkpoint id wins; otherwise the preset decides."""
    if model_name:
        return model_name
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[backbone]


def main():
    args = build_arg_parser().parse_args()
    args.model_name = resolve_model_name(args.backbone, args.model_name)

    if not args.no_pool and args.pool_scope == "fold":
        raise SystemExit("[abort] transductive per-fold pool CV is disabled (D026): "
                         "held-out-text filtering changes training data; use --pool-scope global (D028) or --no-pool")
    if args.pool_scope == "fold" and not args.no_pool and not args.prompt_mapping:
        raise SystemExit("[abort] verified --prompt-mapping is required for per-fold pool diagnostics")

    cfg = {k: getattr(args, k) for k in
           ("backbone", "model_name", "max_len", "batch_size", "epochs_official", "epochs_pool",
            "pool_cap_per_fold", "pool_balance", "lr", "seed", "cos_threshold", "pool_scope")}
    cfg["fold_seed"] = FOLD_SEED
    cfg["model_seed"] = args.seed
    out_dir = Path(args.out)
    (out_dir / "oof").mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir()
    df = load_official(data_dir)
    shipped = Path(__file__).resolve().parent / "folds.json"
    if EMBEDDED_FOLD_IDS.get(FOLD_SEED):
        folds, method = embedded_folds(df, FOLD_SEED)
        print(f"[data] using embedded local folds ({method}) — aligned to local {FOLD_SEED}", flush=True)
    elif shipped.is_file():
        folds, method = load_shipped_folds(shipped, df)
        print(f"[data] using shipped local folds ({method}) — aligned to local cv_s42", flush=True)
    else:
        folds, method = make_grouped_folds(df, args.n_folds, args.seed)
        print("[warn] no embedded/shipped folds; recomputed with Kaggle sklearn "
              "(may differ from local cv_s42)", flush=True)
    print(f"[data] official {len(df)} rows | {method} folds={len(folds)} | data_dir={data_dir}", flush=True)
    ordered_sha = ordered_dataset_fingerprint(df)
    fold_ids = fold_assignments(len(df), folds)

    pool = None
    global_decon = None
    if args.no_pool:
        print("[data] --no-pool: official-only ablation", flush=True)
    else:
        pool_path = find_pool()
        if pool_path is None:
            print("[warn] no ntl_open_pool.parquet attached; running official-only", flush=True)
        else:
            pool = load_pool(pool_path, strict=args.strict_pool)
            print(f"[data] pool {len(pool)} rows from {pool_path}", flush=True)

    # Validate the pool crosswalk before installing or loading model dependencies.
    ensure_deps()
    import torch
    from transformers import AutoTokenizer

    if pool is not None:
        # D028: screen the pool ONCE against the complete official training set, so the
        # eligible rows are identical for every fold and no held-out passage can select
        # which pool rows get trained on (that was the D026 transductive leak).
        keep, global_decon = decontaminate_global(pool, df, cos_threshold=args.cos_threshold)
        pool = pool.loc[keep].reset_index(drop=True)
        print(f"[pool] global decon {global_decon['n_pool']} -> clean {global_decon['kept']} "
              f"(exact {global_decon['exact_dup']} + near {global_decon['near_dup']} removed), "
              f"fold-independent", flush=True)
    pool_cap = args.pool_balance * N_CLASSES if args.pool_balance > 0 else args.pool_cap_per_fold
    cfg["max_pool_rows_per_fold"] = min(len(pool), pool_cap) if pool is not None and pool_cap > 0 else (len(pool) if pool is not None else 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if device.type=='cuda' else 'none'}", flush=True)
    if device.type == "cpu":
        print("[warn] no GPU — this kernel is intended for Kaggle GPU; CPU will be very slow.", flush=True)
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

    results = {"config": cfg, "split_method": method, "ordered_dataset_sha256": ordered_sha, "n_official": len(df),
               "n_pool": int(len(pool)) if pool is not None else 0,
               "decontamination_global": global_decon, "benchmark": bench}

    print("\n=== official-only CV ===", flush=True)
    off_scores, off_pooled, off_folds, _decon, off_stats = run_cv(
        df, folds, None, cfg=cfg, device=device, tokenizer=tokenizer, augment=False)
    np.savez_compressed(out_dir / "oof" / "official_only.npz", scores=off_scores,
                        y=df["model"].map(LABEL_TO_INDEX).to_numpy(),
                        labels=np.array(LABELS, dtype=object),
                        ordered_dataset_sha256=np.asarray(ordered_sha),
                        validation_fold_ids=fold_ids)
    results["official_only"] = {"pooled": off_pooled, "folds": off_folds, "stats": off_stats}
    print(f"[official-only] pooled BA {off_pooled['balanced_accuracy']:.4f}", flush=True)

    if pool is not None:
        print("\n=== official+pool CV ===", flush=True)
        pool_scores, pool_pooled, pool_folds, decon, pool_stats = run_cv(
            df, folds, pool, cfg=cfg, device=device, tokenizer=tokenizer, augment=True,
            pool_scope=args.pool_scope)
        np.savez_compressed(out_dir / "oof" / "official_pool.npz", scores=pool_scores,
                            y=df["model"].map(LABEL_TO_INDEX).to_numpy(),
                            labels=np.array(LABELS, dtype=object),
                            ordered_dataset_sha256=np.asarray(ordered_sha),
                            validation_fold_ids=fold_ids)
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
            # ``pool`` is already the globally-decontaminated, fold-independent set (D028);
            # the final fit only applies the compute-budget cap, no prompt crosswalk.
            pool_final = cap_pool(pool, cfg, args.seed)
            fit_texts += pool_final["text"].tolist()
            fit_y += pool_final["model"].map(LABEL_TO_INDEX).tolist()
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
        (final_dir / "inference_manifest.json").write_text(json.dumps({
            "schema_version": "deberta-inference/v1",
            "class_order": list(LABELS),
            "backbone": args.backbone,
            "model_name": args.model_name,
            "max_len": int(args.max_len),
            "batch_size": int(args.batch_size),
        }, indent=2), encoding="utf-8")
        print(f"[final] saved weights to {final_dir} | {stats}", flush=True)


if __name__ == "__main__":
    main()
