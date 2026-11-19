"""Refit the verified frozen ModernBERT blend and predict offline."""

from __future__ import annotations

import argparse
import json
import os
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.config import load_config
from src.constants import DATA_DIR, LABELS, MAX_NOTEBOOK_HOURS, N_CLASSES, REPO_ROOT, SUBMISSION_FILENAME
from src.data import (
    find_competition_file,
    indices_to_labels,
    labels_to_indices,
    ordered_dataset_fingerprint,
    validate_test_frame,
    validate_train_frame,
)
from src.ensemble import SENTINEL, normalize_scores
from src.frozen_modernbert import MODEL_REPO, MODEL_REVISION, FrozenModernBert
from src.model import estimator_from_config
from src.submission import build_submission


SYNTHETIC_FILENAME = "synthetic_submission.parquet"
# D057: the exact official training snapshot used in the verified OOF runs.
OFFICIAL_TRAIN_SHA256 = "dc60875fd8b536cb8db67bdac8119132c6d0c0cf667fbca262d5ffc47f3e6e5a"
_EXPECTED_SPARSE_MODELS = [
    {"name": "char_tfidf_logreg", "features": ["char"], "classifier": "logreg", "C": 2.0,
     "max_features": 100_000},
    {"name": "style_only", "features": [], "style": True, "classifier": "logreg", "C": 1.0},
]
_EXPECTED_ENCODER = {
    "repo_id": MODEL_REPO, "model_revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION,
    "max_length": 384, "batch_size": 4, "embedding_dim": 1024, "device": "cuda", "dtype": "float16",
    "attention_implementation": "sdpa",
}


def _load_recipe() -> tuple[list[dict], dict]:
    baseline = load_config(REPO_ROOT / "configs" / "submission.yaml")
    modernbert = load_config(REPO_ROOT / "configs" / "frozen_modernbert_probe.yaml")
    if baseline["seed"] != 42 or baseline["models"] != _EXPECTED_SPARSE_MODELS:
        raise ValueError("the sparse submission config differs from the verified baseline")
    if (modernbert["encoder"] != _EXPECTED_ENCODER
            or modernbert["probe"]["blend_weight"] != 0.25
            or modernbert["probe"]["classifier_c"] != 1.0):
        raise ValueError("the encoder submission config differs from the verified recipe")
    return baseline["models"], modernbert["encoder"]


def _check_margins(values: np.ndarray, rows: int) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.shape != (rows, N_CLASSES):
        raise ValueError("score matrix does not cover the expected rows and 18 labels")
    if not np.isfinite(scores).all() or (scores < SENTINEL).any():
        raise ValueError("score matrix contains nonfinite values or missing classes")
    return scores


def blend_margins(char: np.ndarray, style: np.ndarray, encoder: np.ndarray, rows: int) -> np.ndarray:
    """Reproduce the verified row-wise z-score blend in canonical label order."""
    sparse = 0.5 * normalize_scores(_check_margins(char, rows)) + 0.5 * normalize_scores(
        _check_margins(style, rows),
    )
    return 0.75 * sparse + 0.25 * normalize_scores(_check_margins(encoder, rows))


def _synthetic_examples(rows: int) -> pd.DataFrame:
    # Generated text and IDs are unrelated to competition inputs.
    rng = np.random.default_rng(42)
    terms = tuple("a quiet window paper clock turns light and then thought garden walks river "
                  "near through because every so distance amber steady table round".split())
    return pd.DataFrame({
        "id": [f"synthetic-{i:06d}" for i in range(rows)],
        "text": [
            f"Synthetic runtime sample {i}. " + " ".join(rng.choice(terms, int(rng.integers(40, 251)))) + "."
            for i in range(rows)
        ],
    })


def _cuda_runtime() -> tuple[str, str, str, str]:
    import sklearn
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("the verified ModernBERT candidate requires CUDA")
    if sklearn.__version__ != "1.9.1" or transformers.__version__ != "5.23.0":
        raise RuntimeError("the verified recipe needs scikit-learn 1.9.1 and Transformers 5.23.0")
    if find_spec("pyarrow") is None:
        raise RuntimeError("the offline runtime needs PyArrow to validate and write Parquet")
    torch.cuda.reset_peak_memory_stats()
    return torch.cuda.get_device_name(), torch.__version__, transformers.__version__, sklearn.__version__


def _peak_gpu_gib() -> float:
    import torch

    return torch.cuda.max_memory_allocated() / 1024**3


def run_submission(
    checkpoint: str | Path,
    data_dir: str | Path,
    out: str | Path,
    *,
    synthetic_rows: int | None = None,
) -> dict:
    """Fit the three fixed members on verified training rows and emit validated predictions."""
    started = perf_counter()
    deadline = started + MAX_NOTEBOOK_HOURS * 3600
    out = Path(out)
    synthetic = synthetic_rows is not None
    expected_name = SYNTHETIC_FILENAME if synthetic else SUBMISSION_FILENAME
    if out.name != expected_name:
        raise ValueError(f"output in this mode must be named {expected_name}")
    if out.exists():
        raise FileExistsError(f"refusing to replace {out}")
    if synthetic_rows is not None and synthetic_rows < 1:
        raise ValueError("--synthetic-rows must be positive")

    sparse_configs, encoder_config = _load_recipe()
    train = validate_train_frame(pd.read_csv(find_competition_file("train.csv", data_dir)))
    fingerprint = ordered_dataset_fingerprint(train)
    if fingerprint != OFFICIAL_TRAIN_SHA256 or set(train["model"]) != set(LABELS):
        raise ValueError("training data do not match the verified official snapshot and 18 labels")
    texts = train["text"].tolist()
    y = labels_to_indices(train["model"])
    if synthetic_rows is not None:
        test = validate_test_frame(_synthetic_examples(synthetic_rows))
    else:
        test = validate_test_frame(pd.read_csv(
            find_competition_file("test.csv", data_dir), dtype={"id": str},
        ))
    if test.empty or (test["text"].str.strip() == "").any():
        raise ValueError("inference requires nonblank passages and at least one row")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    gpu, torch_version, transformers_version, sklearn_version = _cuda_runtime()
    # This run calls encode directly so neither training nor prediction embeddings
    # are written separately or reused from a cache on another machine.
    encoder = FrozenModernBert(checkpoint, out.parent, **encoder_config)
    fit_started = perf_counter()
    with threadpool_limits(limits=2):
        sparse_models = []
        for config in sparse_configs:
            member = estimator_from_config(config, seed=42).fit(texts, y)
            if tuple(member.class_order) != LABELS:
                raise ValueError("the sparse member must match the canonical label order")
            sparse_models.append(member)
        head = LogisticRegression(C=1.0, class_weight="balanced", random_state=42, max_iter=500)
        head.fit(encoder.encode(texts, deadline=deadline), y)
        if not np.array_equal(head.classes_, np.arange(N_CLASSES)):
            raise ValueError("the encoder head must match the canonical label order")
        fit_seconds = perf_counter() - fit_started
        inference_started = perf_counter()
        passages = test["text"].tolist()
        char, style = (member.decision_function(passages) for member in sparse_models)
        raw = head.decision_function(encoder.encode(passages, deadline=deadline))
        predictions = indices_to_labels(blend_margins(char, style, raw, len(test)).argmax(axis=1))
        inference_seconds = perf_counter() - inference_started
    if perf_counter() >= deadline:
        raise TimeoutError("the submission exceeded the notebook runtime limit")

    ids = test["id"].tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".modernbert-", dir=out.parent) as temporary:
        validated = build_submission(ids, predictions, Path(temporary) / out.name, expected_ids=ids)
        # Publish the validated file on the same volume without replacing any
        # output from another process.
        os.link(validated, out)

    return {
        "mode": "synthetic_inference_smoke" if synthetic else "competition_test_file",
        "inference_rows": len(test),
        "official_training_rows": len(train),
        "ordered_training_sha256": fingerprint,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "total_elapsed_seconds": perf_counter() - started,
        "peak_gpu_allocated_gib": _peak_gpu_gib(),
        "head_n_iter": head.n_iter_.tolist(),
        "head_hit_iteration_limit": bool(np.any(head.n_iter_ >= head.max_iter)),
        "gpu": gpu,
        "torch": torch_version,
        "transformers": transformers_version,
        "sklearn": sklearn_version,
        "output_file": out.name,
        "schema_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="local pinned ModernBERT snapshot")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic-rows", type=int, help=f"runtime check, writes {SYNTHETIC_FILENAME}")
    args = parser.parse_args()
    print(json.dumps(run_submission(args.checkpoint, args.data_dir, args.out,
                                    synthetic_rows=args.synthetic_rows), allow_nan=False))


if __name__ == "__main__":
    main()
