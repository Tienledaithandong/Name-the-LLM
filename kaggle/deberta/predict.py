"""Offline inference for the DeBERTa attribution model (internet-disabled).

Internet-disabled graded notebook. Loads the final model saved by
``kaggle/deberta/deberta_train.py --save-final`` from a LOCAL attached folder,
never downloading. Tokenizes, runs inference in chunks, maps logits to labels,
and writes ``submission.parquet`` using the project's fail-fast submission
checks (``src.submission``).

Expected attached layout (from ``kaggle datasets create`` or the Kaggle input
mount), e.g.:
    model_dir/
        config.json
        model.safetensors        (or pytorch_model.bin)
        tokenizer.json
        tokenizer_config.json
        vocab.json
        label_order.json

Without ``label_order.json`` the official ``src/constants.LABELS`` order is used.

Run (Kaggle GPU, internet ON to fetch the checkpoint, then OFF for grading):
    python kaggle/deberta/deberta_train.py --save-final --model-name microsoft/deberta-v3-base
    # attach /kaggle/working/deberta_cv/final_model as a dataset, then offline:
    python kaggle/deberta/predict.py --model-dir /kaggle/working/deberta_cv/final_model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Force offline providers on import: the competition is internet-disabled and
# there must be no runtime downloads (Code Requirements).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.constants import LABELS  # noqa: E402
from src.submission import (  # noqa: E402
    SubmissionError,
    build_submission,
    validate_submission,
)

DEFAULT_TEST = "/kaggle/input/competitions/name-that-llm/test.csv"
DEFAULT_OUT = "/kaggle/working/submission.parquet"
DEFAULT_MODEL_DIR = "/kaggle/working/deberta_cv/final_model"

REPO_ROOT = Path(__file__).resolve().parents[2]

def find_parent(path: str) -> Path:
    """Resolve *path* (file or dir) to a directory for attached-model/test use.

    Absolute paths win over repo-relative resolution. Returns a directory:
    an existing file input resolves to its parent, an existing directory input
    resolves to itself, and an unknown path resolves to the input unchanged
    (so ``main()`` fails fast with a ``SubmissionError`` rather than silently
    running against a missing artifact).
    """
    p = Path(path)
    if p.is_absolute():
        if p.is_dir():
            return p
        if p.is_file():
            return p.parent
        return p
    for base in (REPO_ROOT, Path("/kaggle/input"), Path.cwd()):
        cand = base / p
        if cand.is_dir():
            return cand
        if cand.is_file():
            return cand.parent
    return p


def find_test_dir(test: str | None) -> Path:
    """Locate the folder containing test.csv; internet-disabled rerun mounts it."""
    pats = [DEFAULT_TEST, "data/test.csv", "test.csv"]
    explicit = Path(test).expanduser() if test else None
    candidates = [explicit] if explicit is not None else pats
    for cand in candidates:
        if cand is None:
            continue
        if cand.is_file():
            return cand.parent
        if cand.is_dir() and (cand / "test.csv").is_file():
            return cand
    raise FileNotFoundError(
        "test.csv not found under the usual mounts; run the Kaggle notebook "
        "or pass --test <path-to-folder>"
    )


def load_label_order(model_dir: Path) -> list[str]:
    """Read the stored label order; fall back to the official constants order."""
    order_file = model_dir / "label_order.json"
    if order_file.is_file():
        try:
            stored = json.loads(order_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return list(LABELS)
        if isinstance(stored, list) and stored and all(isinstance(x, str) for x in stored):
            return [str(x) for x in stored]
    return list(LABELS)


def build(model_dir: Path, device) -> tuple:
    """Load model + tokenizer from a local dir (offline; no download)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    return model.to(device), tokenizer


def infer(
    model, tokenizer, texts, device, max_len: int, batch_size: int
) -> np.ndarray:
    """Chunked log-probability scores (argmax-equivalent), fp32 via autocast."""
    import torch

    from torch.utils.data import DataLoader, Dataset
    from transformers import DataCollatorWithPadding

    class _TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len):
            self.enc = tokenizer(
                list(texts), truncation=True, max_length=max_len,
                padding=False, return_tensors=None,
            )
            self.texts = list(texts)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}

    def _collate(batch):
        coll = DataCollatorWithPadding(tokenizer, padding="longest")
        return coll(batch)

    model.eval()
    probs = []
    t0 = time.perf_counter()
    dl = DataLoader(_TextDataset(texts, tokenizer, max_len),
                    batch_size=batch_size, shuffle=False,
                    collate_fn=_collate, drop_last=False)
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    elapsed = time.perf_counter() - t0
    return np.log(np.clip(np.concatenate(probs, axis=0), 1e-9, 1.0)).astype(np.float32), elapsed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", default=None,
                   help="local folder of the saved DeBERTa model (default: env "
                   "NTL_MODEL_DIR or %s)" % DEFAULT_MODEL_DIR)
    p.add_argument("--test", default=None,
                   help="folder containing test.csv (default: %s)" % DEFAULT_TEST)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default=None,
                   help="output submission.parquet (default: %s)" % DEFAULT_OUT)
    p.add_argument("--checkpoint-version", default=None,
                   help="optional version label for the report line")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    model_dir = Path(args.model_dir) if args.model_dir else Path(
        os.environ.get("NTL_MODEL_DIR", DEFAULT_MODEL_DIR))
    try:
        model_dir = find_parent(str(model_dir))
    except FileNotFoundError:
        raise SubmissionError(f"model folder not found: {model_dir}") from None

    if not model_dir.is_dir():
        raise SubmissionError(f"model folder not found: {model_dir}")

    test_dir = find_test_dir(args.test)
    test_path = test_dir / "test.csv"
    if not test_path.is_file():
        raise SubmissionError(f"test.csv not found in {test_dir}")
    test_df = pd.read_csv(test_path)
    if "id" not in test_df.columns:
        raise SubmissionError("test.csv has no 'id' column")
    # Fail fast on the same checks the graded rerun enforces.
    required = {"id", "text"}
    if not required.issubset(test_df.columns):
        raise SubmissionError(f"test.csv missing columns {required - set(test_df.columns)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[env] no GPU detected; running on CPU (slower).", flush=True)
    model, tokenizer = build(model_dir, device)

    label_order = load_label_order(model_dir)
    if label_order == list(LABELS):
        print("[model] official 18-label order (src/constants.LABELS).", flush=True)
    else:
        print(f"[model] stored label order ({len(label_order)} labels).", flush=True)

    texts = test_df["text"].astype(str).tolist()
    scores, infer_seconds = infer(model, tokenizer, texts, device,
                                  args.max_len, args.batch_size)
    pred_idx = np.argmax(scores, axis=1)
    predictions = np.asarray(label_order, dtype=object)[pred_idx].tolist()

    from src.repro import env_snapshot

    out = Path(args.out) if args.out else Path(
        os.environ.get("NTL_SUBMISSION", DEFAULT_OUT))
    out = out if out.name == "submission.parquet" else out.parent / "submission.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_submission(test_df["id"], predictions, out, expected_ids=test_df["id"])
    validate_submission(out, expected_ids=test_df["id"])

    dist = {label: int((np.asarray(predictions) == label).sum())
            for label in set(predictions)}
    print(f"[predict] wrote {out} ({len(predictions)} rows, "
          f"{len(set(predictions))} distinct labels, {infer_seconds:.1f}s inference)")
    print(f"[predict] device={device} rss_gb={env_snapshot().get('rss_gb')}MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
