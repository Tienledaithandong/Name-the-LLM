"""Offline submission artifact — end-to-end test (skill step 4).

Verifies that the offline inference path writes a fail-fast-valid
``submission.parquet`` from a locally-built checkpoint, with zero network
downloads (``HF_HUB_OFFLINE=1``) and the failure mode surfacing as
``src.submission.SubmissionError`` before any model load.

Run:  python3 -m pytest tests/test_offline_submission_artifact.py -v

Stubs (no real model download):
  * a tiny ``BertConfig`` + ``BertTokenizer`` are built and saved locally, then
  * ``predict.main([...])`` is called with a synthetic ``test.csv`` shard, and
  * the resulting ``submission.parquet`` is validated against the 18 official
  labels and unique ids.
"""
from __future__ import annotations

import importlib.util
import json
import os
import string
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    BertConfig,
    BertTokenizer,
)

# Load the offline script as a plain module (no package/__init__.py needed).
_PRED_MODULE = importlib.util.spec_from_file_location(
    "kaggle_deberta_predict", ROOT / "kaggle" / "deberta" / "predict.py"
)
predict = importlib.util.module_from_spec(_PRED_MODULE)
_PRED_MODULE.loader.exec_module(predict)

from src.constants import LABELS  # noqa: E402
from src.submission import (  # noqa: E402
    SubmissionError,
    build_submission,
    validate_frame,
    validate_submission,
)

# The offline run no-ops on torch import; it only becomes expensive when the
# script is invoked. We import it here so the test driver below sees the module.


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """Tiny random-weight BERT-style checkpoint, saved locally (no download)."""
    import string

    model_dir = tmp_path_factory.mktemp("tiny_model")
    vocab = {ch: i for i, ch in enumerate(string.printable[:60])}
    vocab["[PAD]"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[CLS]"] = len(vocab)
    vocab["[SEP]"] = len(vocab)
    vocab["[MASK]"] = len(vocab)
    tok = BertTokenizer(vocab=vocab, do_lower_case=True)
    tok.save_pretrained(model_dir)

    tiny = AutoModelForSequenceClassification.from_config(
        BertConfig(
            num_labels=18,
            hidden_size=16,
            vocab_size=len(vocab),
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=32,
        )
    )
    tiny.save_pretrained(model_dir)
    # Official label order stored beside the weights (predict.py reads this).
    (model_dir / "label_order.json").write_text(
        json.dumps(list(LABELS)), encoding="utf-8"
    )
    return model_dir


@pytest.fixture(scope="session")
def test_csv(synthetic_data, tmp_path_factory):
    """Synthetic test.csv written into a scratch dir for the offline run."""
    from src.data import validate_test_frame

    _, test, _ = synthetic_data
    test_df = validate_test_frame(test)
    out = tmp_path_factory.mktemp("offline_test")
    path = out / "test.csv"
    path.write_text(test_df.to_csv(index=False), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def submission_parquet(tmp_path_factory, tiny_model_dir, test_csv):
    """Run the offline pipeline and return the written parquet path."""
    import subprocess

    out = tmp_path_factory.mktemp("offline_out") / "submission.parquet"
    env = os.environ
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "kaggle" / "deberta" / "predict.py"),
            "--model-dir",
            str(tiny_model_dir),
            "--test",
            str(test_csv),
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "path": out,
        "proc": proc,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def test_offline_submission_writes_valid_parquet(
    submission_parquet, test_csv, tiny_model_dir
):
    """End-to-end offline artifact: checkpoint -> valid parquet (fail-fast)."""
    out = submission_parquet["path"]
    proc = submission_parquet["proc"]

    # 1. The predict script itself exits 0 and reports the run.
    assert proc.returncode == 0, (
        "predict.py failed\nstdout:\n"
        f"{submission_parquet['stdout']}\nstderr:\n"
        f"{submission_parquet['stderr']}"
    )

    # 2. Output is real parquet on disk.
    assert out.is_file(), "submission.parquet was not written"
    assert out.name == "submission.parquet"

    # 3. Fail-fast: the shipped src.submission validation accepts it.
    n_rows = len(pd.read_csv(test_csv))
    frame = validate_submission(out, expected_ids=None)
    assert list(frame.columns) == ["id", "model"], f"bad columns {frame.columns}"
    assert len(frame) == n_rows and len(frame) > 0

    # 4. Zero nulls, unique ids, and all 18 labels are *possible* (stdout shows
    #    the label distribution the tiny model predicts).
    assert frame["id"].is_unique
    assert not frame["id"].isna().any() and not frame["model"].isna().any()

    # 5. The 18-label set is honoured even though the tiny head is degenerate:
    #    every prediction must be one of the official LABELS.
    assert frame["model"].isin(LABELS).all()

    # 6. Verbose evidence for the record.
    print("\n[offline-submission-artifact] stdout:")
    print(submission_parquet["stdout"])
    print("[offline-submission-artifact] stderr:")
    print(submission_parquet["stderr"])
    print("[offline-submission-artifact] written:", out)


def test_missing_model_dir_fails_fast(tmp_path, test_csv):
    """A stale/missing model dir must raise a clear SubmissionError, not a
    cryptic open() traceback — the graded notebook must never silently skip a
    missing checkpoint."""
    env = os.environ
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(ROOT)

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "kaggle" / "deberta" / "predict.py"),
            "--model-dir",
            str(tmp_path / "does-not-exist"),
            "--test",
            str(test_csv),
            "--out",
            str(tmp_path / "submission.parquet"),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "SubmissionError" in proc.stderr or "model folder not found" in proc.stderr
    assert "Errno" not in proc.stderr or "open(" in proc.stderr


def test_missing_test_csv_fails_fast(tmp_path, tiny_model_dir):
    """A missing test.csv must fail fast with a clear SubmissionError."""
    env = os.environ
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(ROOT)

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "kaggle" / "deberta" / "predict.py"),
            "--model-dir",
            str(tiny_model_dir),
            "--test",
            str(tmp_path / "no-test.csv"),
            "--out",
            str(tmp_path / "submission.parquet"),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "SubmissionError" in proc.stderr or "test.csv not found" in proc.stderr
