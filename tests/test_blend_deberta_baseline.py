"""The blend evaluator must reproduce the adopted MIX baseline from stored OOF.

D037's candidate (cap-40k pool-MIX, 0.4427 / s13 0.4304) and the official-only
fallback (0.3760) are both ``0.5 z(char) + 0.5 z(style)`` over identity-verified
OOF. ``blend_deberta.py --no-deberta`` recomputes that bar from artifacts, so this
test pins the plumbing (member paths, row-wise-z rule, fold-alignment guard) without
needing a GPU kernel run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.constants import LABELS, N_CLASSES
from src.data import indices_to_labels
from src.ensemble import normalize_scores
from src.metrics import balanced_accuracy

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "blend_deberta.py"
SHA = "a" * 64


def _write_split(tmp_path: Path, n_rows: int, n_folds: int = 3) -> Path:
    folds = []
    for f in range(n_folds):
        valid = np.arange(f, n_rows, n_folds)
        train = np.setdiff1d(np.arange(n_rows), valid)
        folds.append({"train": train.tolist(), "valid": valid.tolist()})
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"folds": folds, "ordered_dataset_sha256": SHA}), encoding="utf-8")
    return path


def _fold_ids(n_rows: int, splits_path: Path, n_folds: int = 3) -> np.ndarray:
    ids = np.full(n_rows, -1, dtype=np.int64)
    folds = json.loads(splits_path.read_text(encoding="utf-8"))["folds"]
    for f, fold in enumerate(folds):
        ids[np.asarray(fold["valid"], dtype=int)] = f
    assert (ids >= 0).all()
    return ids


def _write_oof(tmp_path: Path, name: str, y: np.ndarray, fold_ids: np.ndarray, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(len(y), N_CLASSES))
    scores[np.arange(len(y)), y] += 2.5  # a member that is right more often than chance
    path = tmp_path / f"{name}.npz"
    np.savez_compressed(path, scores=scores.astype(np.float32), y=y,
                        labels=np.array(LABELS, dtype=object),
                        ordered_dataset_sha256=np.asarray(SHA), validation_fold_ids=fold_ids)
    return path


def _run(splits: Path, char: Path, style: Path, out: Path, extra: list[str] = ()) -> dict:
    cmd = [sys.executable, str(SCRIPT), "--no-deberta", "--splits", str(splits),
           "--char-oof", str(char), "--style-oof", str(style), "--out", str(out), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads((out / "blend_official_pool.json").read_text(encoding="utf-8"))


def test_no_deberta_recomputes_the_fixed_half_half_mix(tmp_path: Path) -> None:
    n_rows, n_folds = 90, 3
    splits = _write_split(tmp_path, n_rows, n_folds)
    fold_ids = _fold_ids(n_rows, splits, n_folds)
    y = np.tile(np.arange(N_CLASSES), 5)[:n_rows]
    char = _write_oof(tmp_path, "char", y, fold_ids, seed=1)
    style = _write_oof(tmp_path, "style", y, fold_ids, seed=2)

    payload = _run(splits, char, style, tmp_path / "out")
    report = {r["name"]: r for r in payload["reports"]}
    assert set(report) == {"mix_baseline"}, "no DeBERTa arm may be scored when it is not loaded"

    expected = balanced_accuracy(
        indices_to_labels(y),
        indices_to_labels((0.5 * normalize_scores(np.load(char)["scores"].astype(float))
                           + 0.5 * normalize_scores(np.load(style)["scores"].astype(float))
                           ).argmax(axis=1)))
    assert report["mix_baseline"]["balanced_accuracy"] == pytest.approx(expected, abs=1e-12)
    assert report["mix_baseline"]["balanced_accuracy"] > 1.0 / N_CLASSES


def test_baseline_uses_the_named_member_artifacts(tmp_path: Path) -> None:
    n_rows, n_folds = 90, 3
    splits = _write_split(tmp_path, n_rows, n_folds)
    fold_ids = _fold_ids(n_rows, splits, n_folds)
    y = np.tile(np.arange(N_CLASSES), 5)[:n_rows]
    char = _write_oof(tmp_path, "char", y, fold_ids, seed=1)
    style = _write_oof(tmp_path, "style", y, fold_ids, seed=2)
    other_style = _write_oof(tmp_path, "style_official", y, fold_ids, seed=7)

    payload = _run(splits, char, style, tmp_path / "out_a")
    swapped = _run(splits, char, other_style, tmp_path / "out_b")
    # the override actually changes which artifact enters the blend
    assert payload["style_oof"].endswith("style.npz")
    assert swapped["style_oof"].endswith("style_official.npz")
    assert (swapped["reports"][0]["balanced_accuracy"]
            != payload["reports"][0]["balanced_accuracy"])


def test_misaligned_folds_are_rejected(tmp_path: Path) -> None:
    n_rows, n_folds = 90, 3
    splits = _write_split(tmp_path, n_rows, n_folds)
    fold_ids = _fold_ids(n_rows, splits, n_folds)
    y = np.tile(np.arange(N_CLASSES), 5)[:n_rows]
    char = _write_oof(tmp_path, "char", y, fold_ids, seed=1)
    shifted = tmp_path / "style_shifted.npz"
    np.savez_compressed(shifted, scores=np.load(char)["scores"], y=y,
                        labels=np.array(LABELS, dtype=object),
                        ordered_dataset_sha256=np.asarray(SHA),
                        validation_fold_ids=(fold_ids + 1) % n_folds)

    cmd = [sys.executable, str(SCRIPT), "--no-deberta", "--splits", str(splits),
           "--char-oof", str(char), "--style-oof", str(shifted), "--out", str(tmp_path / "out_c")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "validation-fold assignments differ" in proc.stdout + proc.stderr
