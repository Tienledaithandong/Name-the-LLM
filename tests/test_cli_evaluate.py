"""`cli evaluate` scores saved OOF under the registered protocol and fails closed.

Plan §1 requires an ``evaluate`` entry point alongside audit/train/predict/package.
These tests pin its contract: pooled + per-fold metrics recomputed from identity-verified
OOF, a fixed-weight row-z blend with paired fold differences, and rejection of a
fingerprint/fold mismatch, a missing member, bad weights, or a length-band request whose
train frame does not match the splits. Nothing here fits a model or scores a test row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.cli import main
from src.constants import LABELS, N_CLASSES
from src.data import indices_to_labels
from src.ensemble import normalize_scores
from src.metrics import balanced_accuracy

SHA = "a" * 64
N_ROWS, N_FOLDS = 90, 3


def _splits(tmp_path: Path, sha: str = SHA) -> Path:
    folds = []
    for f in range(N_FOLDS):
        valid = np.arange(f, N_ROWS, N_FOLDS)
        folds.append({
            "train": np.setdiff1d(np.arange(N_ROWS), valid).tolist(),
            "valid": valid.tolist(),
        })
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({
        "folds": folds, "ordered_dataset_sha256": sha,
        "method": "stratified_group", "seed": 42, "n_splits": N_FOLDS,
    }), encoding="utf-8")
    return path


def _fold_ids(shift: int = 0) -> np.ndarray:
    ids = np.zeros(N_ROWS, dtype=np.int64)
    for f in range(N_FOLDS):
        ids[np.arange(f, N_ROWS, N_FOLDS)] = f
    return (ids + shift) % N_FOLDS


def _oof_dir(tmp_path: Path, *, sha: str = SHA, shift: int = 0, names=("char", "style")) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, N_CLASSES, size=N_ROWS)
    folder = tmp_path / "oof"
    folder.mkdir(exist_ok=True)
    for i, name in enumerate(names):
        scores = rng.normal(size=(N_ROWS, N_CLASSES))
        scores[np.arange(N_ROWS), y] += 2.0 + i
        np.savez_compressed(
            folder / f"{name}.npz", scores=scores, y=y,
            labels=np.array(LABELS, dtype=object),
            ordered_dataset_sha256=np.asarray(sha), validation_fold_ids=_fold_ids(shift),
        )
    return tmp_path, y


def _run(tmp_path: Path, artifacts: Path, splits: Path, extra: list[str]) -> dict:
    out = tmp_path / "metrics.json"
    code = main([
        "evaluate", "--dataset", "official", "--artifacts", str(artifacts),
        "--members", "char", "style", "--splits", str(splits), "--out", str(out), *extra,
    ])
    assert code == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_pooled_and_fold_metrics_are_recomputed_from_oof(tmp_path: Path) -> None:
    artifacts, y = _oof_dir(tmp_path)
    payload = _run(tmp_path, artifacts, _splits(tmp_path), [])
    truth = indices_to_labels(y)
    assert payload["n_rows"] == N_ROWS
    assert payload["fits_performed"] == 0
    assert payload["transductive"] is False
    assert payload["ordered_dataset_sha256"] == SHA
    assert payload["split_id"] == "stratified_group-s42-n3"
    with np.load(artifacts / "oof" / "char.npz", allow_pickle=True) as data:
        pred = indices_to_labels(np.asarray(data["scores"]).argmax(axis=1))
    entry = payload["results"]["char"]
    assert entry["balanced_accuracy"] == pytest.approx(balanced_accuracy(truth, pred))
    folds = _fold_ids()
    per_fold = [
        balanced_accuracy(truth[folds == k], pred[folds == k]) for k in range(N_FOLDS)
    ]
    assert entry["folds"]["fold_scores"] == pytest.approx(per_fold)
    assert entry["folds"]["mean"] == pytest.approx(float(np.mean(per_fold)))
    assert entry["fold_ci"]["n"] == N_FOLDS
    assert set(entry["per_class_recall"]) == set(LABELS)
    assert np.asarray(entry["confusion_matrix"]).shape == (N_CLASSES, N_CLASSES)
    assert "length_bands" not in entry


def test_fixed_weight_blend_is_row_z_and_reports_paired_fold_diffs(tmp_path: Path) -> None:
    artifacts, y = _oof_dir(tmp_path)
    payload = _run(tmp_path, artifacts, _splits(tmp_path), ["--weights", "0.5", "0.5"])
    matrices = {}
    for name in ("char", "style"):
        with np.load(artifacts / "oof" / f"{name}.npz", allow_pickle=True) as data:
            matrices[name] = normalize_scores(np.asarray(data["scores"], dtype=np.float64))
    expected = indices_to_labels((0.5 * matrices["char"] + 0.5 * matrices["style"]).argmax(axis=1))
    blend = payload["results"]["blend"]
    assert blend["balanced_accuracy"] == pytest.approx(
        balanced_accuracy(indices_to_labels(y), expected)
    )
    assert blend["weights"] == {"char": 0.5, "style": 0.5}
    assert "row_z_mean" in blend["blend_rule"]
    assert set(blend["paired_vs_members"]) == {"char", "style"}
    for name, diff in blend["paired_vs_members"].items():
        single = payload["results"][name]["folds"]["fold_scores"]
        assert diff["mean"] == pytest.approx(
            float(np.mean(np.asarray(blend["folds"]["fold_scores"]) - np.asarray(single)))
        )


def test_fingerprint_mismatch_is_rejected(tmp_path: Path) -> None:
    artifacts, _ = _oof_dir(tmp_path)
    with pytest.raises(ValueError, match="fingerprint differs"):
        _run(tmp_path, artifacts, _splits(tmp_path, sha="b" * 64), [])


def test_fold_assignment_mismatch_is_rejected(tmp_path: Path) -> None:
    artifacts, _ = _oof_dir(tmp_path, shift=1)
    with pytest.raises(ValueError, match="fold assignments differ"):
        _run(tmp_path, artifacts, _splits(tmp_path), [])


def test_missing_member_is_rejected(tmp_path: Path) -> None:
    artifacts, _ = _oof_dir(tmp_path, names=("char",))
    with pytest.raises(FileNotFoundError, match="missing OOF members"):
        _run(tmp_path, artifacts, _splits(tmp_path), [])


@pytest.mark.parametrize("weights", [["0.5"], ["0.5", "nan"], ["0.0", "0.0"], ["-1", "2"]])
def test_invalid_weights_are_rejected(tmp_path: Path, weights: list[str]) -> None:
    artifacts, _ = _oof_dir(tmp_path)
    with pytest.raises(ValueError, match="--weights"):
        _run(tmp_path, artifacts, _splits(tmp_path), ["--weights", *weights])


def test_length_bands_require_a_matching_train_frame(tmp_path: Path) -> None:
    artifacts, _ = _oof_dir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text(
        "id,prompt_id,text,model\n"
        "x0,p0,hello world,gpt-5.5\n"
        "x1,p1,another passage,claude-opus-5\n",
        encoding="utf-8",
    )
    (data_dir / "test.csv").write_text("id,text\nx1,hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the split fingerprint"):
        _run(tmp_path, artifacts, _splits(tmp_path), ["--length-bands", "--data-dir", str(data_dir)])