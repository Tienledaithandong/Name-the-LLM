from __future__ import annotations

import json

import numpy as np

from src.constants import LABELS, N_CLASSES
from src.figures import blend_predictions, make_report_figures


def _write_artifacts(root, names, n=36, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % N_CLASSES
    (root / "oof").mkdir(parents=True)
    (root / "metrics").mkdir(parents=True)
    for name in names:
        scores = rng.normal(size=(n, N_CLASSES)).astype(np.float32)
        scores[np.arange(n), y] += 1.5  # make predictions better than chance
        np.savez_compressed(
            root / "oof" / f"{name}.npz",
            scores=scores,
            y=y.astype(np.int64),
            labels=np.asarray(LABELS, dtype=object),
        )
        fold_scores = [0.3, 0.32, 0.28, 0.31, 0.29]
        payload = {
            "model": name,
            "pooled": {"balanced_accuracy": 0.30, "macro_f1": 0.29},
            "fold_aggregate": {
                "mean": 0.30,
                "std": 0.014,
                "fold_scores": fold_scores,
            },
            "fit_seconds": 1.0,
            "infer_seconds": 0.1,
            "peak_ram_gb": 0.2,
        }
        (root / "metrics" / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return y


def test_make_report_figures_writes_all_five(tmp_path):
    names = ["model_a", "model_b"]
    _write_artifacts(tmp_path, names)
    out = tmp_path / "figs"
    written = make_report_figures(
        tmp_path, out, weights={"model_a": 0.5, "model_b": 0.5}, dataset_label="synthetic"
    )
    assert len(written) == 5
    stems = {p.split("/")[-1].split("\\")[-1] for p in written}
    assert stems == {
        "confusion_matrix_synthetic.png",
        "per_class_recall_synthetic.png",
        "ablation_synthetic.png",
        "runtime_ram_synthetic.png",
        "cv_variation_synthetic.png",
    }
    for path in written:
        from pathlib import Path

        assert Path(path).is_file() and Path(path).stat().st_size > 0


def test_blend_predictions_falls_back_to_best_single(tmp_path):
    _write_artifacts(tmp_path, ["model_a", "model_b"])
    truth, pred, used, name = blend_predictions(tmp_path, weights=None)
    assert len(truth) == len(pred) == 36
    assert sum(used.values()) == 1.0
    assert name in {"model_a", "model_b"}


def test_blend_predictions_uses_requested_weights(tmp_path):
    _write_artifacts(tmp_path, ["model_a", "model_b"])
    _truth, _pred, used, name = blend_predictions(
        tmp_path, weights={"model_a": 0.5, "model_b": 0.5}
    )
    assert set(used) == {"model_a", "model_b"}
    assert name.startswith("blend(")