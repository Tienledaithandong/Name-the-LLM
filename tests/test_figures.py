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
            ordered_dataset_sha256=np.asarray("a" * 64),
            validation_fold_ids=np.arange(n, dtype=np.int64) % 5,
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


def _bands(seed=0):
    from src.metrics import length_band_metrics

    rng = np.random.default_rng(seed)
    truth = np.asarray(LABELS, dtype=object)[np.arange(36) % N_CLASSES]
    pred = truth.copy()
    wrong = rng.random(36) < 0.4
    pred[wrong] = np.asarray(LABELS, dtype=object)[(np.arange(36) + 1) % N_CLASSES][wrong]
    lengths = rng.integers(5, 400, size=36)
    return length_band_metrics(truth, pred, lengths, edges=(0, 100, 200))


def test_make_length_band_figure_writes_one_png_per_label(tmp_path):
    from pathlib import Path

    from src.figures import make_length_band_figure

    bands = {"blend_a": _bands(0), "blend_b": _bands(1)}
    path = make_length_band_figure(bands, tmp_path)
    assert Path(path).name == "length_band_accuracy.png"
    assert Path(path).is_file() and Path(path).stat().st_size > 0
    labelled = make_length_band_figure(bands, tmp_path, dataset_label="pool_mix_cap40k")
    assert Path(labelled).name == "length_band_accuracy_pool_mix_cap40k.png"


def test_make_length_band_figure_rejects_empty_or_mismatched_bands(tmp_path):
    import pytest

    from src.figures import make_length_band_figure
    from src.metrics import length_band_metrics

    with pytest.raises(ValueError, match="at least one variant"):
        make_length_band_figure({}, tmp_path)
    truth = np.asarray(LABELS, dtype=object)[np.arange(36) % N_CLASSES]
    other = length_band_metrics(truth, truth, np.arange(36) * 10, edges=(0, 50))
    with pytest.raises(ValueError, match="same length bands"):
        make_length_band_figure({"a": _bands(), "b": other}, tmp_path)


def _panel_metrics():
    return {
        "char_official": {
            "pooled": {"balanced_accuracy": 0.32},
            "fold_aggregate": {"mean": 0.32, "std": 0.01, "fold_scores": [0.31, 0.33, 0.32]},
            "fit_seconds": 60.0,
            "peak_ram_gb": 0.2,
        },
        "char_pool": {
            "pooled": {"balanced_accuracy": 0.41},
            "fold_aggregate": {"mean": 0.41, "std": 0.02, "fold_scores": [0.40, 0.44, 0.39]},
            "fit_seconds": 1600.0,
            "peak_ram_gb": None,
        },
    }


def test_panel_functions_render_from_a_metrics_mapping(tmp_path):
    from pathlib import Path

    from src.figures import (
        make_ablation_figure,
        make_cv_variation_figure,
        make_runtime_ram_figure,
    )

    metrics = _panel_metrics()
    paths = [
        make_ablation_figure(metrics, tmp_path, dataset_label="candidate"),
        make_runtime_ram_figure(metrics, tmp_path, dataset_label="candidate"),
        make_cv_variation_figure(metrics, tmp_path, dataset_label="candidate"),
    ]
    assert [Path(p).name for p in paths] == [
        "ablation_candidate.png",
        "runtime_ram_candidate.png",
        "cv_variation_candidate.png",
    ]
    for path in paths:
        assert Path(path).is_file() and Path(path).stat().st_size > 0


def test_runtime_ram_figure_tolerates_unrecorded_peak_ram(tmp_path):
    # The pool arm's fit RSS was never recorded; it must render as "not recorded"
    # rather than crash or be silently drawn as a zero-size bubble.
    from pathlib import Path

    from src.figures import make_runtime_ram_figure

    metrics = _panel_metrics()
    assert metrics["char_pool"]["peak_ram_gb"] is None
    path = make_runtime_ram_figure(metrics, tmp_path)
    assert Path(path).is_file() and Path(path).stat().st_size > 0
    del metrics["char_official"]["peak_ram_gb"]
    assert Path(make_runtime_ram_figure(metrics, tmp_path)).is_file()


def test_panel_functions_reject_an_empty_mapping(tmp_path):
    import pytest

    from src.figures import (
        make_ablation_figure,
        make_cv_variation_figure,
        make_runtime_ram_figure,
    )

    for fn in (make_ablation_figure, make_runtime_ram_figure, make_cv_variation_figure):
        with pytest.raises(ValueError, match="at least one model"):
            fn({}, tmp_path)


def test_make_prediction_figures_writes_confusion_and_recall(tmp_path):
    from pathlib import Path

    from src.figures import make_prediction_figures

    truth = np.asarray(LABELS, dtype=object)[np.arange(36) % N_CLASSES]
    pred = np.roll(truth, 1)
    written = make_prediction_figures(
        truth, pred, tmp_path, dataset_label="candidate", blend_name="0.5/0.5"
    )
    assert {Path(p).name for p in written} == {
        "confusion_matrix_candidate.png",
        "per_class_recall_candidate.png",
    }
    for path in written:
        assert Path(path).is_file() and Path(path).stat().st_size > 0
