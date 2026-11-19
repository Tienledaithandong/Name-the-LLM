"""Ensembling must reject missing or inconsistent OOF provenance."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from src.constants import LABELS
from src.ensemble import greedy_blend, load_oof


def _write(path, *, fingerprint="a" * 64, folds=(0, 1), identity=True):
    data = {"scores": np.zeros((2, len(LABELS))), "y": np.array([0, 1]),
            "labels": np.asarray(LABELS)}
    if identity:
        data.update(ordered_dataset_sha256=np.asarray(fingerprint), validation_fold_ids=np.asarray(folds))
    np.savez_compressed(path, **data)


def test_loader_rejects_legacy_oof(tmp_path) -> None:
    folder = tmp_path / "oof"
    folder.mkdir()
    _write(folder / "old.npz", identity=False)
    with pytest.raises(ValueError, match="metadata missing"):
        load_oof(tmp_path)


@pytest.mark.parametrize("fingerprint,folds", [("b" * 64, (0, 1)), ("a" * 64, (1, 0))])
def test_blender_rejects_mismatched_rows_or_folds(tmp_path, fingerprint, folds) -> None:
    folder = tmp_path / "oof"
    folder.mkdir()
    _write(folder / "one.npz")
    _write(folder / "two.npz", fingerprint=fingerprint, folds=folds)
    with pytest.raises(ValueError, match="not aligned"):
        greedy_blend(load_oof(tmp_path))


def test_explicit_members_exclude_unselected_legacy_files(tmp_path) -> None:
    folder = tmp_path / "oof"
    folder.mkdir()
    _write(folder / "selected.npz")
    _write(folder / "rejected.npz", identity=False)
    assert set(load_oof(tmp_path, members=["selected"])) == {"selected"}
    with pytest.raises(FileNotFoundError, match="missing OOF members"):
        load_oof(tmp_path, members=["selected", "not_present"])


def test_cli_writes_candidate_without_overwriting_active_weights(tmp_path, monkeypatch) -> None:
    from src import cli

    folder = tmp_path / "oof"
    folder.mkdir()
    _write(folder / "first.npz")
    _write(folder / "second.npz")
    active = tmp_path / "ensemble" / "weights.json"
    active.parent.mkdir()
    active.write_text("existing active weights", encoding="utf-8")
    monkeypatch.setattr(cli, "load_config", lambda _: {"ensemble": {"members": ["first", "second"], "min_gain": 0.002}})
    args = SimpleNamespace(config="unused", artifacts=str(tmp_path), dataset="official", out=None)
    assert cli.cmd_ensemble(args) == 0
    assert active.read_text(encoding="utf-8") == "existing active weights"
    import json
    candidate = json.loads((active.parent / "candidate_weights.json").read_text(encoding="utf-8"))
    assert candidate["min_gain"] == 0.002
    assert not candidate["meets_min_gain_on_selection_oof"]


def test_ensemble_parser_defaults_to_guarded_member_config() -> None:
    from src.cli import build_parser

    args = build_parser().parse_args(["ensemble"])
    assert args.config == "configs/ensemble.yaml"


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -1.0])
def test_inference_rejects_invalid_blend_weights(tmp_path, value) -> None:
    import json
    from src.inference import load_weights

    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"model": value}), encoding="utf-8")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        load_weights(path)
