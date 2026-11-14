from __future__ import annotations

import json
from argparse import Namespace

from src.cli import cmd_package


def test_package_copies_ensemble_weights(tmp_path, monkeypatch):
    # Regression: weights live under <artifacts>/cv/<dataset>/ensemble, not
    # <final_dir>.parent/cv. A wrong path silently dropped weights.json.
    monkeypatch.setattr("src.cli.ARTIFACTS_DIR", tmp_path)
    final = tmp_path / "final" / "official"
    (final / "models").mkdir(parents=True)
    (final / "manifest.json").write_text(json.dumps({"class_order": []}), encoding="utf-8")
    (final / "models" / "m.joblib").write_bytes(b"dummy")
    weights_dir = tmp_path / "cv" / "official" / "ensemble"
    weights_dir.mkdir(parents=True)
    (weights_dir / "weights.json").write_text(
        json.dumps({"weights": {"a": 0.5, "b": 0.5}}), encoding="utf-8"
    )

    out = tmp_path / "pkg"
    args = Namespace(
        final_dir=str(final), dataset="official", out=str(out), kaggle_user="tester"
    )
    assert cmd_package(args) == 0
    copied = json.loads((out / "weights.json").read_text(encoding="utf-8"))
    assert copied["weights"] == {"a": 0.5, "b": 0.5}
    assert (out / "manifest.json").is_file()
    assert (out / "models" / "m.joblib").is_file()
    assert (out / "src" / "model.py").is_file()
    # default package ships no encoder / crossed contract
    assert not (out / "inference.json").exists()
    assert not (out / "encoder").exists()


def test_package_crossed_ships_encoder_and_inference_contract(tmp_path, monkeypatch):
    monkeypatch.setattr("src.cli.ARTIFACTS_DIR", tmp_path)
    final = tmp_path / "final" / "official"
    (final / "models").mkdir(parents=True)
    (final / "manifest.json").write_text(json.dumps({"class_order": []}), encoding="utf-8")
    (final / "models" / "m.joblib").write_bytes(b"dummy")

    encoder_src = tmp_path / "minilm_checkpoint"
    encoder_src.mkdir()
    (encoder_src / "config.json").write_text("{}", encoding="utf-8")
    (encoder_src / "model.safetensors").write_bytes(b"weights")

    out = tmp_path / "pkg"
    args = Namespace(
        final_dir=str(final), dataset="official", out=str(out), kaggle_user="tester",
        crossed=True, encoder_path=str(encoder_src),
    )
    assert cmd_package(args) == 0
    assert (out / "encoder" / "minilm" / "model.safetensors").is_file()
    inference = json.loads((out / "inference.json").read_text(encoding="utf-8"))
    assert inference["crossed"] is True
    assert inference["encoder_path"] == "encoder/minilm"
    assert inference["distance_threshold"] == 0.4
    assert "encoder/minilm" in (out / "README.md").read_text(encoding="utf-8")