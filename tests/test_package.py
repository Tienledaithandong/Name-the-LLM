from __future__ import annotations

import json
from argparse import Namespace

import pytest

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


def test_package_crossed_fails_before_writing(tmp_path):
    out = tmp_path / "pkg"
    args = Namespace(
        final_dir=str(tmp_path / "missing"), dataset="official", out=str(out),
        kaggle_user="tester", crossed=True, encoder_path=str(tmp_path / "missing_encoder"),
    )
    with pytest.raises(ValueError, match="crossed inference packaging is withdrawn"):
        cmd_package(args)
    assert not out.exists()


def _package_with_fit_modes(tmp_path, monkeypatch, fit_modes):
    monkeypatch.setattr("src.cli.ARTIFACTS_DIR", tmp_path)
    final = tmp_path / "final" / "official"
    (final / "models").mkdir(parents=True)
    (final / "manifest.json").write_text(
        json.dumps({"models": [{"name": n, "fit_mode": m} for n, m in fit_modes.items()]}),
        encoding="utf-8",
    )
    weights_dir = tmp_path / "cv" / "official" / "ensemble"
    weights_dir.mkdir(parents=True)
    (weights_dir / "weights.json").write_text(
        json.dumps({
            "weights": {"char_tfidf_logreg": 0.5, "style_only": 0.5},
            "pooled_oof_balanced_accuracy": 0.3759706686552324,
            "best_single": {"model": "char_tfidf_logreg", "balanced_accuracy": 0.3193},
            "gain_over_best_single": 0.0567,
            "singles": {"char_tfidf_logreg": 0.3193},
        }),
        encoding="utf-8",
    )
    out = tmp_path / "pkg"
    args = Namespace(
        final_dir=str(final), dataset="official", out=str(out), kaggle_user="tester"
    )
    assert cmd_package(args) == 0
    return json.loads((out / "weights.json").read_text(encoding="utf-8"))


def test_package_drops_official_only_score_for_a_pool_trained_member(tmp_path, monkeypatch):
    doc = _package_with_fit_modes(tmp_path, monkeypatch, {
        "char_tfidf_logreg": "train+global_pool(capped=40001)",
        "style_only": "train_only",
    })
    assert doc["weights"] == {"char_tfidf_logreg": 0.5, "style_only": 0.5}
    for key in ("pooled_oof_balanced_accuracy", "best_single", "gain_over_best_single", "singles"):
        assert key not in doc, f"{key} describes the official-only fit, not this package"
    assert doc["fit_modes"]["char_tfidf_logreg"] == "train+global_pool(capped=40001)"
    assert doc["cv_score_of_record"] == "experiments/registry.csv"


def test_package_keeps_official_only_score_when_no_member_uses_the_pool(tmp_path, monkeypatch):
    doc = _package_with_fit_modes(tmp_path, monkeypatch, {
        "char_tfidf_logreg": "train_only",
        "style_only": "train_only",
    })
    assert doc["pooled_oof_balanced_accuracy"] == 0.3759706686552324
    assert "fit_modes" not in doc


def _package_with_manifest(tmp_path, monkeypatch, manifest, dataset="official"):
    monkeypatch.setattr("src.cli.ARTIFACTS_DIR", tmp_path)
    final = tmp_path / "final" / dataset
    (final / "models").mkdir(parents=True)
    (final / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (final / "models" / "m.joblib").write_bytes(b"dummy")
    out = tmp_path / "pkg"
    args = Namespace(
        final_dir=str(final), dataset=dataset, out=str(out), kaggle_user="tester"
    )
    assert cmd_package(args) == 0
    metadata = json.loads((out / "dataset-metadata.json").read_text(encoding="utf-8"))
    return metadata, (out / "README.md").read_text(encoding="utf-8")


def test_package_declares_share_alike_and_credits_the_pool(tmp_path, monkeypatch):
    # D050: the char bundle is a derived adaptation of CC BY-SA pool text, so a
    # CC0 declaration would drop both the licence and the attribution.
    metadata, readme = _package_with_manifest(tmp_path, monkeypatch, {
        "dataset": "official",
        "models": [
            {"name": "char_tfidf_logreg", "fit_mode": "train+global_pool(capped=40001)"},
            {"name": "style_only", "fit_mode": "train_only"},
        ],
        "pool": {"external_data_id": "open-pool-v15"},
    })
    assert metadata["licenses"] == [{"name": "CC-BY-SA-4.0"}]
    assert "creativecommons.org/licenses/by-sa/4.0" in readme
    assert "crtproduction/name-that-llm-open-pool" in readme
    assert "open-pool-v15" in readme
    assert "char_tfidf_logreg" in readme
    assert "share-alike" in readme


def test_package_credits_official_data_without_the_pool(tmp_path, monkeypatch):
    metadata, readme = _package_with_manifest(tmp_path, monkeypatch, {
        "dataset": "official",
        "models": [{"name": "style_only", "fit_mode": "train_only"}],
    })
    assert metadata["licenses"] == [{"name": "CC-BY-SA-4.0"}]
    assert "competitions/name-that-llm" in readme
    assert "crtproduction" not in readme


def test_synthetic_fixture_package_stays_cc0(tmp_path, monkeypatch):
    metadata, readme = _package_with_manifest(
        tmp_path, monkeypatch, {"dataset": "synthetic", "models": []}, dataset="synthetic"
    )
    assert metadata["licenses"] == [{"name": "CC0-1.0"}]
    assert "crtproduction" not in readme
    assert "competitions/name-that-llm" not in readme
