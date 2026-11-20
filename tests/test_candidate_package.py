"""Synthetic-only checks for the offline candidate packager."""
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
import sklearn

from src.constants import LABELS

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("candidate_package", ROOT / "kaggle" / "deberta" / "package_candidate.py")
packager = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(packager)


@pytest.fixture
def sources(tmp_path):
    model = tmp_path / "model"
    sparse = tmp_path / "sparse"
    model.mkdir()
    (sparse / "models").mkdir(parents=True)
    meta = {"schema_version": "deberta-inference/v1", "class_order": list(LABELS),
            "model_name": "microsoft/deberta-v3-large", "max_len": 384, "batch_size": 4}
    (model / "inference_manifest.json").write_text(json.dumps(meta), encoding="utf-8")
    (model / "label_order.json").write_text(json.dumps(list(LABELS)), encoding="utf-8")
    (model / "config.json").write_text(json.dumps({"model_type": "deberta-v2"}), encoding="utf-8")
    (model / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"synthetic-test-not-real-weights")
    (model / "submission.parquet").write_bytes(b"synthetic-should-not-ship")
    (model / "train.csv").write_text("synthetic-should-not-ship", encoding="utf-8")
    names = ("char_tfidf_logreg", "style_only")
    entries = []
    for name in names:
        rel = f"models/{name}.joblib"
        (sparse / rel).write_bytes(b"synthetic-not-real-model")
        entries.append({"name": name, "path": rel, "config": packager.EXPECTED_SPARSE_CONFIGS[name]})
    manifest = {"dataset": "official", "class_order": list(LABELS),
                "seed": 42, "environment": {"sklearn": sklearn.__version__}, "models": entries}
    (sparse / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (sparse / "oof.npz").write_bytes(b"synthetic-should-not-ship")
    return model, sparse


def test_refuses_small_dummy_weights_and_keeps_output_absent(sources, tmp_path):
    model, sparse = sources
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="dummy weights"):
        packager.package_candidate(model, sparse, output)
    assert not output.exists()


def test_package_allowlist_and_existing_output_are_safe(sources, tmp_path, monkeypatch):
    model, sparse = sources
    # Only a synthetic fixture: real callers cannot lower this production guard.
    monkeypatch.setattr(packager, "validate_model_weights", lambda _: None)
    pred = packager._prediction_module()
    monkeypatch.setattr(pred, "load_sparse_bundles", lambda _: {})
    monkeypatch.setattr(packager, "_prediction_module", lambda: pred)
    output = packager.package_candidate(model, sparse, tmp_path / "output")
    assert (output / "model" / "model.safetensors").is_file()
    assert (output / "sparse" / "models" / "style_only.joblib").is_file()
    assert (output / "src" / "inference.py").is_file()
    assert not list(output.rglob("*.parquet"))
    assert not list(output.rglob("*.npz"))
    assert not list(output.rglob("*.csv"))
    with pytest.raises(FileExistsError):
        packager.package_candidate(model, sparse, output)


def test_package_refuses_sparse_manifest_path_traversal(sources, tmp_path, monkeypatch):
    model, sparse = sources
    monkeypatch.setattr(packager, "validate_model_weights", lambda _: None)
    manifest_path = sparse / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["models"][0]["path"] = "../outside.joblib"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected sparse model path"):
        packager.package_candidate(model, sparse, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_package_can_include_only_matching_wheel(sources, tmp_path, monkeypatch):
    model, sparse = sources
    monkeypatch.setattr(packager, "validate_model_weights", lambda _: None)
    pred = packager._prediction_module()
    monkeypatch.setattr(pred, "load_sparse_bundles", lambda _: {})
    monkeypatch.setattr(packager, "_prediction_module", lambda: pred)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / f"scikit_learn-{sklearn.__version__}-cp313-cp313-manylinux_2_28_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("synthetic-wheel-metadata", "test only")
    result = packager.package_candidate(model, sparse, tmp_path / "output", wheels)
    manifest = json.loads((result / "package-manifest.json").read_text(encoding="utf-8"))
    assert (result / "wheels" / wheel.name).is_file()
    assert len(manifest["sklearn_wheel"]["sha256"]) == 64
