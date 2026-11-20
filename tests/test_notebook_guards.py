"""The submit notebook must not silently discard weighted model members."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.constants import LABELS
from src.inference import load_weights


def test_notebook_rejects_weights_with_missing_bundle(tmp_path) -> None:
    notebook = json.loads((Path(__file__).resolve().parents[1] / "kaggle" / "submission.ipynb").read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c["cell_type"] == "code" and "weights_path" in "".join(c["source"]))
    source = "".join(cell["source"])
    (tmp_path / "weights.json").write_text(json.dumps({"known": 0.5, "missing": 0.5}), encoding="utf-8")
    scope = {"load_final": lambda _: ({"known": object()}, {"class_order": list(LABELS)}),
             "load_weights": load_weights, "ARTIFACT_DIR": tmp_path, "LABELS": LABELS, "json": json}
    with pytest.raises(ValueError, match="missing model bundles"):
        exec(compile(source, "submission.ipynb", "exec"), scope)


def test_predict_cli_rejects_weights_for_missing_bundle(tmp_path, monkeypatch) -> None:
    from src import cli

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "manifest.json").write_text("{}", encoding="utf-8")
    weights = tmp_path / "weights.json"
    weights.write_text(json.dumps({"known": 0.5, "missing": 0.5}), encoding="utf-8")
    monkeypatch.setattr(cli, "load_config", lambda _: {"models": []})
    monkeypatch.setattr(cli, "load_data", lambda *_: ([], [], None, "official"))
    monkeypatch.setattr(cli, "load_final", lambda _: ({"known": object()}, {}))
    args = SimpleNamespace(config="unused", data_dir="unused", synthetic=False,
                           final_dir=str(model_dir), fit=False, weights=str(weights), artifacts=None)
    with pytest.raises(ValueError, match="weights reference models absent"):
        cli.cmd_predict(args)


def test_notebook_rejects_ambiguous_artifact_manifests(monkeypatch) -> None:
    import glob

    notebook = json.loads((Path(__file__).resolve().parents[1] / "kaggle" / "submission.ipynb").read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c["cell_type"] == "code" and "ARTIFACT_CANDIDATES" in "".join(c["source"]))
    monkeypatch.setattr(glob, "glob", lambda *args, **kwargs: ["one/manifest.json", "two/manifest.json"])
    with pytest.raises(ValueError, match="exactly one artifact manifest"):
        exec(compile("".join(cell["source"]), "submission.ipynb", "exec"), {})
