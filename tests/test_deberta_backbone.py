"""The Kaggle encoder kernel's backbone switch must work without CLI args.

A Kaggle *script* kernel runs the code file with no arguments, so the module-level
``BACKBONE`` constant is the only reachable control on a pushed run (the same
constraint that forced the folds to be embedded, D023). These tests pin that path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _trainer():
    path = Path(__file__).resolve().parents[1] / "kaggle" / "deberta" / "deberta_train.py"
    spec = importlib.util.spec_from_file_location("deberta_train_backbone", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_backbone_is_the_measured_deberta_run() -> None:
    trainer = _trainer()
    assert trainer.BACKBONE == "deberta-v3-large"
    args = trainer.build_arg_parser().parse_args([])
    assert args.backbone == trainer.BACKBONE
    assert args.model_name is None
    assert trainer.resolve_model_name(args.backbone, args.model_name) == "microsoft/deberta-v3-large"


def test_module_constant_is_what_a_no_arg_run_resolves_to(monkeypatch) -> None:
    trainer = _trainer()
    monkeypatch.setattr(trainer, "BACKBONE", "modernbert-large")
    # The parser reads the constant at call time, so editing it changes a pushed run.
    args = trainer.build_arg_parser().parse_args([])
    assert args.backbone == "modernbert-large"
    assert trainer.resolve_model_name(args.backbone, args.model_name) == "answerdotai/ModernBERT-large"


def test_explicit_model_name_overrides_the_preset() -> None:
    trainer = _trainer()
    assert trainer.resolve_model_name("deberta-v3-large", "org/other") == "org/other"
    args = trainer.build_arg_parser().parse_args(["--model-name", "org/other"])
    assert trainer.resolve_model_name(args.backbone, args.model_name) == "org/other"


def test_unknown_backbone_is_rejected_not_silently_defaulted() -> None:
    trainer = _trainer()
    with pytest.raises(ValueError, match="unknown backbone"):
        trainer.resolve_model_name("not-a-backbone")
    with pytest.raises(SystemExit):
        trainer.build_arg_parser().parse_args(["--backbone", "not-a-backbone"])


def test_every_preset_is_a_huggingface_id_and_the_probe_keeps_the_budget() -> None:
    trainer = _trainer()
    assert set(trainer.BACKBONES) == {"deberta-v3-large", "modernbert-large"}
    for name, repo in trainer.BACKBONES.items():
        assert "/" in repo and " " not in repo, name
    # H8 must change only the backbone: the D038 budget stays fixed across presets.
    for backbone in trainer.BACKBONES:
        args = trainer.build_arg_parser().parse_args(["--backbone", backbone])
        assert (args.max_len, args.batch_size, args.lr) == (384, 4, 1e-5)
        assert args.pool_scope == "global" and args.pool_cap_per_fold == 40000


def test_modernbert_requires_a_newer_transformers_than_the_old_floor() -> None:
    req = (
        Path(__file__).resolve().parents[1] / "kaggle" / "deberta" / "requirements_gpu.txt"
    ).read_text(encoding="utf-8")
    line = next(l for l in req.splitlines() if l.startswith("transformers>="))
    minor = int(line.split(">=")[1].split(".")[1])
    assert minor >= 48, "ModernBERT was added in transformers 4.48"


def test_every_preset_checkpoint_is_registered_with_a_licence() -> None:
    """A selectable backbone must already have a provenance row (D054 register gap)."""
    register = (
        Path(__file__).resolve().parents[1] / "docs" / "external_data_register.md"
    ).read_text(encoding="utf-8")
    rows = [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in register.splitlines()
        if line.startswith("| ")
    ]
    rows = [r for r in rows if len(r) == 10]
    for name, repo in _trainer().BACKBONES.items():
        url = f"https://huggingface.co/{repo}"
        matches = [
            r for r in rows
            if r[1].rstrip("/") == url or r[1].startswith(f"{url}/commit/")
        ]
        assert len(matches) == 1, f"{name} needs exactly one register row for {url}"
        assert matches[0][2] in {"MIT", "Apache-2.0"}, f"{name}: licence not recorded"