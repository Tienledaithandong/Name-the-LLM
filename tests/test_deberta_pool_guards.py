"""Kaggle standalone script enforces the same pool guards as the local code."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.pool import decontaminate_fold as local_decontaminate


def _trainer():
    path = Path(__file__).resolve().parents[1] / "kaggle" / "deberta" / "deberta_train.py"
    spec = importlib.util.spec_from_file_location("deberta_train_guards", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_rejects_unproven_mapping() -> None:
    trainer = _trainer()
    pool = pd.DataFrame({"text": ["some other sample"], "prompt_canon": ["external-x"]})
    held = pd.DataFrame({"text": ["held out example"], "prompt_id": ["official-x"]})
    with pytest.raises(ValueError, match="mapping"):
        trainer.decontaminate_fold(pool, held, fit_texts=["training document"])


def test_standalone_and_local_remove_verified_shared_prompt() -> None:
    trainer = _trainer()
    pool = pd.DataFrame({"text": ["different response", "another response", "third response"],
                         "prompt_canon": ["external-x", "external-y", None]})
    held = pd.DataFrame({"text": ["held out example"], "prompt_id": ["official-x"]})
    kw = {"fit_texts": ["training document", "training response"],
          "prompt_mapping": {"official-x": ["external-x"]},
          "mapping_provenance": "synthetic mapping"}
    local_keep, local_stats = local_decontaminate(pool, held, **kw)
    kaggle_keep, kaggle_stats = trainer.decontaminate_fold(pool, held, **kw)
    assert local_keep.tolist() == kaggle_keep.tolist() == [False, True, False]
    assert local_stats["shared_prompt"] == kaggle_stats["shared_prompt"] == 1
    assert local_stats["unmapped_pool_prompt"] == kaggle_stats["unmapped_pool_prompt"] == 1
    assert kaggle_stats["similarity_fit_scope"] == "fold_training_only"


def test_balanced_pool_sampling_limits_each_class() -> None:
    trainer = _trainer()
    pool = pd.DataFrame({"model": ["a"] * 10 + ["b"] * 2})
    chosen = trainer.cap_pool_balanced(pool, per_class_cap=3, seed=42)
    assert chosen["model"].value_counts().to_dict() == {"a": 3, "b": 2}


def test_training_loss_warning_is_diagnostic_only() -> None:
    trainer = _trainer()
    assert trainer.training_loss_warning([2.90, 2.89], n_classes=18)
    assert not trainer.training_loss_warning([2.9, 1.8], n_classes=18)
    with pytest.raises(ValueError, match="finite"):
        trainer.training_loss_warning([float("nan")], n_classes=18)


def test_transductive_pool_cv_is_disabled_global_route_is_open(monkeypatch) -> None:
    """D026 still blocks the per-fold (transductive) screen; D028 opens the global route."""
    trainer = _trainer()
    # fold scope stays rejected (held-out text must not select pool rows)
    with pytest.raises(ValueError, match="transductive per-fold pool CV is disabled"):
        trainer.run_cv(None, None, None, cfg={}, device=None, tokenizer=None,
                       augment=True, pool_scope="fold",
                       prompt_mapping={}, mapping_provenance="synthetic")
    # global scope needs a pre-screened pool, else it fails closed
    with pytest.raises(ValueError, match="globally-decontaminated"):
        trainer.run_cv(None, None, None, cfg={}, device=None, tokenizer=None,
                       augment=True, pool_scope="global")
    # the CLI aborts the transductive scope before any data loading
    monkeypatch.setattr(sys, "argv", ["deberta_train.py", "--pool-scope", "fold"])
    with pytest.raises(SystemExit, match="transductive per-fold pool CV is disabled"):
        trainer.main()


def test_global_screen_is_fold_independent_and_text_only() -> None:
    """decontaminate_global removes exact + near duplicates vs the FULL train, once."""
    trainer = _trainer()
    train = pd.DataFrame({"text": ["the quick brown fox jumps over the lazy dog",
                                    "a completely unrelated sentence about astronomy and stars"]})
    pool = pd.DataFrame({"text": ["the quick brown fox jumps over the lazy dog",
                                  "the quick brown fox leaps over the lazy dog",
                                  "quantum entanglement enables teleportation of states",
                                  "the quick brown fox jumps over the lazy dog"]})
    keep1, stats1 = trainer.decontaminate_global(pool, train, cos_threshold=0.3)
    # re-screening against the same train in a different order gives the same mask
    keep2, _ = trainer.decontaminate_global(pool, train.iloc[::-1].reset_index(drop=True),
                                            cos_threshold=0.3)
    assert keep1.tolist() == keep2.tolist()
    assert stats1["decontamination_scope"] == "global_full_official_train"
    assert stats1["validation_independent"] is True
    # exact duplicates (rows 0 and 3) and the near-duplicate sibling (row 1) are removed;
    # the unrelated quantum row (2) survives the text-level screen.
    assert keep1.tolist() == [False, False, True, False]
