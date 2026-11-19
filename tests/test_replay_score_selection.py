"""Exercise selection and data isolation without official rows or model fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import replay_score_selection as replay
from src.constants import LABELS, N_CLASSES
from src.data import ordered_dataset_fingerprint
from src.nested_validation import make_nested_splits
from src.oof_identity import fold_assignments


def _frame(n_prompts: int = 15) -> pd.DataFrame:
    return pd.DataFrame([
        {"id": f"{prompt}-{index}", "text": f"prompt-{prompt}:{index}",
         "prompt_id": f"prompt-{prompt}", "model": label}
        for prompt in range(n_prompts)
        for index, label in enumerate(LABELS)
    ])


def _rows(texts) -> np.ndarray:
    return np.asarray([int(text.split(":")[1]) for text in texts])


def _scores(rows: np.ndarray, perfect: bool) -> np.ndarray:
    # Keep the control imperfect, with every class occurring in every prompt.
    predictions = rows.copy() if perfect else np.where(rows % 2 == 0, rows, (rows + 1) % N_CLASSES)
    return np.eye(N_CLASSES)[predictions]


def _fake_control(configs, train, valid, y, seed):
    assert set(text.split(":")[0] for text in train).isdisjoint(text.split(":")[0] for text in valid)
    np.testing.assert_array_equal(y, _rows(train))
    assert seed == 42
    control = _scores(_rows(valid), perfect=False)
    return control, control


@pytest.mark.parametrize(
    ("passing_cpu", "passing_encoder_folds", "expected", "expected_encoder_calls"),
    [
        (replay.CPU_ARMS, 5, "provider_specialist", 0),
        (("masked_blend",), 5, "masked_blend", 0),
        ((), 1, "baseline", 2),  # Fold 1 fails the pilot.
        ((), 2, "baseline", 5),  # Pilot passes, but the 5-fold gate fails.
        ((), 4, "modernbert_blend", 5),
    ],
)
def test_cpu_priority_and_complete_gpu_gate(
    monkeypatch, passing_cpu, passing_encoder_folds, expected, expected_encoder_calls,
) -> None:
    simulated = _frame()
    requests = []
    encoder_calls = []
    monkeypatch.setattr(replay, "refit_control", _fake_control)

    def fake_cpu(arm, configs, train, valid, y, char, baseline):
        return _scores(_rows(valid), perfect=arm in passing_cpu)

    def fake_encode(texts, _deadline):
        requests.append(list(texts))
        return _rows(texts)[:, None]

    def fake_encoder(train, valid, y, baseline):
        np.testing.assert_array_equal(train[:, 0], y)
        encoder_calls.append(len(valid))
        raw = _scores(valid[:, 0], perfect=True)
        blended = _scores(valid[:, 0], perfect=len(encoder_calls) <= passing_encoder_folds)
        return raw, blended

    monkeypatch.setattr(replay, "_fit_cpu", fake_cpu)
    monkeypatch.setattr(replay, "_fit_encoder", fake_encoder)
    result, _ = replay.choose_arm(
        simulated, outer_fold=2, configs={}, encode=fake_encode, pilot_seconds=30, full_seconds=30,
    )
    assert result["selected_arm"] == expected
    assert result["inner_seed"] == 214
    assert len(encoder_calls) == expected_encoder_calls
    assert requests == ([simulated["text"].tolist()] if expected_encoder_calls else [])


def _manifest(path: Path, frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    plans = make_nested_splits(frame)
    result = {
        "schema_version": "nested-grouped-splits/v1",
        "ordered_dataset_sha256": ordered_dataset_fingerprint(frame),
        "outer_seed": 121, "inner_seed": 212,
        "folds": [
            {"train": plan["train"].tolist(), "valid": plan["valid"].tolist(),
             "inner": [{"train": tr.tolist(), "valid": va.tolist()} for tr, va in plan["inner"]]}
            for plan in plans
        ],
    }
    path.write_text(json.dumps(result), encoding="utf-8")
    return [(plan["train"], plan["valid"]) for plan in plans]


def test_replay_keeps_outer_prompts_out_of_selection_and_saves_identity(tmp_path, monkeypatch) -> None:
    simulated = _frame()
    simulated.to_csv(tmp_path / "train.csv", index=False)
    path = tmp_path / "nested.json"
    folds = _manifest(path, simulated)
    assignments = fold_assignments(len(simulated), folds)
    arg = argparse.Namespace(
        out=tmp_path / "out", data_dir=tmp_path, nested_plan=path, threads=1,
        checkpoint=None, cache_dir=tmp_path / "unused",
    )
    calls = []
    select = replay.choose_arm
    monkeypatch.setattr(replay, "refit_control", _fake_control)

    def track_inner(frame, *, outer_fold, **kwargs):
        # The real selection function can only see this frame, which contains
        # neither the held-out texts nor their model labels.
        outer_train, outer_valid = folds[outer_fold]
        assert frame["id"].tolist() == simulated.iloc[outer_train]["id"].tolist()
        assert set(frame["prompt_id"]).isdisjoint(simulated.iloc[outer_valid]["prompt_id"])
        return select(frame, outer_fold=outer_fold, **kwargs)

    def fake_cpu(arm, configs, train, valid, y, char, baseline):
        calls.append((arm, len(train)))
        return _scores(_rows(valid), perfect=True)

    monkeypatch.setattr(replay, "choose_arm", track_inner)
    monkeypatch.setattr(replay, "_fit_cpu", fake_cpu)
    output = replay.run_replay(arg)
    assert [d["selected_arm"] for d in output["outer_decisions"]] == ["provider_specialist"] * 3
    # Exactly one chosen arm refit per held-out fold, in addition to CPU screens.
    assert calls.count(("provider_specialist", len(folds[0][0]))) == 3
    assert calls.count(("masked_blend", len(folds[0][0]))) == 0
    assert output["test_rows_read"] == 0
    assert output["results"]["selection_procedure"]["balanced_accuracy"] == 1.0
    assert output["results"]["selection_procedure"]["paired_prompt_ci_95"]["low"] > 0
    with np.load(arg.out / "oof" / "selection_procedure.npz", allow_pickle=False) as oof:
        assert oof["ordered_dataset_sha256"].item() == ordered_dataset_fingerprint(simulated)
        np.testing.assert_array_equal(oof["labels"], LABELS)
        np.testing.assert_array_equal(oof["validation_fold_ids"], assignments)
