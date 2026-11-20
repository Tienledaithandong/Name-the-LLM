"""Exercise final-blend guards and Parquet validation without official rows or CUDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import modernbert_submission as submission
from src.constants import LABELS, N_CLASSES
from src.data import SchemaError, ordered_dataset_fingerprint, validate_train_frame
from src.ensemble import normalize_scores


def test_blend_reproduces_the_three_verified_weights() -> None:
    rng = np.random.default_rng(11)
    char, style, head = rng.normal(size=(3, 4, N_CLASSES))
    expected = 0.75 * (0.5 * normalize_scores(char) + 0.5 * normalize_scores(style))
    expected += 0.25 * normalize_scores(head)
    np.testing.assert_array_equal(submission.blend_margins(char, style, head, 4), expected)
    with pytest.raises(ValueError, match="expected rows"):
        submission.blend_margins(char[:-1], style, head, 4)
    head[0, 0] = -1e9
    with pytest.raises(ValueError, match="missing classes"):
        submission.blend_margins(char, style, head, 4)


@pytest.fixture
def simulated_runtime(tmp_path, monkeypatch):
    train = pd.DataFrame({
        "id": [f"train-{i}" for i in range(N_CLASSES)],
        "prompt_id": ["synthetic-training-prompt"] * N_CLASSES,
        "text": [f"Synthetic training passage {i}." for i in range(N_CLASSES)],
        "model": LABELS,
    })
    train.to_csv(tmp_path / "train.csv", index=False)
    texts = train["text"].tolist()
    fits = []
    requests = []

    class SparseMember:
        class_order = LABELS

        def __init__(self, name):
            self.name = name

        def fit(self, passages, y):
            assert passages == texts
            np.testing.assert_array_equal(y, np.arange(N_CLASSES))
            fits.append(self.name)
            return self

        def decision_function(self, passages):
            assert all(text not in texts for text in passages)
            return np.tile(np.arange(N_CLASSES), (len(passages), 1))

    class FrozenEncoder:
        def __init__(self, checkpoint, cache_dir, **config):
            assert config["max_length"] == 384
            assert config["batch_size"] == 4

        def encode(self, passages, *, deadline):
            requests.append(list(passages))
            return np.arange(len(passages), dtype=float)[:, None]

    class LogisticHead:
        classes_ = np.arange(N_CLASSES)
        n_iter_ = np.array([500])
        max_iter = 500

        def __init__(self, **config):
            assert config == {"C": 1.0, "class_weight": "balanced", "random_state": 42, "max_iter": 500}

        def fit(self, matrix, y):
            assert matrix.shape == (N_CLASSES, 1)
            np.testing.assert_array_equal(y, np.arange(N_CLASSES))
            fits.append("encoder")

        def decision_function(self, matrix):
            return np.tile(np.arange(N_CLASSES), (len(matrix), 1))

    monkeypatch.setattr(submission, "OFFICIAL_TRAIN_SHA256", ordered_dataset_fingerprint(validate_train_frame(train)))
    monkeypatch.setattr(submission, "estimator_from_config", lambda config, seed: SparseMember(config["name"]))
    monkeypatch.setattr(submission, "FrozenModernBert", FrozenEncoder)
    monkeypatch.setattr(submission, "LogisticRegression", LogisticHead)
    monkeypatch.setattr(submission, "_cuda_runtime", lambda: ("synthetic", "test", "test", "test"))
    monkeypatch.setattr(submission, "_peak_gpu_gib", lambda: 0.0)
    # Restore the environment after run_submission forces offline mode.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    return tmp_path, texts, fits, requests


def test_competition_mode_preserves_text_ids_in_supplied_order(simulated_runtime) -> None:
    path, train, fits, requests = simulated_runtime
    inputs = ["Synthetic prediction passage B.", "Synthetic prediction passage A."]
    ids = ["010", "001"]
    pd.DataFrame({"id": ids, "text": inputs}).to_csv(path / "test.csv", index=False)
    out = path / "submission.parquet"
    report = submission.run_submission(path / "checkpoint", path, out)
    assert fits == ["char_tfidf_logreg", "style_only", "encoder"]
    assert requests == [train, inputs]
    result = pd.read_parquet(out)
    assert result.columns.tolist() == ["id", "model"]
    assert result["id"].tolist() == ids
    assert result["model"].tolist() == [LABELS[-1], LABELS[-1]]
    assert report["schema_verified"] and report["head_hit_iteration_limit"]
    assert report["mode"] == "competition_test_file"


def test_runtime_mode_has_distinct_name_and_never_requires_test_file(simulated_runtime) -> None:
    path, train, fits, requests = simulated_runtime
    assert not (path / "test.csv").exists()
    with pytest.raises(ValueError, match="synthetic_submission.parquet"):
        submission.run_submission(path / "checkpoint", path, path / "submission.parquet", synthetic_rows=3)
    assert fits == []
    out = path / "synthetic_submission.parquet"
    report = submission.run_submission(path / "checkpoint", path, out, synthetic_rows=3)
    assert requests[0] == train and len(requests[1]) == 3
    assert report["mode"] == "synthetic_inference_smoke" and report["inference_rows"] == 3
    assert pd.read_parquet(out)["id"].tolist() == ["synthetic-000000", "synthetic-000001", "synthetic-000002"]
    with pytest.raises(FileExistsError, match="refusing to replace"):
        submission.run_submission(path / "checkpoint", path, out, synthetic_rows=3)


def test_training_snapshot_guard_stops_before_fitting(simulated_runtime) -> None:
    path, _, fits, requests = simulated_runtime
    frame = pd.read_csv(path / "train.csv")
    frame.loc[0, "text"] = "Unexpected training snapshot."
    frame.to_csv(path / "train.csv", index=False)
    out = path / "synthetic_submission.parquet"
    with pytest.raises(ValueError, match="verified official snapshot"):
        submission.run_submission(path / "checkpoint", path, out, synthetic_rows=3)
    assert fits == requests == []
    assert not out.exists()


def test_invalid_input_does_not_start_training_or_write_output(simulated_runtime) -> None:
    path, _, fits, requests = simulated_runtime
    pd.DataFrame({"id": ["duplicate", "duplicate"], "text": ["one", "two"]}).to_csv(
        path / "test.csv", index=False,
    )
    out = path / "submission.parquet"
    with pytest.raises(SchemaError, match="duplicate ids"):
        submission.run_submission(path / "checkpoint", path, out)
    assert fits == requests == []
    assert not out.exists()
