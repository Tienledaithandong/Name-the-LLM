"""Validate provider reranking using synthetic prompt groups."""

from __future__ import annotations

from copy import copy
from string import ascii_lowercase

import joblib
import numpy as np
import pytest
from scipy.special import softmax

from src.constants import LABELS, N_CLASSES
from src.metrics import provider_of
from src.model import ModelError
from src.provider_specialist import ProviderSpecialist


def _passage(label: int, prompt: str) -> str:
    marker = ascii_lowercase[label] * 8
    return f"{prompt}: {marker}. {marker}! {marker}?"


@pytest.fixture(scope="module")
def specialist() -> ProviderSpecialist:
    prompts = ("train group one", "train group two", "train group three")
    order = np.arange(N_CLASSES)[::-1]
    return ProviderSpecialist(max_features=2000).fit(
        [_passage(int(label), prompt) for prompt in prompts for label in order],
        np.tile(order, len(prompts)),
    )


def test_provider_mass_is_conserved_and_singletons_unchanged(specialist: ProviderSpecialist) -> None:
    texts = [_passage(label, "held-out group") for label in (1, 7, 17)]
    scores = np.random.default_rng(13).normal(size=(len(texts), N_CLASSES))
    hierarchy, flat = specialist.compare(texts, scores)
    baseline = softmax(scores, axis=1)
    providers = np.array([provider_of(label) for label in LABELS])

    assert hierarchy.shape == flat.shape == scores.shape
    np.testing.assert_allclose(flat.sum(axis=1), 1)
    for provider in np.unique(providers):
        columns = providers == provider
        if columns.sum() == 1:
            np.testing.assert_array_equal(hierarchy[:, columns], baseline[:, columns])
        else:
            np.testing.assert_allclose(hierarchy[:, columns].sum(axis=1), baseline[:, columns].sum(axis=1))


def test_specialist_learns_canonical_version_columns_without_validation_features(
    specialist: ProviderSpecialist,
) -> None:
    target = LABELS.index("gpt-5.6-terra")
    provider_columns = [i for i, label in enumerate(LABELS) if provider_of(label) == "OpenAI"]
    scores = np.full((1, N_CLASSES), -10.0)
    scores[:, provider_columns] = 0

    assert scores.argmax(axis=1).item() != target
    hierarchy, _ = specialist.compare([_passage(target, "held-out ☃☃ group")], scores)
    assert hierarchy.argmax(axis=1).item() == target
    assert specialist.builder is not None
    assert not any("☃" in name for name in specialist.builder.get_feature_names_out())
    np.testing.assert_array_equal(specialist.provider_heads["OpenAI"].classes_, provider_columns)


def test_scoring_does_not_depend_on_other_passages(specialist: ProviderSpecialist) -> None:
    texts = [_passage(label, "held-out group") for label in (15, 10)]
    scores = np.random.default_rng(42).normal(size=(2, N_CLASSES))
    together = specialist.compare(texts, scores)
    solo = specialist.compare(texts[:1], scores[:1])
    reversed_batch = specialist.compare(texts[::-1], scores[::-1])
    for all_rows, one_row, reverse_rows in zip(together, solo, reversed_batch, strict=True):
        np.testing.assert_allclose(all_rows[:1], one_row)
        np.testing.assert_allclose(all_rows, reverse_rows[::-1])


def test_zero_weight_preserves_exact_argmax_even_at_near_tie(specialist: ProviderSpecialist) -> None:
    control = copy(specialist)
    control.alpha = 0
    scores = np.full((1, N_CLASSES), -1.0)
    scores[0, 0] = 0.0
    scores[0, 1] = np.nextafter(0.0, 1.0)
    for blend in control.compare(["the first"], scores):
        np.testing.assert_array_equal(blend, scores)
        assert blend.argmax(axis=1).item() == 1


def test_reject_incomplete_training_classes_and_missing_baseline_columns(
    specialist: ProviderSpecialist,
) -> None:
    with pytest.raises(ModelError, match="all 18 canonical"):
        ProviderSpecialist().fit(["training alpha", "training beta"], [0, 1])
    with pytest.raises(ModelError, match="18 labels"):
        specialist.compare(["held-out group"], np.zeros((1, N_CLASSES - 1)))
    scores = np.zeros((1, N_CLASSES))
    scores[0, 2] = -1e9
    with pytest.raises(ModelError, match="absent classes"):
        specialist.compare(["held-out group"], scores)


def test_artifact_roundtrip_rejects_wrong_class_order(
    specialist: ProviderSpecialist, tmp_path,
) -> None:
    path = specialist.save(tmp_path / "provider.joblib")
    reloaded = ProviderSpecialist.load(path)
    texts = ["the first held-out group"]
    scores = np.zeros((1, N_CLASSES))
    for actual, expected in zip(reloaded.compare(texts, scores), specialist.compare(texts, scores), strict=True):
        np.testing.assert_array_equal(actual, expected)

    reloaded.class_order = LABELS[::-1]
    joblib.dump(reloaded, path)
    with pytest.raises(ModelError, match="class order"):
        ProviderSpecialist.load(path)
