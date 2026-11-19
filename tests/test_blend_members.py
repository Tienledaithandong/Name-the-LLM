"""The third-member screen must pair only identity-verified OOF and stay descriptive."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.blend_members import (
    THIRD_WEIGHT,
    fold_balanced_accuracies,
    load_members,
    screen_members,
)
from src.constants import LABELS
from src.data import indices_to_labels
from src.ensemble import normalize_scores

A, B = LABELS[0], LABELS[1]


def _base(n: int = 40):
    """char and style agree, but only narrowly beat a fixed distractor class."""
    y = np.array([0 if i % 2 == 0 else 1 for i in range(n)], dtype=np.int64)
    truth = indices_to_labels(y)
    folds = np.array([i % 4 for i in range(n)], dtype=np.int64)
    scores = np.zeros((n, len(LABELS)))
    scores[np.arange(n), y] = 1.0
    scores[:, 2] = 0.9
    z = normalize_scores(scores)
    return z, z.copy(), truth, folds


def _distractor_scores(n: int = 40):
    scores = np.full((n, len(LABELS)), -3.0)
    scores[:, 2] = 3.0
    return scores


def test_fold_balanced_accuracies_matches_manual_per_fold_scoring():
    cz, sz, truth, folds = _base()
    z = 0.5 * cz + 0.5 * sz
    pooled, per = fold_balanced_accuracies(z, truth, folds)
    assert pooled == pytest.approx(1.0)
    assert len(per) == 4 and all(v == pytest.approx(1.0) for v in per)
    # a member that is wrong on exactly one fold shows up only in that fold
    bad = z.copy()
    m = folds == 2
    bad[m] = -bad[m]
    pooled_bad, per_bad = fold_balanced_accuracies(bad, truth, folds)
    assert per_bad[2] < per[2]
    assert [per_bad[i] for i in (0, 1, 3)] == [per[i] for i in (0, 1, 3)]
    assert pooled_bad < pooled


def test_adding_the_blend_itself_as_a_third_member_changes_nothing():
    cz, sz, truth, folds = _base()
    base_z = 0.5 * cz + 0.5 * sz
    out = screen_members(cz, sz, truth, folds, {"self": base_z})
    rec = out["members"][0]
    for tag in ("thirds", "keep_ratio", "swap"):
        assert rec[tag]["pooled_ba"] == pytest.approx(out["base_pooled_ba"])
        assert rec[tag]["paired_diff_vs_base"]["mean"] == pytest.approx(0.0)
        assert rec[tag]["all_folds_non_negative"] is True


def test_a_weak_anticorrelated_third_member_dilutes_the_blend():
    cz, sz, truth, folds = _base()
    out = screen_members(cz, sz, truth, folds, {"distractor": _distractor_scores(len(cz))})
    rec = out["members"][0]
    assert rec["solo_pooled_ba"] == pytest.approx(0.0)
    assert rec["thirds"]["pooled_ba"] < out["base_pooled_ba"]
    assert rec["thirds"]["paired_diff_vs_base"]["mean"] < 0.0
    assert rec["thirds"]["all_folds_non_negative"] is False
    assert rec["keep_ratio"]["pooled_ba"] < out["base_pooled_ba"]
    assert rec["keep_ratio"]["paired_diff_vs_base"]["ci_high"] < 0.0


def test_keep_ratio_preserves_the_adopted_char_to_style_ratio():
    cz, sz, truth, folds = _base()
    rec = screen_members(cz, sz, truth, folds, {"m": cz})["members"][0]
    assert rec["keep_ratio"]["third_weight"] == THIRD_WEIGHT
    assert rec["keep_ratio"]["char_weight"] == pytest.approx((1.0 - THIRD_WEIGHT) * 0.5)
    assert rec["thirds"]["char_weight"] == pytest.approx(1.0 / 3.0)
    assert rec["thirds"]["third_weight"] == pytest.approx(1.0 / 3.0)
    assert rec["swap"]["style_weight"] == 0.0
    assert rec["swap"]["char_weight"] == pytest.approx(0.5)
    assert rec["swap"]["third_weight"] == pytest.approx(0.5)


def test_swapping_the_style_member_back_in_reproduces_the_base_exactly():
    cz, sz, truth, folds = _base()
    out = screen_members(cz, sz, truth, folds, {"style": sz})
    rec = out["members"][0]
    assert rec["swap"]["pooled_ba"] == pytest.approx(out["base_pooled_ba"])
    assert rec["swap"]["fold_ba"] == pytest.approx(out["base_fold_ba"])
    assert rec["swap"]["paired_diff_vs_base"]["mean"] == pytest.approx(0.0)


def _write_oof(path, scores, y, folds, fingerprint="fp1", labels=None):
    np.savez(path, scores=scores, y=y,
             labels=np.array(LABELS if labels is None else labels, dtype=object),
             ordered_dataset_sha256=np.array(fingerprint), validation_fold_ids=folds)


def test_load_members_accepts_only_matching_identity(tmp_path):
    cz, _, _, folds = _base()
    y = np.array([0 if i % 2 == 0 else 1 for i in range(len(cz))], dtype=np.int64)
    _write_oof(tmp_path / "good.npz", cz, y, folds)
    _write_oof(tmp_path / "other_fp.npz", cz, y, folds, fingerprint="fp2")
    _write_oof(tmp_path / "other_folds.npz", cz, y, (folds + 1) % 4)
    _write_oof(tmp_path / "bad_order.npz", cz, y, folds, labels=list(reversed(LABELS)))
    np.savez(tmp_path / "no_provenance.npz", scores=cz, y=y, labels=np.array(LABELS, dtype=object))

    members, skipped = load_members(
        ["good.npz", "other_fp.npz", "other_folds.npz", "bad_order.npz",
         "no_provenance.npz", "absent.npz"], "fp1", y, folds, root=tmp_path)
    assert list(members) == ["good"]
    reasons = {s["member"]: s["reason"] for s in skipped}
    assert set(reasons) == {"other_fp", "other_folds", "bad_order", "no_provenance", "absent"}
    assert reasons["other_fp"] == "identity/fold mismatch"
    assert reasons["other_folds"] == "identity/fold mismatch"
    assert reasons["absent"] == "file missing"
    assert "provenance" in reasons["no_provenance"]
    assert "class order" in reasons["bad_order"]


def test_load_members_returns_scores_untouched_for_normalisation_downstream(tmp_path):
    cz, _, _, folds = _base()
    y = np.array([0 if i % 2 == 0 else 1 for i in range(len(cz))], dtype=np.int64)
    raw = cz * 7.0 + 2.0
    _write_oof(tmp_path / "m.npz", raw, y, folds)
    members, skipped = load_members(["m.npz"], "fp1", y, folds, root=tmp_path)
    assert skipped == []
    assert np.allclose(members["m"], raw)


def test_colliding_member_names_are_qualified_not_overwritten(tmp_path):
    """An official-only and a pool-trained export of the same model must both survive."""
    cz, _, _, folds = _base()
    y = np.array([0 if i % 2 == 0 else 1 for i in range(len(cz))], dtype=np.int64)
    for run, scale in (("cv_official", 1.0), ("pool_cv", 2.0)):
        d = tmp_path / run / "oof"
        d.mkdir(parents=True)
        _write_oof(d / "word_tfidf_linearsvc.npz", cz * scale, y, folds)
    members, skipped = load_members(
        ["cv_official/oof/word_tfidf_linearsvc.npz", "pool_cv/oof/word_tfidf_linearsvc.npz"],
        "fp1", y, folds, root=tmp_path)
    assert skipped == []
    assert set(members) == {"cv_official/word_tfidf_linearsvc", "pool_cv/word_tfidf_linearsvc"}
    assert np.allclose(members["cv_official/word_tfidf_linearsvc"], cz)
    assert np.allclose(members["pool_cv/word_tfidf_linearsvc"], cz * 2.0)