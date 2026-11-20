from __future__ import annotations

import numpy as np
import pytest

from src.constants import LABELS, PROVIDER_OF
from src.ensemble import SENTINEL
from src.typography import MARKERS, apply_vetoes, marker_hits, mine_vetoes


def _families(labels):
    return np.asarray([PROVIDER_OF[l] for l in labels], dtype=object)


def test_marker_hits_detects_raw_characters_without_normalising():
    texts = ["a\u2014b", "a \u2014 b", "it's", "it\u2019s", 'say "hi"', "say \u201chi\u201d"]
    hits = marker_hits(texts)
    assert hits.shape == (len(texts), len(MARKERS))
    idx = {name: i for i, name in enumerate(MARKERS)}
    assert hits[0, idx["em_dash"]] and hits[0, idx["em_dash_unspaced"]]
    assert hits[1, idx["em_dash"]] and not hits[1, idx["em_dash_unspaced"]]
    assert hits[2, idx["straight_apostrophe"]] and not hits[2, idx["curly_apostrophe"]]
    assert hits[3, idx["curly_apostrophe"]] and not hits[3, idx["straight_apostrophe"]]
    assert hits[4, idx["straight_dquote"]] and not hits[4, idx["curly_dquote"]]
    assert hits[5, idx["curly_dquote"]] and not hits[5, idx["straight_dquote"]]


def test_mine_vetoes_requires_zero_family_hits_and_enough_support():
    # marker present in 40 rows of family A, never in the 40 rows of family B
    families = np.asarray(["A"] * 40 + ["B"] * 40, dtype=object)
    hits = np.zeros((80, 1), dtype=bool)
    hits[:40, 0] = True
    rules = mine_vetoes(hits, families, marker_names=["m"], min_support=20)
    assert [r["family"] for r in rules] == ["B"]
    assert rules[0]["family_hits"] == 0
    assert rules[0]["family_n"] == 40
    assert rules[0]["support"] == 40
    assert 0.0 < rules[0]["chance"] < 1e-4


def test_mine_vetoes_rejects_a_rule_the_family_actually_uses():
    families = np.asarray(["A"] * 40 + ["B"] * 40, dtype=object)
    hits = np.zeros((80, 1), dtype=bool)
    hits[:40, 0] = True
    hits[40, 0] = True  # one B text uses it -> not an invariant
    assert mine_vetoes(hits, families, marker_names=["m"], min_support=20) == []


def test_mine_vetoes_rejects_low_support_and_high_chance():
    families = np.asarray(["A"] * 40 + ["B"] * 40, dtype=object)
    hits = np.zeros((80, 1), dtype=bool)
    hits[:3, 0] = True
    assert mine_vetoes(hits, families, marker_names=["m"], min_support=20) == []
    # enough support, but B is tiny so absence is unremarkable
    small = np.asarray(["A"] * 40 + ["B"] * 2, dtype=object)
    h = np.zeros((42, 1), dtype=bool)
    h[:40, 0] = True
    assert mine_vetoes(h, small, marker_names=["m"], min_support=20, max_chance=1e-4) == []


def test_apply_vetoes_masks_only_the_vetoed_family():
    labels = np.asarray(LABELS, dtype=object)
    families = _families(labels)
    google = [i for i, l in enumerate(labels) if PROVIDER_OF[l] == "Google"]
    scores = np.zeros((2, len(LABELS)))
    scores[:, google] = 5.0  # blend wants Google for both rows
    hits = np.zeros((2, 1), dtype=bool)
    hits[0, 0] = True
    rules = [{"marker": "m", "marker_index": 0, "family": "Google"}]
    out, diag = apply_vetoes(scores, hits, rules, families)
    assert np.all(out[0, google] == SENTINEL)
    assert out[0].argmax() not in google
    assert np.array_equal(out[1], scores[1])  # marker absent -> untouched
    assert diag["rows_affected"] == 1
    assert diag["rows_fully_vetoed"] == 0


def test_apply_vetoes_never_removes_every_class():
    labels = np.asarray(LABELS, dtype=object)
    families = _families(labels)
    scores = np.zeros((1, len(LABELS)))
    hits = np.ones((1, 1), dtype=bool)
    rules = [
        {"marker": "m", "marker_index": 0, "family": f}
        for f in sorted(set(families.tolist()))
    ]
    out, diag = apply_vetoes(scores, hits, rules, families)
    assert np.array_equal(out, scores)
    assert diag["rows_fully_vetoed"] == 1
    assert diag["rows_affected"] == 0


def test_apply_vetoes_is_a_noop_without_rules():
    scores = np.arange(12, dtype=float).reshape(2, 6)
    out, diag = apply_vetoes(scores, np.zeros((2, 1), dtype=bool), [], np.asarray(["A"] * 6))
    assert np.array_equal(out, scores)
    assert diag["rows_affected"] == 0


def test_group_aware_chance_counts_prompts_not_rows():
    # 40 A rows and 40 B rows, but B's rows come from only 2 prompts.
    families = np.asarray(["A"] * 40 + ["B"] * 40, dtype=object)
    groups = np.asarray([f"p{i}" for i in range(40)] + ["q0"] * 20 + ["q1"] * 20, dtype=object)
    hits = np.zeros((80, 1), dtype=bool)
    hits[:40, 0] = True
    row_level = mine_vetoes(hits, families, marker_names=["m"], min_support=20, max_chance=1e-6)
    grouped = mine_vetoes(
        hits, families, marker_names=["m"], min_support=20, max_chance=1e-6, groups=groups
    )
    assert [r["family"] for r in row_level] == ["B"]
    assert row_level[0]["family_trials"] == 40
    assert grouped == []  # 2 independent prompts cannot rule out a 50% marker
    loose = mine_vetoes(
        hits, families, marker_names=["m"], min_support=20, max_chance=0.3, groups=groups
    )
    assert loose[0]["family_trials"] == 2
    assert loose[0]["chance"] == pytest.approx(0.25)
    assert loose[0]["family_n"] == 40


def test_mine_vetoes_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="marker_names"):
        mine_vetoes(np.zeros((4, 2), dtype=bool), np.asarray(["A"] * 4), marker_names=["m"])
    with pytest.raises(ValueError, match="rows"):
        mine_vetoes(np.zeros((4, 1), dtype=bool), np.asarray(["A"] * 3), marker_names=["m"])
    with pytest.raises(ValueError, match="groups"):
        mine_vetoes(
            np.zeros((4, 1), dtype=bool), np.asarray(["A"] * 4), marker_names=["m"], groups=["p"] * 3
        )