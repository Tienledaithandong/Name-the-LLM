"""Hard typographic vetoes at provider-family level (hypothesis H9, ADR D052).

A stylistic *rate* is soft evidence: a model can trade a high em-dash rate off
against other features.  Some punctuation habits are closer to invariants — a
provider whose renderer never emits a curly quote will not emit one by accident.
This module turns "family f never uses marker m in training" into a hard mask on
the blend's scores, which is a different kind of signal from the soft rates in
:mod:`src.features_style`.

Two safeguards make the rule set defensible rather than a lucky lookup table:

* **Rules are mined per fold.**  :func:`mine_vetoes` only ever sees the training
  rows of one fold.  Mining once on all of train and applying to validation is
  the transductive leak class ADR D026 banned.
* **Absence must be surprising.**  With marker prevalence ``p`` over the fold's
  training rows and ``n_f`` rows in family f, pure chance produces zero hits with
  probability ``(1 - p) ** n_f``.  A rule is kept only when that probability is
  below ``max_chance``, so a two-row family is not "vetoed" for lacking a marker
  it had almost no opportunity to show.  Support and chance are returned with
  every rule so the evidence is auditable.

Markers are tested on the raw string with no case folding or unicode
normalisation: the distinction between ``'`` and ``\\u2019`` *is* the signal.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np

from src.ensemble import SENTINEL

_EM_DASH_UNSPACED = re.compile(r"\S—\S")

_MARKER_TESTS: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("em_dash", lambda t: "—" in t),
    ("em_dash_unspaced", lambda t: _EM_DASH_UNSPACED.search(t) is not None),
    ("en_dash", lambda t: "–" in t),
    ("curly_apostrophe", lambda t: "’" in t),
    ("straight_apostrophe", lambda t: "'" in t),
    ("curly_dquote", lambda t: "“" in t or "”" in t),
    ("straight_dquote", lambda t: '"' in t),
    ("ellipsis_char", lambda t: "…" in t),
)

MARKERS: tuple[str, ...] = tuple(name for name, _ in _MARKER_TESTS)


def marker_hits(texts: Sequence[str]) -> np.ndarray:
    """Boolean ``(n_texts, len(MARKERS))`` presence matrix."""
    rows = [
        [test(str(text)) for _, test in _MARKER_TESTS]
        for text in texts
    ]
    if not rows:
        return np.zeros((0, len(MARKERS)), dtype=bool)
    return np.asarray(rows, dtype=bool)


def mine_vetoes(
    hits: np.ndarray,
    families: Sequence[str],
    marker_names: Sequence[str] | None = None,
    min_support: int = 20,
    max_chance: float = 1e-3,
    groups: Sequence | None = None,
) -> list[dict]:
    """Mine "family never uses marker" rules from ONE fold's training rows.

    Pass ``groups`` (the prompt id of each row) whenever the data is grouped.
    Rows sharing a prompt are near-replicates of the same generation, so counting
    them as independent trials inflates confidence by orders of magnitude; the
    number of *distinct prompts* in the family is the honest trial count.
    """
    hits = np.asarray(hits, dtype=bool)
    families = np.asarray(families, dtype=object)
    names = tuple(marker_names) if marker_names is not None else MARKERS
    if hits.ndim != 2 or hits.shape[1] != len(names):
        raise ValueError(f"marker_names has {len(names)} entries but hits has {hits.shape} shape")
    if hits.shape[0] != families.shape[0]:
        raise ValueError(f"hits has {hits.shape[0]} rows but families has {families.shape[0]}")
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        if groups.shape[0] != hits.shape[0]:
            raise ValueError(f"groups has {groups.shape[0]} entries but hits has {hits.shape[0]} rows")

    n_train = hits.shape[0]
    rules: list[dict] = []
    if not n_train:
        return rules
    for m, name in enumerate(names):
        support = int(hits[:, m].sum())
        if support < min_support:
            continue
        prevalence = support / n_train
        for family in sorted(set(families.tolist())):
            in_family = families == family
            n_f = int(in_family.sum())
            family_hits = int(hits[in_family, m].sum())
            if family_hits:
                continue
            n_trials = (
                n_f if groups is None else int(np.unique(groups[in_family]).size)
            )
            chance = float((1.0 - prevalence) ** n_trials)
            if chance > max_chance:
                continue
            rules.append(
                {
                    "marker": name,
                    "marker_index": m,
                    "family": str(family),
                    "support": support,
                    "prevalence": float(prevalence),
                    "family_n": n_f,
                    "family_trials": n_trials,
                    "family_hits": family_hits,
                    "chance": chance,
                }
            )
    return rules


def apply_vetoes(
    scores: np.ndarray,
    hits: np.ndarray,
    rules: Sequence[dict],
    class_families: Sequence[str],
) -> tuple[np.ndarray, dict]:
    """Mask vetoed families to ``SENTINEL``; never mask every class in a row."""
    scores = np.asarray(scores, dtype=float)
    hits = np.asarray(hits, dtype=bool)
    class_families = np.asarray(class_families, dtype=object)
    if scores.shape[0] != hits.shape[0]:
        raise ValueError(f"scores has {scores.shape[0]} rows but hits has {hits.shape[0]}")
    if scores.shape[1] != class_families.shape[0]:
        raise ValueError("class_families must have one entry per score column")

    out = scores.copy()
    diag = {
        "rows_total": int(scores.shape[0]),
        "rows_affected": 0,
        "rows_fully_vetoed": 0,
        "rules_applied": len(rules),
    }
    if not len(rules) or not scores.size:
        return out, diag

    by_family: dict[str, list[int]] = {}
    for rule in rules:
        by_family.setdefault(str(rule["family"]), []).append(int(rule["marker_index"]))

    mask = np.zeros(scores.shape, dtype=bool)
    for family, indices in by_family.items():
        cols = np.flatnonzero(class_families == family)
        if not cols.size:
            continue
        present = hits[:, indices].any(axis=1)
        mask[np.ix_(present, cols)] = True

    fully = mask.all(axis=1)
    mask[fully] = False
    out[mask] = SENTINEL
    diag["rows_affected"] = int(mask.any(axis=1).sum())
    diag["rows_fully_vetoed"] = int(fully.sum())
    return out, diag