"""Compare blend rules on saved OOF (plan §7, third bullet).

The adopted rule is the row-wise z-score mean of two members. Simple averaging,
rank-based blending and min-max blending were never measured, so this script
compares them on the same identity-verified OOF rows for both member pairs: the
official-only fallback and the cap-40k pool-MIX candidate. Nothing is fit here and
no test row is scored.

Margins from LinearSVC / logistic-regression ``decision_function`` are **not
probabilities**; the softmax arm is a monotone-transform diagnostic, not a
calibrated probability, and is labelled as such in the output.

    python experiments/blend_rules.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import LABELS  # noqa: E402
from src.data import indices_to_labels, load_competition_data, ordered_dataset_fingerprint  # noqa: E402
from src.ensemble import SENTINEL, load_oof, normalize_scores  # noqa: E402
from src.metrics import balanced_accuracy, mean_ci, paired_difference  # noqa: E402
from src.oof_identity import fold_assignments  # noqa: E402

RULES = ("raw_mean", "row_z_mean", "rank_mean", "minmax_mean", "softmax_mean")
ADOPTED = "row_z_mean"
WEIGHTS = {"char_tfidf_logreg": 0.5, "style_only": 0.5}


def mask_sentinel(scores: np.ndarray) -> np.ndarray:
    """Replace the absent-class placeholder with nan, as ``normalize_scores`` does."""
    out = np.asarray(scores, dtype=np.float64).copy()
    out[out < SENTINEL] = np.nan
    return out


def _fill(values: np.ndarray, fill_value: float) -> np.ndarray:
    return np.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value)


def transform_rows(scores: np.ndarray, rule: str) -> np.ndarray:
    """Apply one per-row blend transform; absent classes get the row's neutral value.

    The neutral value is the mean of the transformed row (0 for z, 0.5 for rank and
    min-max, 1/k for softmax), matching ``normalize_scores`` so the rules differ only
    in scale, not in how they treat a class missing from a fold.
    """
    if rule == "row_z_mean":
        return normalize_scores(scores)
    masked = mask_sentinel(scores)
    out = np.zeros_like(masked)
    for i, row in enumerate(masked):
        finite = np.isfinite(row)
        values = row[finite]
        k = int(values.size)
        if k == 0:
            continue
        if rule == "raw_mean":
            transformed, neutral = values, float(values.mean())
        elif rule == "rank_mean":
            ranks = rankdata(values, method="average")
            transformed = (ranks - 1.0) / (k - 1.0) if k > 1 else np.full(k, 0.5)
            neutral = 0.5
        elif rule == "minmax_mean":
            span = float(values.max() - values.min())
            transformed = (values - values.min()) / span if span > 1e-12 else np.full(k, 0.5)
            neutral = 0.5
        elif rule == "softmax_mean":
            shifted = np.exp(values - values.max())
            transformed = shifted / shifted.sum()
            neutral = 1.0 / k
        else:
            raise ValueError(f"unknown blend rule {rule!r}")
        row_out = np.full(row.shape, neutral, dtype=np.float64)
        row_out[finite] = transformed
        out[i] = row_out
    return out


def blend(members: dict[str, np.ndarray], rule: str) -> np.ndarray:
    return sum(
        transform_rows(members[name], rule) * weight
        for name, weight in WEIGHTS.items()
    )


def fold_scores(scores: np.ndarray, truth: np.ndarray, folds_valid) -> list[float]:
    out = []
    for valid in folds_valid:
        pred = indices_to_labels(scores[valid].argmax(axis=1))
        out.append(balanced_accuracy(truth[valid], pred))
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", default="artifacts/splits/splits_official.json")
    ap.add_argument("--official-oof", default="artifacts/cv_identity/official")
    ap.add_argument("--pool-char-oof", default="artifacts/pool_cv_cap40k/official")
    ap.add_argument("--out", default="artifacts/blend_rules")
    args = ap.parse_args()

    train, _test = load_competition_data(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    fingerprint = ordered_dataset_fingerprint(train)
    if fingerprint != splits.get("ordered_dataset_sha256"):
        raise SystemExit("[abort] train frame does not match the split fingerprint")

    official = load_oof(args.official_oof, list(WEIGHTS))
    pool = load_oof(args.pool_char_oof, ["char_tfidf_logreg"])
    n_rows = int(official["style_only"]["y"].size)
    folds_full = [
        (np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int))
        for f in splits["folds"]
    ]
    folds_valid = [np.asarray(f["valid"], dtype=int) for f in splits["folds"]]
    expected = fold_assignments(n_rows, folds_full)
    for source in (official, pool):
        for name, item in source.items():
            if item["ordered_dataset_sha256"] != fingerprint:
                raise SystemExit(f"[abort] {name}: OOF fingerprint differs from the splits")
            if not np.array_equal(item["validation_fold_ids"], expected):
                raise SystemExit(f"[abort] {name}: OOF folds differ from the splits")
            if not np.array_equal(item["y"], official["style_only"]["y"]):
                raise SystemExit(f"[abort] {name}: OOF labels differ across members")

    truth = indices_to_labels(official["style_only"]["y"])
    pairs = {
        "official_only": {n: official[n]["scores"] for n in WEIGHTS},
        "pool_mix_cap40k": {
            "char_tfidf_logreg": pool["char_tfidf_logreg"]["scores"],
            "style_only": official["style_only"]["scores"],
        },
    }

    results: dict[str, dict] = {}
    for pair_name, members in pairs.items():
        adopted_scores = blend(members, ADOPTED)
        adopted_folds = fold_scores(adopted_scores, truth, folds_valid)
        per_rule = {}
        for rule in RULES:
            scores = blend(members, rule)
            pred = indices_to_labels(scores.argmax(axis=1))
            folds = fold_scores(scores, truth, folds_valid)
            diff = paired_difference(adopted_folds, folds)
            per_rule[rule] = {
                "pooled_balanced_accuracy": balanced_accuracy(truth, pred),
                "fold_ba": folds,
                "paired_diff_vs_adopted": [float(d) for d in diff],
                "paired_diff_ci": mean_ci(diff),
                "sentinel_cells": int((members["char_tfidf_logreg"] < SENTINEL).sum()
                                      + (members["style_only"] < SENTINEL).sum()),
            }
        results[pair_name] = {
            "adopted_rule": ADOPTED,
            "weights": WEIGHTS,
            "rules": per_rule,
            "best_rule": max(per_rule, key=lambda r: per_rule[r]["pooled_balanced_accuracy"]),
        }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "ordered_dataset_sha256": fingerprint,
        "split_method": splits.get("method"),
        "split_seed": splits.get("seed"),
        "n_rows": n_rows,
        "fits_performed": 0,
        "transductive": False,
        "caveat": (
            "Margins are not probabilities; softmax_mean is a monotone-transform diagnostic. "
            "All rules are scored on the same OOF rows, so the paired fold difference removes "
            "shared fold noise. Selecting a rule on this OOF is mildly optimistic: any winner "
            "must also beat row-z on the alternative seed and the nested outer gate before adoption."
        ),
        "pairs": results,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    for pair_name, payload_pair in results.items():
        print(f"\n== {pair_name} (adopted {ADOPTED})")
        print("| rule | pooled BA | fold BA | paired diff vs row-z [95% CI] |")
        print("|---|---|---|---|")
        for rule in RULES:
            item = payload_pair["rules"][rule]
            ci = item["paired_diff_ci"]
            folds = " ".join(f"{v:.4f}" for v in item["fold_ba"])
            print(
                f"| {rule} | {item['pooled_balanced_accuracy']:.6f} | {folds} | "
                f"{ci['mean']:+.4f} [{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}] |"
            )
        print(f"best on this OOF: {payload_pair['best_rule']}")
    print(f"\nwrote {out / 'results.json'}")


if __name__ == "__main__":
    main()