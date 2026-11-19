"""H9 (ADR D052): do hard typographic family vetoes improve the adopted blend?

The Discussion sweep reported participants using punctuation *invariants* — "this
provider never emits a curly quote" — as hard filters rather than soft features.
Our style block already carries the corresponding rates, so the open question is
whether a hard mask adds anything on top of the adopted row-z blend.

Protocol (identical rows for both arms, so the comparison is paired):

* rules are mined **per fold, on that fold's official training rows only**;
* every rule records its training support, the family's row count and the chance
  probability ``(1 - p) ** n_f`` of seeing zero hits by luck;
* the mask is applied to the validation rows of that fold only;
* a row whose every class would be vetoed is left untouched and counted.

Post-hoc only: fits nothing, reads no pool row and no test row. The blend is
rebuilt from the identity-verified OOF and cross-checked against the published
candidate score before any veto is applied.

    python experiments/typo_vetoes.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import LABELS, PROVIDER_OF  # noqa: E402
from src.data import (  # noqa: E402
    indices_to_labels,
    load_competition_data,
    ordered_dataset_fingerprint,
)
from src.ensemble import load_oof, normalize_scores  # noqa: E402
from src.metrics import balanced_accuracy, mean_ci, paired_difference  # noqa: E402
from src.oof_identity import fold_assignments  # noqa: E402
from src.typography import MARKERS, apply_vetoes, marker_hits, mine_vetoes  # noqa: E402

WEIGHTS = {"char_tfidf_logreg": 0.5, "style_only": 0.5}
MEMBERS = ["char_tfidf_logreg", "style_only"]
CLASS_FAMILIES = np.asarray([PROVIDER_OF[label] for label in LABELS], dtype=object)
TOL = 1e-6
PRIMARY = {"min_support": 20, "max_chance": 1e-3}
GRID = [
    {"min_support": 20, "max_chance": 1e-3},
    {"min_support": 10, "max_chance": 1e-3},
    {"min_support": 40, "max_chance": 1e-3},
    {"min_support": 20, "max_chance": 1e-2},
    {"min_support": 20, "max_chance": 1e-6},
]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def blend(char: np.ndarray, style: np.ndarray) -> np.ndarray:
    return WEIGHTS["char_tfidf_logreg"] * char + WEIGHTS["style_only"] * style


def fold_bas(scores: np.ndarray, truth, folds_valid) -> list[float]:
    return [
        balanced_accuracy(truth[valid], indices_to_labels(scores[valid].argmax(axis=1)))
        for valid in folds_valid
    ]


def pooled_ba(scores: np.ndarray, truth) -> float:
    return balanced_accuracy(truth, indices_to_labels(scores.argmax(axis=1)))


def veto_folds(scores, hits, families, folds_train, folds_valid, y_indices, params, groups) -> dict:
    """Mine on each fold's training rows, mask only that fold's validation rows."""
    out = scores.copy()
    per_fold = []
    for k, (train_idx, valid_idx) in enumerate(zip(folds_train, folds_valid, strict=True)):
        rules = mine_vetoes(
            hits[train_idx], families[train_idx], marker_names=MARKERS,
            groups=groups[train_idx], **params
        )
        masked, diag = apply_vetoes(scores[valid_idx], hits[valid_idx], rules, CLASS_FAMILIES)
        out[valid_idx] = masked

        counterexamples = []
        for rule in rules:
            violated = valid_idx[
                (families[valid_idx] == rule["family"]) & hits[valid_idx, rule["marker_index"]]
            ]
            for row in violated:
                counterexamples.append(
                    {
                        "row": int(row),
                        "true_label": str(LABELS[int(y_indices[row])]),
                        "marker": rule["marker"],
                        "vetoed_family": rule["family"],
                        "rule_chance": rule["chance"],
                    }
                )

        before = scores[valid_idx].argmax(axis=1)
        after = masked.argmax(axis=1)
        changed = before != after
        was_right = before == y_indices[valid_idx]
        per_fold.append(
            {
                "fold": k,
                "n_rules": len(rules),
                "rules": rules,
                "rows_valid": int(valid_idx.size),
                **diag,
                "prediction_flips": int(changed.sum()),
                "flips_from_correct": int((changed & was_right).sum()),
                "flips_from_wrong": int((changed & ~was_right).sum()),
                "validation_counterexamples": counterexamples,
            }
        )
    return {"scores": out, "folds": per_fold}


def summarise(base, vetoed, truth, folds_valid) -> dict:
    base_folds = fold_bas(base, truth, folds_valid)
    new_folds = fold_bas(vetoed, truth, folds_valid)
    diff = paired_difference(base_folds, new_folds)
    return {
        "pooled_ba": pooled_ba(vetoed, truth),
        "fold_bas": new_folds,
        "fold_diff": diff.tolist(),
        "paired": mean_ci(diff),
        "all_folds_non_negative": bool(np.all(diff >= -TOL)),
    }


def oracle_ceiling(base, vetoed, y_indices, truth, folds_valid) -> dict:
    """Best score any veto rule set could reach — USES LABELS, never a submission.

    On rows where a veto fired and the true class survived it, hand the row to the
    truth. This bounds the whole H9 direction: if the ceiling is negligible, no
    amount of rule refinement is worth the leakage risk of chasing it.
    """
    best = vetoed.copy()
    control = base.copy()
    fired_rows = 0
    rescued = 0
    for valid in folds_valid:
        fired = np.any(vetoed[valid] != base[valid], axis=1)
        survivable = fired & (vetoed[valid][np.arange(valid.size), y_indices[valid]] > -1e7)
        idx = valid[survivable]
        best[idx, y_indices[idx]] = np.max(base) + 1.0
        control[valid[fired], y_indices[valid[fired]]] = np.max(base) + 1.0
        fired_rows += int(fired.sum())
        rescued += int(survivable.sum())
    ceiling = pooled_ba(best, truth)
    without = pooled_ba(control, truth)
    return {
        "is_oracle_uses_labels": True,
        "rows_where_a_veto_fired": fired_rows,
        "rows_the_truth_survives_the_veto": rescued,
        "upper_bound_pooled_ba": ceiling,
        "same_rows_oracle_without_any_veto": without,
        "gain_attributable_to_the_veto": ceiling - without,
    }


def markdown_rules(per_fold: list[dict]) -> str:
    lines = [
        "| fold | marker | vetoed family | support | family rows / prompts | chance |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fold in per_fold:
        for rule in fold["rules"]:
            lines.append(
                f"| {fold['fold']} | `{rule['marker']}` | {rule['family']} | "
                f"{rule['support']} | {rule['family_n']} / {rule['family_trials']} | {rule['chance']:.3g} |"
            )
    if len(lines) == 2:
        lines.append("| — | — | no rule survived the support/chance filters | — | — | — |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", default="artifacts/splits/splits_official.json")
    ap.add_argument("--official-oof", default="artifacts/cv_identity/official")
    ap.add_argument("--pool-char-oof", default="artifacts/pool_cv_cap40k/official")
    ap.add_argument("--expected-candidate-ba", type=float, default=0.442675)
    ap.add_argument("--out", default="artifacts/typo_vetoes")
    args = ap.parse_args()

    train, _test = load_competition_data(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    fingerprint = ordered_dataset_fingerprint(train)
    if fingerprint != splits.get("ordered_dataset_sha256"):
        raise SystemExit("[abort] train frame does not match the split fingerprint")

    official = load_oof(args.official_oof, MEMBERS)
    pool = load_oof(args.pool_char_oof, ["char_tfidf_logreg"])
    n_rows = int(official["style_only"]["y"].size)
    if n_rows != len(train):
        raise SystemExit(f"[abort] OOF has {n_rows} rows but train has {len(train)}")
    folds_valid = [np.asarray(f["valid"], dtype=int) for f in splits["folds"]]
    folds_train = [np.asarray(f["train"], dtype=int) for f in splits["folds"]]
    expected_folds = fold_assignments(
        n_rows, list(zip(folds_train, folds_valid, strict=True))
    )
    for source in (official, pool):
        for name, item in source.items():
            if item["ordered_dataset_sha256"] != fingerprint:
                raise SystemExit(f"[abort] {name}: OOF fingerprint differs from the splits")
            if not np.array_equal(item["validation_fold_ids"], expected_folds):
                raise SystemExit(f"[abort] {name}: OOF fold assignments differ from the splits")
            if not np.array_equal(item["y"], official["style_only"]["y"]):
                raise SystemExit(f"[abort] {name}: OOF labels differ across members")

    y = official["style_only"]["y"]
    truth = indices_to_labels(y)
    style_norm = normalize_scores(official["style_only"]["scores"])
    arms = {
        "official_blend": blend(
            normalize_scores(official["char_tfidf_logreg"]["scores"]), style_norm
        ),
        "pool_mix_cap40k": blend(
            normalize_scores(pool["char_tfidf_logreg"]["scores"]), style_norm
        ),
    }
    candidate_ba = pooled_ba(arms["pool_mix_cap40k"], truth)
    if abs(candidate_ba - args.expected_candidate_ba) > 1e-6:
        raise SystemExit(
            f"[abort] rebuilt candidate BA {candidate_ba:.6f} != published "
            f"{args.expected_candidate_ba:.6f}; the saved OOF no longer reproduces the candidate"
        )

    texts = train["text"].astype(str).tolist()
    groups = train["prompt_id"].to_numpy(dtype=object)
    hits = marker_hits(texts)
    families = np.asarray([PROVIDER_OF[label] for label in truth], dtype=object)
    prevalence = {
        name: {"texts_with_marker": int(hits[:, i].sum()), "share": float(hits[:, i].mean())}
        for i, name in enumerate(MARKERS)
    }

    results = {}
    for arm_name, base in arms.items():
        primary = veto_folds(
            base, hits, families, folds_train, folds_valid, y, PRIMARY, groups
        )
        entry = {
            "base_pooled_ba": pooled_ba(base, truth),
            "base_fold_bas": fold_bas(base, truth, folds_valid),
            "primary": summarise(base, primary["scores"], truth, folds_valid),
            "primary_folds": primary["folds"],
            "primary_oracle_ceiling": oracle_ceiling(
                base, primary["scores"], y, truth, folds_valid
            ),
            "sensitivity": [],
        }
        for params in GRID:
            run = veto_folds(base, hits, families, folds_train, folds_valid, y, params, groups)
            entry["sensitivity"].append(
                {
                    "params": params,
                    "n_rules_total": sum(f["n_rules"] for f in run["folds"]),
                    "rows_affected": sum(f["rows_affected"] for f in run["folds"]),
                    **summarise(base, run["scores"], truth, folds_valid),
                }
            )
        results[arm_name] = entry

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "hypothesis": "H9 (D052): hard typographic family vetoes on top of the row-z blend",
        "fits_performed": 0,
        "test_rows_read": 0,
        "pool_rows_read": 0,
        "transductive": False,
        "rule_mining": "per fold, on that fold's official training rows only",
        "chance_trials": "distinct prompt_id within the family, not rows (rows share a prompt)",
        "n_prompt_groups": int(np.unique(groups).size),
        "mining_caveat": (
            "the char member of pool_mix_cap40k also saw screened pool rows; the veto rules "
            "are mined on the official training rows of the same fold, a subset of that "
            "member's training data, never on its validation rows"
        ),
        "primary_params": PRIMARY,
        "grid_is_sensitivity_not_selection": True,
        "ordered_dataset_sha256": fingerprint,
        "n_rows": n_rows,
        "markers": MARKERS,
        "marker_prevalence_all_train": prevalence,
        "arms": results,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# H9 — hard typographic family vetoes (post-hoc, no fit)",
        "",
        f"Generated {payload['generated_utc']} · commit `{payload['git_commit']}` · "
        f"{n_rows} official OOF rows · 0 fits · 0 test rows · 0 pool rows read.",
        "",
        f"Primary rule filters: support ≥ {PRIMARY['min_support']} texts, "
        f"chance ≤ {PRIMARY['max_chance']}. The grid below is a sensitivity check, "
        "not a selection over the reported score.",
        "",
        "| arm | base pooled BA | vetoed pooled BA | Δ pooled | paired Δ fold mean | 95% CI | all folds ≥ 0 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in results.items():
        primary = entry["primary"]
        paired = primary["paired"]
        lines.append(
            f"| `{name}` | {entry['base_pooled_ba']:.6f} | {primary['pooled_ba']:.6f} | "
            f"{primary['pooled_ba'] - entry['base_pooled_ba']:+.6f} | "
            f"{paired['mean']:+.6f} | [{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}] | "
            f"{'yes' if primary['all_folds_non_negative'] else 'no'} |"
        )
    lines += ["", "## Oracle upper bound (uses labels — a bound, not a score)", ""]
    lines += [
        "| arm | rows a veto fired | truth survives | oracle BA | same rows, NO veto | veto's share |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in results.items():
        ceiling = entry["primary_oracle_ceiling"]
        lines.append(
            f"| `{name}` | {ceiling['rows_where_a_veto_fired']} | "
            f"{ceiling['rows_the_truth_survives_the_veto']} | "
            f"{ceiling['upper_bound_pooled_ba']:.6f} | "
            f"{ceiling['same_rows_oracle_without_any_veto']:.6f} | "
            f"{ceiling['gain_attributable_to_the_veto']:+.6f} |"
        )
    lines += [
        "",
        "The control column matters: handing the *same rows* to their true label "
        "without applying any veto scores almost as high, because those rows are "
        "simply where the blend is wrong. Only the last column is attributable to "
        "the veto itself, and it bounds what any refinement of these rules could add.",
        "",
        "## Rule recurrence across folds (descriptive only)",
        "",
        "Counting how many folds mined the same rule is **descriptive**: fold k's "
        "validation rows sit inside fold j's training rows for j != k, so selecting "
        "rules by cross-fold recurrence would feed validation information back into "
        "the rule set — the transductive class ADR D026 bans. No score here uses it.",
        "",
        "| marker | vetoed family | folds mining it |",
        "| --- | --- | --- |",
    ]
    recurrence: dict[tuple[str, str], int] = {}
    for fold in results["pool_mix_cap40k"]["primary_folds"]:
        for rule in fold["rules"]:
            key = (rule["marker"], rule["family"])
            recurrence[key] = recurrence.get(key, 0) + 1
    for (marker, family), count in sorted(recurrence.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| `{marker}` | {family} | {count}/5 |")
    lines += ["", "## Validation rows that violate a mined rule", ""]
    lines += [
        "A rule with an astronomically small chance level can still be wrong. These "
        "are the validation rows that break an 'invariant', with the chance the "
        "miner assigned to it.",
        "",
        "| arm | fold | row | true label | marker | vetoed family | rule chance |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    any_counterexample = False
    for name, entry in results.items():
        for fold in entry["primary_folds"]:
            for case in fold["validation_counterexamples"]:
                any_counterexample = True
                lines.append(
                    f"| `{name}` | {fold['fold']} | {case['row']} | `{case['true_label']}` | "
                    f"`{case['marker']}` | {case['vetoed_family']} | {case['rule_chance']:.3g} |"
                )
    if not any_counterexample:
        lines.append("| — | — | — | — | — | no rule was violated | — |")
    lines += ["", "## Rules mined per fold (primary filters)", ""]
    for name, entry in results.items():
        lines += [f"### `{name}`", "", markdown_rules(entry["primary_folds"]), ""]
        lines += [
            "| fold | rules | rows vetoed | fully vetoed (skipped) | flips | flips from correct | flips from wrong |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for fold in entry["primary_folds"]:
            lines.append(
                f"| {fold['fold']} | {fold['n_rules']} | {fold['rows_affected']} | "
                f"{fold['rows_fully_vetoed']} | {fold['prediction_flips']} | "
                f"{fold['flips_from_correct']} | {fold['flips_from_wrong']} |"
            )
        lines.append("")
    (out_dir / "veto_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines[:14]))
    print(f"\n[written] {out_dir / 'results.json'}")
    print(f"[written] {out_dir / 'veto_report.md'}")


if __name__ == "__main__":
    main()