"""Per-prompt-group error concentration (plan §3 L118).

Post-hoc analysis of saved, identity-verified out-of-fold predictions: no model is
fit, no pool row is read and no test row is scored. The unit of analysis is the
prompt (17-18 passages, one per model), which is the grouping the CV splits use.

A prompt has only ~18 rows, so its accuracy is noisy. Every concentration claim is
therefore compared against a binomial null with the same total error count, which
is what "errors spread evenly across prompts" would actually look like.

    python experiments/prompt_groups.py --data-dir data

Writes ``artifacts/prompt_groups/results.json`` plus a markdown summary. Stores
prompt ids, counts and scores only — never passage text.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (  # noqa: E402
    indices_to_labels,
    load_competition_data,
    ordered_dataset_fingerprint,
)
from src.ensemble import load_oof, normalize_scores  # noqa: E402
from src.metrics import balanced_accuracy, mean_ci  # noqa: E402
from src.oof_identity import fold_assignments  # noqa: E402
from src.repro import env_snapshot, git_commit  # noqa: E402

WEIGHTS = {"char_tfidf_logreg": 0.5, "style_only": 0.5}
MEMBERS = ["char_tfidf_logreg", "style_only"]
TOP_FRACTION = 0.10
NULL_DRAWS = 2000
NULL_SEED = 20261117
WORST_LIMIT = 10


def blend(normalized: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(normalized[name] * weight for name, weight in weights.items())


def prompt_table(y_true, y_pred, prompt_ids) -> dict:
    """Per-prompt row count, correct count and accuracy, ordered by prompt id."""
    truth = np.asarray(y_true, dtype=str)
    pred = np.asarray(y_pred, dtype=str)
    ids = np.asarray(prompt_ids, dtype=str)
    if truth.shape != pred.shape or truth.shape != ids.shape:
        raise ValueError("truth, prediction and prompt arrays must have the same shape")
    if truth.ndim != 1 or truth.size == 0:
        raise ValueError("need a nonempty 1-D array of rows")
    if np.any(ids == "") or np.any(pd.isna(ids)):
        raise ValueError("prompt ids must not be missing or empty")
    unique, inverse = np.unique(ids, return_inverse=True)
    n = np.bincount(inverse, minlength=unique.size)
    correct = np.bincount(inverse, weights=(truth == pred).astype(float), minlength=unique.size)
    correct = correct.astype(np.int64)
    return {
        "prompt_ids": unique.tolist(),
        "n": n.astype(np.int64),
        "correct": correct,
        "accuracy": correct / n,
    }


def error_concentration(table: dict, top_fraction: float = TOP_FRACTION) -> dict:
    """Share of all errors sitting in the worst ``top_fraction`` of prompts."""
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    n, correct = table["n"], table["correct"]
    errors = n - correct
    total = int(errors.sum())
    k = max(1, int(round(top_fraction * n.size)))
    worst = np.argsort(-errors, kind="stable")[:k]
    return {
        "n_prompts": int(n.size),
        "k_prompts": int(k),
        "total_errors": total,
        "top_share_of_errors": float(errors[worst].sum() / total) if total else 0.0,
        "zero_accuracy_prompts": int((correct == 0).sum()),
        "perfect_prompts": int((correct == n).sum()),
        "rows_per_prompt_min": int(n.min()),
        "rows_per_prompt_max": int(n.max()),
    }


def binomial_null(
    n,
    error_rate: float,
    top_fraction: float = TOP_FRACTION,
    draws: int = NULL_DRAWS,
    seed: int = NULL_SEED,
) -> dict:
    """What the concentration statistics look like if errors fell uniformly at random.

    Each prompt keeps its real row count; its error count is drawn from
    Binomial(n_p, error_rate) with the observed overall rate. This is the honest
    comparison for "do a few prompts carry the errors", because ~18 rows per prompt
    produce a wide spread even with no prompt-level difficulty at all.
    """
    n = np.asarray(n, dtype=np.int64)
    if draws <= 0:
        raise ValueError("draws must be positive")
    if not 0 <= error_rate <= 1:
        raise ValueError("error_rate must be in [0, 1]")
    k = max(1, int(round(top_fraction * n.size)))
    rng = np.random.default_rng(seed)
    shares = np.empty(draws, dtype=np.float64)
    zeros = np.empty(draws, dtype=np.float64)
    perfect = np.empty(draws, dtype=np.float64)
    for d in range(draws):
        err = rng.binomial(n, error_rate)
        total = int(err.sum())
        shares[d] = np.sort(err)[::-1][:k].sum() / total if total else 0.0
        zeros[d] = float((err == n).sum())
        perfect[d] = float((err == 0).sum())
    return {
        "draws": int(draws),
        "seed": int(seed),
        "error_rate": float(error_rate),
        "top_share_mean": float(shares.mean()),
        "top_share_p05": float(np.quantile(shares, 0.05)),
        "top_share_p95": float(np.quantile(shares, 0.95)),
        "zero_accuracy_mean": float(zeros.mean()),
        "zero_accuracy_p95": float(np.quantile(zeros, 0.95)),
        "perfect_mean": float(perfect.mean()),
        "perfect_p95": float(np.quantile(perfect, 0.95)),
    }


def prediction_spread(y_pred, prompt_ids) -> dict:
    """Per prompt: how many distinct labels were predicted and the largest share.

    A prompt whose passages all collapse to one predicted label is a prompt-level
    failure; a prompt with many distinct predictions fails row by row.
    """
    frame = pd.DataFrame({"prompt": np.asarray(prompt_ids, dtype=str),
                          "pred": np.asarray(y_pred, dtype=str)})
    grouped = frame.groupby("prompt", sort=True)["pred"]
    distinct = grouped.nunique().to_numpy(dtype=np.int64)
    modal = grouped.agg(lambda s: float(s.value_counts().iloc[0]) / len(s)).to_numpy(dtype=np.float64)
    return {
        "distinct_predictions_mean": float(distinct.mean()),
        "distinct_predictions_min": int(distinct.min()),
        "distinct_predictions_max": int(distinct.max()),
        "modal_prediction_share_mean": float(modal.mean()),
        "modal_prediction_share_max": float(modal.max()),
        "prompts_collapsed_to_one_label": int((distinct == 1).sum()),
    }


def wrong_prediction_spread(y_true, y_pred, prompt_ids) -> dict:
    """Among a prompt's WRONG rows, how concentrated is the wrong answer?"""
    truth = np.asarray(y_true, dtype=str)
    pred = np.asarray(y_pred, dtype=str)
    ids = np.asarray(prompt_ids, dtype=str)
    wrong = truth != pred
    if not wrong.any():
        return {"prompts_with_errors": 0, "modal_wrong_share_mean": None,
                "prompts_one_wrong_answer": 0}
    frame = pd.DataFrame({"prompt": ids[wrong], "pred": pred[wrong]})
    grouped = frame.groupby("prompt", sort=True)["pred"]
    counts = grouped.size()
    modal = grouped.agg(lambda s: float(s.value_counts().iloc[0]) / len(s))
    return {
        "prompts_with_errors": int(counts.size),
        "modal_wrong_share_mean": float(modal.mean()),
        "prompts_one_wrong_answer": int((modal == 1.0).sum()),
        "prompts_with_one_error": int((counts == 1).sum()),
    }


def accuracy_summary(table: dict) -> dict:
    acc = table["accuracy"]
    return {
        "mean": float(acc.mean()),
        "std": float(acc.std(ddof=1)) if acc.size > 1 else 0.0,
        "min": float(acc.min()),
        "p10": float(np.quantile(acc, 0.10)),
        "median": float(np.median(acc)),
        "p90": float(np.quantile(acc, 0.90)),
        "max": float(acc.max()),
    }


def worst_prompts(table: dict, other: dict | None, words_mean, limit: int = WORST_LIMIT) -> list[dict]:
    order = np.argsort(table["accuracy"], kind="stable")[:limit]
    rows = []
    for i in order:
        row = {
            "prompt_id": table["prompt_ids"][int(i)],
            "n": int(table["n"][i]),
            "correct": int(table["correct"][i]),
            "accuracy": float(table["accuracy"][i]),
            "mean_words": float(words_mean[i]),
        }
        if other is not None:
            row["pool_mix_accuracy"] = float(other["accuracy"][i])
        rows.append(row)
    return rows


def markdown_summary(variants: dict[str, dict]) -> str:
    lines = [
        "| variant | pooled BA | prompt acc mean ± std | min | median | max | "
        "worst-10% share of errors | binomial null mean [p05, p95] | 0-acc prompts (null mean) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, item in variants.items():
        acc, conc, null = item["accuracy"], item["concentration"], item["binomial_null"]
        lines.append(
            f"| {name} | {item['pooled_balanced_accuracy']:.4f} | "
            f"{acc['mean']:.4f} ± {acc['std']:.4f} | {acc['min']:.4f} | "
            f"{acc['median']:.4f} | {acc['max']:.4f} | "
            f"{conc['top_share_of_errors']:.4f} | "
            f"{null['top_share_mean']:.4f} [{null['top_share_p05']:.4f}, {null['top_share_p95']:.4f}] | "
            f"{conc['zero_accuracy_prompts']} ({null['zero_accuracy_mean']:.1f}) |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", default="artifacts/splits/splits_official_verified_s42.json")
    ap.add_argument("--official-oof", default="artifacts/cv_identity/official")
    ap.add_argument("--pool-char-oof", default="artifacts/pool_cv_cap40k/official")
    ap.add_argument("--out", default="artifacts/prompt_groups")
    ap.add_argument("--null-draws", type=int, default=NULL_DRAWS)
    args = ap.parse_args()

    train, _test = load_competition_data(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    fingerprint = ordered_dataset_fingerprint(train)
    if fingerprint != splits.get("ordered_dataset_sha256"):
        raise SystemExit("[abort] local train frame does not match the split fingerprint")

    official = load_oof(args.official_oof, MEMBERS)
    pool = load_oof(args.pool_char_oof, ["char_tfidf_logreg"])
    n_rows = int(official["style_only"]["y"].size)
    if n_rows != len(train):
        raise SystemExit(f"[abort] OOF has {n_rows} rows but train has {len(train)}")
    folds_full = [(np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int))
                  for f in splits["folds"]]
    expected_folds = fold_assignments(n_rows, folds_full)
    for source in (official, pool):
        for name, item in source.items():
            if item["ordered_dataset_sha256"] != fingerprint:
                raise SystemExit(f"[abort] {name}: OOF fingerprint differs from the splits")
            if not np.array_equal(item["validation_fold_ids"], expected_folds):
                raise SystemExit(f"[abort] {name}: OOF fold assignments differ from the splits")
            if not np.array_equal(item["y"], official["style_only"]["y"]):
                raise SystemExit(f"[abort] {name}: OOF labels differ across members")

    truth = indices_to_labels(official["style_only"]["y"])
    prompts = train["prompt_id"].astype(str).to_numpy()
    words = train["text"].astype(str).str.split().str.len().to_numpy(dtype=np.int64)
    norm_official = {n: normalize_scores(official[n]["scores"]) for n in MEMBERS}
    norm_pool_char = normalize_scores(pool["char_tfidf_logreg"]["scores"])
    scores = {
        "official_blend": blend(norm_official, WEIGHTS),
        "pool_mix_cap40k": blend(
            {"char_tfidf_logreg": norm_pool_char, "style_only": norm_official["style_only"]},
            WEIGHTS,
        ),
        "char_official": norm_official["char_tfidf_logreg"],
        "char_pool_cap40k": norm_pool_char,
        "style_only": norm_official["style_only"],
    }

    words_frame = pd.DataFrame({"prompt": prompts, "words": words})
    words_mean = words_frame.groupby("prompt", sort=True)["words"].mean().to_numpy(dtype=np.float64)

    variants: dict[str, dict] = {}
    tables: dict[str, dict] = {}
    for name, matrix in scores.items():
        pred = indices_to_labels(matrix.argmax(axis=1))
        table = prompt_table(truth, pred, prompts)
        tables[name] = table
        conc = error_concentration(table)
        error_rate = conc["total_errors"] / n_rows
        rho, pvalue = spearmanr(table["accuracy"], words_mean)
        variants[name] = {
            "pooled_balanced_accuracy": balanced_accuracy(truth, pred),
            "accuracy": accuracy_summary(table),
            "concentration": conc,
            "binomial_null": binomial_null(table["n"], error_rate, draws=args.null_draws),
            "prediction_spread": prediction_spread(pred, prompts),
            "wrong_prediction_spread": wrong_prediction_spread(truth, pred, prompts),
            "spearman_accuracy_vs_mean_words": {"rho": float(rho), "p_value": float(pvalue)},
        }

    base, candidate = tables["official_blend"], tables["pool_mix_cap40k"]
    if base["prompt_ids"] != candidate["prompt_ids"]:
        raise SystemExit("[abort] prompt ordering differs between variants")
    delta = candidate["accuracy"] - base["accuracy"]
    paired = {
        "unit": "prompt (150 groups, 17-18 rows each)",
        "delta_definition": "pool_mix_cap40k prompt accuracy - official_blend prompt accuracy",
        **mean_ci(delta),
        "prompts_improved": int((delta > 0).sum()),
        "prompts_worsened": int((delta < 0).sum()),
        "prompts_unchanged": int((delta == 0).sum()),
        "max_gain": float(delta.max()),
        "max_loss": float(delta.min()),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment": env_snapshot(),
        "analysis": "per_prompt_group_error_concentration",
        "plan_item": "§3 L118 (performance by prompt group)",
        "transductive": False,
        "fits_performed": 0,
        "predictors_fit": 0,
        "test_rows_read": 0,
        "ordered_dataset_sha256": fingerprint,
        "split_method": splits.get("method"),
        "split_seed": splits.get("seed"),
        "n_rows": n_rows,
        "n_prompts": int(base["n"].size),
        "blend_weights": WEIGHTS,
        "top_fraction": TOP_FRACTION,
        "variants": variants,
        "paired_prompt_delta": paired,
        "worst_prompts_official_blend": worst_prompts(base, candidate, words_mean),
        "note": (
            "Per-prompt accuracy uses ~18 rows, so it is noisy by construction. The binomial null "
            "keeps each prompt's real row count and draws its error count at the observed overall "
            "error rate; concentration is only meaningful relative to that null, not to a uniform "
            "ideal. Within a prompt every passage is a different model, so per-prompt accuracy and "
            "per-prompt balanced accuracy nearly coincide. Descriptive: nothing is fit, no score is "
            "adopted and no validation row influences a training decision. Prompt ids, counts and "
            "scores only - no passage text is stored."
        ),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    table_md = markdown_summary(variants)
    (out / "prompt_groups.md").write_text(table_md + "\n", encoding="utf-8")

    print(table_md)
    print()
    for name in ("official_blend", "pool_mix_cap40k"):
        item = variants[name]
        spread, wrong = item["prediction_spread"], item["wrong_prediction_spread"]
        print(
            f"{name:18s} distinct preds/prompt {spread['distinct_predictions_mean']:.2f} "
            f"modal share {spread['modal_prediction_share_mean']:.3f} "
            f"collapsed {spread['prompts_collapsed_to_one_label']} | "
            f"modal wrong share {wrong['modal_wrong_share_mean']:.3f} "
            f"one-wrong-answer prompts {wrong['prompts_one_wrong_answer']} | "
            f"rho(acc, words) {item['spearman_accuracy_vs_mean_words']['rho']:+.3f}"
        )
    print(
        f"\npaired prompt delta {paired['mean']:+.6f} "
        f"[{paired['ci_low']:+.6f}, {paired['ci_high']:+.6f}] "
        f"improved {paired['prompts_improved']} worsened {paired['prompts_worsened']} "
        f"unchanged {paired['prompts_unchanged']}"
    )
    print(f"\nwrote {out / 'results.json'}")


if __name__ == "__main__":
    main()