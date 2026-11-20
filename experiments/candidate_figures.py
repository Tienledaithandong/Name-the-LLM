"""Ablation, runtime/RAM and grouped-CV variation panels for the pool-MIX candidate.

`src.figures.make_report_figures` reads the metrics directory of a *single*
training run, so it can only ever draw the official-only baseline: the adopted
cap-40k candidate's two members were fitted in separate runs (char with the
globally screened pool under ``artifacts/pool_cv_cap40k``, style official-only
under ``artifacts/cv_identity``). This rebuilds all five arms from the
identity-verified OOF, cross-checks every pooled and per-fold score against the
value already recorded in the artifacts, and renders the three missing panels.

Post-hoc only: fits nothing, reads no pool row, scores no test row. Peak fit RSS
is reported as ``null`` where the originating run never recorded it — it is never
estimated.
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

from src.data import (  # noqa: E402
    indices_to_labels,
    load_competition_data,
    ordered_dataset_fingerprint,
)
from src.ensemble import load_oof, normalize_scores  # noqa: E402
from src.figures import (  # noqa: E402
    make_ablation_figure,
    make_cv_variation_figure,
    make_runtime_ram_figure,
)
from src.metrics import balanced_accuracy  # noqa: E402
from src.oof_identity import fold_assignments  # noqa: E402

WEIGHTS = {"char_tfidf_logreg": 0.5, "style_only": 0.5}
MEMBERS = ["char_tfidf_logreg", "style_only"]
TOL = 1e-6


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def blend(normalized: dict[str, np.ndarray]) -> np.ndarray:
    return sum(normalized[name] * weight for name, weight in WEIGHTS.items())


def fold_scores(scores: np.ndarray, truth, folds_valid) -> list[float]:
    return [
        balanced_accuracy(truth[valid], indices_to_labels(scores[valid].argmax(axis=1)))
        for valid in folds_valid
    ]


def check(recorded: float | list[float], computed: float | list[float], what: str) -> None:
    a = np.atleast_1d(np.asarray(recorded, dtype=float))
    b = np.atleast_1d(np.asarray(computed, dtype=float))
    if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=TOL):
        raise SystemExit(
            f"[abort] {what}: recomputed {b.tolist()} != recorded {a.tolist()}; "
            "the saved OOF no longer reproduces the published candidate"
        )


def aggregate_costs(
    costs: dict[str, tuple[float, float | None]], parts: tuple[str, ...]
) -> tuple[float, float | None]:
    """A blend has no fit of its own: seconds add, peaks do not.

    The member fits run sequentially, so the blend's peak is the larger member
    peak, not the sum. One unrecorded member peak makes the blend peak unknown.
    """
    peaks = [costs[p][1] for p in parts]
    return (
        sum(costs[p][0] for p in parts),
        None if any(p is None for p in peaks) else max(peaks),
    )


def markdown_table(arms: dict[str, dict]) -> str:
    lines = [
        "| arm | pooled OOF BA | fold mean ± std | fit s (all folds) | peak fit RAM GB |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, item in arms.items():
        agg = item["fold_aggregate"]
        ram = item["peak_ram_gb"]
        lines.append(
            f"| `{name}` | {item['pooled']['balanced_accuracy']:.6f} | "
            f"{agg['mean']:.4f} ± {agg['std']:.4f} | "
            f"{item['fit_seconds']:.2f} | "
            f"{'not recorded' if ram is None else f'{ram:.3f}'} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", default="artifacts/splits/splits_official.json")
    ap.add_argument("--official-oof", default="artifacts/cv_identity/official")
    ap.add_argument("--pool-char-oof", default="artifacts/pool_cv_cap40k/official")
    ap.add_argument("--pool-results", default="artifacts/pool_cv_cap40k/official/results.json")
    ap.add_argument("--mix-verify", default="artifacts/mix_baseline_verify_cap40k")
    ap.add_argument("--base-verify", default="artifacts/mix_baseline_verify_official_only")
    ap.add_argument("--out", default="artifacts/candidate_figures")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    train, _test = load_competition_data(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    fingerprint = ordered_dataset_fingerprint(train)
    if fingerprint != splits.get("ordered_dataset_sha256"):
        raise SystemExit(
            "[abort] the local train frame does not match the split fingerprint; "
            "the saved OOF rows would be mislabelled"
        )

    official = load_oof(args.official_oof, MEMBERS)
    pool = load_oof(args.pool_char_oof, ["char_tfidf_logreg"])
    n_rows = int(official["style_only"]["y"].size)
    if n_rows != len(train):
        raise SystemExit(f"[abort] OOF has {n_rows} rows but train has {len(train)}")
    folds_valid = [np.asarray(f["valid"], dtype=int) for f in splits["folds"]]
    expected_folds = fold_assignments(
        n_rows, [(np.asarray(f["train"], dtype=int), v) for f, v in
                 zip(splits["folds"], folds_valid, strict=True)]
    )
    for source in (official, pool):
        for name, item in source.items():
            if item["ordered_dataset_sha256"] != fingerprint:
                raise SystemExit(f"[abort] {name}: OOF fingerprint differs from the splits")
            if not np.array_equal(item["validation_fold_ids"], expected_folds):
                raise SystemExit(f"[abort] {name}: OOF fold assignments differ from the splits")
            if not np.array_equal(item["y"], official["style_only"]["y"]):
                raise SystemExit(f"[abort] {name}: OOF labels differ across members")

    truth = indices_to_labels(official["style_only"]["y"])
    norm_official = {n: normalize_scores(official[n]["scores"]) for n in MEMBERS}
    norm_pool_char = normalize_scores(pool["char_tfidf_logreg"]["scores"])
    scores = {
        "char_official": norm_official["char_tfidf_logreg"],
        "char_pool_cap40k": norm_pool_char,
        "style_only": norm_official["style_only"],
        "official_blend": blend(norm_official),
        "pool_mix_cap40k": blend(
            {"char_tfidf_logreg": norm_pool_char, "style_only": norm_official["style_only"]}
        ),
    }

    metrics_dir = Path(args.official_oof) / "metrics"
    char_meta = json.loads((metrics_dir / "char_tfidf_logreg.json").read_text(encoding="utf-8"))
    style_meta = json.loads((metrics_dir / "style_only.json").read_text(encoding="utf-8"))
    pool_rows = json.loads(Path(args.pool_results).read_text(encoding="utf-8"))
    pool_meta = next(r for r in pool_rows if r["model"] == "char_tfidf_logreg")
    mix_report = json.loads(
        (Path(args.mix_verify) / "blend_official_pool.json").read_text(encoding="utf-8")
    )["reports"][0]
    base_report = json.loads(
        (Path(args.base_verify) / "blend_official_pool.json").read_text(encoding="utf-8")
    )["reports"][0]

    recorded = {
        "char_official": (
            char_meta["pooled"]["balanced_accuracy"],
            char_meta["fold_aggregate"]["fold_scores"],
        ),
        "char_pool_cap40k": (pool_meta["official_pool_ba"], pool_meta["pool_fold_ba"]),
        "style_only": (
            style_meta["pooled"]["balanced_accuracy"],
            style_meta["fold_aggregate"]["fold_scores"],
        ),
        "official_blend": (base_report["balanced_accuracy"], base_report["fold_ba"]),
        "pool_mix_cap40k": (mix_report["balanced_accuracy"], mix_report["fold_ba"]),
    }
    # A blend has no fit of its own; see aggregate_costs.
    costs = {
        "char_official": (char_meta["fit_seconds"], char_meta.get("peak_ram_gb")),
        "char_pool_cap40k": (pool_meta["fit_seconds"], pool_meta.get("peak_ram_gb")),
        "style_only": (style_meta["fit_seconds"], style_meta.get("peak_ram_gb")),
    }
    for name, parts in (
        ("official_blend", ("char_official", "style_only")),
        ("pool_mix_cap40k", ("char_pool_cap40k", "style_only")),
    ):
        costs[name] = aggregate_costs(costs, parts)

    arms: dict[str, dict] = {}
    for name, matrix in scores.items():
        pooled = balanced_accuracy(truth, indices_to_labels(matrix.argmax(axis=1)))
        folds = fold_scores(matrix, truth, folds_valid)
        check(recorded[name][0], pooled, f"{name} pooled BA")
        check(recorded[name][1], folds, f"{name} per-fold BA")
        fit_seconds, peak_ram = costs[name]
        arms[name] = {
            "pooled": {"balanced_accuracy": pooled},
            "fold_aggregate": {
                "n_folds": len(folds),
                "mean": float(np.mean(folds)),
                "std": float(np.std(folds)),
                "min": float(np.min(folds)),
                "max": float(np.max(folds)),
                "fold_scores": folds,
            },
            "fit_seconds": fit_seconds,
            "peak_ram_gb": peak_ram,
            "recorded_pooled_balanced_accuracy": recorded[name][0],
        }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "analysis": "candidate_ablation_runtime_cv_panels",
        "transductive": False,
        "fits_performed": 0,
        "competition_rows_scored": n_rows,
        "test_rows_read": 0,
        "ordered_dataset_sha256": fingerprint,
        "split_method": splits.get("method"),
        "split_seed": splits.get("seed"),
        "n_rows": n_rows,
        "blend_weights": WEIGHTS,
        "blend_rule": "row-wise z (src.ensemble.normalize_scores), fixed 0.5/0.5",
        "sources": {
            "char_official": str(Path(args.official_oof) / "oof" / "char_tfidf_logreg.npz"),
            "char_pool_cap40k": str(Path(args.pool_char_oof) / "oof" / "char_tfidf_logreg.npz"),
            "style_only": str(Path(args.official_oof) / "oof" / "style_only.npz"),
            "timings_official": str(metrics_dir),
            "timings_pool": args.pool_results,
            "recorded_blend_candidate": args.mix_verify,
            "recorded_blend_fallback": args.base_verify,
        },
        "arms": arms,
        "peak_ram_note": (
            "null means the originating run never recorded peak fit RSS; it is not "
            "estimated. A blend's fit_seconds is the sum of its members' fit seconds "
            "and its peak is the larger member peak, because the member fits are "
            "sequential rather than co-resident. Inference cost is a separate "
            "measurement (artifacts/inference_benchmark_refresh/results.json)."
        ),
        "reproduction_check": f"every pooled and per-fold score matched the recorded value within {TOL}",
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    table = markdown_table(arms)
    (out / "candidate_panels.md").write_text(table + "\n", encoding="utf-8")

    figures: list[str] = []
    if not args.no_figures:
        figures = [
            make_ablation_figure(arms, args.figures_dir, dataset_label="pool_mix_cap40k"),
            make_runtime_ram_figure(arms, args.figures_dir, dataset_label="pool_mix_cap40k"),
            make_cv_variation_figure(arms, args.figures_dir, dataset_label="pool_mix_cap40k"),
        ]
        payload["figures"] = figures
        (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(table)
    print(f"\n{payload['reproduction_check']}")
    print(f"wrote {out / 'results.json'}")
    for path in figures:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()