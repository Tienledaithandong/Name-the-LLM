"""Text-length sensitivity of the adopted blends (plan §9 item 9).

Post-hoc error analysis on saved, identity-verified out-of-fold predictions: no
model is fit here, no external pool row is read and no test row is scored. Bands
are word counts of the official training passage, so the same rows are compared
across variants and the per-band difference is paired.

    python experiments/length_sensitivity.py --data-dir data

Writes ``artifacts/length_sensitivity/results.json`` plus a markdown table, and
renders the length-band figure and the candidate's confusion/recall figures.
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

from src.constants import LABELS  # noqa: E402
from src.data import (  # noqa: E402
    indices_to_labels,
    load_competition_data,
    ordered_dataset_fingerprint,
)
from src.ensemble import load_oof, normalize_scores  # noqa: E402
from src.figures import make_length_band_figure, make_prediction_figures  # noqa: E402
from src.metrics import (  # noqa: E402
    balanced_accuracy,
    confusion_matrix,
    length_band_metrics,
    per_class_recall,
    within_provider_confusions,
)
from src.oof_identity import fold_assignments  # noqa: E402

# The official corpus spans 36-274 words, so the top band is left open-ended to
# keep every band above 300 rows instead of scoring a 25-row tail.
EDGES = (0, 50, 100, 150, 200)
WEIGHTS = {"char_tfidf_logreg": 0.5, "style_only": 0.5}
MEMBERS = ["char_tfidf_logreg", "style_only"]


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def blend(normalized: dict[str, np.ndarray], members: dict[str, float]) -> np.ndarray:
    return sum(normalized[name] * weight for name, weight in members.items())


def top_confusion_pairs(truth, pred, limit: int = 6) -> list[dict]:
    matrix = confusion_matrix(truth, pred)
    pairs = [
        {"true": LABELS[i], "pred": LABELS[j], "count": int(matrix[i, j])}
        for i in range(len(LABELS))
        for j in range(len(LABELS))
        if i != j and matrix[i, j] > 0
    ]
    return sorted(pairs, key=lambda p: -p["count"])[:limit]


def markdown_table(variants: dict[str, dict]) -> str:
    lines = [
        "| words | n | classes | " + " | ".join(variants) + " | MIX − official |",
        "|---|---|---|" + "---|" * (len(variants) + 1),
    ]
    reference = variants["official_blend"]["bands"]
    candidate = variants["pool_mix_cap40k"]["bands"]
    for i, band in enumerate(reference):
        cells = []
        for payload in variants.values():
            value = payload["bands"][i]["balanced_accuracy"]
            cells.append("—" if value is None else f"{value:.4f}")
        base = band["balanced_accuracy"]
        other = candidate[i]["balanced_accuracy"]
        delta = None if base is None or other is None else other - base
        shown = "—" if delta is None else f"{delta:+.4f}"
        lines.append(
            f"| {band['band']} | {band['n']} | {band['classes_present']} | "
            + " | ".join(cells)
            + f" | {shown} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", default="artifacts/splits/splits_official.json")
    ap.add_argument("--official-oof", default="artifacts/cv_identity/official")
    ap.add_argument("--pool-char-oof", default="artifacts/pool_cv_cap40k/official")
    ap.add_argument("--out", default="artifacts/length_sensitivity")
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
    folds_full = [
        (np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int))
        for f in splits["folds"]
    ]
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

    variants: dict[str, dict] = {}
    for name, matrix in scores.items():
        pred = indices_to_labels(matrix.argmax(axis=1))
        variants[name] = {
            "pooled_balanced_accuracy": balanced_accuracy(truth, pred),
            "within_provider": within_provider_confusions(truth, pred),
            "per_class_recall": per_class_recall(truth, pred),
            "top_confusion_pairs": top_confusion_pairs(truth, pred),
            "bands": length_band_metrics(truth, pred, words, edges=EDGES),
        }

    band_delta = []
    for a, b in zip(
        variants["official_blend"]["bands"], variants["pool_mix_cap40k"]["bands"],
        strict=True,
    ):
        base, other = a["balanced_accuracy"], b["balanced_accuracy"]
        band_delta.append(
            {
                "band": a["band"],
                "n": a["n"],
                "official_blend": base,
                "pool_mix_cap40k": other,
                "delta": None if base is None or other is None else other - base,
            }
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "analysis": "length_band_error_analysis",
        "transductive": False,
        "fits_performed": 0,
        "ordered_dataset_sha256": fingerprint,
        "split_method": splits.get("method"),
        "split_seed": splits.get("seed"),
        "n_rows": n_rows,
        "band_edges_words": list(EDGES),
        "length_words_min": int(words.min()),
        "length_words_max": int(words.max()),
        "length_words_median": float(np.median(words)),
        "blend_weights": WEIGHTS,
        "length_source": "official train text, whitespace word count (src.audit rule)",
        "variants": variants,
        "paired_band_delta": band_delta,
        "note": (
            "Band balanced accuracy averages only the classes present in that band, "
            "so bands are not directly comparable to each other or to the pooled score. "
            "The delta column is paired: both blends score the same rows."
        ),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    table = markdown_table(variants)
    (out / "length_bands.md").write_text(table + "\n", encoding="utf-8")

    figures: list[str] = []
    if not args.no_figures:
        figures.append(
            make_length_band_figure(
                {k: variants[k]["bands"] for k in ("official_blend", "pool_mix_cap40k")},
                args.figures_dir,
            )
        )
        figures.extend(
            make_prediction_figures(
                truth,
                indices_to_labels(scores["pool_mix_cap40k"].argmax(axis=1)),
                args.figures_dir,
                dataset_label="pool_mix_cap40k",
                blend_name="0.5 z(char pool cap40k) + 0.5 z(style official)",
            )
        )
        payload["figures"] = figures
        (out / "results.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(table)
    print()
    for name, item in variants.items():
        wp = item["within_provider"]
        print(
            f"{name:18s} pooled BA {item['pooled_balanced_accuracy']:.6f} "
            f"errors {wp['errors']} within {wp['within_provider_errors']} "
            f"cross {wp['cross_provider_errors']}"
        )
    print(f"\nwrote {out / 'results.json'}")
    for path in figures:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()