"""Fold-aligned DeBERTa + sparse blend test (D022 step 2, baseline updated to D037).

Loads the DeBERTa OOF (log-probability scores) and two sparse members and tests
whether DeBERTa adds diversity to the **adopted candidate**. With the default
``--sparse-oof`` that baseline is the official-only 0.376 char+style blend; pointing
``--char-oof`` at the pool-trained cap-40k artifact makes it the adopted pool-MIX
candidate (D037, 0.4427), which is the bar any new member must beat. Blending
is only valid when the OOF artifacts have the same ordered dataset fingerprint
and exact validation folds. Legacy artifacts without this provenance must be
rerun. Reports pooled BA, per-class recall,
within-provider errors and paired per-fold diffs + 95% CI for each blend.

Verify the adopted baseline reproduces from stored artifacts (no GPU run needed):
    python experiments/blend_deberta.py --no-deberta \
        --char-oof artifacts/pool_cv_cap40k/official/oof/char_tfidf_logreg.npz \
        --style-oof artifacts/cv_identity/official/oof/style_only.npz

Score a DeBERTa arm against it (after a kernel run with embedded/shipped folds):
    python experiments/blend_deberta.py \
        --deberta-dir artifacts/deberta_kernel_output/deberta_cv \
        --char-oof artifacts/pool_cv_cap40k/official/oof/char_tfidf_logreg.npz \
        --style-oof artifacts/cv_identity/official/oof/style_only.npz
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import LABELS, N_CLASSES
from src.data import indices_to_labels
from src.ensemble import normalize_scores
from src.oof_identity import fold_assignments
from src.metrics import (
    balanced_accuracy,
    mean_ci,
    paired_difference,
    per_class_recall,
    within_provider_confusions,
)


def load_oof(path: Path):
    with np.load(path, allow_pickle=True) as d:
        required = {"scores", "y", "labels", "ordered_dataset_sha256", "validation_fold_ids"}
        missing = required - set(d.files)
        if missing:
            raise ValueError(f"{path}: OOF provenance missing {sorted(missing)}; rerun the model")
        scores = np.asarray(d["scores"], dtype=np.float64)
        y = np.asarray(d["y"], dtype=np.int64)
        labels = [str(v) for v in d["labels"]]
        fingerprint = str(np.asarray(d["ordered_dataset_sha256"]).item())
        folds = np.asarray(d["validation_fold_ids"], dtype=np.int64)
    if labels != list(LABELS):
        raise ValueError(f"{path}: class order is not the official order")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError(f"{path}: invalid ordered dataset fingerprint")
    if y.ndim != 1 or scores.shape != (len(y), N_CLASSES) or folds.shape != y.shape:
        raise ValueError(f"{path}: inconsistent OOF dimensions")
    if not np.isfinite(scores).all():
        raise ValueError(f"{path}: OOF scores must be finite")
    return scores, y, fingerprint, folds


def verify_oof_alignment(artifacts, expected_folds, expected_fingerprint):
    """Require all models and the split artifact to describe the same rows/folds."""
    if not artifacts:
        raise ValueError("at least one OOF artifact is required")
    if not isinstance(expected_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint):
        raise ValueError("split ordered dataset fingerprint is missing or invalid; regenerate splits")
    _, reference_y, reference_hash, reference_folds = artifacts[0]
    if reference_hash != expected_fingerprint:
        raise ValueError("OOF ordered dataset fingerprint differs from requested splits")
    if not np.array_equal(reference_folds, expected_folds):
        raise ValueError("OOF validation folds differ from requested splits")
    for _, y, fingerprint, folds in artifacts[1:]:
        if fingerprint != reference_hash or not np.array_equal(y, reference_y):
            raise ValueError("OOF ordered dataset fingerprint or labels differ")
        if not np.array_equal(folds, reference_folds):
            raise ValueError("OOF validation-fold assignments differ across models")


def fold_ba(scores, y, folds_valid):
    truth = indices_to_labels(y)
    out = []
    for va in folds_valid:
        pred = indices_to_labels(scores[va].argmax(axis=1))
        out.append(balanced_accuracy(truth[va], pred))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deberta-dir", default="artifacts/deberta_kernel_output/deberta_cv")
    ap.add_argument("--sparse-oof", default="artifacts/cv/official/oof")
    ap.add_argument("--char-oof", default=None,
                    help="override the char member OOF path; point it at the pool-trained "
                         "cap-40k artifact to make the baseline the adopted MIX candidate (D037)")
    ap.add_argument("--style-oof", default=None,
                    help="override the style member OOF path (official-only in the MIX recipe)")
    ap.add_argument("--no-deberta", action="store_true",
                    help="score only the baseline blend, without loading a DeBERTa kernel run")
    ap.add_argument("--splits", default="artifacts/splits/splits_official.json")
    ap.add_argument("--variant", default="official_pool",
                    choices=["official_only", "official_pool"])
    ap.add_argument("--out", default="artifacts/deberta_blend")
    args = ap.parse_args()

    artifacts = []
    deb_artifact = None
    if not args.no_deberta:
        ddir = Path(args.deberta_dir)
        results = json.loads((ddir / "results.json").read_text(encoding="utf-8"))
        method = results.get("split_method", "")
        if not method.endswith(("-shipped", "-embedded")):
            raise SystemExit(
                f"[abort] DeBERTa split_method={method!r} is not '*-shipped'/'*-embedded'; its "
                "folds are not aligned to local cv_s42 (Kaggle sklearn shift). Re-run the kernel "
                "with the embedded/shipped local folds before blending."
            )
        deb_artifact = load_oof(ddir / "oof" / f"{args.variant}.npz")
        artifacts.append(deb_artifact)
    else:
        method = "not-loaded"

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    folds_valid = [np.asarray(f["valid"], dtype=int) for f in splits["folds"]]
    folds_full = [(np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int))
                  for f in splits["folds"]]

    sparse_dir = Path(args.sparse_oof)
    char_path = Path(args.char_oof) if args.char_oof else sparse_dir / "char_tfidf_logreg.npz"
    style_path = Path(args.style_oof) if args.style_oof else sparse_dir / "style_only.npz"
    char_artifact = load_oof(char_path)
    style_artifact = load_oof(style_path)
    artifacts += [char_artifact, style_artifact]
    char_scores, char_y, _, _ = char_artifact
    style_scores, _, _, _ = style_artifact
    verify_oof_alignment(artifacts, fold_assignments(len(char_y), folds_full),
                         splits.get("ordered_dataset_sha256"))

    truth = indices_to_labels(char_y)
    norm = {"char": normalize_scores(char_scores), "style": normalize_scores(style_scores)}
    if deb_artifact is not None:
        norm["deberta"] = normalize_scores(deb_artifact[0])
    sparse_blend = 0.5 * norm["char"] + 0.5 * norm["style"]

    def report(name, scores):
        pred = indices_to_labels(scores.argmax(axis=1))
        ba = balanced_accuracy(truth, pred)
        fba = fold_ba(scores, char_y, folds_valid)
        return {"name": name, "balanced_accuracy": ba, "fold_ba": fba,
                "per_class_recall": per_class_recall(truth, pred),
                "within_provider": within_provider_confusions(truth, pred)}

    candidates = {"mix_baseline": sparse_blend}
    if deb_artifact is not None:
        candidates.update({
            "deberta_only": norm["deberta"],
            "mix+deberta_50": 0.5 * sparse_blend + 0.5 * norm["deberta"],
            "mix+deberta_70": 0.7 * sparse_blend + 0.3 * norm["deberta"],
            "mix+deberta_30": 0.3 * sparse_blend + 0.7 * norm["deberta"],
        })
    reports = [report(n, s) for n, s in candidates.items()]
    base = reports[0]
    for r in reports[1:]:
        diffs = paired_difference(base["fold_ba"], r["fold_ba"])
        r["paired_vs_mix"] = mean_ci(diffs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"variant": args.variant, "split_method": method,
               "char_oof": str(char_path), "style_oof": str(style_path), "reports": reports}
    (out_dir / f"blend_{args.variant}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"DeBERTa variant={args.variant} split={method}")
    print(f"char={char_path}\nstyle={style_path}")
    print(f"{'blend':24s} {'BA':>8s} {'vs MIX baseline (paired 95% CI)':>34s}")
    for r in reports:
        ci = r.get("paired_vs_mix")
        cis = "" if ci is None else f"{ci['mean']:+.4f} [{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}]"
        print(f"{r['name']:24s} {r['balanced_accuracy']:8.4f} {cis:>34s}")
    print(f"wrote {out_dir / f'blend_{args.variant}.json'}")


if __name__ == "__main__":
    main()
