"""Fold-aligned DeBERTa + sparse blend test (D022 step 2).

Loads the DeBERTa OOF (log-probability scores) and the local sparse OOF
(char_tfidf_logreg + style_only margins) and tests whether DeBERTa adds diversity
to the adopted 0.376 sparse blend, even when its standalone BA is lower. Blending
is only valid when both OOFs use the SAME grouped folds, so this script REFUSES to
blend unless the DeBERTa run reports a ``*-shipped`` split method (i.e. it loaded
``kaggle/deberta/folds.json`` = local cv_s42). Reports pooled BA, per-class recall,
within-provider errors and paired per-fold diffs + 95% CI for each blend.

Run (after a DeBERTa kernel run with shipped folds):
    python experiments/blend_deberta.py \
        --deberta-dir artifacts/deberta_kernel_output/deberta_cv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import LABELS, N_CLASSES
from src.data import indices_to_labels
from src.ensemble import normalize_scores
from src.metrics import (
    balanced_accuracy,
    mean_ci,
    paired_difference,
    per_class_recall,
    within_provider_confusions,
)


def load_oof(path: Path):
    d = np.load(path, allow_pickle=True)
    scores = np.asarray(d["scores"], dtype=np.float64)
    y = np.asarray(d["y"], dtype=np.int64)
    labels = [str(v) for v in d["labels"]]
    if labels != list(LABELS):
        raise ValueError(f"{path}: class order is not the official order")
    return scores, y


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
    ap.add_argument("--splits", default="artifacts/splits/splits_official.json")
    ap.add_argument("--variant", default="official_pool",
                    choices=["official_only", "official_pool"])
    ap.add_argument("--out", default="artifacts/deberta_blend")
    args = ap.parse_args()

    ddir = Path(args.deberta_dir)
    results = json.loads((ddir / "results.json").read_text(encoding="utf-8"))
    method = results.get("split_method", "")
    if not method.endswith(("-shipped", "-embedded")):
        raise SystemExit(
            f"[abort] DeBERTa split_method={method!r} is not '*-shipped'/'*-embedded'; its "
            "folds are not aligned to local cv_s42 (Kaggle sklearn shift). Re-run the kernel "
            "with the embedded/shipped local folds before blending."
        )

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    folds_valid = [np.asarray(f["valid"], dtype=int) for f in splits["folds"]]

    deb_scores, deb_y = load_oof(ddir / "oof" / f"{args.variant}.npz")
    sparse_dir = Path(args.sparse_oof)
    char_scores, char_y = load_oof(sparse_dir / "char_tfidf_logreg.npz")
    style_scores, style_y = load_oof(sparse_dir / "style_only.npz")
    if not (np.array_equal(deb_y, char_y) and np.array_equal(char_y, style_y)):
        raise SystemExit("[abort] OOF row labels are not aligned across models")

    truth = indices_to_labels(deb_y)
    norm = {
        "deberta": normalize_scores(deb_scores),
        "char": normalize_scores(char_scores),
        "style": normalize_scores(style_scores),
    }
    sparse_blend = 0.5 * norm["char"] + 0.5 * norm["style"]

    def report(name, scores):
        pred = indices_to_labels(scores.argmax(axis=1))
        ba = balanced_accuracy(truth, pred)
        fba = fold_ba(scores, deb_y, folds_valid)
        return {"name": name, "balanced_accuracy": ba, "fold_ba": fba,
                "per_class_recall": per_class_recall(truth, pred),
                "within_provider": within_provider_confusions(truth, pred)}

    candidates = {
        "sparse_adopted": sparse_blend,
        "deberta_only": norm["deberta"],
        "sparse+deberta_50": 0.5 * sparse_blend + 0.5 * norm["deberta"],
        "sparse+deberta_70": 0.7 * sparse_blend + 0.3 * norm["deberta"],
        "sparse+deberta_30": 0.3 * sparse_blend + 0.7 * norm["deberta"],
    }
    reports = [report(n, s) for n, s in candidates.items()]
    base = reports[0]
    for r in reports[1:]:
        diffs = paired_difference(base["fold_ba"], r["fold_ba"])
        r["paired_vs_sparse"] = mean_ci(diffs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"variant": args.variant, "split_method": method, "reports": reports}
    (out_dir / f"blend_{args.variant}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"DeBERTa variant={args.variant} split={method}")
    print(f"{'blend':24s} {'BA':>8s} {'vs sparse (paired 95% CI)':>32s}")
    for r in reports:
        ci = r.get("paired_vs_sparse")
        cis = "" if ci is None else f"{ci['mean']:+.4f} [{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}]"
        print(f"{r['name']:24s} {r['balanced_accuracy']:8.4f} {cis:>32s}")
    print(f"wrote {out_dir / f'blend_{args.variant}.json'}")


if __name__ == "__main__":
    main()