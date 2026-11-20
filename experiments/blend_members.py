"""Does a THIRD member help the cap-40k pool-MIX blend? (priority 5, larger ensembles)

The adopted recipe is ``0.5 z(char_tfidf_logreg + global pool) + 0.5 z(style_only)``.
The pool lifted the char member from 0.3193 to 0.4104 while style stayed
official-only at 0.2904, so every cheap official-only member is now much weaker
than the blend it would join. This script measures that dilution instead of
assuming it: for each candidate it reports the solo pooled BA, the row-z
correlation with the adopted blend (redundancy), and three **pre-specified**
recipes -- equal thirds, 0.4/0.4/0.2 (which preserves the adopted char:style
ratio) and a swap of the style member -- each with a paired five-fold CI against
the two-member base.

DESCRIPTIVE ONLY. Picking the best arm off these curves and shipping it would be
selection on the same OOF that reports the score (D026). An arm could only be
adopted after its own grouped outer-fold gate plus an alternative seed. Members
whose OOF lacks the D024 provenance keys are skipped, never blended: an unverified
fold assignment cannot be paired against the base.

Three pre-specified recipes are compared: equal thirds, 0.4/0.4/0.2 (which keeps
the adopted char:style ratio), and a **swap** that replaces the style member
outright (0.5 char + 0.5 candidate). The swap answers a different question from
the additions -- whether the candidate is a better *partner* for the pool-trained
char member than style is -- and matters most for a candidate that was itself
trained on the pool.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.blend_weights import load_identity_oof
from src.constants import DEFAULT_REGISTRY
from src.data import indices_to_labels
from src.ensemble import normalize_scores
from src.metrics import evaluate, mean_ci, paired_difference
from src.repro import git_commit
from src.train import append_registry

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHAR_OOF = "artifacts/pool_cv_cap40k/official/oof/char_tfidf_logreg.npz"
DEFAULT_STYLE_OOF = "artifacts/cv_identity/official/oof/style_only.npz"
DEFAULT_MEMBER_OOFS = (
    "artifacts/cv_blend_members/oof/word_tfidf_linearsvc.npz",
    "artifacts/cv_blend_members/oof/char_wb_tfidf_linearsvc.npz",
    "artifacts/cv_style_svm/oof/style_svm.npz",
    "artifacts/cv_style_catboost/oof/style_catboost.npz",
    "artifacts/cv_blend_embed/oof/minilm_embedding_style_logreg.npz",
    "artifacts/cv_blend_embed/oof/minilm_embedding_logreg.npz",
    "artifacts/pool_cv_word40k/oof/word_tfidf_linearsvc.npz",
)
THIRD_WEIGHT = 0.2


def fold_balanced_accuracies(z: np.ndarray, truth: np.ndarray, folds: np.ndarray) -> tuple[float, list[float]]:
    """Pooled BA and per-fold BA of a row-wise-z score matrix under argmax."""
    per = [evaluate(truth[folds == f], indices_to_labels(z[folds == f].argmax(1)))["balanced_accuracy"]
           for f in np.unique(folds)]
    pooled = evaluate(truth, indices_to_labels(z.argmax(1)))["balanced_accuracy"]
    return float(pooled), [float(v) for v in per]


def screen_members(char_z: np.ndarray, style_z: np.ndarray, truth: np.ndarray, folds: np.ndarray,
                   members: dict[str, np.ndarray]) -> dict:
    """Compare pre-specified three-member recipes against the adopted 0.5/0.5 blend."""
    base_z = 0.5 * char_z + 0.5 * style_z
    base_ba, base_per = fold_balanced_accuracies(base_z, truth, folds)
    out = {"base_pooled_ba": base_ba, "base_fold_ba": base_per, "members": []}
    for name, scores in members.items():
        mz = normalize_scores(scores)
        solo, _ = fold_balanced_accuracies(mz, truth, folds)
        rec = {"member": name, "solo_pooled_ba": solo,
               "rowz_corr_with_base": float(np.corrcoef(mz.ravel(), base_z.ravel())[0, 1])}
        for tag, z, weights in (
            ("thirds", (char_z + style_z + mz) / 3.0, (1 / 3, 1 / 3, 1 / 3)),
            ("keep_ratio", (1.0 - THIRD_WEIGHT) * 0.5 * (char_z + style_z) + THIRD_WEIGHT * mz,
             ((1.0 - THIRD_WEIGHT) * 0.5, (1.0 - THIRD_WEIGHT) * 0.5, THIRD_WEIGHT)),
            ("swap", 0.5 * char_z + 0.5 * mz, (0.5, 0.0, 0.5)),
        ):
            ba, per = fold_balanced_accuracies(z, truth, folds)
            rec[tag] = {
                "char_weight": weights[0], "style_weight": weights[1], "third_weight": weights[2],
                "pooled_ba": ba, "fold_ba": per,
                "paired_diff_vs_base": mean_ci(paired_difference(base_per, per)),
                "all_folds_non_negative": bool(np.all(np.asarray(per) >= np.asarray(base_per))),
            }
        out["members"].append(rec)
    return out


def load_members(paths: list[str], fingerprint: str, y: np.ndarray, folds: np.ndarray,
                 root: Path = ROOT) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Load only OOF whose fingerprint, labels and fold ids match the base members.

    Two runs can export the same member name (e.g. an official-only and a
    pool-trained ``word_tfidf_linearsvc``). Colliding stems are qualified by their
    run directory instead of overwriting each other, so no arm is silently lost.
    """
    loaded, skipped = [], []
    for rel in paths:
        path = root / rel
        name = path.stem
        if not path.is_file():
            skipped.append({"member": name, "path": rel, "reason": "file missing"})
            continue
        try:
            scores, my, fp, mfolds = load_identity_oof(path)
        except ValueError as exc:
            skipped.append({"member": name, "path": rel, "reason": str(exc)})
            continue
        if fp != fingerprint or not np.array_equal(my, y) or not np.array_equal(mfolds, folds):
            skipped.append({"member": name, "path": rel, "reason": "identity/fold mismatch"})
            continue
        loaded.append((name, rel, scores))
    counts = Counter(name for name, _, _ in loaded)
    members = {}
    for name, rel, scores in loaded:
        key = name if counts[name] == 1 else f"{Path(rel).parent.parent.name}/{name}"
        members[key] = scores
    return members, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--char-oof", default=DEFAULT_CHAR_OOF)
    parser.add_argument("--style-oof", default=DEFAULT_STYLE_OOF)
    parser.add_argument("--member-oof", action="append", default=None,
                        help="candidate OOF path; repeatable (default: the seven verified arms)")
    parser.add_argument("--seed", type=int, default=42,
                        help="label only: must be the seed the supplied OOF was produced with")
    parser.add_argument("--out", type=Path, default=Path("artifacts/blend_members/results.json"))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--no-registry", action="store_true")
    args = parser.parse_args()

    char, y, fp, folds = load_identity_oof(ROOT / args.char_oof)
    style, sy, sfp, sfolds = load_identity_oof(ROOT / args.style_oof)
    if (fp, sy.tolist(), sfolds.tolist()) != (sfp, y.tolist(), folds.tolist()):
        raise ValueError("char and style OOF are not the same rows/folds")
    paths = args.member_oof if args.member_oof else list(DEFAULT_MEMBER_OOFS)
    members, skipped = load_members(paths, fp, y, folds)
    if not members:
        raise ValueError("no candidate OOF passed the identity check; nothing to screen")

    started = time.perf_counter()
    truth = indices_to_labels(y)
    result = screen_members(normalize_scores(char), normalize_scores(style), truth, folds, members)
    result.update({
        "mode": "descriptive third-member screen",
        "warning": ("DESCRIPTIVE ONLY: choosing an arm from these curves is selection on the reporting "
                    "OOF (D026). Adoption needs its own grouped outer-fold gate plus an alternative seed."),
        "dataset": "official", "ordered_dataset_sha256": fp, "n_rows": int(len(y)),
        "seed": args.seed,
        "char_oof": args.char_oof, "style_oof": args.style_oof,
        "skipped": skipped, "elapsed_seconds": round(time.perf_counter() - started, 2),
    })

    print(f"base 0.5/0.5 pool-MIX pooled {result['base_pooled_ba']:.6f}", flush=True)

    def _fmt(rec: dict, tag: str) -> str:
        v = rec[tag]
        ci = v["paired_diff_vs_base"]
        return (f"{tag} {v['pooled_ba']:.6f} ({ci['mean']:+.4f}"
                f" [{ci['ci_low']:+.4f},{ci['ci_high']:+.4f}])")

    for rec in result["members"]:
        print(f"  {rec['member']:32s} solo {rec['solo_pooled_ba']:.4f}"
              f" corr {rec['rowz_corr_with_base']:.3f} | "
              + " | ".join(_fmt(rec, t) for t in ("thirds", "keep_ratio", "swap")), flush=True)
    for s in skipped:
        print(f"  SKIP {s['member']}: {s['reason']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    if not args.no_registry:
        best = max(result["members"], key=lambda r: r["keep_ratio"]["pooled_ba"])
        bk = best["keep_ratio"]
        bswap = max(result["members"], key=lambda r: r["swap"]["pooled_ba"])
        bs = bswap["swap"]
        append_registry(args.registry, {
            "experiment_id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-official-blend_members-s{args.seed}",
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(ROOT),
            "method": "blend_third_member_screen",
            "features": "0.5 z(char+pool40k) + 0.5 z(style) + third member [descriptive]",
            "seed": args.seed, "split_id": f"stratified_group-s{args.seed}-n5",
            "fold_scores": " ".join(f"{v:.4f}" for v in bk["fold_ba"]),
            "pooled_oof_balanced_accuracy": round(bk["pooled_ba"], 6),
            "macro_f1": "", "fit_seconds": result["elapsed_seconds"],
            "infer_seconds": "", "peak_ram_gb": "", "external_data_id": "open-pool-v15",
            "artifact_path": str(args.out), "public_lb_score": "",
            "notes": (f"Priority 5 (larger ensembles) vs the cap-40k pool-MIX base "
                      f"{result['base_pooled_ba']:.6f}. {len(result['members'])} identity-verified arms, "
                      f"{len(skipped)} skipped for missing D024 provenance. Pre-specified equal thirds, "
                      f"0.4/0.4/0.2 and swap (0.5 char + 0.5 candidate, style dropped); best keep-ratio arm "
                      f"{best['member']} {bk['pooled_ba']:.6f} "
                      f"(paired {bk['paired_diff_vs_base']['mean']:+.6f} "
                      f"[{bk['paired_diff_vs_base']['ci_low']:+.6f},{bk['paired_diff_vs_base']['ci_high']:+.6f}]); "
                      f"best swap arm {bswap['member']} {bs['pooled_ba']:.6f} "
                      f"(paired {bs['paired_diff_vs_base']['mean']:+.6f} "
                      f"[{bs['paired_diff_vs_base']['ci_low']:+.6f},{bs['paired_diff_vs_base']['ci_high']:+.6f}]). "
                      f"DESCRIPTIVE screen, not a selection score (D026); no blend/config/package change."),
            "status": "done",
        })


if __name__ == "__main__":
    main()