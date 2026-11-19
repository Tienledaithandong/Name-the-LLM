"""Persist nested group splits without training, selection, or scoring.

The plan is prospective scaffolding; earlier model choices already used all
official prompts, so this alone is not a pristine independent test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_competition_data, ordered_dataset_fingerprint  # noqa: E402
from src.nested_validation import make_nested_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", type=Path, default=Path("artifacts/splits/nested_grouped.json"))
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--outer-seed", type=int, default=121)
    parser.add_argument("--inner-seed", type=int, default=212)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing split plan: {args.out}")
    df, _ = load_competition_data(args.data_dir)
    plans = make_nested_splits(df, outer_splits=args.outer_splits, inner_splits=args.inner_splits,
                               outer_seed=args.outer_seed, inner_seed=args.inner_seed)
    payload = {
        "schema_version": "nested-grouped-splits/v1",
        "ordered_dataset_sha256": ordered_dataset_fingerprint(df),
        "outer_seed": args.outer_seed, "inner_seed": args.inner_seed,
        "note": "Split plan only; outer evaluation is not complete.",
        "folds": [{"outer_fold": p["outer_fold"], "outer_method": p["outer_method"],
                   "train": p["train"].tolist(), "valid": p["valid"].tolist(),
                   "inner_method": p["inner_method"],
                   "inner": [{"train": tr.tolist(), "valid": va.tolist()}
                             for tr, va in p["inner"]]} for p in plans],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"Prepared {len(plans)} outer folds with {args.inner_splits} inner folds each: {args.out}")
    print("No training, selection or scoring performed.")


if __name__ == "__main__":
    main()
