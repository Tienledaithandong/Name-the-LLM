"""External-pool audit + per-fold decontamination (Phase 1a).

Audits the Open Pool's schema and recorded flags and computes fold-specific
screening masks using an evidence-backed prompt crosswalk. Similarity screening
is heuristic and does not establish that all prompt overlaps were detected.

This is an audit, not a training run: no score is produced. The pool is public
CC BY-SA 4.0 community data; downloading is an owner-account action (the owner
delegated this audit). No competition rows leave the machine.

Run:
    python experiments/pool_audit.py --data-dir data --prompt-mapping <reviewed-crosswalk.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import ARTIFACTS_DIR
from src.data import load_competition_data
from src.pool import PoolError, audit_pool, load_pool, load_prompt_mapping, per_fold_decontamination
from src.validation import make_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--pool", default="artifacts/pool_audit/ntl_open_pool.parquet")
    parser.add_argument("--prompt-mapping", required=True,
                        help="reviewed JSON crosswalk: provenance and official prompt IDs to pool canonical IDs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cos-threshold", type=float, default=0.3)
    parser.add_argument("--strict", action="store_true",
                        help="also drop flag_kb_proxy_after_ruling rows")
    args = parser.parse_args()

    mapping, provenance = load_prompt_mapping(args.prompt_mapping)

    train_df, _ = load_competition_data(args.data_dir)
    missing = set(train_df["prompt_id"].astype(str)) - set(mapping)
    if missing:
        raise PoolError(f"verified prompt mapping lacks {len(missing)} official prompt(s)")
    pool = load_pool(args.pool, strict=args.strict)
    print(f"pool: {len(pool)} rows (strict={args.strict})", flush=True)

    audit = audit_pool(pool, train_df)
    print("\n=== Pool audit (verified against actual rows) ===")
    print(f"rows={audit['n_rows']} cols={audit['n_columns']} labels_complete={audit['labels_complete']}")
    print(f"max_cos_train max={audit['max_cos_train_max']:.4f} mean={audit['max_cos_train_mean']:.4f}")
    print(f"exact text duplicates vs train: {audit['exact_text_duplicates_vs_train']}")
    print(f"null text: {audit['null_text']}; n_words {audit['n_words_min']}-{audit['n_words_max']}")
    for col, vals in audit["flags"].items():
        print(f"  {col}: {vals}")

    folds, method = make_splits(train_df, n_splits=5, seed=args.seed)
    print(f"\n=== Per-fold decontamination ({method}, seed {args.seed}, cos>{args.cos_threshold}) ===",
          flush=True)
    decon = per_fold_decontamination(
        pool, train_df, folds, cos_threshold=args.cos_threshold,
        prompt_mapping=mapping, mapping_provenance=provenance,
    )
    print(f"{'fold':>4} {'holdout':>8} {'exact':>6} {'shared':>7} {'near':>6} {'removed':>8} {'kept':>8} {'maxcos':>8}")
    for f in decon["per_fold"]:
        print(f"{f['fold']:>4} {f['n_holdout']:>8} {f['exact_dup']:>6} {f['shared_prompt']:>7} "
              f"{f['near_dup']:>6} {f['removed']:>8} {f['kept']:>8} {f['max_cos_holdout_max']:>8.4f}")
    print(f"\nunion removed across folds: {decon['union_removed']} | screened pool rows: {decon['clean_pool_rows']}")
    print("Similarity screening and crosswalk structure do not prove the source evidence is complete.")

    out_dir = ARTIFACTS_DIR / "pool_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"pool_path": str(args.pool), "strict": args.strict, "seed": args.seed,
               "split_method": method, "audit": audit, "decontamination": decon}
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    # Diagnostic only: union depends on validation rows and must not feed CV.
    removed = set(decon["union_removed_ids"])
    clean_ids = pool.loc[~pool["id"].astype(str).isin(removed), "id"].astype(str).tolist()
    (out_dir / "clean_pool_ids.json").write_text(json.dumps(clean_ids), encoding="utf-8")
    print(f"\nwrote {out_dir / 'audit.json'} and diagnostic-only clean_pool_ids.json ({len(clean_ids)} ids); do not use for CV")


if __name__ == "__main__":
    main()
