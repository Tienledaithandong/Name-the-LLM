"""Cross-fold near-duplicate audit of the official training text (plan §2 L94).

The data audit reports exact duplicates and the grouped split guarantees zero
prompt overlap, but near-duplicate similarity *within* the official corpus had
never been measured. This is a post-hoc audit: it fits no predictor, produces no
score, reads no test row and stores ids and similarities only — never passage
text. The similarity vectorizer uses the same settings and 0.3 threshold as the
external-pool screen (``src.pool.decontaminate_global``), so the two audits are
comparable.

    python experiments/near_duplicates.py --data-dir data

Writes ``artifacts/near_duplicates/results.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_competition_data, ordered_dataset_fingerprint  # noqa: E402
from src.oof_identity import fold_assignments  # noqa: E402
from src.repro import env_snapshot, git_commit  # noqa: E402

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
MAX_FEATURES = 30000
CHUNK = 500
TOP_PAIRS = 15


def pair_audit(
    texts: list[str],
    folds: np.ndarray,
    prompts: np.ndarray,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    chunk: int = CHUNK,
) -> dict:
    """Count similar official-train pairs and split them by fold and prompt."""
    n = len(texts)
    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=MAX_FEATURES,
        sublinear_tf=True, norm="l2", dtype=np.float64,
    )
    matrix = vectorizer.fit_transform(texts)
    counts = {float(t): {"pairs": 0, "cross_fold": 0, "same_prompt": 0, "cross_fold_different_prompt": 0}
              for t in thresholds}
    max_cos = np.zeros(n, dtype=np.float64)
    top: list[tuple[float, int, int]] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = (matrix[start:stop] @ matrix.T).toarray()
        rows = np.arange(start, stop)
        sims[rows - start, rows] = 0.0  # a passage is not its own near-duplicate
        max_cos[start:stop] = sims.max(axis=1)
        for threshold in thresholds:
            ii, jj = np.nonzero(sims > threshold)
            ii = ii + start
            upper = jj > ii  # count each unordered pair once
            ii, jj = ii[upper], jj[upper]
            entry = counts[float(threshold)]
            entry["pairs"] += int(ii.size)
            cross = folds[ii] != folds[jj]
            same_prompt = prompts[ii] == prompts[jj]
            entry["cross_fold"] += int(cross.sum())
            entry["same_prompt"] += int(same_prompt.sum())
            entry["cross_fold_different_prompt"] += int((cross & ~same_prompt).sum())
        if thresholds:
            ii, jj = np.nonzero(sims > min(thresholds))
            for a, b in zip(ii + start, jj, strict=True):
                if a < b:
                    top.append((float(sims[a - start, b]), int(a), int(b)))
    top.sort(key=lambda item: (-item[0], item[1], item[2]))
    return {
        "n_rows": n,
        "n_pairs_examined": n * (n - 1) // 2,
        "pair_counting": "unordered pairs (row_a < row_b); self-pairs excluded",
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "thresholds": {str(t): entry for t, entry in counts.items()},
        "max_offdiagonal_cosine_quantiles": {
            str(q): float(np.quantile(max_cos, q)) for q in (0.5, 0.9, 0.99, 1.0)
        },
        "top_pairs": [
            {"cosine": cos, "row_a": a, "row_b": b,
             "fold_a": int(folds[a]), "fold_b": int(folds[b]),
             "same_prompt": bool(prompts[a] == prompts[b]),
             "cross_fold": bool(folds[a] != folds[b])}
            for cos, a, b in top[:TOP_PAIRS]
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--splits", default="artifacts/splits/splits_official_verified_s42.json")
    ap.add_argument("--out", default="artifacts/near_duplicates/results.json")
    args = ap.parse_args()

    df, _test = load_competition_data(args.data_dir)
    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    fingerprint = ordered_dataset_fingerprint(df)
    if splits.get("ordered_dataset_sha256") != fingerprint:
        raise SystemExit("[abort] the local train frame does not match the split fingerprint")
    folds_full = [
        (np.asarray(f["train"], dtype=int), np.asarray(f["valid"], dtype=int)) for f in splits["folds"]
    ]
    folds = fold_assignments(len(df), folds_full)
    prompts = df["prompt_id"].astype(str).to_numpy()
    texts = df["text"].astype(str).tolist()

    exact = len(texts) - len(set(texts))
    prompt_fold_counts: dict[str, set[int]] = {}
    for prompt, fold in zip(prompts, folds, strict=True):
        prompt_fold_counts.setdefault(str(prompt), set()).add(int(fold))
    shared_prompts = sorted(p for p, seen in prompt_fold_counts.items() if len(seen) > 1)

    audit = pair_audit(texts, folds, prompts)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment": env_snapshot(),
        "analysis": "official_train_near_duplicate_audit",
        "plan_item": "§2 L94",
        "transductive": False,
        "fits_performed": 0,
        "predictors_fit": 0,
        "test_rows_read": 0,
        "ordered_dataset_sha256": fingerprint,
        "split_id": f"{splits['method']}-s{splits['seed']}-n{splits['n_splits']}",
        "n_prompts": int(len(prompt_fold_counts)),
        "exact_duplicate_texts": int(exact),
        "prompts_appearing_in_more_than_one_fold": shared_prompts,
        "similarity": "word TF-IDF (1-2 grams), sublinear tf, l2 norm, max_features 30000 — "
                      "the same settings as src.pool.decontaminate_global",
        "audit": audit,
        "note": (
            "The vectorizer is fit on the whole official corpus because this is a descriptive "
            "audit of the corpus, not a learned transform inside a model: no predictor is fit, "
            "no score is reported and no validation row influences any training decision. "
            "Similarity is a heuristic, not proof that every augmented variant was found. "
            "Only ids, fold numbers and cosine values are stored; no passage text."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"rows {audit['n_rows']} | prompts {payload['n_prompts']} | pairs examined {audit['n_pairs_examined']}")
    print(f"exact duplicate texts: {exact}")
    print(f"prompts spanning more than one fold: {len(shared_prompts)}")
    for threshold, entry in audit["thresholds"].items():
        print(
            f"cos > {threshold}: pairs {entry['pairs']} | cross-fold {entry['cross_fold']} "
            f"| same-prompt {entry['same_prompt']} | cross-fold different-prompt {entry['cross_fold_different_prompt']}"
        )
    print(f"max off-diagonal cosine quantiles: {audit['max_offdiagonal_cosine_quantiles']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()