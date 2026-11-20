"""Between-prompt variance headroom pre-check (spec 2026-11-19, section 2.2).

Reports ``between_prompt_share`` for every representation we might neutralise:
the share of feature variance explained by prompt identity.  A representation
with a low share has no prompt component to remove and cannot benefit from the
self-neutralised residual; a high share says the headroom exists but says
nothing yet about whether it is recoverable from one passage.

This is a property of the data, not a model score: the representations are
fitted once on the full labelled training frame and nothing here is compared
against a validation fold.  Never cite these numbers as a CV result.

    python -m experiments.residual_headroom --out artifacts/residual/headroom.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import find_competition_file, validate_train_frame  # noqa: E402
from src.features_tfidf import TextMatrixBuilder  # noqa: E402
from src.prompt_invariance import prompt_centering_diagnostic  # noqa: E402

SPACES = ("style", "svd128", "embedding")

NOTE = (
    "Descriptive property of the labelled training data, not a CV score. "
    "Every representation was fitted on the complete training frame."
)


def _builder(space: str) -> TextMatrixBuilder:
    if space == "style":
        return TextMatrixBuilder(kinds=(), style=True)
    if space == "svd128":
        return TextMatrixBuilder(kinds=("char",), svd=128)
    if space == "embedding":
        return TextMatrixBuilder(kinds=(), embedding={"backend": "hashing", "dim": 384})
    raise ValueError(f"unknown space {space!r}; choose from {SPACES}")


def representation_matrix(texts, space: str) -> np.ndarray:
    """Fit the named representation on ``texts`` and return a dense matrix."""
    builder = _builder(space).fit(list(texts))
    matrix = builder.transform(list(texts))
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def headroom_table(
    texts, groups, spaces: tuple[str, ...] = SPACES
) -> dict[str, dict]:
    """``between_prompt_share`` per representation, with its width."""
    texts = list(texts)
    groups = [str(g) for g in groups]
    if len(texts) != len(groups):
        raise ValueError("texts and groups must have the same length")
    out: dict[str, dict] = {}
    for space in spaces:
        matrix = representation_matrix(texts, space)
        diagnostic = prompt_centering_diagnostic(matrix, groups)
        out[space] = {
            "between_prompt_share": float(diagnostic["between_prompt_share"]),
            "n_prompts": int(diagnostic["n_prompts"]),
            "n_features": int(matrix.shape[1]),
        }
    return out


def load_official_train(data_dir: str | Path | None) -> "object":
    """Read the labelled file only; the test file is never opened."""
    import pandas as pd

    path = find_competition_file("train.csv", data_dir)
    return validate_train_frame(pd.read_csv(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default="artifacts/residual/headroom.json")
    parser.add_argument("--spaces", nargs="+", default=list(SPACES))
    args = parser.parse_args(argv)

    frame = load_official_train(args.data_dir)
    table = headroom_table(
        frame["text"].tolist(),
        frame["prompt_id"].astype(str).tolist(),
        spaces=tuple(args.spaces),
    )
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "official",
        "n_rows": int(len(frame)),
        "n_prompts": int(frame["prompt_id"].nunique()),
        "note": NOTE,
        "spaces": table,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"[headroom] {len(frame)} rows / {payload['n_prompts']} prompts")
    for space, cell in table.items():
        print(
            f"[headroom] {space:9s} share={cell['between_prompt_share']:.4f} "
            f"features={cell['n_features']}"
        )
    print(f"[headroom] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
