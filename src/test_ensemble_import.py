from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.constants import ARTIFACTS_DIR
from src.data import labels_to_indices
from src.inference import load_final, predict_dataframe, fit_final
from src.model import ModelBundle, ModelSpec
from src.submission import build_submission, validate_submission

FINAL = ARTIFACTS_DIR / "final" / "official"
OUT = FINAL / "submission.parquet"

def main():
    sys.setrecursionlimit(128_000)

    # Load weights artifact
    weights_data = json.loads((ARTIFACTS_DIR / "cv" / "official" / "ensemble" / "weights.json").read_text(encoding="utf-8"))
    weights = weights_data["weights"] if "weights" in weights_data else weights_data
    assert isinstance(weights, dict), f"unsupported weights payload: {weights_data}"

    # Fit style-only if missing
    spec = ModelSpec(name="style_only",features=(),style=True,classifier="logreg",C=1.0,seed=42)
    df = pd.read_csv(Path("data") / "train.csv")
    texts = df["text"].tolist()
    y = labels_to_indices(df["model"].tolist())
    bundle = ModelBundle(spec).fit(texts, y)
    bundle.save(FINAL / "models" / "style_only.joblib")
    print("style_only saved",file=sys.stderr)

    bundles, manifest = load_final(FINAL)
    print("final models loaded:", list(bundles.keys()), file=sys.stderr)

    # Build weight map: model name -> weight
    wmap = {name: float(weights[name]) for name in weights if name in bundles}
    missing = set(weights) - set(bundles)
    if missing:
        raise SystemExit(f"weights reference unknown models: {sorted(missing)}")

    test = pd.read_csv(Path("data") / "test.csv")
    preds = predict_dataframe(bundles, test, wmap)
    print("used models in preds:", sorted(preds.unique()), file=sys.stderr)
    print("count:", len(preds), "distinct labels:", len(preds.unique()), file=sys.stderr)

    build_submission(test["id"], preds, OUT, expected_ids=test["id"])
    validate_submission(OUT)
    print("OK written and validated", file=sys.stderr)

if __name__ == "__main__":
    main()
