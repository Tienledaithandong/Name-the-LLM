from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import LABELS
from src.inference import fit_final, load_final, predict_dataframe

A, B = LABELS[0], LABELS[1]
CONFIGS = [
    {"name": "char_tfidf_logreg", "features": ["char"], "classifier": "logreg",
     "C": 1.0, "max_features": 2000},
    {"name": "style_only", "features": [], "style": True, "classifier": "logreg", "C": 1.0},
]


def _train(n_groups: int = 6, per: int = 4):
    rows = []
    vocab = ["alpha beta gamma delta", "quantum lattice gauge field",
             "compiler parsing token grammar", "market revenue growth quarter",
             "chlorophyll photosynthesis leaf", "recursion stack frame heap"]
    for g in range(n_groups):
        for r in range(per):
            label = A if (g + r) % 2 == 0 else B
            cue = "model style marker one" if label == A else "model style marker two"
            base = vocab[(g + r) % len(vocab)]
            rows.append({"id": f"t{g}-{r}", "prompt_id": f"g{g}",
                         "text": f"{base} {cue} filler {r} more varied tokens here",
                         "model": label})
    return pd.DataFrame(rows)


def _pool(n: int = 12):
    return pd.DataFrame({
        "id": [f"p{i}" for i in range(n)],
        "text": [f"pool external passage number {i} with model style marker {'one' if i % 2 else 'two'} filler"
                 for i in range(n)],
        "model": [A if i % 2 == 0 else B for i in range(n)],
        "prompt_canon": [f"c{i % 4}" for i in range(n)],
        "max_cos_train": [0.0] * n,
    })


def test_fit_final_trains_only_selected_member_with_pool(tmp_path):
    train = _train()
    meta_path = fit_final(CONFIGS, train, tmp_path, seed=0, pool_df=_pool(),
                          pool_models=["char_tfidf_logreg"], cos_threshold=0.3, pool_cap=8)
    import json
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    assert manifest["pool"]["pool_models"] == ["char_tfidf_logreg"]
    assert manifest["pool"]["reference"] == "full_official_train"
    assert manifest["pool"]["capped_rows"] <= 8
    modes = {m["name"]: m["fit_mode"] for m in manifest["models"]}
    assert modes["char_tfidf_logreg"].startswith("train+global_pool")
    assert modes["style_only"] == "train_only"


def test_fit_final_without_pool_records_no_pool_meta(tmp_path):
    meta_path = fit_final(CONFIGS, _train(), tmp_path, seed=0)
    import json
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    assert manifest["pool"] is None
    assert all(m["fit_mode"] == "train_only" for m in manifest["models"])


def test_fit_final_pool_predictions_round_trip(tmp_path):
    fit_final(CONFIGS, _train(), tmp_path, seed=0, pool_df=_pool(),
              pool_models=["char_tfidf_logreg"], pool_cap=12)
    bundles, manifest = load_final(tmp_path)
    assert manifest["class_order"] == list(LABELS)
    assert set(bundles) == {"char_tfidf_logreg", "style_only"}
    test_df = pd.DataFrame({"id": ["x1", "x2"],
                            "text": ["alpha beta gamma model style marker one filler",
                                     "quantum lattice gauge model style marker two filler"]})
    preds = predict_dataframe(bundles, test_df, {"char_tfidf_logreg": 0.5, "style_only": 0.5})
    assert len(preds) == 2
    assert set(preds) <= set(LABELS)


def test_fit_final_rejects_pool_models_without_pool(tmp_path):
    with pytest.raises(ValueError, match="non-empty pool_df"):
        fit_final(CONFIGS, _train(), tmp_path, seed=0, pool_models=["char_tfidf_logreg"])
