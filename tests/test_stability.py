from __future__ import annotations

from src.stability import holdout_check, run_stability, seed_stability

MEMBERS = [
    {"name": "tiny_char", "features": ["char"], "classifier": "linearsvc",
     "max_features": 2000, "char_ngram_range": [2, 3], "min_df": 1},
    {"name": "tiny_style", "features": [], "style": True, "classifier": "logreg", "C": 1.0},
]
WEIGHTS = {"tiny_char": 0.5, "tiny_style": 0.5}


def test_seed_stability_reports_gain_per_seed(train_df, tmp_path):
    rows = seed_stability(
        train_df, MEMBERS, seeds=[13, 21], out_root=tmp_path,
        dataset_label="synthetic", registry_path=None, n_splits=3,
    )
    assert [r["seed"] for r in rows] == [13, 21]
    for row in rows:
        assert 0.0 <= row["blend_pooled_oof"] <= 1.0
        assert abs(sum(row["weights"].values()) - 1.0) < 1e-9
        assert (tmp_path / f"seed_{row['seed']}" / "splits.json").is_file()


def test_holdout_check_scores_fixed_weights(train_df):
    result = holdout_check(train_df, MEMBERS, WEIGHTS, seed=7, n_groups=8)
    assert result["n_valid"] > 0 and result["n_train"] > 0
    assert 0.0 <= result["blend_balanced_accuracy"] <= 1.0
    assert set(result["singles"]) == {"tiny_char", "tiny_style"}
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-9


def test_run_stability_writes_report_and_registry(train_df, tmp_path):
    registry = tmp_path / "registry.csv"
    summary = run_stability(
        train_df, MEMBERS, WEIGHTS, seeds=[13], out_root=tmp_path / "stab",
        dataset_label="synthetic", registry_path=registry,
        holdout_seed=7, holdout_groups=8,
    )
    assert (tmp_path / "stab" / "stability.md").is_file()
    assert (tmp_path / "stab" / "stability.json").is_file()
    assert len(summary["seed_stability"]) == 1
    text = registry.read_text(encoding="utf-8")
    assert "holdout_blend" in text
    assert "not used for selection" in text