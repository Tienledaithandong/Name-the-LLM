"""Configuration loading.

Configs are plain YAML; missing sections fall back to the defaults below so a
minimal config only needs to name its models.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.constants import RANDOM_STATE

DEFAULTS: dict[str, Any] = {
    "seed": RANDOM_STATE,
    "cv": {"n_splits": 5, "shuffle": True, "method": "auto"},
    "features": {
        "max_features": 300_000,
        "min_df": 2,
        "sublinear_tf": True,
        "char_ngram_range": [2, 5],
        "word_ngram_range": [1, 2],
    },
    "models": [
        {"name": "char_tfidf_linearsvc", "features": ["char"], "classifier": "linearsvc"}
    ],
    "ensemble": {"method": "greedy", "max_models": 4},
}


class ConfigError(ValueError):
    """Raised when a config file is unusable."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config merged over ``DEFAULTS`` (no file -> defaults)."""
    if path is None:
        return copy.deepcopy(DEFAULTS)
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"config must be a mapping, got {type(loaded).__name__}")
    cfg = _deep_merge(DEFAULTS, loaded)
    if not cfg.get("models"):
        raise ConfigError("config defines no models")
    names = [str(m.get("name")) for m in cfg["models"]]
    if len(set(names)) != len(names):
        raise ConfigError(f"duplicate model names in config: {names}")
    return cfg


def feature_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("features", {}))
