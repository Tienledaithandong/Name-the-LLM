"""Verified constants for the Kaggle *Name That LLM* competition.

Every value in this module was read from the official competition pages on
2026-11-14 (see ``docs/competition_spec.md`` for the URLs and the retrieval
notes).  Do not change a label, the metric or the submission filename without
re-verifying the source.
"""

from __future__ import annotations

import os
from pathlib import Path

COMPETITION_SLUG = "name-that-llm"
COMPETITION_URL = "https://www.kaggle.com/competitions/name-that-llm"
SPEC_RETRIEVED = "2026-11-14"

#: The official metric: mean per-class recall over the 18 candidate models.
METRIC = "balanced_accuracy"

#: Valid submission labels, in the order published on the Evaluation page.
LABELS: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
)

N_CLASSES: int = len(LABELS)

PROVIDER_OF: dict[str, str] = {
    "gpt-5.5": "OpenAI",
    "gpt-5.6-luna": "OpenAI",
    "gpt-5.6-sol": "OpenAI",
    "gpt-5.6-terra": "OpenAI",
    "gpt-6-astra": "OpenAI",
    "gemini-3.1-pro-preview": "Google",
    "gemini-3.6-flash": "Google",
    "gemini-3.7-flash": "Google",
    "gemini-3.8-flash": "Google",
    "claude-fable-5": "Anthropic",
    "claude-opus-4-6": "Anthropic",
    "claude-opus-4-8": "Anthropic",
    "claude-opus-5": "Anthropic",
    "claude-sonnet-5": "Anthropic",
    "qwen3.8-max": "Qwen",
    "glm-5.3": "ZAI",
    "glm-5.3-flash": "ZAI",
    "kimi-k3": "Moonshot AI",
}

LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(LABELS)}

SUBMISSION_FILENAME = "submission.parquet"
#: Path used by the graded Kaggle rerun (from the official Submission section).
KAGGLE_INPUT_DIR = Path("/kaggle/input/competitions/name-that-llm")
#: Max notebook runtime stated on the Code Requirements page (both CPU and GPU).
MAX_NOTEBOOK_HOURS = 9

RANDOM_STATE = 42

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NTL_DATA_DIR", REPO_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("NTL_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
FIGURES_DIR = REPO_ROOT / "figures"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
DEFAULT_REGISTRY = EXPERIMENTS_DIR / "registry.csv"
