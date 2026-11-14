"""Synthetic *stand-in* data for tests and pipeline smoke runs.

This module never produces competition predictions.  It exists because the real
``train.csv`` requires accepting the competition rules on Kaggle, so the
repository must be testable without it.  Outputs are written under
``data/synthetic`` and every consumer labels them as synthetic.

The generator gives each of the 18 labels a small probabilistic style profile
(punctuation habits, sentence length, markdown, hedging) on top of shared
topic text, so attribution is learnable but far from trivial — good enough to
exercise the whole pipeline and catch shape/leakage bugs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import LABELS

TOPICS = [
    "the nature of consciousness",
    "the origin of the universe",
    "the limits of formal proof",
    "how memory works in the brain",
    "the ethics of automation",
    "why galaxies rotate",
    "the structure of DNA",
    "the problem of free will",
    "how compilers optimise loops",
    "the history of the printing press",
    "entropy in closed systems",
    "the mathematics of prime numbers",
    "a rainy harbour at dawn",
    "a cartographer's last journey",
    "the silence of an empty library",
    "a machine that learns to dream",
    "the economics of attention",
    "whether language shapes thought",
]

OPENERS = [
    "Let me think about this carefully.",
    "This is a genuinely interesting question.",
    "Here is one way to approach it.",
    "The short answer is complicated.",
    "I want to separate two things here.",
    "There is a useful distinction to draw.",
]

CONNECTORS = [
    "Moreover,",
    "That said,",
    "In practice,",
    "By contrast,",
    "More precisely,",
    "On the other hand,",
]

BODY = (
    "the argument turns on a tradeoff between structure and flexibility, and the "
    "evidence is suggestive rather than decisive; a careful account has to hold "
    "several competing explanations in view at once while remaining honest about "
    "what the available measurements can and cannot support"
).split()

HEDGES = ["perhaps", "arguably", "it seems", "roughly", "in some sense", "likely"]

MODEL_PROFILE: dict[str, dict] = {
    "gpt-5.5": {"sent": (14, 4), "comma": 3.0, "markdown": 0.05, "hedge": 0.4, "dash": 0.2},
    "gpt-5.6-luna": {"sent": (20, 6), "comma": 1.6, "markdown": 0.20, "hedge": 0.2, "dash": 0.4},
    "gpt-5.6-sol": {"sent": (11, 3), "comma": 2.2, "markdown": 0.30, "hedge": 0.1, "dash": 0.5},
    "gpt-5.6-terra": {"sent": (17, 5), "comma": 2.6, "markdown": 0.02, "hedge": 0.5, "dash": 0.1},
    "gpt-6-astra": {"sent": (24, 7), "comma": 1.2, "markdown": 0.15, "hedge": 0.3, "dash": 0.7},
    "gemini-3.1-pro-preview": {"sent": (16, 4), "comma": 2.8, "markdown": 0.35, "hedge": 0.2, "dash": 0.2},
    "gemini-3.6-flash": {"sent": (9, 2), "comma": 1.0, "markdown": 0.10, "hedge": 0.1, "dash": 0.1},
    "gemini-3.7-flash": {"sent": (12, 3), "comma": 1.4, "markdown": 0.25, "hedge": 0.2, "dash": 0.3},
    "gemini-3.8-flash": {"sent": (8, 2), "comma": 0.8, "markdown": 0.40, "hedge": 0.05, "dash": 0.1},
    "claude-fable-5": {"sent": (22, 6), "comma": 2.4, "markdown": 0.05, "hedge": 0.6, "dash": 0.8},
    "claude-opus-4-6": {"sent": (19, 5), "comma": 2.9, "markdown": 0.18, "hedge": 0.5, "dash": 0.3},
    "claude-opus-4-8": {"sent": (21, 5), "comma": 2.1, "markdown": 0.08, "hedge": 0.45, "dash": 0.6},
    "claude-opus-5": {"sent": (25, 8), "comma": 3.2, "markdown": 0.12, "hedge": 0.7, "dash": 0.9},
    "claude-sonnet-5": {"sent": (15, 4), "comma": 1.9, "markdown": 0.22, "hedge": 0.35, "dash": 0.4},
    "qwen3.8-max": {"sent": (13, 4), "comma": 1.1, "markdown": 0.30, "hedge": 0.15, "dash": 0.2},
    "glm-5.3": {"sent": (18, 5), "comma": 2.0, "markdown": 0.06, "hedge": 0.25, "dash": 0.5},
    "glm-5.3-flash": {"sent": (10, 3), "comma": 0.9, "markdown": 0.16, "hedge": 0.1, "dash": 0.2},
    "kimi-k3": {"sent": (23, 7), "comma": 2.7, "markdown": 0.14, "hedge": 0.4, "dash": 1.0},
}


def _sentence(rng: np.random.Generator, profile: dict, topic: str) -> str:
    target = max(4, int(rng.normal(profile["sent"][0], profile["sent"][1])))
    words = list(rng.choice(BODY, size=target, replace=True))
    if rng.random() < profile["hedge"]:
        words.insert(int(rng.integers(1, max(2, len(words)))), rng.choice(HEDGES))
    if rng.random() < min(0.9, profile["comma"] / 3.5):
        words.insert(int(rng.integers(2, max(3, len(words)))), ",")
    if rng.random() < profile["dash"] / 6:
        words.insert(int(rng.integers(2, max(3, len(words)))), rng.choice(["—", "–"]))
    text = " ".join(words).replace(" ,", ",")
    if rng.random() < 0.15:
        text = text.capitalize()
    return text


def _passage(
    rng: np.random.Generator, model: str, topic: str, target_words: int
) -> str:
    profile = MODEL_PROFILE[model]
    parts: list[str] = []
    if rng.random() < 0.5:
        parts.append(str(rng.choice(OPENERS)))
    paragraphs = 1 + (1 if rng.random() < 0.35 else 0)
    for p in range(paragraphs):
        sentences = []
        for s in range(1 + int(target_words // max(6, profile["sent"][0]))):
            if s and rng.random() < 0.25:
                sentences.append(f"{rng.choice(CONNECTORS)} {_sentence(rng, profile, topic)}.")
            else:
                sentences.append(_sentence(rng, profile, topic).rstrip(".") + ".")
        block = " ".join(sentences)
        if p == 0:
            block = f"{topic.capitalize()} — {block[0].lower()}{block[1:]}"
        parts.append(block)
    text = "\n\n".join(parts)

    if rng.random() < profile["markdown"]:
        text = text.replace(". ", ".\n\n- ", 1)
    if rng.random() < profile["markdown"] * 0.6:
        text += "\n\n**Key point:** the details matter more than the summary."
    if rng.random() < 0.12:
        text += " See `notes.md` for the derivation."
    if rng.random() < 0.10:
        text += " In short: it depends!"
    if rng.random() < 0.08:
        text += " (This is a simplification…)"
    return text


def _make_prompts(rng: np.random.Generator, n: int, prefix: str) -> list[str]:
    prompts = []
    for i in range(n):
        topic = TOPICS[rng.integers(len(TOPICS))]
        angle = str(rng.choice(["explain", "argue", "describe", "compare", "speculate"]))
        prompts.append(f"{prefix}{i:04d}: {angle} {topic}")
    return prompts


def make_synthetic_dataset(
    n_prompts: int = 150,
    test_prompts: int = 60,
    seed: int = 42,
    inclusion_prob: float = 0.62,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(train, test, truth)`` with the official columns.

    ``test`` has only ``id``/``text`` (as the real one does); ``truth`` carries
    the withheld labels for local scoring only.
    """
    rng = np.random.default_rng(seed)
    train_prompts = _make_prompts(rng, n_prompts, "p")
    test_prompt_ids = _make_prompts(rng, test_prompts, "q")

    train_rows = []
    for pid in train_prompts:
        topic = pid.split(": ", 1)[1].split(" ", 1)[1]
        for model in LABELS:
            if rng.random() > inclusion_prob:
                continue
            words = int(np.clip(rng.normal(120, 45), 40, 250))
            train_rows.append(
                {
                    "id": f"tr{rng.integers(0, 10**12):012x}",
                    "prompt_id": pid,
                    "text": _passage(rng, model, topic, words),
                    "model": model,
                }
            )
    train = pd.DataFrame(train_rows, columns=["id", "prompt_id", "text", "model"])
    train = train.drop_duplicates(subset=["id"]).reset_index(drop=True)

    test_rows, truth_rows = [], []
    for pid in test_prompt_ids:
        topic = pid.split(": ", 1)[1].split(" ", 1)[1]
        for model in LABELS:
            if rng.random() > inclusion_prob:
                continue
            words = int(np.clip(rng.normal(120, 45), 40, 250))
            row_id = f"te{rng.integers(0, 10**12):012x}"
            test_rows.append({"id": row_id, "text": _passage(rng, model, topic, words)})
            truth_rows.append({"id": row_id, "model": model})
    test = pd.DataFrame(test_rows, columns=["id", "text"]).drop_duplicates(subset=["id"])
    truth = pd.DataFrame(truth_rows, columns=["id", "model"])
    truth = test[["id"]].merge(truth, on="id", how="left")
    return train.reset_index(drop=True), test.reset_index(drop=True), truth


def write_synthetic_dataset(dest, **kwargs) -> tuple:
    """Write ``train.csv``/``test.csv``/``test_truth.csv`` under ``dest``."""
    from pathlib import Path

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    train, test, truth = make_synthetic_dataset(**kwargs)
    train.to_csv(dest / "train.csv", index=False)
    test.to_csv(dest / "test.csv", index=False)
    truth.to_csv(dest / "test_truth.csv", index=False)
    return train, test, truth
