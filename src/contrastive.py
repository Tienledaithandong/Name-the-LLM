"""Prompt-aware batching and supervised contrastive loss for GPU experiments.

No data from validation prompts should ever be passed to the sampler. Torch is
an optional dependency and is imported only by the loss function.
"""
from __future__ import annotations

import numpy as np


class CrossedBatchSampler:
    """Sample matching authors from two different prompts per batch.

    Every selected author contributes one passage per prompt, providing a
    cross-prompt positive and same-prompt, different-author hard negatives.
    Sampling is with replacement across batches, never within a batch.
    """

    def __init__(self, labels, groups, authors_per_batch: int = 8, seed: int = 42):
        labels = np.asarray(labels)
        groups = np.asarray(groups).astype(str)
        if labels.ndim != 1 or groups.ndim != 1 or len(labels) != len(groups):
            raise ValueError("labels and prompt groups must be aligned 1-D arrays")
        if authors_per_batch < 2:
            raise ValueError("authors_per_batch must be at least 2")
        self.rng = np.random.default_rng(seed)
        self.authors_per_batch = authors_per_batch
        self.by_group: dict[str, dict[int, list[int]]] = {}
        for index, (label, group) in enumerate(zip(labels, groups, strict=True)):
            self.by_group.setdefault(group, {}).setdefault(int(label), []).append(index)
        keys = sorted(self.by_group)
        self.pairs = []
        for i, first in enumerate(keys):
            for second in keys[i + 1:]:
                common = sorted(set(self.by_group[first]) & set(self.by_group[second]))
                if len(common) >= authors_per_batch:
                    self.pairs.append((first, second, common))
        if not self.pairs:
            raise ValueError("no prompt pair has enough shared authors for a batch")

    def sample(self) -> np.ndarray:
        first, second, common = self.pairs[int(self.rng.integers(len(self.pairs)))]
        chosen = self.rng.choice(common, size=self.authors_per_batch, replace=False)
        indices = []
        for group in (first, second):
            for label in chosen:
                indices.append(int(self.rng.choice(self.by_group[group][int(label)])))
        return np.asarray(indices, dtype=np.int64)


def prompt_contrastive_loss(embeddings, labels, groups, temperature=0.1, hard_weight=1.0):
    """SupCon with cross-prompt positives and weighted same-prompt negatives.

    ``hard_weight=1`` is ordinary supervised contrastive training; larger
    weights emphasize same-prompt, different-class negatives. Each anchor must
    have a positive in another prompt, so malformed batches fail loudly.
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0 or hard_weight < 1:
        raise ValueError("temperature must be positive and hard_weight at least 1")
    labels = labels.reshape(-1)
    groups = groups.reshape(-1)
    n = labels.numel()
    if embeddings.ndim != 2 or embeddings.shape[0] != n or groups.numel() != n:
        raise ValueError("embeddings, labels and groups must align")
    same_class = labels[:, None] == labels[None, :]
    same_prompt = groups[:, None] == groups[None, :]
    diagonal = torch.eye(n, device=embeddings.device, dtype=torch.bool)
    positives = same_class & ~same_prompt
    if not positives.any(dim=1).all():
        raise ValueError("each anchor requires a same-class cross-prompt positive")
    # Exclude other same-class, same-prompt observations from the denominator.
    allowed = ~diagonal & (~same_class | positives)
    hard = ~same_class & same_prompt
    logits = F.normalize(embeddings.float(), dim=1) @ F.normalize(embeddings.float(), dim=1).T
    logits = logits / temperature
    log_weight = torch.zeros_like(logits)
    if hard_weight != 1:
        log_weight = torch.where(hard, torch.as_tensor(hard_weight, device=logits.device).log(), log_weight)
    denominator = torch.logsumexp((logits + log_weight).masked_fill(~allowed, -torch.inf), dim=1)
    positive_log_probs = logits - denominator[:, None]
    return -(positive_log_probs.masked_fill(~positives, 0).sum(dim=1) / positives.sum(dim=1)).mean()


def paired_prompt_bootstrap(labels, groups, baseline_predictions, candidate_predictions,
                            n_resamples: int = 1000, seed: int = 42) -> dict:
    """Paired balanced-accuracy difference, resampling whole prompt groups."""
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    baseline = np.asarray(baseline_predictions, dtype=np.int64)
    candidate = np.asarray(candidate_predictions, dtype=np.int64)
    if labels.ndim != 1 or not len(labels) or any(x.shape != labels.shape for x in (groups, baseline, candidate)):
        raise ValueError("labels, groups and predictions must be aligned nonempty 1-D arrays")
    if n_resamples < 1 or np.any(labels < 0):
        raise ValueError("n_resamples must be positive and class indices nonnegative")
    _, inverse = np.unique(groups, return_inverse=True)
    ng, nc = int(inverse.max()) + 1, int(labels.max()) + 1
    counts = np.zeros((ng, nc), dtype=np.int64)
    diff = np.zeros((ng, nc), dtype=np.int64)
    np.add.at(counts, (inverse, labels), 1)
    np.add.at(diff, (inverse, labels), (candidate == labels).astype(int) - (baseline == labels).astype(int))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, ng, size=ng)
        support = counts[idx].sum(axis=0)
        change = diff[idx].sum(axis=0)
        samples.append(float(np.mean(change[support > 0] / support[support > 0])))
    return {'n_resamples': n_resamples, 'seed': seed,
            'mean_difference': float(np.mean(samples)),
            'lower_95': float(np.quantile(samples, 0.025)),
            'upper_95': float(np.quantile(samples, 0.975))}


def crop_token_ids(token_ids, content_limit: int, mode: str, rng=None) -> list[int]:
    """Select a contiguous content-token window, excluding special tokens.

    Random windows are for training only; deterministic head and tail windows
    can be averaged at inference without accessing prompt labels or other rows.
    """
    if content_limit < 1:
        raise ValueError("content_limit must be positive")
    if mode not in {"head", "tail", "random"}:
        raise ValueError(f"unknown crop mode: {mode}")
    ids = list(token_ids)
    remaining = max(0, len(ids) - content_limit)
    if mode == "tail":
        start = remaining
    elif mode == "random":
        if rng is None:
            raise ValueError("random crops require an explicit random generator")
        start = int(rng.integers(remaining + 1))
    else:
        start = 0
    return ids[start:start + content_limit]
