"""Report figures built from saved CV artifacts (Section 9).

Reads the per-model metrics JSON and fold-aligned OOF score matrices written by
``src.train``, reconstructs the adopted blend from ``ensemble/weights.json``, and
renders the confusion matrix, per-class recall, ablation, runtime/RAM tradeoff
and grouped-CV variation figures. Plotting is offline and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.constants import LABELS, PROVIDER_OF
from src.data import indices_to_labels
from src.ensemble import load_oof, normalize_scores
from src.metrics import confusion_matrix, per_class_recall

_PROVIDER_COLORS = {
    "OpenAI": "#3b6ea5",
    "Google": "#4c9a6a",
    "Anthropic": "#c0703f",
    "Qwen": "#7a5aa0",
    "ZAI": "#b5893b",
    "Moonshot AI": "#5a8a9a",
    "unknown": "#888888",
}


def load_metrics(artifacts_dir: str | Path) -> dict[str, dict]:
    metrics_dir = Path(artifacts_dir) / "metrics"
    out: dict[str, dict] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    if not out:
        raise FileNotFoundError(f"no metrics JSON in {metrics_dir}; run training first")
    return out


def blend_predictions(
    artifacts_dir: str | Path, weights: dict[str, float] | None
) -> tuple[np.ndarray, np.ndarray, dict[str, float], str]:
    """Return (truth, pred, used_weights, name) for the blend or best single."""
    oof = load_oof(artifacts_dir)
    names = sorted(oof)
    y = oof[names[0]]["y"]
    normalized = {n: normalize_scores(oof[n]["scores"]) for n in names}
    truth = indices_to_labels(y)

    if weights:
        used = {k: float(v) for k, v in weights.items() if k in normalized and v > 0}
        if used:
            total = sum(used.values())
            used = {k: v / total for k, v in used.items()}
            scores = sum(normalized[k] * v for k, v in used.items())
            pred = indices_to_labels(scores.argmax(axis=1))
            return truth, pred, used, "blend(" + "+".join(sorted(used)) + ")"

    from src.metrics import balanced_accuracy

    singles = {
        n: balanced_accuracy(truth, indices_to_labels(normalized[n].argmax(axis=1)))
        for n in names
    }
    best = max(singles, key=singles.get)
    pred = indices_to_labels(normalized[best].argmax(axis=1))
    return truth, pred, {best: 1.0}, best


def _short(label: str) -> str:
    return label.replace("gemini-", "gem-").replace("claude-", "cl-").replace("preview", "prev")


def make_report_figures(
    artifacts_dir: str | Path,
    out_dir: str | Path,
    weights: dict[str, float] | None = None,
    dataset_label: str = "official",
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts_dir = Path(artifacts_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(artifacts_dir)
    truth, pred, used_weights, blend_name = blend_predictions(artifacts_dir, weights)
    written: list[str] = []
    suffix = "" if dataset_label == "official" else f"_{dataset_label}"

    # 1. Confusion matrix of the adopted blend (row = true, col = predicted).
    cm = confusion_matrix(truth, pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels([_short(l) for l in LABELS], rotation=90, fontsize=7)
    ax.set_yticklabels([_short(l) for l in LABELS], fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — {blend_name} (pooled OOF, {dataset_label})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    fig.tight_layout()
    path = out_dir / f"confusion_matrix{suffix}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    # 2. Per-class recall, sorted, coloured by provider.
    recalls = per_class_recall(truth, pred)
    order = sorted(LABELS, key=lambda l: (np.nan_to_num(recalls[l], nan=-1.0)))
    values = [recalls[l] for l in order]
    colors = [_PROVIDER_COLORS[PROVIDER_OF[l]] for l in order]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([_short(l) for l in order], values, color=colors)
    ax.set_xlabel("recall")
    ax.set_xlim(0, 1)
    ax.set_title(f"Per-class recall — {blend_name} ({dataset_label})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in dict.fromkeys(colors)
    ]
    providers = list(dict.fromkeys(PROVIDER_OF[l] for l in order))
    ax.legend(handles, providers, fontsize=7, loc="lower right")
    fig.tight_layout()
    path = out_dir / f"per_class_recall{suffix}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    # 3. Ablation: pooled OOF per model.
    rows = sorted(
        metrics.items(), key=lambda kv: kv[1]["pooled"]["balanced_accuracy"]
    )
    names = [k for k, _ in rows]
    pooled = [v["pooled"]["balanced_accuracy"] for _, v in rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, pooled, color="#3b6ea5")
    ax.set_xlabel("pooled out-of-fold balanced accuracy")
    ax.set_title(f"Ablation — pooled OOF by model ({dataset_label})")
    ax.axvline(1.0 / len(LABELS), color="#999999", linestyle="--", linewidth=1)
    fig.tight_layout()
    path = out_dir / f"ablation{suffix}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    # 4. Runtime / RAM tradeoff: fit seconds vs pooled OOF, bubble = peak RSS.
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, payload in metrics.items():
        fit = max(payload.get("fit_seconds", 0.0), 0.01)
        acc = payload["pooled"]["balanced_accuracy"]
        ram = max(payload.get("peak_ram_gb", 0.0), 0.01)
        ax.scatter(fit, acc, s=ram * 900 + 20, alpha=0.6, color="#4c9a6a")
        ax.annotate(name, (fit, acc), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("fit seconds, all folds (log scale)")
    ax.set_ylabel("pooled OOF balanced accuracy")
    ax.set_title(f"Runtime / RAM tradeoff (bubble area ∝ peak RSS, {dataset_label})")
    fig.tight_layout()
    path = out_dir / f"runtime_ram{suffix}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    # 5. Grouped-CV variation: per-fold balanced accuracy spread per model.
    fig, ax = plt.subplots(figsize=(10, 6))
    fold_rows = sorted(
        metrics.items(), key=lambda kv: kv[1]["fold_aggregate"]["mean"]
    )
    positions = np.arange(len(fold_rows))
    for pos, (name, payload) in zip(positions, fold_rows, strict=True):
        scores = payload["fold_aggregate"]["fold_scores"]
        ax.scatter([pos] * len(scores), scores, color="#3b6ea5", alpha=0.5, s=18)
        mean = payload["fold_aggregate"]["mean"]
        std = payload["fold_aggregate"]["std"]
        ax.errorbar(pos, mean, yerr=std, fmt="o", color="#c0703f", capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([k for k, _ in fold_rows], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("per-fold balanced accuracy")
    ax.set_title(f"Grouped-CV variation — mean ± std and fold points ({dataset_label})")
    fig.tight_layout()
    path = out_dir / f"cv_variation{suffix}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    return written