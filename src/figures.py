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


def make_prediction_figures(
    truth,
    pred,
    out_dir: str | Path,
    dataset_label: str = "official",
    blend_name: str = "blend",
) -> list[str]:
    """Confusion matrix and per-class recall for one set of OOF predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    suffix = "" if dataset_label == "official" else f"_{dataset_label}"

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
    return written


def make_length_band_figure(
    bands_by_variant: dict[str, list[dict]],
    out_dir: str | Path,
    dataset_label: str = "official",
) -> str:
    """Balanced accuracy per passage-length band, one line per blend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not bands_by_variant:
        raise ValueError("at least one variant is required")
    reference = next(iter(bands_by_variant.values()))
    for bands in bands_by_variant.values():
        if [b["band"] for b in bands] != [b["band"] for b in reference]:
            raise ValueError("all variants must use the same length bands")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = ["#3b6ea5", "#c0703f", "#4c9a6a", "#7a5aa0", "#b5893b"]
    positions = np.arange(len(reference))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, bands) in enumerate(bands_by_variant.items()):
        values = [b["balanced_accuracy"] for b in bands]
        ax.plot(positions, values, marker="o", color=palette[i % len(palette)],
                label=name, linewidth=1.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{b['band']}\nn={b['n']}, {b['classes_present']} classes" for b in reference],
        fontsize=8,
    )
    ax.set_ylabel("balanced accuracy within band")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(
        "Text-length sensitivity — pooled OOF by passage word count "
        f"({dataset_label}); each band averages only the classes it contains"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    suffix = "" if dataset_label == "official" else f"_{dataset_label}"
    path = out_dir / f"length_band_accuracy{suffix}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _panel_axes(out_dir: str | Path, dataset_label: str, stem: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if dataset_label == "official" else f"_{dataset_label}"
    return plt, out_dir / f"{stem}{suffix}.png"


def make_ablation_figure(
    metrics: dict[str, dict], out_dir: str | Path, dataset_label: str = "official"
) -> str:
    """Pooled OOF balanced accuracy per model/arm, sorted ascending."""
    if not metrics:
        raise ValueError("at least one model is required")
    plt, path = _panel_axes(out_dir, dataset_label, "ablation")
    rows = sorted(metrics.items(), key=lambda kv: kv[1]["pooled"]["balanced_accuracy"])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([k for k, _ in rows], [v["pooled"]["balanced_accuracy"] for _, v in rows],
            color="#3b6ea5")
    ax.set_xlabel("pooled out-of-fold balanced accuracy")
    ax.set_title(f"Ablation — pooled OOF by model ({dataset_label})")
    ax.axvline(1.0 / len(LABELS), color="#999999", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def make_runtime_ram_figure(
    metrics: dict[str, dict], out_dir: str | Path, dataset_label: str = "official"
) -> str:
    """Fit seconds vs pooled OOF; bubble area ∝ peak fit RSS where recorded.

    An arm whose peak RSS was never recorded is drawn hollow and labelled, never
    as a zero-area bubble that would read as "no memory used".
    """
    if not metrics:
        raise ValueError("at least one model is required")
    plt, path = _panel_axes(out_dir, dataset_label, "runtime_ram")
    fig, ax = plt.subplots(figsize=(9, 6))
    unrecorded = False
    for name, payload in metrics.items():
        fit = max(payload.get("fit_seconds") or 0.0, 0.01)
        acc = payload["pooled"]["balanced_accuracy"]
        ram = payload.get("peak_ram_gb")
        if ram is None:
            unrecorded = True
            ax.scatter(fit, acc, s=90, facecolors="none", edgecolors="#4c9a6a",
                       linewidths=1.4)
            label = f"{name} (RAM n/r)"
        else:
            ax.scatter(fit, acc, s=max(ram, 0.01) * 900 + 20, alpha=0.6, color="#4c9a6a")
            label = name
        ax.annotate(label, (fit, acc), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    fits = [max(p.get("fit_seconds") or 0.0, 0.01) for p in metrics.values()]
    ax.set_xlim(min(fits) / 1.8, max(fits) * 6)
    ax.set_xlabel("fit seconds, all folds (log scale)")
    ax.set_ylabel("pooled OOF balanced accuracy")
    title = f"Runtime / RAM tradeoff (bubble area ∝ peak RSS, {dataset_label})"
    if unrecorded:
        title += "\nhollow = peak fit RSS not recorded for that run"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def make_cv_variation_figure(
    metrics: dict[str, dict], out_dir: str | Path, dataset_label: str = "official"
) -> str:
    """Per-fold balanced accuracy points plus mean ± std, sorted by mean."""
    if not metrics:
        raise ValueError("at least one model is required")
    plt, path = _panel_axes(out_dir, dataset_label, "cv_variation")
    fold_rows = sorted(metrics.items(), key=lambda kv: kv[1]["fold_aggregate"]["mean"])
    positions = np.arange(len(fold_rows))
    fig, ax = plt.subplots(figsize=(10, 6))
    for pos, (_name, payload) in zip(positions, fold_rows, strict=True):
        scores = payload["fold_aggregate"]["fold_scores"]
        ax.scatter([pos] * len(scores), scores, color="#3b6ea5", alpha=0.5, s=18)
        ax.errorbar(pos, payload["fold_aggregate"]["mean"],
                    yerr=payload["fold_aggregate"]["std"], fmt="o",
                    color="#c0703f", capsize=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([k for k, _ in fold_rows], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("per-fold balanced accuracy")
    ax.set_title(f"Grouped-CV variation — mean ± std and fold points ({dataset_label})")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def make_report_figures(
    artifacts_dir: str | Path,
    out_dir: str | Path,
    weights: dict[str, float] | None = None,
    dataset_label: str = "official",
) -> list[str]:
    metrics = load_metrics(artifacts_dir)
    truth, pred, _used_weights, blend_name = blend_predictions(artifacts_dir, weights)
    written = make_prediction_figures(truth, pred, out_dir, dataset_label, blend_name)
    written.append(make_ablation_figure(metrics, out_dir, dataset_label))
    written.append(make_runtime_ram_figure(metrics, out_dir, dataset_label))
    written.append(make_cv_variation_figure(metrics, out_dir, dataset_label))
    return written