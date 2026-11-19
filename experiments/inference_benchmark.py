"""Cold-start inference benchmark at the expected hidden-test size (plan §8, L208).

Report §9 claimed "seconds for ~4,000 test passages" as an *estimate*. This measures
it: a fresh Python process loads the packaged bundles exactly as
``kaggle/submission.ipynb`` does (attached ``src`` on ``sys.path``, offline env,
``load_final`` → ``load_weights`` → ``predict_dataframe`` → ``build_submission`` →
read-back validation) and reports per-stage wall time and peak memory.

The test frame is **synthetic** (``src.synthetic``, k≈2 passages/prompt, the shape
``docs/strategy_review.md`` infers for the hidden test). No competition row is read,
nothing is fit, and no leaderboard number is claimed. The measurement is on the local
CPU/OS, not on Kaggle's runtime, so it bounds the *work*, not the platform.

    python experiments/inference_benchmark.py --sizes 4000 8000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
CPU_BUDGET_HOURS = 9.0
PASSAGES_PER_PROMPT = 2.0


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def synthetic_test_frame(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Exactly ``n_rows`` synthetic ``id``/``text`` rows at k≈2 passages per prompt."""
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    sys.path.insert(0, str(REPO_ROOT))
    from src.synthetic import make_synthetic_dataset

    inclusion = PASSAGES_PER_PROMPT / 18.0
    prompts = max(1, int(n_rows / PASSAGES_PER_PROMPT) + 8)
    for _ in range(6):
        _, test, _ = make_synthetic_dataset(
            n_prompts=1, test_prompts=prompts, seed=seed, inclusion_prob=inclusion
        )
        if len(test) >= n_rows:
            return test.head(n_rows).reset_index(drop=True)
        prompts = int(prompts * 1.25) + 8
    raise RuntimeError(f"could not generate {n_rows} synthetic rows")


def _peak_memory_gb(process) -> dict:
    info = process.memory_info()
    out = {"rss_gb": round(info.rss / 1024**3, 4)}
    peak = getattr(info, "peak_wset", None)  # Windows: true peak working set
    if peak is not None:
        out["peak_wset_gb"] = round(peak / 1024**3, 4)
    return out


def worker_main(package: Path, test_csv: Path, report_path: Path) -> None:
    """Notebook-equivalent inference in a cold process; writes a timing report."""
    import psutil

    process = psutil.Process()
    stages: dict[str, float] = {}
    marks: dict[str, float] = {}

    def mark(name: str, start: float) -> float:
        now = time.perf_counter()
        stages[name] = round(now - start, 4)
        marks[name] = round(process.memory_info().rss / 1024**3, 4)
        return now

    t_start = time.perf_counter()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(package))  # the attached artifact dir, as in the notebook

    from src.constants import LABELS, N_CLASSES
    from src.inference import load_final, load_weights, predict_dataframe
    from src.submission import build_submission
    import src as shipped_src

    t = mark("imports", t_start)
    if not str(Path(shipped_src.__file__).resolve()).startswith(str(package.resolve())):
        raise RuntimeError(
            f"benchmark must use the packaged src, got {shipped_src.__file__}"
        )

    test = pd.read_csv(test_csv)
    assert {"id", "text"} <= set(test.columns), f"unexpected columns: {list(test.columns)}"
    assert test["id"].is_unique, "test.csv has duplicate ids"
    assert test["text"].notna().all(), "test.csv has null text"
    t = mark("read_test_csv", t)

    bundles, manifest = load_final(package)
    assert manifest["class_order"] == list(LABELS), "artifact class order mismatch"
    t = mark("load_bundles", t)

    weights_path = package / "weights.json"
    if not weights_path.is_file():
        raise FileNotFoundError("validated weights.json is required for submission")
    weights = load_weights(weights_path)
    missing = sorted(set(weights) - set(bundles))
    if missing:
        raise ValueError(f"weights reference missing model bundles: {missing}")
    inference_path = package / "inference.json"
    inference = (
        json.loads(inference_path.read_text(encoding="utf-8"))
        if inference_path.is_file()
        else {"crossed": False}
    )
    if inference.get("crossed"):
        raise ValueError("crossed inference is withdrawn and cannot be submitted under D024")
    t = mark("load_weights_and_guards", t)

    predictions = predict_dataframe(bundles, test, weights)
    assert len(predictions) == len(test)
    assert set(predictions) <= set(LABELS)
    t = mark("predict", t)

    out = build_submission(test["id"], predictions, "submission.parquet", expected_ids=test["id"])
    t = mark("write_parquet", t)

    check = pd.read_parquet(out)
    assert list(check.columns) == ["id", "model"]
    assert len(check) == len(test)
    assert check["id"].is_unique and not check.isna().any().any()
    assert check["model"].isin(LABELS).all()
    t = mark("read_back_and_validate", t)

    words = test["text"].astype(str).str.split().str.len()
    report = {
        "n_rows": int(len(test)),
        "n_classes": int(N_CLASSES),
        "models": sorted(bundles),
        "weights": weights,
        "packaged_src": str(Path(shipped_src.__file__).resolve()),
        "stage_seconds": stages,
        "stage_rss_gb": marks,
        "worker_total_seconds": round(t - t_start, 4),
        "peak_memory_gb": _peak_memory_gb(process),
        "output_bytes": int(out.stat().st_size),
        "output_rows": int(len(check)),
        "synthetic_word_count": {
            "min": int(words.min()), "max": int(words.max()),
            "mean": round(float(words.mean()), 2), "median": float(words.median()),
        },
        "prediction_counts": {str(k): int(v) for k, v in pd.Series(predictions).value_counts().items()},
        "prediction_counts_note": (
            "Runtime diagnostic only. The fixture's template styles are not real model "
            "styles, so this distribution says nothing about the hidden test's class mix "
            "and must not be read as a prior or an accuracy result."
        ),
        "validation_passed": True,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_cold_start(package: Path, test_csv: Path, work_dir: Path) -> dict:
    """Run the worker in a fresh process and time the whole cold start."""
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = (work_dir / "worker_report.json").resolve()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # so `src` can only come from the attached package
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        sys.executable, str(SCRIPT), "--worker",
        "--package", str(package), "--test-csv", str(test_csv),
        "--report", str(report_path),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True)
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"cold-start worker failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cold_start_wall_seconds"] = round(wall, 4)
    report["interpreter_startup_seconds"] = round(wall - report["worker_total_seconds"], 4)
    report["returncode"] = proc.returncode
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="artifacts/kaggle_dataset")
    ap.add_argument("--sizes", type=int, nargs="+", default=[4000, 8000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/inference_benchmark")
    ap.add_argument("--data-dir", default="data", help="official train, for a length reference only")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--test-csv", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--report", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker_main(Path(args.package), Path(args.test_csv), Path(args.report))
        return

    sys.path.insert(0, str(REPO_ROOT))
    from src.repro import env_snapshot

    package = Path(args.package).resolve()
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    official_words = None
    train_path = Path(args.data_dir) / "train.csv"
    if train_path.is_file():
        official = pd.read_csv(train_path, usecols=["text"])["text"].astype(str).str.split().str.len()
        official_words = {
            "n_rows": int(len(official)), "min": int(official.min()), "max": int(official.max()),
            "mean": round(float(official.mean()), 2), "median": float(official.median()),
        }
    for size in args.sizes:
        frame = synthetic_test_frame(size, seed=args.seed)
        test_csv = (out_dir / f"test_synthetic_{size}.csv").resolve()
        frame.to_csv(test_csv, index=False)
        report = run_cold_start(package, test_csv, out_dir / f"work_{size}")
        report["requested_rows"] = size
        report["headroom_vs_9h_budget"] = round(
            CPU_BUDGET_HOURS * 3600.0 / report["cold_start_wall_seconds"], 1
        )
        runs.append(report)
        print(
            f"n={size}: cold start {report['cold_start_wall_seconds']:.2f}s "
            f"(predict {report['stage_seconds']['predict']:.2f}s, "
            f"load {report['stage_seconds']['load_bundles']:.2f}s), "
            f"peak {report['peak_memory_gb'].get('peak_wset_gb', report['peak_memory_gb']['rss_gb']):.3f} GB"
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "label": "synthetic",
        "purpose": "plan §8 L208 cold-start inference benchmark at the expected hidden-test size",
        "package": str(package),
        "package_git_commit": manifest.get("git_commit"),
        "package_fit_modes": {m["name"]: m.get("fit_mode") for m in manifest["models"]},
        "package_pool_cap": (manifest.get("pool") or {}).get("pool_cap"),
        "environment": env_snapshot(),
        "cpu_budget_hours": CPU_BUDGET_HOURS,
        "passages_per_prompt_assumed": PASSAGES_PER_PROMPT,
        "fits_performed": 0,
        "competition_rows_scored": 0,
        "competition_train_rows_read_for_length_reference": (
            official_words["n_rows"] if official_words else 0
        ),
        "official_train_word_count": official_words,
        "synthetic_vs_official_mean_words": (
            round(runs[0]["synthetic_word_count"]["mean"] / official_words["mean"], 3)
            if official_words and runs else None
        ),
        "caveat": (
            "Synthetic test rows (src.synthetic), not competition data; no fitting and no "
            "leaderboard claim. Timings are local CPU/OS, NOT Kaggle's runtime, so they bound "
            "the work rather than the platform. Peak working set is a Windows counter; on "
            "Linux only sampled RSS is available. The 9h budget is the competition CPU limit. "
            "Fixture passages are LONGER than official ones (see synthetic_vs_official_mean_words), "
            "so the measured time is conservative for official-length text. The synthetic "
            "prediction distribution is a runtime artefact, not an accuracy or class-prior result. "
            "Only train.csv word counts are read as a length reference; no label is used."
        ),
        "runs": runs,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n| rows | cold start s | imports s | load bundles s | predict s | write+validate s | peak GB | ×9h budget |")
    print("|---|---|---|---|---|---|---|---|")
    for r in runs:
        s = r["stage_seconds"]
        peak = r["peak_memory_gb"].get("peak_wset_gb", r["peak_memory_gb"]["rss_gb"])
        print(
            f"| {r['n_rows']} | {r['cold_start_wall_seconds']:.2f} | {s['imports']:.2f} | "
            f"{s['load_bundles']:.2f} | {s['predict']:.2f} | "
            f"{s['write_parquet'] + s['read_back_and_validate']:.2f} | {peak:.3f} | "
            f"{r['headroom_vs_9h_budget']:.0f} |"
        )
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()