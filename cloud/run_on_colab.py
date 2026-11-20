"""SASCR on the cloud (Kaggle / Google Colab).

End-to-end orchestrator for the SASCR (Sibling-Aware Source-Conditioned
Readout) contribution. It deliberately runs on the platform's GPU and pulls the
causal backbone down into that runtime, never on the user's local machine.

Usage
-----
1. Set the flags below, then run the whole file as one Colab / Kaggle Notebook
   cell.

    BACKEND        "transformers" (default): frozen Qwen3-8B-Base multi-layer
                    causal readout; ~6 GB downloaded into the runtime cache on
                    first run. "hashing"       : CPU-only single-view reader,
                    protocol sanity check, no download.
    USE_EXTERNAL_POOL=True   : attach a Kaggle dataset id holding the
                    competition pool (open-pool-v15) so Member A can use it.
                    False (default): sparse trunk + SASCR fusion, no pool.

Everything installs CPU-only torch on a CPU-only box and never touches a
GPU-only path. The heavy back-ends live behind cloud-only `import torch` /
`import transformers` blocks.

The heavy primitives are the repo's own tested ones:

    src.train.run_cv                     5-fold StratifiedGroupKFold on
                                          prompt_id, official-train-only OOF
    src.inference.fit_final              refit every member on all official rows
    src.inference.predict_dataframe      row-z blend -> 18-column predictions
    src.features_multilayer.MultiLayerReadoutBuilder  Member B
    src.model.SiblingResidualClassifier   Member C

Syncs with `python -m src.cli train --config configs/sascr.yaml`, except the
backbone and external pool are supplied by the cloud run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# --- 1. Repo root ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- 2. Dependencies ------------------------------------------------------
def _pip_install(packages: list[str]) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *packages],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

# A CPU-only box gets the CPU wheel; Colab/Kaggle GPU gets the cu121 wheels.
if os.environ.get("COLAB_RESIDENT_ENV") == "1":
    _pip_install([
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])
else:
    _pip_install(["torch", "torchvision", "torchaudio"])

_pip_install([
    "transformers", "sentence-transformers", "kaggle", "pandas", "pyarrow",
    "scikit-learn", "scipy",
])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.constants import ARTIFACTS_DIR, RANDOM_STATE  # noqa: E402
from src.data import load_competition_data  # noqa: E402
from src.ensemble import normalize_scores  # noqa: E402
from src.inference import (  # noqa: E402
    fit_final,
    load_final,
    load_weights,
    predict_dataframe,
)
from src.model import estimator_from_config  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.repro import env_snapshot, git_commit  # noqa: E402
from src.submission import build_submission, validate_submission  # noqa: E402
from src.train import run_cv  # noqa: E402
from src.validation import make_splits  # noqa: E402

# --- 3. Flags -------------------------------------------------------------
BACKEND = os.getenv("SASCR_BACKEND", "transformers")  # transformers | hashing
USE_EXTERNAL_POOL = os.getenv("SASCR_USE_POOL", "False").strip().lower() == "true"
EXTERNAL_POOL_ID = os.getenv("SASCR_POOL_ID", "").strip()  # Kaggle dataset id
KAGGLE_TRAIN_CSV = os.getenv("SASCR_TRAIN_CSV", "").strip()

assert BACKEND in ("transformers", "hashing"), BACKEND
print(f"[cloud] BACKEND={BACKEND}  USE_EXTERNAL_POOL={USE_EXTERNAL_POOL}")

# --- 4. Data ----------------------------------------------------------------
if KAGGLE_TRAIN_CSV:
    TRAIN_CSV = Path(KAGGLE_TRAIN_CSV)
elif USE_EXTERNAL_POOL and EXTERNAL_POOL_ID:
    # python -m kaggle datasets download -d OPEN_POOL_ID -p /tmp && unzip
    TRAIN_CSV = Path("/content") / "train.csv"
else:
    # Colab / Kaggle: upload train.csv to /content or /kaggle/working/
    TRAIN_CSV = Path("/content") / "train.csv"

train, test = load_competition_data(str(TRAIN_CSV))
print(f"[cloud] loaded {len(train)} official train rows, "
      f"{train['prompt_id'].nunique()} prompts, "
      f"{test['prompt_id'].nunique()} test prompts")

splits, method = make_splits(
    train, n_splits=5, seed=RANDOM_STATE, method="stratified_group"
)
print(f"[cloud] folds={method}, {len(splits)} folds, "
      f"git={git_commit(REPO_ROOT)}")

# --- 5. Fold OOF (fair comparison) ------------------------------------------
cfg = load_config("configs/sascr.yaml")
print(f"[cloud] SASCR config: {[m['name'] for m in cfg['models']]}")

oof: dict[str, float] = {}
for raw in cfg["models"]:
    name = str(raw["name"])
    results = run_cv(
        train, splits, [raw], ARTIFACTS_DIR, seed=RANDOM_STATE,
        dataset_label="cloud", registry_path=None,
        split_id=f"cloud-s{RANDOM_STATE}",
    )
    oof[name] = results[0]["pooled"]["balanced_accuracy"]
    print(f"[cloud] OOF {name:32s} {oof[name]:.4f}")

# --- 6. Final refit on all official rows -------------------------------------
final_dir = REPO_ROOT / "artifacts" / "cloud_final"
final_dir.mkdir(parents=True, exist_ok=True)

pool_df = None
if USE_EXTERNAL_POOL and EXTERNAL_POOL_ID:
    from src.pool import decontaminate_global  # noqa: F401

    pool_df = pd.read_csv(EXTERNAL_POOL_ID)
    keep, _ = decontaminate_global(pool_df, train, cos_threshold=0.3)
    pool_df = pool_df.loc[keep].reset_index(drop=True)
    print(f"[cloud] external pool kept {len(pool_df)} rows "
          f"(cos<=0.3, capped)")

fit_final(
    cfg["models"], train, final_dir, seed=RANDOM_STATE,
    dataset_label="cloud", pool_df=pool_df,
    pool_models=["char_tfidf_logreg_pool40k"] if USE_EXTERNAL_POOL else (),
)
print(f"[cloud] refit members into {final_dir}")

# --- 7. Blend + submit -------------------------------------------------------
bundles, _manifest = load_final(final_dir)
weights_path = ARTIFACTS_DIR / "cv" / "cloud" / "ensemble" / "weights.json"
weights = load_weights(weights_path) if weights_path.is_file() else None
submission = predict_dataframe(bundles, test, weights)

out = REPO_ROOT / "submission.parquet"
build_submission(test["id"], submission, out, expected_ids=test["id"])
validate_submission(out, expected_ids=test["id"])
print(f"[cloud] wrote {out} ({len(submission)} rows)")
