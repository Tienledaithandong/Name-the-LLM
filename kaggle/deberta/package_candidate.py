"""Prepare the evaluated DeBERTa+sparse candidate for an offline Kaggle run.

Only an explicit allowlist of inference files is copied.  This command never
uploads or submits.  It refuses an existing output directory and requires real
DeBERTa-v3-large weights, matching metadata, and both fitted sparse models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import LABELS  # noqa: E402

MIN_MODEL_SIZE_BYTES = 500_000_000
MODEL_FILES = (
    "config.json", "inference_manifest.json", "label_order.json",
    "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "spm.model", "vocab.json", "merges.txt",
)
EXPECTED_SPARSE_CONFIGS = {
    "char_tfidf_logreg": {"name": "char_tfidf_logreg", "features": ["char"],
                           "classifier": "logreg", "C": 2.0, "max_features": 100000},
    "style_only": {"name": "style_only", "features": [], "style": True,
                   "classifier": "logreg", "C": 1.0},
}


def _prediction_module():
    path = ROOT / "kaggle" / "deberta" / "predict.py"
    spec = importlib.util.spec_from_file_location("candidate_packager_predict", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the offline predictor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_file(root: Path, rel: str) -> Path:
    path = root / rel
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"missing or unsafe artifact file: {rel}")
    return path


def validate_model_weights(model_dir: Path) -> None:
    """Check a non-pickle large-model checkpoint and its classifier shape."""
    from safetensors import safe_open

    weights = _safe_file(model_dir, "model.safetensors")
    if weights.stat().st_size < MIN_MODEL_SIZE_BYTES:
        raise ValueError("weight file is too small to be the evaluated large model; dummy weights are refused")
    try:
        with safe_open(str(weights), framework="np") as checkpoint:
            names = set(checkpoint.keys())
            if not {"deberta.embeddings.word_embeddings.weight", "classifier.weight"} <= names:
                raise ValueError("checkpoint does not contain expected DeBERTa weights")
            shape = checkpoint.get_slice("classifier.weight").get_shape()
            if len(shape) != 2 or shape[0] != len(LABELS):
                raise ValueError("checkpoint classifier does not have 18 output classes")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid or unreadable safetensors checkpoint") from exc


def validate_sources(model_dir: Path, sparse_dir: Path) -> dict:
    """Reject dummy, incompatible or incomplete inference artifacts."""
    predictor = _prediction_module()
    predictor.load_label_order(model_dir)
    metadata = predictor.load_inference_manifest(model_dir)
    if metadata["model_name"] != "microsoft/deberta-v3-large" or metadata["max_len"] != 384:
        raise ValueError("candidate requires the evaluated deberta-v3-large model and max_len 384")
    config_path = _safe_file(model_dir, "config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "deberta-v2":
        raise ValueError("model config is not DeBERTa-v3")
    if (model_dir / "pytorch_model.bin").exists():
        raise ValueError("candidate requires one unsharded safetensors file, not pickle weights")
    validate_model_weights(model_dir)
    _safe_file(model_dir, "tokenizer_config.json")
    if not any((model_dir / n).is_file() for n in ("tokenizer.json", "spm.model")):
        raise ValueError("model tokenizer files are missing")
    sparse_manifest = json.loads(_safe_file(sparse_dir, "manifest.json").read_text(encoding="utf-8"))
    if sparse_manifest.get("dataset") != "official":
        raise ValueError("candidate requires sparse bundles fitted on official data")
    if sparse_manifest.get("class_order") != list(LABELS):
        raise ValueError("sparse class order mismatch")
    if sparse_manifest.get("seed") != 42:
        raise ValueError("sparse artifact seed does not match the evaluated candidate")
    sparse_files = []
    for entry in sparse_manifest["models"]:
        if entry.get("config") != EXPECTED_SPARSE_CONFIGS.get(entry.get("name")):
            raise ValueError("sparse model configuration does not match the evaluated candidate")
        rel = entry.get("path")
        if (not isinstance(rel, str) or Path(rel).is_absolute() or
                ".." in Path(rel).parts or len(Path(rel).parts) != 2 or
                Path(rel).parts[0] != "models" or Path(rel).suffix != ".joblib"):
            raise ValueError("unexpected sparse model path")
        _safe_file(sparse_dir, rel)
        sparse_files.append(rel)
    predictor.load_sparse_bundles(sparse_dir)
    return {"model": metadata, "sparse_files": sparse_files,
            "sklearn_version": sparse_manifest["environment"]["sklearn"]}


def package_candidate(model_dir: Path, sparse_dir: Path, out: Path,
                      wheel_dir: Path | None = None) -> Path:
    """Build a new local package without training rows, predictions, or OOF."""
    model_dir, sparse_dir, out = Path(model_dir), Path(sparse_dir), Path(out)
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    details = validate_sources(model_dir, sparse_dir)
    wheel = None
    if wheel_dir is not None:
        wheel_root = Path(wheel_dir)
        matches = list(wheel_root.glob(f"scikit_learn-{details['sklearn_version']}-cp313-cp313-manylinux*x86_64.whl"))
        if len(matches) != 1 or not zipfile.is_zipfile(matches[0]):
            raise ValueError("wheel directory must contain exactly one valid matching CPython 3.13 Linux scikit-learn wheel")
        wheel = _safe_file(wheel_root, matches[0].name)
    out_parent = out.parent.resolve()
    out_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".candidate-staging-", dir=out_parent)).resolve()
    if stage.parent != out_parent:
        raise ValueError("staging path is outside the intended output directory")
    try:
        (stage / "model").mkdir()
        for name in MODEL_FILES:
            if (model_dir / name).exists():
                shutil.copy2(_safe_file(model_dir, name), stage / "model" / name)
        (stage / "sparse" / "models").mkdir(parents=True)
        shutil.copy2(_safe_file(sparse_dir, "manifest.json"), stage / "sparse" / "manifest.json")
        for rel in details["sparse_files"]:
            dest = stage / "sparse" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_safe_file(sparse_dir, rel), dest)
        (stage / "src").mkdir()
        for source in (ROOT / "src").glob("*.py"):
            if not source.name.startswith("test_"):
                shutil.copy2(source, stage / "src" / source.name)
        shutil.copy2(ROOT / "kaggle" / "deberta" / "predict.py", stage / "predict.py")
        shutil.copy2(ROOT / "kaggle" / "deberta" / "requirements_gpu.txt", stage / "requirements_gpu.txt")
        package_manifest = {"schema_version": "candidate-package/v1", "model": details["model"],
                            "provenance_status": "unverified",
                            "sparse_sklearn_version": details["sklearn_version"],
                            "weights": {"deberta": 0.5, "char_tfidf_logreg": 0.25, "style_only": 0.25}}
        if wheel is not None:
            (stage / "wheels").mkdir()
            shutil.copy2(wheel, stage / "wheels" / wheel.name)
            with wheel.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            package_manifest["sklearn_wheel"] = {"filename": wheel.name, "sha256": digest}
        (stage / "package-manifest.json").write_text(json.dumps(package_manifest, indent=2) + "\n", encoding="utf-8")
        (stage / "README.md").write_text(
            "# Offline inference candidate\n\n"
            "Attach this package and the competition data to an internet-disabled Kaggle GPU notebook. "
            "Use the same scikit-learn version recorded in package-manifest.json; attach vetted wheels "
            "if the notebook image does not contain it. Verify all other dependencies offline.\n\n"
            "If included, install the bundled wheel without network access before starting the predictor:\n\n"
            "```sh\npython -m pip install --no-index --no-deps ./wheels/scikit_learn-*.whl\n```\n\n"
            "From the package directory, run:\n\n"
            "```sh\npython predict.py --mode blend --model-dir ./model --sparse-dir ./sparse "
            "--test /kaggle/input/competitions/name-that-llm/test.csv "
            "--out /kaggle/working/submission.parquet\n```\n\n"
            "Run a realistic synthetic-size smoke test with runtime and memory measurements before submission. "
            "This package is an unvalidated candidate, not a leaderboard result. "
            "Packaging does not independently verify checkpoint provenance, pool decontamination, "
            "or equivalence to the CV training run. Resolve those gates before submission.\n", encoding="utf-8")
        stage.rename(out_parent / out.name)
    finally:
        if stage.exists():
            if stage.parent != out_parent:
                raise RuntimeError("refusing cleanup outside the staging parent")
            shutil.rmtree(stage)
    return out_parent / out.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--sparse-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--wheel-dir", type=Path,
                        help="optional directory with the matching Linux CPython 3.13 scikit-learn wheel")
    args = parser.parse_args()
    print(f"Prepared candidate package: {package_candidate(args.model_dir, args.sparse_dir, args.out, args.wheel_dir)}")


if __name__ == "__main__":
    main()
