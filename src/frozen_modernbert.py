"""Offline frozen ModernBERT-large features for the official-data pilot."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


MODEL_REPO = "answerdotai/ModernBERT-large"
MODEL_REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
# Hugging Face records small files as Git blob IDs and weights as a Git LFS SHA-256.
_GIT_BLOBS = {
    "config.json": "f58291adedf9dc2fe5a5d80f8e2221bd1f135a23",
    "special_tokens_map.json": "6bf65ea14185a88eef7e24b41c86a953a58dcbad",
    "tokenizer.json": "de3f3e316b66f3885df49ab2c0fff3e477ff1680",
    "tokenizer_config.json": "d78aa50c890c167d8700301e5580164466e51fa5",
}
_WEIGHT_SHA256 = "44510fec5d3a81a1877f225637b869495f18e55f6f23a09abb9be0acc030295f"
_MAX_LENGTH = 384


class FrozenModernBertError(RuntimeError):
    """The pinned encoder, runtime or cached embeddings failed a check."""


def verify_checkpoint(checkpoint_dir: Path) -> None:
    """Verify the exact local model and tokenizer files in the pinned revision."""
    missing = [
        name for name in (*_GIT_BLOBS, "model.safetensors")
        if not (checkpoint_dir / name).is_file()
    ]
    if missing:
        raise FrozenModernBertError(f"pinned ModernBERT snapshot is missing: {', '.join(missing)}")
    if (checkpoint_dir / "adapter_config.json").exists():
        raise FrozenModernBertError("the frozen ModernBERT snapshot must not include an adapter")
    for filename, git_blob in _GIT_BLOBS.items():
        data = (checkpoint_dir / filename).read_bytes()
        content = f"blob {len(data)}\0".encode("ascii") + data
        if hashlib.sha1(content, usedforsecurity=False).hexdigest() != git_blob:
            raise FrozenModernBertError(f"{filename} does not match the pinned ModernBERT snapshot")
    with (checkpoint_dir / "model.safetensors").open("rb") as weights:
        actual = hashlib.file_digest(weights, "sha256").hexdigest()
    if actual != _WEIGHT_SHA256:
        raise FrozenModernBertError("model.safetensors does not match the pinned ModernBERT snapshot")


def masked_mean(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Average only nonpadding tokens, accumulating in float32."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise FrozenModernBertError("hidden states and attention mask disagree")
    include = attention_mask.bool()
    n_tokens = include.sum(dim=1, keepdim=True)
    if (n_tokens == 0).any().item():
        raise FrozenModernBertError("cannot pool a passage with no encoded tokens")
    values = hidden.float().masked_fill(~include.unsqueeze(-1), 0)
    return values.sum(dim=1) / n_tokens


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _require_time(deadline: float | None) -> None:
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError("the frozen ModernBERT encoding deadline expired")


class FrozenModernBert:
    """Encode individual passages with the pinned model and no fitted state."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        cache_dir: str | Path,
        *,
        repo_id: str = MODEL_REPO,
        model_revision: str = MODEL_REVISION,
        tokenizer_revision: str = MODEL_REVISION,
        max_length: int = _MAX_LENGTH,
        batch_size: int = 4,
        embedding_dim: int = 1024,
        device: str = "cuda",
        dtype: str = "float16",
        attention_implementation: str = "sdpa",
    ) -> None:
        if repo_id != MODEL_REPO or model_revision != MODEL_REVISION or tokenizer_revision != MODEL_REVISION:
            raise FrozenModernBertError("model and tokenizer must use the pinned ModernBERT-large revision")
        if not 1 <= max_length <= _MAX_LENGTH or batch_size < 1 or embedding_dim < 1:
            raise FrozenModernBertError("invalid token cap, embedding dimension or batch size")
        if dtype != "float16" or attention_implementation != "sdpa":
            raise FrozenModernBertError("the frozen pilot requires float16 and SDPA")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cache_dir = Path(cache_dir)
        self.max_length = max_length
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.device = device
        self._runtime: tuple[PreTrainedTokenizerBase, PreTrainedModel] | None = None
        self._cache_settings = {
            "format": 1,
            "repo_id": repo_id,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "max_length": max_length,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "padding": "right, fixed length",
            "pooling": "mean over nonpadding tokens in float32",
            "device": device,
            "dtype": dtype,
            "attention_implementation": attention_implementation,
            "torch": _installed_version("torch"),
            "transformers": _installed_version("transformers"),
            "tokenizers": _installed_version("tokenizers"),
        }

    def load(self) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
        """Load the snapshot offline using the Transformers 5.23 `dtype` API."""
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise FrozenModernBertError("install CUDA PyTorch and Transformers for the frozen pilot") from exc
        if torch.device(self.device).type != "cuda" or not torch.cuda.is_available():
            raise FrozenModernBertError("the frozen ModernBERT pilot requires CUDA-enabled PyTorch")
        verify_checkpoint(self.checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_dir), local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        tokenizer.padding_side = "right"
        model = AutoModel.from_pretrained(
            str(self.checkpoint_dir),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        if model.config.model_type != "modernbert" or model.config.hidden_size != self.embedding_dim:
            raise FrozenModernBertError("unexpected ModernBERT architecture or hidden size")
        if model.config._attn_implementation != "sdpa" or next(model.parameters()).dtype != torch.float16:
            raise FrozenModernBertError("ModernBERT did not load with float16 and SDPA")
        model.to(self.device)
        model.eval()
        model.requires_grad_(False)
        self._runtime = tokenizer, model
        return self._runtime

    def _validated_matrix(self, matrix: np.ndarray, n_rows: int) -> np.ndarray:
        if matrix.shape != (n_rows, self.embedding_dim) or matrix.dtype != np.float32:
            raise FrozenModernBertError("embedding matrix has an unexpected shape or dtype")
        if not np.isfinite(matrix).all():
            raise FrozenModernBertError("embedding matrix contains nonfinite values")
        return matrix

    def encode(self, texts: Sequence[str], *, deadline: float | None = None) -> np.ndarray:
        """Encode a batch in input order without reading or writing the disk cache."""
        _require_time(deadline)
        if len(texts) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        tokenizer, model = self.load()
        import torch

        result = np.empty((len(texts), self.embedding_dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                _require_time(deadline)
                stop = min(start + self.batch_size, len(texts))
                tokens = tokenizer(
                    list(texts[start:stop]),
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                ids = tokens["input_ids"].to(self.device)
                mask = tokens["attention_mask"].to(self.device)
                hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
                values = masked_mean(hidden, mask).cpu().numpy()
                result[start:stop] = self._validated_matrix(values, stop - start)
        _require_time(deadline)
        return result

    def _cache_path(self, texts: Sequence[str]) -> Path:
        passage_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        key = json.dumps(
            {"settings": self._cache_settings, "passage_hashes": passage_hashes},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return self.cache_dir / f"modernbert_{hashlib.sha256(key).hexdigest()}.npy"

    def encode_cached(self, texts: Sequence[str], *, deadline: float | None = None) -> np.ndarray:
        """Reuse an exact ordered embedding batch, storing no raw text or labels."""
        _require_time(deadline)
        if len(texts) == 0:
            return self.encode(texts, deadline=deadline)
        path = self._cache_path(texts)
        if path.is_file():
            _require_time(deadline)
            matrix = self._validated_matrix(np.load(path, allow_pickle=False), len(texts))
            _require_time(deadline)
            return matrix

        matrix = self.encode(texts, deadline=deadline)
        _require_time(deadline)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = None
        try:
            with NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as output:
                tmp_path = Path(output.name)
                np.save(output, matrix, allow_pickle=False)
            _require_time(deadline)
            os.replace(tmp_path, path)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        return matrix
