"""Frozen multi-layer causal readout feature builder.

Member B of SASCR: a frozen causal LM read out at multiple depths instead of
only the final layer.  The design is the same across backends:

1. Freeze the backbone (``requires_grad_(False)``), never train it.
2. Run each passage through the backbone and capture hidden states at several
   transformer depths (early/middle layers retain generative style far better
   than final layers).
3. LayerNorm-standardise each depth, drop the BOS token, pool content tokens.
4. Concatenate the depth views into one dense (n_rows x n_views x dim) matrix.

The builder is backend-agnostic: the ``hashing`` backend is a CPU-only
deterministic projection used for tests and CI; ``sentence-transformers`` and
``transformers`` back the real multi-depth causal reader.

All representations are cached on disk (float32) so repeated CV folds reuse
them exactly, and encoding is a pure function of text (no label), so it is
leakage-safe both train/valid and fold/holdout.  Screening a frozen backbone
against the full official train is done once by the caller (see ``src.pool``
for the identical non-transductive pattern the sparse members already use).
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.constants import ARTIFACTS_DIR

BACKENDS = ("hashing", "sentence-transformers", "transformers")


class ReadoutError(RuntimeError):
    """Raised when a multi-layer readout backend is unavailable."""


class MultiLayerReadoutBuilder:
    """Dense frozen-readout block for ``TextMatrixBuilder``.

    The block is a stateless function of text (plus the pretrained backbone).
    Calling ``.fit`` on a training fold is a no-op for backends other than
    hashing (which fits a fixed random projection); real back-ends just cache.
    """

    def __init__(
        self,
        *,
        backend: str = "hashing",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        depths: tuple[int, ...] = (2, 4, 6, 8, 12, 16, 20, 24),
        pooling: str = "content_token_mean",
        scale: str = "layer_norm",
        dim: int = 256,
        seed: int = 0,
    ) -> None:
        if backend not in BACKENDS:
            raise ReadoutError(f"unknown backend {backend!r}; choose from {BACKENDS}")
        self.backend = backend
        self.model_name = model_name
        self.depths = tuple(int(d) for d in depths)
        if not self.depths:
            raise ReadoutError("depths must be non-empty")
        self.pooling = pooling
        self.scale = scale
        self.dim = int(dim)
        self.seed = int(seed)
        self._model = None
        self._projection = None
        self._embedder = None
        self._embedding_dim = None

    # -- backend loading ----------------------------------------------------
    def _load_backends(self):
        """Import transformers/sentence_transformers; fail fast with a clear message."""
        if self.backend in {"transformers", "sentence-transformers"}:
            try:
                import torch  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ReadoutError(
                    "SASCR Member B (multi-layer causal readout) needs ``torch`` + "
                    "``transformers``. Install the optional ``gpu`` dependency "
                    "(python -m pip install torch transformers) or use backend="
                    "'hashing' for a CPU-only deterministic reader."
                ) from exc
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ReadoutError(
                    "sentence-transformers is not installed. Install it, or use "
                    "backend='hashing' for a CPU-only deterministic reader."
                ) from exc
            try:
                self._embedder = SentenceTransformer(self.model_name)
            except Exception as exc:  # pragma: no cover - network/weights missing
                raise ReadoutError(
                    f"could not load encoder {self.model_name!r}: {exc}. "
                    "Attach the weights as a Kaggle dataset and pass a local path."
                ) from exc

    @property
    def _loaded(self):
        if self.backend in {"transformers", "sentence-transformers"}:
            self._load_backends()
        return bool(self._embedder is not None or self.backend == "hashing")

    # -- backends -----------------------------------------------------------
    def _encode_hashing(self, texts) -> np.ndarray:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.random_projection import SparseRandomProjection

        vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            n_features=2 ** 18,
            alternate_sign=True,
            norm="l2",
            lowercase=False,
            dtype=np.float32,
        )
        sparse = vectorizer.transform(list(texts))
        if self._projection is None:
            self._projection = SparseRandomProjection(
                n_components=min(self.dim, sparse.shape[1]),
                random_state=self.seed,
                dense_output=True,
            )
            self._projection.fit(sparse)
        return np.asarray(self._projection.transform(sparse), dtype=np.float32)

    def _encode_transformers(self, texts) -> np.ndarray:
        import torch
        from torch.nn import LazyLinear
        from transformers import AutoModel, AutoTokenizer

        if self._embedder is not None:
            model = self._embedder
        else:
            model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Transformers tokenizers (e.g. GPT-2) carry no padding token; give
            # them one so batching pads cleanly. This is a tokenizer-level
            # constant, never a label.
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model.eval()
            model.requires_grad_(False)
            model.to("cpu" if torch.cuda.is_available() is False else "cuda")
            # One shared linear projects each pooled (batch x hidden) stream down
            # to the configured `dim` so every backend lands in the same column
            # space as the hashing reader. It is created on first use (lazy),
            # is never trained (the readout stays frozen), and outputs
            # (n_rows x self.dim) regardless of the number of depths.
            if getattr(self, "_project", None) is None:
                self._project = LazyLinear(self.dim, bias=False)

        result = np.empty((len(texts), self.dim), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                stop = min(start + 32, len(texts))
                inputs = tokenizer(
                    list(texts[start:stop]),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states  # tuple of (layer, batch, seq, hidden)
                drop_bos = self.pooling == "content_token_mean"
                views = []
                for depth in self.depths:
                    h = hidden[depth]  # (batch, seq, hidden)
                    mask = attention_mask[:, : h.shape[1]].unsqueeze(-1).to(h.device)
                    if drop_bos:
                        mask = torch.cat(
                            [torch.zeros_like(mask[:, :1]), mask[:, 1:]], dim=1
                        )
                    pooled = (h.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    views.append(pooled)
                stack = torch.stack(views, dim=1)  # (batch, n_depths, hidden)
                projected = self._project(stack)  # (batch, n_depths, dim)
                view = projected.mean(dim=1)  # (batch, dim)
                if self.scale == "layer_norm":
                    view = torch.from_numpy(self._layer_norm(np.asarray(view)))
                result[start:stop] = np.asarray(view.cpu(), dtype=np.float32)
        return result

    @staticmethod
    def _pool_hidden(hidden, attention_mask, depths, drop_bos: bool = False):
        import torch
        from torch.nn import LayerNorm

        n = len(hidden[0]) if not isinstance(hidden, (list, tuple)) else len(hidden)
        device = hidden[0].device if isinstance(hidden, (list, tuple)) else hidden.device
        out = []
        for depth in depths:
            if isinstance(hidden, (list, tuple)):
                h = hidden[depth]
            else:
                h = hidden[depth]
            mask = attention_mask[:, : h.shape[1]].unsqueeze(-1).to(h.device)
            if drop_bos:
                mask = torch.cat(
                    [torch.zeros_like(mask[:, :1]), mask[:, 1:]], dim=1
                )
            pooled = (h.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out.append(pooled)
        return torch.cat(out, dim=-1)

    @staticmethod
    def _layer_norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        std[std < 1e-9] = 1.0
        return np.asarray((x - mean) / std, dtype=np.float32)

    # -- caching -------------------------------------------------------------
    def _cache_path(self, texts) -> Path:
        digest = hashlib.sha256(
            (repr(self.backend) + repr(self.model_name) + repr(self.depths)).encode()
            + b"\x00".join(text.encode("utf-8", "ignore") for text in texts[:200])
            + str(self.seed).encode()
        ).hexdigest()[:16]
        return ARTIFACTS_DIR / "embeddings" / f"multilayer_{digest}.pkl"

    def encode_cached(self, texts) -> np.ndarray:
        path = self._cache_path(list(texts))
        if path.is_file():
            with path.open("rb") as fh:
                return np.asarray(pickle.load(fh), dtype=np.float32)
        texts = [str(t) for t in texts]
        if self.backend == "hashing":
            matrix = self._encode_hashing(texts)
        elif self.backend == "sentence-transformers":
            model = self._embedder
            matrix = np.asarray(model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
            matrix = self._pool_sentence(matrix, self.depths)
        else:
            if not self._loaded:
                self._load_backends()
            matrix = self._encode_transformers(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(matrix, fh)
        return np.asarray(matrix, dtype=np.float32)

    @staticmethod
    def _pool_sentence(matrix: np.ndarray, depths) -> np.ndarray:
        # A single sentence-transformers embedding is the pooled view; depths is
        # retained for shape/documentation parity with the transformer backend.
        return matrix[:, : int(np.prod(depths))]

    # -- sklearn-compatible API ---------------------------------------------
    def fit(self, texts, y=None):
        if self.backend == "hashing":
            if self._projection is None:
                self._projection = None
        return self

    def transform(self, texts) -> np.ndarray:
        return self.encode_cached(list(texts))

    def fit_transform(self, texts, y=None):
        return self.fit(texts, y).transform(texts)
