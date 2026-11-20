"""Exercise the frozen encoder without a checkpoint or a GPU."""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src import frozen_modernbert as modernbert


def test_verify_checkpoint_checks_tokenizer_and_weights(tmp_path, monkeypatch) -> None:
    tokens = b'{"fixture": true}'
    fake_weights = b"fixture weights"
    git_oid = hashlib.sha1(f"blob {len(tokens)}\0".encode() + tokens).hexdigest()
    monkeypatch.setattr(modernbert, "_GIT_BLOBS", {"tokenizer.json": git_oid})
    monkeypatch.setattr(modernbert, "_WEIGHT_SHA256", hashlib.sha256(fake_weights).hexdigest())
    (tmp_path / "tokenizer.json").write_bytes(tokens)
    (tmp_path / "model.safetensors").write_bytes(fake_weights)
    modernbert.verify_checkpoint(tmp_path)

    (tmp_path / "model.safetensors").write_bytes(b"altered weights")
    with pytest.raises(modernbert.FrozenModernBertError, match="model.safetensors"):
        modernbert.verify_checkpoint(tmp_path)

    (tmp_path / "tokenizer.json").write_bytes(b"altered tokenizer")
    with pytest.raises(modernbert.FrozenModernBertError, match="tokenizer.json"):
        modernbert.verify_checkpoint(tmp_path)


def test_load_uses_only_verified_local_safetensors_with_float16_sdpa(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    verify = Mock()
    monkeypatch.setattr(modernbert, "verify_checkpoint", verify)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    tokenizer = SimpleNamespace(padding_side="left")
    model = Mock()
    model.config = SimpleNamespace(
        model_type="modernbert", hidden_size=1024, _attn_implementation="sdpa",
    )
    model.parameters.return_value = iter([torch.zeros(1, dtype=torch.float16)])
    get_tokenizer = Mock(return_value=tokenizer)
    get_model = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=get_tokenizer),
            AutoModel=SimpleNamespace(from_pretrained=get_model),
        ),
    )

    encoder = modernbert.FrozenModernBert(tmp_path, tmp_path / "cache")
    assert encoder.load() == (tokenizer, model)
    assert encoder.load() == (tokenizer, model)
    verify.assert_called_once_with(tmp_path)
    get_tokenizer.assert_called_once_with(
        str(tmp_path), local_files_only=True, trust_remote_code=False, use_fast=True,
    )
    get_model.assert_called_once_with(
        str(tmp_path), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=torch.float16, attn_implementation="sdpa",
    )
    assert tokenizer.padding_side == "right"
    model.to.assert_called_once_with("cuda")
    model.eval.assert_called_once_with()
    model.requires_grad_.assert_called_once_with(False)


def test_mean_pooling_ignores_padding_and_matches_single_passage(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    def tokenize(passages, *, padding, truncation, max_length, return_attention_mask, return_tensors):
        assert padding == "max_length" and truncation and return_attention_mask and return_tensors == "pt"
        ids = torch.full((len(passages), max_length), 99)
        mask = torch.zeros_like(ids)
        for row, text in enumerate(passages):
            values = [len(token) for token in text.split()][:max_length]
            ids[row, :len(values)] = torch.tensor(values, dtype=ids.dtype)
            mask[row, :len(values)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def model(input_ids, attention_mask):
        assert torch.is_inference_mode_enabled()
        hidden = torch.stack((input_ids, input_ids * 2), dim=-1).to(torch.float16)
        return SimpleNamespace(last_hidden_state=hidden)

    encoder = modernbert.FrozenModernBert(
        tmp_path, tmp_path / "cache", max_length=3, embedding_dim=2, batch_size=2, device="cpu",
    )
    encoder._runtime = tokenize, model
    passages = ["a xyz", "abc", "abc abcd abcde abcdef"]
    batch = encoder.encode(passages)
    singles = np.concatenate([encoder.encode([text]) for text in passages])
    np.testing.assert_array_equal(batch, [[2, 4], [3, 6], [4, 8]])
    np.testing.assert_array_equal(batch, singles)
    assert batch.dtype == np.float32

    with pytest.raises(modernbert.FrozenModernBertError, match="no encoded tokens"):
        encoder.encode([""])


def test_cache_tracks_order_and_settings_and_checks_shape(tmp_path, monkeypatch) -> None:
    encoder = modernbert.FrozenModernBert(tmp_path, tmp_path / "cache", embedding_dim=2)
    evaluate = Mock(return_value=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    monkeypatch.setattr(encoder, "encode", evaluate)
    passages = ["first synthetic passage", "second synthetic passage"]
    expected = encoder.encode_cached(passages)
    np.testing.assert_array_equal(encoder.encode_cached(passages), expected)
    evaluate.assert_called_once_with(passages, deadline=None)
    path = encoder._cache_path(passages)
    assert path != encoder._cache_path(passages[::-1])
    different_cap = modernbert.FrozenModernBert(tmp_path, path.parent, max_length=192, embedding_dim=2)
    assert path != different_cap._cache_path(passages)
    assert path.read_bytes().find(passages[0].encode()) == -1

    np.save(path, np.ones((1, 2), dtype=np.float32), allow_pickle=False)
    with pytest.raises(modernbert.FrozenModernBertError, match="shape or dtype"):
        encoder.encode_cached(passages)
    np.save(path, np.full((2, 2), np.inf, dtype=np.float32), allow_pickle=False)
    with pytest.raises(modernbert.FrozenModernBertError, match="nonfinite"):
        encoder.encode_cached(passages)


def test_expired_deadline_prevents_model_load_and_cache_read(tmp_path, monkeypatch) -> None:
    encoder = modernbert.FrozenModernBert(tmp_path, tmp_path)
    passages = ["fixture passage"]
    encoder._cache_path(passages).touch()
    load_model = Mock()
    load_cache = Mock()
    monkeypatch.setattr(encoder, "load", load_model)
    monkeypatch.setattr(np, "load", load_cache)
    monkeypatch.setattr(modernbert, "perf_counter", lambda: 10.0)

    with pytest.raises(TimeoutError, match="deadline expired"):
        encoder.encode(passages, deadline=10.0)
    with pytest.raises(TimeoutError, match="deadline expired"):
        encoder.encode_cached(passages, deadline=10.0)
    load_model.assert_not_called()
    load_cache.assert_not_called()


def test_encoding_stops_between_batches_without_publishing_cache(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    clock = [0.0]
    monkeypatch.setattr(modernbert, "perf_counter", lambda: clock[0])
    tokens = torch.ones((1, 2), dtype=torch.int64)
    tokenizer = Mock(return_value={"input_ids": tokens, "attention_mask": tokens})

    def run_batch(**inputs):
        clock[0] = 11.0
        return SimpleNamespace(last_hidden_state=torch.ones((1, 2, 1), dtype=torch.float16))

    model = Mock(side_effect=run_batch)
    cache_dir = tmp_path / "cache"
    encoder = modernbert.FrozenModernBert(
        tmp_path, cache_dir, max_length=2, batch_size=1, embedding_dim=1, device="cpu",
    )
    encoder._runtime = tokenizer, model

    with pytest.raises(TimeoutError, match="deadline expired"):
        encoder.encode_cached(["first batch", "second batch"], deadline=10.0)
    assert tokenizer.call_count == model.call_count == 1
    assert not cache_dir.exists()
