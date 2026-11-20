from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contrastive import CrossedBatchSampler, prompt_contrastive_loss
from src.constants import LABELS


def test_sampler_builds_cross_prompt_positives_and_hard_negatives():
    labels = np.tile(np.arange(8), 3)
    groups = np.repeat(np.arange(3), 8)
    a = CrossedBatchSampler(labels, groups, authors_per_batch=4, seed=5)
    b = CrossedBatchSampler(labels, groups, authors_per_batch=4, seed=5)
    for _ in range(20):
        idx = a.sample()
        assert np.array_equal(idx, b.sample())
        assert len(set(idx)) == 8
        assert len(set(groups[idx])) == 2
        for i in idx:
            assert np.any((labels[idx] == labels[i]) & (groups[idx] != groups[i]))
            assert np.any((labels[idx] != labels[i]) & (groups[idx] == groups[i]))


def test_sampler_refuses_inadequate_or_misaligned_groups():
    with pytest.raises(ValueError, match='aligned'):
        CrossedBatchSampler([0], [0, 1])
    with pytest.raises(ValueError, match='no prompt pair'):
        CrossedBatchSampler([0, 1, 2, 3], ['a', 'a', 'b', 'b'])


def test_loss_rewards_cross_prompt_matches_and_backpropagates():
    torch = pytest.importorskip('torch')
    vectors = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]], requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    prompts = torch.tensor([0, 0, 1, 1])
    good = prompt_contrastive_loss(vectors, labels, prompts)
    wrong = prompt_contrastive_loss(vectors, torch.tensor([0, 1, 1, 0]), prompts)
    hard = prompt_contrastive_loss(vectors, labels, prompts, hard_weight=3)
    assert good < wrong and hard > good and torch.isfinite(hard)
    hard.backward()
    assert torch.isfinite(vectors.grad).all()
    with pytest.raises(ValueError, match='cross-prompt'):
        prompt_contrastive_loss(vectors, torch.arange(4), prompts)


def test_tiny_offline_training_end_to_end(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import BertConfig, BertModel, BertTokenizerFast
    from experiments.contrastive_gpu import run
    from src.ensemble import load_oof

    torch.set_num_threads(1)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nsample\ntext\n', encoding='utf-8')
    BertTokenizerFast(vocab_file=str(checkpoint / 'vocab.txt')).save_pretrained(checkpoint)
    BertModel(BertConfig(vocab_size=7, hidden_size=16, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=32)).save_pretrained(checkpoint)
    rows = [{'id': f'{g}-{i}', 'prompt_id': f'p{g}', 'text': 'sample text', 'model': label}
            for g in range(5) for i, label in enumerate(LABELS)]
    cfg = dict(seed=42, holdout_seed=7, holdout_groups=1, n_splits=2,
               authors_per_batch=2, steps_per_epoch=1, epochs=1, max_length=16,
               learning_rate=2e-5, weight_decay=0.01, contrastive_weight=0.2,
               temperature=0.1, hard_negative_weight=3.0,
               variants=['classification', 'contrastive', 'hard_negative'])
    output = tmp_path / 'result'
    run(pd.DataFrame(rows), checkpoint, output, cfg, allow_cpu=True)
    oof = load_oof(output)
    assert set(oof) == set(cfg['variants'])
    assert all(x['scores'].shape == (72, len(LABELS)) for x in oof.values())
    with pytest.raises(FileExistsError):
        run(pd.DataFrame(rows), checkpoint, output, cfg, allow_cpu=True)


def test_paired_bootstrap_resamples_prompts_and_is_reproducible():
    from src.contrastive import paired_prompt_bootstrap
    y = [0, 1, 0, 1]
    groups = ['a', 'a', 'b', 'b']
    baseline = [1, 0, 1, 0]
    candidate = y
    first = paired_prompt_bootstrap(y, groups, baseline, candidate, n_resamples=100, seed=9)
    assert first == paired_prompt_bootstrap(y, groups, baseline, candidate, n_resamples=100, seed=9)
    assert first['lower_95'] == first['upper_95'] == 1.0
