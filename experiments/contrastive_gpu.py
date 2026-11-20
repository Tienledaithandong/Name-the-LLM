"""Grouped-CV GPU experiment for prompt-aware authorship representation.

Requires a local Hugging Face checkpoint and CUDA. Does not perform submission,
load test.csv, or evaluate the reserved prompt groups. See docs/contrastive_gpu.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.constants import LABELS, N_CLASSES  # noqa: E402
from src.contrastive import CrossedBatchSampler, paired_prompt_bootstrap, prompt_contrastive_loss  # noqa: E402
from src.data import dataset_fingerprint, labels_to_indices, validate_train_frame  # noqa: E402
from src.metrics import evaluate  # noqa: E402
from src.repro import env_snapshot, git_commit  # noqa: E402
from src.validation import make_holdout, make_splits  # noqa: E402



def checkpoint_fingerprint(path: Path) -> str:
    """Hash relative names and checkpoint contents to identify exact local weights."""
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob('*') if p.is_file()):
        digest.update(str(file.relative_to(path)).encode('utf-8') + b'\0')
        with file.open('rb') as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()

def run(train, checkpoint, output, cfg, allow_cpu=False):
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError('CUDA is required; enable a GPU in the Kaggle notebook')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for key in ("epochs", "steps_per_epoch", "authors_per_batch", "max_length", "n_splits"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not cfg["variants"] or len(set(cfg["variants"])) != len(cfg["variants"]):
        raise ValueError("variants must be nonempty and unique")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f'attach a local Hugging Face checkpoint directory: {checkpoint}')
    checkpoint_hash = checkpoint_fingerprint(checkpoint)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a fresh run directory')
    train = validate_train_frame(train)
    if set(train.model) != set(LABELS):
        raise ValueError('training data must contain all 18 official classes')
    selection_idx, reserved_idx = make_holdout(train, cfg['holdout_groups'], cfg['holdout_seed'])
    selection = train.iloc[selection_idx].reset_index(drop=True)
    reserved = train.iloc[reserved_idx].reset_index(drop=True)
    folds, method = make_splits(selection, cfg['n_splits'], cfg['seed'])
    y = labels_to_indices(selection.model)
    groups, _ = pd.factorize(selection.prompt_id, sort=True)
    texts = selection.text.tolist()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    encoded = tokenizer(texts, truncation=True, max_length=cfg['max_length'], padding=False)
    output.mkdir(parents=True)
    (output / 'oof').mkdir()
    metadata = {'dataset_fingerprint': dataset_fingerprint(train),
                'selection_fingerprint': dataset_fingerprint(selection),
                'reserved_fingerprint': dataset_fingerprint(reserved),
                'selection_rows': len(selection), 'reserved_rows': len(reserved),
                'reserved_used_for_training_or_scoring': False,
                'warning': 'Historical experiments may have used reserved prompts; not a pristine holdout.',
                'config': cfg, 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash, 'method': method,
                'git_commit': git_commit(ROOT), 'environment': env_snapshot(),
                'torch': torch.__version__, 'class_order': list(LABELS)}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    fold_ids = np.empty(len(selection), dtype=np.int64)
    for f, (_, valid) in enumerate(folds):
        fold_ids[valid] = f

    class EncoderClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(str(checkpoint), local_files_only=True)
            self.head = nn.Linear(self.encoder.config.hidden_size, N_CLASSES)

        def forward(self, batch):
            hidden = self.encoder(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.head(pooled), pooled

    def tensor_batch(indices):
        items = [{key: values[int(i)] for key, values in encoded.items()} for i in indices]
        return {key: value.to(device) for key, value in tokenizer.pad(items, padding=True, return_tensors='pt').items()}

    summary = {}
    for variant in cfg['variants']:
        if variant not in {'classification', 'contrastive', 'hard_negative'}:
            raise ValueError(f'unknown variant {variant!r}')
        scores = np.empty((len(selection), N_CLASSES), dtype=np.float32)
        for fold, (tr, va) in enumerate(folds):
            if set(y[tr]) != set(range(N_CLASSES)):
                raise ValueError(f'fold {fold} training data does not contain every class')
            seed = int(cfg['seed']) + fold
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if device == 'cuda' else None
            sampler = CrossedBatchSampler(y[tr], groups[tr], cfg['authors_per_batch'], seed)
            model = EncoderClassifier().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']), weight_decay=float(cfg['weight_decay']))
            model.train()
            for epoch in range(cfg['epochs']):
                total_loss = 0.0
                for _ in range(cfg['steps_per_epoch']):
                    indices = tr[sampler.sample()]
                    target = torch.as_tensor(y[indices], device=device, dtype=torch.long)
                    prompts = torch.as_tensor(groups[indices], device=device, dtype=torch.long)
                    optimizer.zero_grad(set_to_none=True)
                    logits, pooled = model(tensor_batch(indices))
                    loss = nn.functional.cross_entropy(logits, target)
                    if variant != 'classification':
                        weight = cfg['hard_negative_weight'] if variant == 'hard_negative' else 1.0
                        loss = loss + cfg['contrastive_weight'] * prompt_contrastive_loss(pooled, target, prompts, cfg['temperature'], weight)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += float(loss.detach())
                print(f'{variant} fold={fold} epoch={epoch + 1} loss={total_loss / cfg["steps_per_epoch"]:.4f}', flush=True)
            model.eval()
            with torch.no_grad():
                for start in range(0, len(va), cfg['authors_per_batch'] * 2):
                    idx = va[start:start + cfg['authors_per_batch'] * 2]
                    logits, _ = model(tensor_batch(idx))
                    scores[idx] = logits.float().cpu().tolist()
            del model, optimizer
            if device == 'cuda':
                torch.cuda.empty_cache()
        result = evaluate(selection.model.tolist(), [LABELS[i] for i in scores.argmax(axis=1)], model=variant)
        summary[variant] = result
        np.savez_compressed(output / 'oof' / f'{variant}.npz', scores=scores, y=y,
                            labels=np.asarray(LABELS, dtype=str), row_ids=selection.id.to_numpy(dtype=str),
                            fold_ids=fold_ids, dataset_fingerprint=dataset_fingerprint(selection),
                            config=json.dumps({'name': variant, 'experiment': cfg, 'checkpoint_sha256': checkpoint_hash}, sort_keys=True))
        print(f'{variant}: pooled grouped-CV balanced accuracy={result["balanced_accuracy"]:.4f}', flush=True)
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if 'classification' in cfg['variants']:
        base = np.load(output / 'oof' / 'classification.npz', allow_pickle=False)['scores'].argmax(axis=1)
        comparisons = {}
        for variant in cfg['variants']:
            if variant == 'classification':
                continue
            pred = np.load(output / 'oof' / f'{variant}.npz', allow_pickle=False)['scores'].argmax(axis=1)
            comparisons[variant] = paired_prompt_bootstrap(y, groups, base, pred, seed=cfg['seed'])
        (output / 'paired_bootstrap.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
    print(f'Experiment completed: {output}. Reserved prompts were not scored.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-cpu', action='store_true', help='allow slow CPU execution for synthetic smoke tests')
    parser.add_argument('--config', default=str(ROOT / 'configs/contrastive_gpu.yaml'))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    run(pd.read_csv(args.train), args.checkpoint, args.output, cfg, allow_cpu=args.allow_cpu)


if __name__ == '__main__':
    main()
