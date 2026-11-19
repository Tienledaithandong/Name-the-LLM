from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.cli import cmd_package, cmd_reserve_holdout, cmd_score_reserved
from src.config import load_config, resolved_models
from src.data import dataset_fingerprint
from src.ensemble import load_oof, normalize_scores, run_ensemble
from src.inference import fit_final, load_final, validate_artifact_provenance, validate_weight_provenance
from src.metrics import macro_f1
from src.train import run_cv
from src.validation import LeakageError, load_splits, make_splits, save_splits


def test_saved_splits_reject_reordered_and_corrupt_data(train_df, folds, tmp_path):
    path = save_splits(tmp_path / 'splits.json', train_df, folds, 'group', 42)
    with pytest.raises(LeakageError, match='fingerprint'):
        load_splits(path, train_df.iloc[::-1].reset_index(drop=True))
    payload = json.loads(path.read_text())
    payload['folds'][0]['valid'].append(payload['folds'][0]['valid'][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(LeakageError):
        load_splits(path, train_df)


def test_macro_f1_zero_predictions_and_all_wrong():
    for truth, pred in [(['a','b','b'], ['a','a','a']), (['a','b'], ['b','a'])]:
        assert macro_f1(truth, pred, ['a','b']) == pytest.approx(
            f1_score(truth, pred, labels=['a','b'], average='macro', zero_division=0))


def test_absent_class_cannot_beat_present_scores():
    scores = np.array([[1., -1., -1e9]])
    normalized = normalize_scores(scores)
    assert normalized.argmax(axis=1)[0] == 0
    assert normalized[0, 2] < normalized[0, 1]


def test_oof_provenance_and_min_gain(train_df, folds, tmp_path):
    cfg = [{'name':'majority', 'baseline':'majority'}, {'name':'random', 'baseline':'stratified_random'}]
    run_cv(train_df, folds, cfg, tmp_path, registry_path=None)
    result = run_ensemble(tmp_path, min_gain=2.0)
    assert result['weights'] == {result['best_single']['model']: 1.0}
    path = tmp_path / 'oof' / 'random.npz'
    with np.load(path, allow_pickle=False) as data:
        values = {k: data[k] for k in data.files}
    values['fold_ids'] = np.roll(values['fold_ids'], 1)
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match='provenance'):
        load_oof(tmp_path)


def test_package_protects_existing_and_overlapping_paths(tmp_path):
    final = tmp_path / 'final'
    final.mkdir()
    (final / 'manifest.json').write_text('{}')
    args = Namespace(final_dir=str(final), dataset='synthetic', out=str(final), kaggle_user='test')
    with pytest.raises(ValueError, match='overlap'):
        cmd_package(args)
    args.out = str(tmp_path / 'existing')
    Path(args.out).mkdir()
    with pytest.raises(FileExistsError):
        cmd_package(args)
    assert (final / 'manifest.json').exists()


def test_reserved_holdout_workflow_and_offline_notebook(train_df, test_df, tmp_path, monkeypatch):
    # Reserve before selection, then train/CV using selection rows only.
    data = tmp_path / 'data'; data.mkdir()
    train_df.to_csv(data / 'train.csv', index=False)
    test_df.to_csv(data / 'test.csv', index=False)
    reserve = tmp_path / 'reserved'
    cmd_reserve_holdout(Namespace(data_dir=str(data), synthetic=True, out=str(reserve), holdout_groups=5, seed=7))
    selection = pd.read_csv(reserve / 'selection.csv')
    holdout = pd.read_csv(reserve / 'holdout.csv')
    assert set(selection.prompt_id).isdisjoint(holdout.prompt_id)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('seed: 42\nmodels:\n  - name: majority\n    baseline: majority\n')
    cfg = load_config(config_path)
    models = resolved_models(cfg)
    folds, _ = make_splits(selection, n_splits=3)
    root = tmp_path / 'artifacts'
    cv = root / 'cv' / 'synthetic'
    run_cv(selection, folds, models, cv, dataset_label='synthetic', registry_path=None)
    run_ensemble(cv)
    final = root / 'final' / 'synthetic'
    fit_final(models, selection, final, dataset_label='synthetic')
    weights = cv / 'ensemble' / 'weights.json'
    bundles, manifest = load_final(final)
    validate_artifact_provenance(manifest, selection, models, 42)
    validate_weight_provenance(weights, manifest, bundles)
    changed = selection.copy(); changed.loc[0, 'text'] += ' changed'
    assert dataset_fingerprint(changed) != dataset_fingerprint(selection)
    with pytest.raises(ValueError, match='fingerprint'):
        validate_artifact_provenance(manifest, changed, models, 42)
    args = Namespace(selection=str(reserve/'selection.csv'), holdout=str(reserve/'holdout.csv'),
                     final_dir=str(final), weights=str(weights), config=str(config_path), seed=None)
    assert cmd_score_reserved(args) == 0
    args.holdout = args.selection
    with pytest.raises(LeakageError, match='overlap'):
        cmd_score_reserved(args)
    monkeypatch.setattr('src.cli.ARTIFACTS_DIR', root)
    package = tmp_path / 'package'
    cmd_package(Namespace(final_dir=str(final), dataset='synthetic', out=str(package), kaggle_user='test'))
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'kaggle/submission.ipynb').read_text())
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    env = {**os.environ, 'NTL_ARTIFACT_DIR': str(package), 'NTL_TEST_PATH': str(data / 'test.csv'),
           'PYTHONPATH': str(package)}
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert pd.read_parquet(tmp_path / 'submission.parquet').id.astype(str).tolist() == test_df.id.astype(str).tolist()


def test_optional_offline_minilm_package(train_df, test_df, tmp_path, monkeypatch):
    checkpoint = os.environ.get('NTL_MINILM_DIR')
    if not checkpoint:
        pytest.skip('set NTL_MINILM_DIR to a local MiniLM checkpoint to run the offline integration')
    pytest.importorskip('sentence_transformers')
    models = [{'name': 'majority', 'baseline': 'majority'}]
    root = tmp_path / 'artifacts'; final = root / 'final' / 'synthetic'
    fit_final(models, train_df, final, dataset_label='synthetic')
    monkeypatch.setattr('src.cli.ARTIFACTS_DIR', root)
    package = tmp_path / 'package'
    cmd_package(Namespace(final_dir=str(final), dataset='synthetic', out=str(package),
                          kaggle_user='test', crossed=True, encoder_path=checkpoint))
    test_path = tmp_path / 'test.csv'; test_df.head(20).to_csv(test_path, index=False)
    notebook = json.loads((Path(__file__).resolve().parents[1] / 'kaggle/submission.ipynb').read_text())
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    env = {**os.environ, 'NTL_ARTIFACT_DIR': str(package), 'NTL_TEST_PATH': str(test_path),
           'PYTHONPATH': str(package), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert len(pd.read_parquet(tmp_path / 'submission.parquet')) == 20


def test_config_controls_cv_and_features(train_df, tmp_path, monkeypatch):
    from src import cli
    config = tmp_path / 'config.yaml'
    config.write_text('seed: 19\ncv:\n  n_splits: 3\n  shuffle: false\n  method: stratified_group\nfeatures:\n  min_df: 7\nmodels:\n  - name: majority\n    baseline: majority\n')
    cfg = load_config(config)
    assert resolved_models(cfg)[0]['min_df'] == 7
    monkeypatch.setattr(cli, 'ARTIFACTS_DIR', tmp_path / 'artifacts')
    args = Namespace(seed=None, n_splits=None, splits=None)
    folds, split_id = cli.ensure_splits(train_df, args, 'synthetic', cfg)
    assert len(folds) == 3 and '-s19-n3' in split_id
    payload = json.loads((tmp_path / 'artifacts/splits/splits_synthetic.json').read_text())
    assert payload['shuffle'] is False and payload['method'] == 'stratified_group'
    args.seed = 20
    with pytest.raises(ValueError, match='seed'):
        cli.ensure_splits(train_df, args, 'synthetic', cfg)


def test_weight_provenance_rejects_missing_model_and_mismatch(train_df, tmp_path):
    from src.model import ModelError
    cfg = [{'name': 'majority', 'baseline': 'majority'}]
    final = tmp_path / 'final'
    fit_final(cfg, train_df, final, dataset_label='synthetic')
    bundles, manifest = load_final(final)
    path = tmp_path / 'weights.json'
    path.write_text(json.dumps({'weights': {'missing': 1},
                                'dataset_fingerprint': manifest['dataset_fingerprint'],
                                'model_configs': {}}))
    with pytest.raises(ValueError, match='unknown models'):
        validate_weight_provenance(path, manifest, bundles)
    path.write_text(json.dumps({'weights': {'majority': 1}, 'dataset_fingerprint': 'stale',
                                'model_configs': {'majority': cfg[0]}}))
    with pytest.raises(ValueError, match='provenance'):
        validate_weight_provenance(path, manifest, bundles)
    with pytest.raises(ModelError, match='config or seed'):
        validate_artifact_provenance(manifest, train_df, cfg, 17)
