PYTHON ?= python
DATA ?= data
CONFIG ?= configs/baseline.yaml

.PHONY: help install test demo audit splits baseline features embeddings frozen-encoder ensemble figures stability stress predict package predict-crossed package-crossed clean

help:
	@echo "make install     install runtime dependencies"
	@echo "make test        run the unit test suite"
	@echo "make demo        synthetic end-to-end smoke run (no competition data)"
	@echo "make audit       audit the competition data in $(DATA)"
	@echo "make splits      build frozen prompt-grouped folds"
	@echo "make baseline    cross-validate configs/baseline.yaml"
	@echo "make features    cross-validate the stylometry ablations"
	@echo "make embeddings  cross-validate dense/nuisance variants"
	@echo "make frozen-encoder  H3 benchmark (needs optional sentence-transformers)"
	@echo "make ensemble    greedy blend from saved OOF scores"
	@echo "make figures     render report figures from saved CV artifacts"
	@echo "make stability   blend robustness: alternative seeds + secondary holdout"
	@echo "make stress      primary full OOF + k-subsampled paired-difference stress test"
	@echo "make predict     refit on all data and write submission.parquet"
	@echo "make package     build an upload-ready Kaggle dataset folder"
	@echo "make predict-crossed   WITHDRAWN (D017/D018) research-only crossed inference"
	@echo "make package-crossed   WITHDRAWN research-only dataset + offline MiniLM encoder"

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m src.cli demo --config $(CONFIG)

audit:
	$(PYTHON) -m src.cli audit --data-dir $(DATA)

splits:
	$(PYTHON) -m src.cli splits --data-dir $(DATA)

baseline: splits
	$(PYTHON) -m src.cli train --config configs/baseline.yaml --data-dir $(DATA)

features:
	$(PYTHON) -m src.cli train --config configs/features.yaml --data-dir $(DATA)

embeddings:
	$(PYTHON) -m src.cli train --config configs/embeddings.yaml --data-dir $(DATA)

# Optional H3 research benchmark; requires `pip install sentence-transformers`
# and a local checkpoint. Not part of the submission pipeline.
frozen-encoder:
	$(PYTHON) -m src.cli train --config configs/frozen_encoder.yaml --data-dir $(DATA)

ensemble:
	$(PYTHON) -m src.cli ensemble --dataset official

figures:
	$(PYTHON) -m src.cli figures --dataset official

stability:
	$(PYTHON) -m src.cli stability --config configs/submission.yaml --data-dir $(DATA)

# Primary full grouped OOF + inference-only k-subsampled stress test (paired
# differences + 95% CIs). Add --reserve-fresh-holdout to persist the seed-2027
# reserve without scoring it. Needs optional sentence-transformers for clustering.
stress:
	$(PYTHON) experiments/stress_k.py --data-dir $(DATA)

predict:
	$(PYTHON) -m src.cli predict --config configs/submission.yaml --data-dir $(DATA) --fit

package:
	$(PYTHON) -m src.cli package --dataset official

# WITHDRAWN (D017/D018) — research-only, NOT the submission default. The same blend
# (configs/submission.yaml) plus test-time prompt-group recovery + one-per-model
# assignment; the gain is a k=18 artifact that does not deploy to a k~2 test. Needs
# the optional sentence-transformers dependency and a local MiniLM checkpoint
# (auto-resolved from the HF cache, or set ENCODER_PATH=/path/to/checkpoint).
ENCODER_PATH ?=
predict-crossed:
	$(PYTHON) -m src.cli predict --config configs/submission.yaml --data-dir $(DATA) --fit --crossed $(if $(ENCODER_PATH),--encoder-path $(ENCODER_PATH),)

package-crossed:
	$(PYTHON) -m src.cli package --dataset official --crossed $(if $(ENCODER_PATH),--encoder-path $(ENCODER_PATH),)

clean:
	rm -rf artifacts/demo artifacts/cv artifacts/final artifacts/submissions .pytest_cache
