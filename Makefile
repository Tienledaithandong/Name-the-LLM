PYTHON ?= python
DATA ?= data
CONFIG ?= configs/baseline.yaml

.PHONY: help install test demo audit splits baseline features embeddings ensemble predict package clean

help:
	@echo "make install     install runtime dependencies"
	@echo "make test        run the unit test suite"
	@echo "make demo        synthetic end-to-end smoke run (no competition data)"
	@echo "make audit       audit the competition data in $(DATA)"
	@echo "make splits      build frozen prompt-grouped folds"
	@echo "make baseline    cross-validate configs/baseline.yaml"
	@echo "make features    cross-validate the stylometry ablations"
	@echo "make embeddings  cross-validate dense/nuisance variants"
	@echo "make ensemble    greedy blend from saved OOF scores"
	@echo "make predict     refit on all data and write submission.parquet"
	@echo "make package     build an upload-ready Kaggle dataset folder"

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

ensemble:
	$(PYTHON) -m src.cli ensemble --dataset official

predict:
	$(PYTHON) -m src.cli predict --config $(CONFIG) --data-dir $(DATA) --fit

package:
	$(PYTHON) -m src.cli package --dataset official

clean:
	rm -rf artifacts/demo artifacts/cv artifacts/final artifacts/submissions .pytest_cache
