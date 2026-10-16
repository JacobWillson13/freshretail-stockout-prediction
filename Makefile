# Pipeline entry points. Override the interpreter with: make all PYTHON=.venv/bin/python
PYTHON ?= python

.PHONY: data warehouse checks train backtest report test lint all

data:        ## download train and eval parquet into data/raw/ (skips when present)
	$(PYTHON) scripts/download_data.py

warehouse:   ## raw parquet -> DuckDB warehouse (data/warehouse.duckdb)
	$(PYTHON) -m src.run_sql

checks:      ## data checks -> reports/tables/data_checks_*.csv
	$(PYTHON) -m src.checks

train:       ## final fit on train, threshold and calibration on validation, test scored once
	$(PYTHON) -m src.modeling

backtest:    ## rolling-origin backtest over the train period (selects the model for test)
	$(PYTHON) -m src.backtest

report:      ## bootstrap CIs, capacity metrics, calibration, slices, SHAP, figures
	$(PYTHON) -m src.report

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

all: data warehouse checks backtest train report
