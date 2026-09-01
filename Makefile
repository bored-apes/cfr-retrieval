VENV ?= .venv
PY   := $(VENV)/bin/python
export PYTHONPATH := src

.PHONY: install build eval pool calibrate bench serve test clean

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e .

build:        ## fetch, chunk, embed
	$(PY) -m cfr.cli build

pool:         ## gather candidates to judge
	$(PY) -m cfr.cli pool

eval:         ## run the ablation
	$(PY) -m cfr.cli eval --json-out evaldata/ablation.json --markdown-out evaldata/ablation.md
	$(PY) scripts/judge.py export

qrels-import: ## restore the answer key into a freshly built db
	$(PY) scripts/judge.py import

calibrate:    ## sweep the abstention threshold
	$(PY) -m cfr.cli calibrate

bench:        ## latency per stage, across rerank depths
	$(PY) -m cfr.cli bench

serve:
	$(PY) -m cfr.cli serve --reload

test:
	$(PY) -m pytest tests -q

clean:
	rm -rf data/*.db data/*.db-wal data/*.db-shm
