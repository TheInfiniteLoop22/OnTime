.PHONY: setup data build features baselines train evaluate explain impact monitor rolling stage2 scores bundle report notebooks app test lint format all

PY = uv run python

setup:
	uv sync
	uv run pre-commit install

data:
	$(PY) -m ontime.pipeline data

build:
	$(PY) -m ontime.pipeline build

features:
	$(PY) -m ontime.pipeline features

baselines:
	$(PY) -m ontime.pipeline baselines

train:
	$(PY) -m ontime.pipeline train

evaluate:
	$(PY) -m ontime.pipeline evaluate

explain:
	$(PY) -m ontime.pipeline explain

impact:
	$(PY) -m ontime.pipeline impact

scores:
	$(PY) -m ontime.pipeline scores

bundle:
	$(PY) -m ontime.pipeline bundle

stage2:
	$(PY) -m ontime.pipeline stage2

rolling:
	$(PY) -m ontime.pipeline rolling

monitor:
	$(PY) -m ontime.pipeline monitor

report:
	$(PY) -m ontime.pipeline report

notebooks:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_impact_analysis.ipynb
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling_walkthrough.ipynb
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_explainability.ipynb
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_stage2_handover.ipynb

app:
	uv run streamlit run app/Home.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

all: data build impact features baselines train evaluate explain monitor rolling stage2 scores report
