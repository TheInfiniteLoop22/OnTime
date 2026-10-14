.PHONY: setup data build features train evaluate explain impact app test lint format all

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

train:
	$(PY) -m ontime.pipeline train

evaluate:
	$(PY) -m ontime.pipeline evaluate

explain:
	$(PY) -m ontime.pipeline explain

impact:
	$(PY) -m ontime.pipeline impact

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

all: data build features train evaluate explain impact
