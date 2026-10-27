.PHONY: install data build tune train experiments score serve test lint all

install:
	pip install -e ".[dev]"

data:
	cvf ingest

build:
	cvf build

tune:
	cvf tune

train:
	cvf train

experiments:
	cvf experiments

score:
	cvf score

serve:
	cvf serve --host 0.0.0.0 --port 8000

test:
	pytest

lint:
	ruff check src tests

all: data build train score
