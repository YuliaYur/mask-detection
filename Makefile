.PHONY: install format format-check lint test check

PYTHON ?= python

# Our code lives in src/; the vendored subtrees (yolo7_face, spiga_project, code_former)
# and the research notebooks are intentionally excluded from formatting and linting.
SOURCES = src

install:
	$(PYTHON) -m pip install -r docker/requirements.txt -r requirements-dev.txt

format:
	$(PYTHON) -m black $(SOURCES)

format-check:
	$(PYTHON) -m black --check $(SOURCES)

lint:
	$(PYTHON) -m flake8 $(SOURCES)
	$(PYTHON) -m pylint $(SOURCES)

test:
	$(PYTHON) -m pytest

check: format-check lint
