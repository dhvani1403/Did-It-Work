PY ?= python3

.PHONY: all test data simulate validate analysis figures

all: test simulate validate analysis figures

test:
	$(PY) -m pytest -q

data:            # optional: re-download page views (needs internet)
	$(PY) scripts/fetch_italy.py

simulate:        # a few minutes, uses all CPU cores
	$(PY) scripts/simulate.py

validate:        # ~2.5 min
	$(PY) scripts/validate_against_library.py

analysis:        # under a minute, offline (reads data/processed/italy_*)
	$(PY) scripts/italy_analysis.py

figures:
	$(PY) scripts/make_figures.py
