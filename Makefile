CONFIG ?= config.yaml
PYTHON ?= python3

.PHONY: help show setup setup-evaluation verify-data doctor test fit-preprocessing smoke train evaluate
help:
	$(PYTHON) scripts/manage.py --help

show setup setup-evaluation verify-data doctor test fit-preprocessing smoke train evaluate:
	$(PYTHON) scripts/manage.py $@ --config "$(CONFIG)"
