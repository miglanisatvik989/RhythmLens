CONFIG ?= config.yaml
PYTHON ?= python3

.PHONY: show setup verify-data doctor test fit-preprocessing smoke train evaluate
show setup verify-data doctor test fit-preprocessing smoke train evaluate:
	$(PYTHON) scripts/manage.py $@ --config "$(CONFIG)"
