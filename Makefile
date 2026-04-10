PYTHON ?= $(shell \
	if command -v python3.12 >/dev/null 2>&1; then echo python3.12; \
	elif command -v python3.11 >/dev/null 2>&1; then echo python3.11; \
	elif command -v python3.10 >/dev/null 2>&1; then echo python3.10; \
	else echo python3; \
	fi)
AUTH_FILE ?=
PREFLIGHT_AUTH_ARG := $(if $(AUTH_FILE),--auth-file $(AUTH_FILE),)

.PHONY: syntax test validate preflight smoke doctor format

syntax:
	$(PYTHON) -m compileall -q ai_scientist *.py tests
	bash -n run_stable_daemon.sh
	bash -n start_research.sh

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

validate:
	$(PYTHON) validate_repo.py --full-import-smoke

preflight:
	$(PYTHON) preflight_check.py --strict $(PREFLIGHT_AUTH_ARG)

smoke: syntax test validate

doctor: smoke preflight

format:
	$(PYTHON) -m black ai_scientist tests *.py
