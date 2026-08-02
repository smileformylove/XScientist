PYTHON ?= $(shell \
	if command -v python3.12 >/dev/null 2>&1; then echo python3.12; \
	elif command -v python3.11 >/dev/null 2>&1; then echo python3.11; \
	elif command -v python3.10 >/dev/null 2>&1; then echo python3.10; \
	else echo python3; \
	fi)
AUTH_FILE ?=
PREFLIGHT_AUTH_ARG := $(if $(AUTH_FILE),--auth-file $(AUTH_FILE),)

.PHONY: syntax test validate preflight smoke doctor format package perf-record perf-compare executor-image

syntax:
	$(PYTHON) -m compileall -q ai_scientist xscientist compat scripts tools tests
	bash -n run_stable_daemon.sh
	bash -n start_research.sh

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

validate:
	$(PYTHON) -m xscientist validate --full-import-smoke

preflight:
	$(PYTHON) -m xscientist preflight --strict $(PREFLIGHT_AUTH_ARG)

smoke: syntax test validate

doctor: smoke preflight

format:
	$(PYTHON) -m black ai_scientist xscientist compat scripts tools tests

package:
	$(PYTHON) tools/build_distribution.py

perf-record:
	@test -n "$(OUTPUT)" || (echo "OUTPUT=/path/to/result.json is required" && exit 2)
	$(PYTHON) tools/performance_regression.py record --output "$(OUTPUT)"

perf-compare:
	@test -n "$(BASELINE)" || (echo "BASELINE=/path/to/baseline.json is required" && exit 2)
	@test -n "$(CANDIDATE)" || (echo "CANDIDATE=/path/to/candidate.json is required" && exit 2)
	$(PYTHON) tools/performance_regression.py compare --baseline "$(BASELINE)" --candidate "$(CANDIDATE)"

executor-image:
	docker build -f docker/Dockerfile.executor -t xscientist-exec:latest .
