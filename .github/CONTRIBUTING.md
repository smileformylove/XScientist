# Contributing

Thanks for your interest in contributing!

This repository is intended to be **safe to run and safe to publish**. Please help us keep it that way:

## Ground rules

- Do **not** commit secrets (API keys, tokens, cookies, auth files).
- Do **not** commit personal information (local absolute paths, usernames, emails, machine hostnames).
- Prefer portable paths (`~`, `XDG_*`, relative paths) and config via environment variables.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[full,service,dev]" -c requirements/constraints-ci.txt
```

## Running tests / checks

```bash
make syntax
make engineering
make test
make coverage
make package-check
```

`make smoke` also runs the repository validator and requires a valid local
login session. See `docs/ENGINEERING.md` for CI lanes, dependency policy, and
the release checklist.

## Code style

- Format: Black (see `pyproject.toml`).
- Keep changes focused and add/adjust tests when behavior changes.

## Submitting changes

1. Create a topic branch.
2. Ensure tests pass locally.
3. Open a PR with:
   - Problem statement
   - What changed
   - How you tested
   - Research/ARA artifact impact
   - Risk and rollback
   - Any compatibility notes (Python version, OS, GPU requirements)

Changes to protocol schemas, claim/evidence binding, re-execution, CAS
reachability, or ContextPack selection must include focused compatibility
tests. Do not weaken a quality gate solely to make CI pass.

## Reporting issues

- Use GitHub Issues for bugs/feature requests.
- For security issues, please follow `.github/SECURITY.md`.
