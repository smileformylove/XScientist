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
pip install -r requirements.txt
```

## Running tests / checks

```bash
python -m compileall -q ai_scientist *.py tests
python -m pytest -q
python validate_repo.py
```

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
   - Any compatibility notes (Python version, OS, GPU requirements)

## Reporting issues

- Use GitHub Issues for bugs/feature requests.
- For security issues, please follow `SECURITY.md`.

