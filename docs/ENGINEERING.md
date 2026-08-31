# Engineering Guide

This document defines the repository's engineering contract. Scientific claims
and artifact semantics are governed separately by the ARA specification and
research-integrity documents.

## Supported environments

- Python: 3.10, 3.11, and 3.12.
- Core SDK/protocol CI platforms: Ubuntu, macOS, and Windows on Python 3.11,
  plus Ubuntu on every supported Python version.
- The complete research-runtime lane runs on Ubuntu/Python 3.11; portable
  SDK/protocol code must not assume POSIX paths.
- The full research runtime may require optional system tools such as LaTeX,
  Poppler, Docker, or GPU-specific libraries.

The compatibility CI lane installs only the core package and exercises the
public SDK and protocol across supported interpreters and operating systems.
The full lane uses Ubuntu/Python 3.11 and runs the complete test, validator,
ARA conformance, coverage, and distribution checks.

## Dependency policy

`pyproject.toml` is authoritative for package dependencies:

- core dependencies keep the SDK and protocol lightweight;
- `full` contains the research runtime;
- `service` contains the HTTP API;
- `dev` contains test, coverage, build, and release tools.

Minimum versions express the supported API floor. Upper compatibility bounds
live in `requirements/constraints-ci.txt` so CI cannot drift across untested
major versions. `requirements.txt` applies that constraint file automatically.

These bounds are not a bit-for-bit research environment lock. Every concrete
research run must persist its resolved interpreter, package versions, external
model identifiers, container digest, and data hashes in its ARA environment
and manifests. CI constraints protect software maintenance; ARA fingerprints
protect scientific replay.

Update dependencies in this order:

1. Change package minimums or CI bounds.
2. Run `make engineering` to verify every dependency has a bound.
3. Run `make coverage` and `make package-check`.
4. Record user-visible changes in `CHANGELOG.md`.

## Local quality gates

```bash
make syntax          # compile Python and parse shell entrypoints
make engineering     # metadata, dependencies, protocol docs, links, CI policy
make test            # complete pytest suite, including unittest-style cases
make coverage        # branch-aware coverage; fails below the regression floor
make package-check   # build wheel/sdist, inspect them, isolated wheel smoke
make smoke           # syntax + engineering + tests + repository validation
```

The whole-repository coverage floor is 45%. It is intentionally slightly below
the measured 47.3% baseline: it prevents large regressions without pretending
that legacy orchestration code already has ideal coverage. New or changed
behavior should receive focused tests even when the aggregate floor still
passes. Raise the floor as low-coverage modules are retired or tested.

## Packaging contract

The wheel contains only runtime packages, compatibility entrypoints, schemas,
templates, and packaged defaults. It must not contain tests, repository tools,
or documentation trees. The source distribution additionally contains
maintenance tools, documentation, examples, citation metadata, and governance
files.

`tools/check_distribution.py` checks both inventories and installs the wheel
with `--no-deps` into an isolated target before importing the public package,
protocol schemas, configuration resources, and LaTeX templates. This catches
the common failure where imports succeed from the checkout but packaged assets
are missing.

## Versioning and releases

The package version is defined once in `xscientist/_version.py`. Before tagging
a release:

1. Move the relevant `Unreleased` entries into a dated version section.
2. Update `CITATION.cff` to the same version.
3. Run all local quality gates.
4. Tag exactly `v<package-version>`.
5. Push the tag and review the GitHub Actions environment approval.

The release workflow fails if the tag, changelog, and citation version disagree,
or if the `Unreleased` section still contains entries. It builds, checks,
smoke-installs, and hashes the artifacts before PyPI trusted publishing. A
manual workflow dispatch builds artifacts but cannot publish a branch
accidentally.

## Protocol changes

Every enumerated protocol kind must have a loadable JSON Schema. Additive
optional fields remain compatible; required fields or semantic changes require
the protocol-version process documented in `ai_scientist/protocol/SPEC.md`.

Do not write the number of schemas into prose. The inventory changes as
lifecycle and view schemas are added, so engineering checks derive it directly
from the registry and reject stale numeric claims in the main READMEs.

## Pull-request expectations

Every PR should state the problem, risk, tests, artifact/protocol impact, and
rollback path. Changes to claims, evidence, re-execution, storage reachability,
or ContextPack selection must include a compatibility note and focused tests.
Do not lower coverage, validation, safety, or release gates merely to make a PR
green; document and fix the underlying mismatch.
