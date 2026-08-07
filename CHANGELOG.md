# Changelog

All notable changes are recorded here. XScientist follows semantic versioning
for its Python package; the ARA protocol has its own version in
`ai_scientist/protocol/constants.py`.

## [Unreleased]

### Added

- Serverless local research Git repositories with scientific checkpoint
  commits, safe staging policies, local CAS object pointers, offline Git/CAS
  bundles, and commit-scoped reproduction worktrees.
- Optional `--research-git local` integration for project runs, with manual,
  stage, and milestone checkpoint policies and no automatic remote or push.
- Research repository `fsck`, tamper-detecting CAS reproduction, verified
  bundle restore, branch-aware multi-parent checkpoints, repository mutation
  locks, failure-safe checkpoint transactions, environment receipts, strict
  reproduction compatibility checks, and structured scientific diffs.
- Installed-package-first `xscientist init` workspaces with provider/model
  validation, secret-free environment templates, pinned isolated-executor
  Dockerfiles, packaged BFTS profiles, and overwrite protection.

### Fixed

- `xscientist project --help` and `xscientist batch --help` now work from the
  lightweight core install without importing optional scientific dependencies.

## [0.1.0] - 2026-08-06

### Added

- Installable `xscientist` Python package, public CLI, SDK, and optional HTTP
  service.
- Long-running research workflows, structured review/repair artifacts, and ARA
  protocol tooling.
- Cross-version CI coverage for Python 3.10, 3.11, and 3.12.
- Branch-aware whole-repository coverage regression checks.
- Automated engineering, distribution inventory, isolated wheel, and release
  metadata checks.
- GitHub issue/PR templates, dependency update policy, citation metadata, and
  an engineering maintenance guide.

### Changed

- GitHub Actions are pinned to immutable commits.
- Release publishing is restricted to version tags and validates artifacts
  before trusted publishing.
- Dependency minimums and CI compatibility windows are explicit and checked.
- Protocol documentation no longer duplicates a hand-maintained schema count.

[Unreleased]: https://github.com/smileformylove/XScientist/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/smileformylove/XScientist/releases/tag/v0.1.0
