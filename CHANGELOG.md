# Changelog

All notable changes are recorded here. XScientist follows semantic versioning
for its Python package; the ARA protocol has its own version in
`ai_scientist/protocol/constants.py`.

## [Unreleased]

### Added

- Beginner-oriented `research start`, exploratory `research plan`, and
  bilingual `research guide` commands that create a falsifiable lineage and
  explain the next scientific action.
- A unified, content-addressed Research VCS + ARA evidence DAG with explicit
  trace/replay/verify checks, contested-evidence states, JSON export, an
  accessible offline browser, and a metadata-only HTTP endpoint.
- A versioned `xscientist.research_adapters` entry-point contract with safe
  discovery, explicit doctor/sync operations, atomic filesystem publication,
  and hash-bound platform receipts.
- Schema-bound `research ingest` receipts for MLflow, DVC, notebooks, ELNs,
  instruments, and other tools; imports remain unverified until reviewed.

## [0.1.1] - 2026-08-08

### Added

- Unified `xscientist setup`, `doctor`, and `capability` commands with
  task-aware readiness checks and exact opt-in installation guidance.
- Deterministic `xscientist research decide` policy for checkpoint, fork,
  merge, hold, and release transitions, including stable decision identifiers
  and explicit trace requirements.
- Payload-free long-term research technology trees spanning every local
  research line, with topology, frontier, cycle, and missing-reference views.
- Stable scientific merge-conflict identifiers, actionable resolution
  guidance, and an explicit opposed-evidence preservation path that creates a
  rejected hold gate while retaining both sides.
- Provider-specific (`openai`, `anthropic`, `zhipu`, `openai-compatible`,
  `bedrock`, `vertex`) and capability-specific (`research`, `plot`, `pdf`,
  `pdf-layout`, `ml`, `service`) package extras; the legacy `full` profile is
  unchanged.
- Native `xscientist git doctor/add/commit` commands for backend capability
  probing and familiar Research VCS workflows without raw Git passthrough.
- One-command `hypothesis`, `preregister`, `experiment`, `evidence`, `review`,
  and `claim` workflows for ordinary researchers, including automatic exact
  checkpoints and hard gates for confirmatory experiments and verified claims.
- Native Research VCS objects and public Python APIs for semantic staging,
  research lines, immutable tags, typed diff, provenance blame, safe
  multi-parent merge, and integrity-aware repository verification.
- Evidence-gated lifecycle APIs covering hypotheses, plans, locked
  preregistrations, successful/failed/timed-out attempts, evidence, claims,
  independent review, gate decisions, and manuscripts.
- Versioned self-evolution lines with shadow candidates, sealed evaluation,
  canary-bound promotion, stable-line admission checks, and append-only
  rollback receipts.
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
- Config-aware preflight checks for the selected models, credentials, client
  packages, Docker daemon, and exact isolated-executor image.
- Read-only `xscientist info` installation profiles covering runtime/service
  readiness, missing packages, output resolution, and local login status.
- Secure `xscientist provider` setup and switching for all supported model
  backends, with hidden credential prompts, private workspace env files,
  secret-free metadata, readiness inspection, and OpenAI-compatible endpoints.

### Changed

- First-run workspaces now select only the capabilities required by the chosen
  research task; provider-neutral tasks avoid model clients and ML runtimes.
- Research VCS milestone history now records ideation as well as experiments,
  evidence, review, paper, evolution, merge, and release boundaries.
- Provider readiness now distinguishes credentials from installed client
  packages, prints a path-safe installation command, and discovers workspace
  configuration from nested directories.
- Generated workspaces and executor images install only the selected provider
  plus the requested research capabilities instead of every provider SDK.
- Preflight's common runtime gate is provider-neutral; configured models retain
  exact client and credential checks.
- End-to-end project runs now enable local Research VCS milestone history by
  default; `--research-vcs off` remains the explicit opt-out and all legacy
  `--research-git*` spellings remain accepted.
- Documentation and generated workspaces present Git as a replaceable local
  persistence adapter rather than equating Research VCS with source hosting.
- Project, batch, direct console, daemon, and preflight entrypoints now inherit
  the active workspace provider and default model while preserving explicit
  per-role model overrides.

### Fixed

- Distribution smoke checks now install declared core dependencies, preventing
  developer site packages from hiding failures in clean release environments.
- Unified diagnostics recursively redact credentials, personal identifiers,
  and host-local paths, including text returned by deep runtime preflight.
- New first-run command parsers are constructed lazily so top-level CLI startup
  retains its lightweight import and help path.
- PDF layout extraction is loaded only when used, so the basic PDF fallback
  remains available without the large `pymupdf4llm` dependency stack.
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

[Unreleased]: https://github.com/smileformylove/XScientist/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/smileformylove/XScientist/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/smileformylove/XScientist/releases/tag/v0.1.0
