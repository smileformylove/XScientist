# Changelog

All notable changes are recorded here. XScientist follows semantic versioning
for its Python package; the ARA protocol has its own version in
`ai_scientist/protocol/constants.py`.

## [Unreleased]

### Added

- Added a compact `xscientist status` research-output view that pairs the
  latest experiment with its own result contract, surfaces paper/review/repair
  readiness, distinguishes checkpointed, pending, and unbound artifacts, and
  emits workspace-bound next actions that remain safe to copy from any working
  directory.
- Made experiment completion receipts replay-safe and auditable: distinct
  terminal executions retain distinct registry records, confirmatory limits
  are checked before CAS mutation, and result artifacts can be bound directly
  into Research Git alongside typed experiment attempts.
- Extended end-to-end paper provenance so manuscript, PDF, review, critic, and
  repair artifacts survive every publication-gate outcome as workspace-relative
  records and enter the same local content-addressed handoff. Research Git now
  preserves multiple logical paths for identical content and privacy-scans
  eligible text before permanent CAS admission.
- Added an EvoTrainer-inspired offline evolution harness audit and
  `xscientist evolution harness-audit`: bounded content-addressed version
  evidence is diagnosed across score, reward signal, behavior, and version
  integrity; fixed blockers fail closed while failed branches and next-epoch
  challenges are retained. Harness/evaluator changes are incomparable within
  the current epoch, and neither the audit nor its skills can authorize
  automatic production or evaluator mutation. Skills require independent
  historical plus holdout validation in one domain, and two validated domains
  for cross-domain status. This is an independent diagnostic contract, not an
  implementation of EvoTrainer's online RL or automatic training runtime.
- Bound every harness comparison to one epoch and canonical policy hash, retain
  normalized evidence for deterministic replay validation, and keep rounded
  presentation metrics out of threshold decisions.
- Retain every distinct project harness audit in an idempotent, hash-chained
  JSONL history. Same-epoch current updates now require an exact evidence-version
  prefix extension unless an operator explicitly passes `--supersede`.
- Made the structured scientific trajectory an explicit protocol invariant:
  observable hypotheses, decisions, tool executions, attempts, evidence,
  objections, failures, recoveries and gates are typed, content-addressed and
  related so Research VCS operations apply to scientific meaning rather than
  hidden reasoning or opaque chat logs.
- Added `research trajectory`, a bounded payload-free view that validates exact
  checkpoints and projects typed object identities, relations, actors, hashes
  and checkpoint-parent edges with its own canonical projection hash.
- Defined trajectory chronology from the authoritative checkpoint-parent DAG:
  projections are parent-before-child, content-derived sibling order is
  presentation-only, and `created_at` never selects or authorizes scientific
  state. Branch-local `@latest:<kind>` follows first-parent introduction order
  and fails closed when one checkpoint makes the latest kind ambiguous.
- Enforced immutable correction semantics: ordinary checkpoints cannot modify
  or delete existing Research Objects; typed reverts bind the exact target
  commit/checkpoint, verify an exact inverse material diff, and append rollback
  edges without rewriting history. Historical blame is explicitly scoped to
  the selected ref/commit's reachable history.
- Documented that `--research-vcs off` and non-strict adapter execution are
  legacy exploration/recovery modes without publication, `trace`, `replay`, or
  `verify` standing until migrated into a strict hash-valid closure.
- Added publication-grade trajectory closure: every confirmatory/reproduction
  registry row must bind bijectively to one typed attempt and its hash-valid
  origin checkpoint through `research trajectory-bind`; unsuccessful attempts
  remain immutable and require a typed `research attempt-disposition` decision
  before the verification report can pass. Trajectory hashes, frozen/lineage
  heads and object/checkpoint sets are included in external authority receipts.
- Closed unsuccessful-attempt authority gaps: failed confirmatory and
  reproduction rows now participate in completeness and in every producer set;
  trajectory gates require both structural validity and publication readiness,
  while audit-only deviation/exclusion dispositions never clear a blocker.
  `terminal_negative` no longer accepts a preservation boolean: it requires the
  exact `scientific_negative_result` class, a bounded host-rehashed repository
  artifact matching both registry and attempt hashes, and a metric-bearing
  evidence assessment derived from that attempt. Those identities are covered
  by the final externally signed trajectory receipt.
- Bound experiment object actors to payload producers and canonicalized
  role-prefixed principals for all local provenance-disjointness checks. Generic
  in-workspace `research review` is now explicitly advisory-only: a requested
  pass produces a hold gate and cannot mint a verified claim without the
  separate externally trusted authority flow.
- Added an explicit adoption boundary for version-control decisions. Read-only
  transition previews remain available, while adopted decisions can be stored
  as deterministic gate objects linked to the exact, bounded evidence-context
  snapshot visible at decision time; policy reasons are retained without
  persisting hidden chain-of-thought.
- Added `xscientist research confirm`, which validates the empirical snapshot
  and draft preregistration, locks the exploration-to-confirmation transition,
  and emits a hash-bound multi-task confirmatory queue. Post-freeze attempts
  must retain the frozen research/code/protocol/data identities, while
  ablation and robustness records must disclose the exact configuration
  transformation they performed.
- Bound the complete empirical data contract and every primary or reproduction
  producer, including unsuccessful attempts, into verification reports and external authority
  receipts. Canonical principal matching now prevents role-prefix aliases from
  impersonating an independent verifier, and trust roots inside the research
  workspace are rejected.
- Tightened numerical evidence admission: point estimates and effect sizes no
  longer masquerade as uncertainty; confirmatory results need a valid interval,
  p-value, standard error, dispersion statistic or repeated measurements.
- Added a fail-closed NeurIPS/ICML publication contract: typed and comparable
  primary/ablation/robustness evidence, a host-attested Research VCS freeze,
  content-addressed non-synthetic empirical data, pinned official venue/year
  templates with source-byte receipts, and zero tolerated submission blockers.
- Added external verifier-authority receipts and
  `research verifier-authority prepare|finalize|verify`. The top-venue gate
  accepts only an Ed25519 signer whose trusted principal matches the report
  verifier and is disjoint from research producers; report, registry,
  manuscript, evidence-snapshot, preregistration and record hashes are bound.
- Added secret-free provider configuration explanations, stricter custom-model
  identity/tool probes, transaction-safe environment handling, and clearer
  GLM-5.3 executor-only guidance without granting the model planning or audit
  authority.
- Replaced redistributed paper/template examples with project-authored
  synthetic fixtures, pinned runtime downloads for official templates, and
  explicit derivative-code, MIT-lineage and acquisition-provenance notices.
- Clarified the Recuris comparison: its structured step tuples localize
  execution failures and support memory evolution, whereas XScientist versions
  hypotheses, decisions, model/tool attempts, evidence, negative results,
  reviews and gates across branches as a scientific VCS trajectory.

- Added a bounded GLM-5.3 research-policy benchmark with a fixed Simpson's
  paradox task, four deterministic statistical tools, strict JSON actions,
  per-turn model identity checks, redacted traces, an offline verifier, and
  non-compensable gates for evidence grounding, causal restraint, and negative
  result preservation. Passing the synthetic contract never enables quality,
  generalization, real-world truth, or cross-model comparison claims.
- Upgraded scientific closure audits to `research-closure.v2`: every claim now
  discloses its direct and same-study attempts, evidence, states, plans,
  preregistrations, and selected experiment priorities. Failed or unfinished
  attempts, inactive evidence, unbound negative results, invalid plan/priority
  state, cross-attempt provenance borrowing, and non-protocol hash anchors fail
  closed instead of permitting claim verification.
- Added a Faraday-inspired, metadata-only rollout policy projection with
  contiguous budget accounting, decision ownership, failure/recovery signals,
  and deterministic strategy hashes. Completed rollouts now require complete,
  in-boundary budget continuity and successful required recovery (or an
  explicit terminal stop) before verification eligibility.
- Added a fail-closed `research-rollout-audit` contract and
  `xscientist research rollout-audit`, binding successful executor artifacts to
  resolved evaluator evidence and actor-disjoint provenance receipts. The
  audit never enables quality or causal claims.
- Added an optional hash-bound harness/resource/evaluator comparison boundary
  so tool-swap eligibility cannot silently ignore network, seed, starting
  artifact, or evaluator-protocol differences.
- Added bilingual rollout documentation and regression coverage for strategy
  tampering, artifact binding, evidence resolution, and the audit CLI. The
  `research rollout --json` wrapper can be passed directly to
  `research rollout-audit`.
- Added a BCG-inspired, read-only belief-context projection over immutable
  Research VCS closures, with ordinal evidence states, source-family
  deduplication, temporal invalidation, explicit conflicts, hard graph limits,
  payload-free audit output, and `research belief` / `belief-audit` commands.
  Historical `as_of` projections exclude future evidence and future
  invalidation events, while claim-to-evidence `depends_on` edges receive a
  type-constrained support binding without promoting arbitrary lineage.
- Added versioned belief-context and audit schemas, a v4 context-retrieval
  receipt that hash-binds the projection, bilingual boundary documentation,
  and compatibility validation for historical v3 context snapshots.
- Hardened completed rollout verification: a caller-supplied
  `identity_verified` boolean is observational only, while independent
  evaluation now requires a signed attestation verified against a local trust
  store and bound to all observed producer identities and executor artifacts.
- Added fail-closed strict tool-swap eligibility, requiring a union evidence
  hash resolver and local trust store to audit both rollouts before comparing
  distinct tool signatures.
- Made guided next actions rival-first: before selecting a study mode, the
  guide asks for a falsifiable competitor and then a locked hypothesis
  portfolio. Portable JSON actions carry exact object IDs, an explicit
  `{workspace}` binding, and `input_binding`; unresolved human placeholders
  remain non-executable.
- Hardened checkpoint identity and direct saves. Checkpoint-sensitive reads,
  reproduction, tags, bundles, exports, and semantic merges require the exact
  selected commit to carry a valid checkpoint binding rather than inheriting
  one from an ancestor. Commit-enabled one-command recorders require a clean
  research worktree and checkpoint only their newly created paths.
- Made initialization additive in existing projects: preserve and append to
  `.gitignore` instead of overwriting it, reject conflicting managed files,
  pre-existing staged work, or tracked managed-file dirt, and restrict an
  existing repository's first checkpoint to XScientist-managed paths.
- Made `init`/`setup`/`start` failure handling transaction-safe and
  concurrency-preserving: privacy, provider, and diagnostic gates run before
  publishing checkpoints; rollback removes only unchanged invocation-owned
  outputs and preserves concurrent files, Git configuration, refs, index
  intent, and history. Unsafe Git control paths, special managed files, and
  credential-shaped model metadata fail before persistence, while JSON errors
  stay redacted.
- Hardened semantic merge against base-versus-branch support/refutation and
  metric-definition conflicts, exact source/target tips, staged/worktree drift
  around privacy scanning, and privacy findings. Preserved opposing evidence
  still produces a hold; merge success does not imply scientific agreement.
- Reproduction now replaces inherited environment variables, bounds retained
  output, and applies a timeout without claiming an OS sandbox. Receipts state
  `isolated=false`, `security_boundary=false`, `environment_scope=variables_only`,
  `filesystem=host_visible`, and `network=host_unrestricted`; POSIX process-group
  cleanup is best-effort, while Windows terminates only the parent process. The
  boundary is persisted and hash-bound in v2 receipts; upgraded v1 receipts use
  explicit `legacy_unknown` fields instead of inferred isolation claims.
- Reproduction receipts now distinguish full retained tails from truncated
  output by binding the capture mode, character limit, and stdout/stderr
  truncation flags; legacy v1 upgrades keep output scope explicitly unknown.
- Added reproduction receipt v2, binding the resolved source checkpoint, exact
  reproduced objects, active claim closure at that same audit checkpoint, and
  execution result. When a verified record is requested, the lifecycle
  atomically upgrades a valid generated v1 receipt; historical v1 objects
  remain readable but cannot satisfy `verify`.
- Expanded research bundles to an `all-advertised-refs` closure, including
  pointers and historical CAS objects reachable only from non-current branches
  or tags. Verification imports the embedded Git bundle and locally recomputes
  that closure before checking pointer and CAS hashes/sizes.
- Strengthened verified-claim closure so one independent verified review must
  itself cover all active evidence, reasoning, challenges, and recorded
  resolutions; active refutation/challenge blocks verification, and split
  partial reviews cannot be combined to bypass the rule.
- Bound literature evidence as locked plan → retrieval receipt → uniquely
  selected source. Retractions remain active until a later, same-provider,
  notice-bearing reinstatement explicitly supersedes the active retraction;
  historical `as_of` projections exclude future signals, lineage, and source
  status events.

### Fixed

- Confined model- or caller-supplied project, idea, batch, experiment, and
  paper labels to their artifact roots, with bounded slugs and content-derived
  identities preventing traversal and duplicate-name collisions.
- Prevented installed/PyPI executor builds from sending the research workspace
  to the Docker daemon; only a temporary Dockerfile-only context is used, while
  source-checkout builds retain their explicit source context.
- Added a process-local workspace reservation so duplicate HTTP jobs cannot
  concurrently mutate one project, with a stable HTTP 409 response and release
  on completion, failure, or cancellation.
- Required release tags to be reachable from `origin/main` and to pass the full
  test suite before distribution build; Ubuntu CI now covers declared Python
  3.13 support.
- Restored Windows initialization, onboarding, and provider transactions by
  writing canonical encoded bytes without CRLF rewriting. Compatibility smoke
  commands now fail independently and exercise the cross-platform byte
  contract instead of hiding earlier native-command failures.
- Replaced selector-based bounded Git capture with a cross-platform dual-pipe
  reader that preserves the combined byte ceiling, wall-clock deadline, and
  fail-closed audit semantics on Windows.
- Resized Full Smoke for the expanded integration suite, installed its declared
  pytest dependency, and made pytest execute every unittest-style and
  pytest-native case with per-test progress and slow-test diagnostics instead
  of silently omitting pytest functions or timing out behind buffered dots.
- Kept the isolated wheel journey bounded while giving cold Windows runners a
  120-second budget; observed successful and timed-out runs straddled the old
  60-second ceiling even though the same offline journey was making progress.
- Aligned the full repository validator with fail-closed evidence gates: prose
  terminology cannot stand in for an experiment registry, and manuscript
  numbers cannot become key results without bound experimental evidence.

## [0.1.4] - 2026-08-23

### Changed

- Benchmark pilots now emit a fixed-vocabulary P0/P1/P2 diagnostics backlog,
  label stage coverage as structural-only, and expose an atomic `--output`
  path for retaining the redacted report without copying raw ARA/CAS payloads.
- Added a bilingual benchmark optimization status report with capability-gap
  mapping and explicit blockers (no dated delivery plan); public SDK exports
  now include the benchmark helpers, report persistence, and offline report
  verification function.
- Benchmark workspace audits now include a bounded, read-only SHA-256 evidence
  index for Research VCS, ARA/CAS, and generated views, plus a redacted
  exploration-graph summary and deterministic input fingerprint.
- Evidence/process scanners now cap directory-entry traversal before parsing,
  reject symlink escapes, expose prefix-versus-complete scopes, and validate
  exploration/process/evidence subcontracts during offline report verification.
- Shell completion now includes `benchmark verify` and its `--report`/`--output`
  options; malformed exploration nodes and inconsistent repeated manifest
  digests fail closed instead of looking like complete evidence.
- Feedback reports now label evaluator links as independence-unverified and
  mark health scores as observational heuristics; process fairness reports
  expose fixed unverified-reason codes.
- Process fairness metadata now fails closed for invalid task counts (including
  non-integral or non-finite values) instead of producing schema-invalid audit
  reports.
- The first-run benchmark now avoids duplicate DAG/guide rendering and has a
  versioned `first_run_benchmark` schema plus atomic-output regression coverage.

- Provider diagnostics now track workspace-owned environment values, expose
  secret-free resolved model/endpoint fingerprints, distinguish configuration
  checks from live verification, and return structured unsupported live-probe
  results for non-OpenAI-compatible clients.
- `provider check` keeps its zero-request default and now supports an explicit
  `--live` minimal probe with model-identity scope and no response-content
  recording.
- Closure audits and workspace status expose one shared trace/replay/verify
  summary with per-level blocker and warning counts; durable feedback can link
  interventions to outcomes and accountable evaluators without treating the
  link as automatic causal proof.
- LLM call provenance records the resolved provider contract without API keys;
  historical `research blame` selectors resolve against the requested commit.
- Added an offline `benchmark autoresearch` pilot that checks local task
  manifests, six-stage typed-artifact coverage, closure levels, and contained
  versus shipped review debt. It never calls a provider and is explicitly not
  comparable to the official AutoResearchEval rollout score.
- Added a read-only ARFT observability contract (`arft_coverage.json`) with all
  45 pattern IDs, A–F/X stages, root-cause rollups, and an explicit
  `covered/partial/unassessed` boundary.
- The process pilot now distinguishes open review debt from a typed hold/reject
  gate and reports unreadable contract inputs without exposing their contents.
- AutoResearch benchmark reports now expose a bounded Git-like checkpoint and
  branch topology, typed decision/failure signals, and explicit fairness
  checks while redacting free-form branch, commit, payload, and gold text.
- Added a source-audited bilingual human-baseline inventory. It keeps direct
  participant runs (including versioned BAISBench, Mind2Web 2, and WebArena)
  separate from leaderboard references, expert validation, and judge
  calibration; missing or non-tabulated scores remain `not_reported`.
- Versioned `process_audit` schema coverage now keeps available and unavailable
  process reports machine-valid and distinguishes bounded samples from source
  totals.
- Process timelines now retain Git short hashes and use bounded parent-first
  ordering when checkpoint timestamps collide, making branch reasoning easier
  to review without exposing commit subjects.
- Benchmark documentation now states the read-only evidence/ARA retention
  boundary and gives explicit fsck, ARA-audit-bundle, and payload-export
  commands; it also documents the controls required for a human baseline.
- Added a source-audited external human-baseline inventory that separates
  measured participant runs from SOTA/leaderboard proxies, expert validation,
  judge calibration, ground truth, and explicit `not_reported` cases; no
  external number is treated as an XScientist score.
- Autoresearch pilot JSON now carries an explicit `human_baseline` record with
  `status: "not_reported"`, no matched arm, and a null score until a registered
  local human run exists.
- Added a bilingual, source-audited comparison matrix and
  `xscientist benchmark systems`. It separates end-to-end agents from MLE,
  review, writing, and figure components; includes primary-source links and a
  redacted Git-like process view; and hard-codes that no cross-system score or
  quality claim is made. Audit-only rollout/cost scope is explicit, and the
  human inventory now includes separate ideation and web-process references.
- Added a FAR-inspired, source-audited opportunity funnel covering research
  directions, complete candidate pools, attempts, independent judgments,
  grades, deterministic allocation, and lineage/coverage summaries. Stage gates
  and fail-closed allocation prevent incomplete or non-open pools from being
  presented as scientific results; the protocol does not claim to reproduce
  FAR's corpus-wide pilot or human performance.
- Added the `literature_opportunity` schema, Research VCS validation, CLI
  commands under `xscientist research opportunity`, and bilingual usage docs;
  probability semantics, evaluator independence, evidence references, and
  override reasons are explicit and hash-bound.
- Source-checkout test discovery is deterministic through the configured
  project path and local `tests` package.

### Documentation

- Added a current Chinese project audit covering usability, auditability,
  scientific quality, exploration, clue transparency, feedback, and
  self-evolution limits.
- Added bilingual benchmark documentation with a measured zero-cost demo
  baseline and links to the official AutoResearchEval task release.

## [0.1.3] - 2026-08-19

### Added

- A provider-free `xscientist explore` journey that records a user's own idea,
  asks plain-language falsifiability questions, permits honest incomplete
  states, and creates no invented evidence or conclusions.
- Compact `status`, `audit`, and `history` review surfaces with hash-checked
  checkpoint listing, show, semantic diff, policy-safe save, payload-free
  rollback preview, and append-only reversal commands plus matching public SDKs.
- Persistent local detached-run controls and HTTP job log/cancel/resume
  endpoints, including bounded live output, graceful process-group shutdown,
  exact private resume commands, and PID-reuse protection.
- Interactive first-run provider/model/evidence selection, progressive root
  help, shell completion, version-matched executor management, and a
  deterministic zero-cost Autopilot fixture with a public first-run benchmark.
- Read-only offline upgrade/schema compatibility checks, an offline protocol
  producer conformance kit, and disabled-by-default fixed-shape local usage
  counters that never transmit research data.
- A MkDocs documentation site configuration, reproducible example gallery,
  MLflow/DVC/notebook adapter cookbook, protocol conformance guide, and strict
  documentation build workflow.
- Provider discovery before workspace creation, including safe local Ollama
  detection and copyable setup commands without reading unrelated credentials.
- Explicit workspace operational states, durable cross-process feedback, and
  bounded immutable scientific-strategy follow-ups with stop conditions.
- Machine-readable, import-free discovery for the versioned
  `xscientist.research_adapters` extension point.

### Changed

- Default `status` is now the single research review surface for checkpoint and
  worktree state, `trace → replay → verify` gates, generated-DAG freshness,
  bounded next actions, and optional agent-evolution receipt counts.
- Worktree safety and closure classification are shared by status, branch
  transitions, rollback, bundling, and audit so those surfaces cannot silently
  disagree.
- English and Chinese entry documentation now uses one short task-first model,
  explains Git-like evidence history without treating Git as scientific
  authority, and keeps advanced protocol controls under `research`.
- Model-backed `start` reuses an existing `explore` question and adds missing
  runtime files without replacing the workspace's research history.
- Interactive setup uses one readiness-led provider picker; non-interactive
  starts require an explicit provider instead of claiming an implicit route.
- Workspace status uses a stable path-free repository identity, detects
  malformed runtime state instead of treating it as absent, and directs
  contested claims to a boundary-resolving next experiment.
- The provider-free demo now records a bounded inference and can populate the
  same progress, budget, insight, negative-result, and resumability surfaces
  used by Autopilot without executing generated code.
- Doctor human output renders every capability row consistently and the
  installed-wheel smoke journey now covers human status/doctor, completion,
  upgrade compatibility, and protocol conformance.
- The English README now leads with zero-cost proof, local Ollama and hosted
  provider routes, Docker expectations, paid-run checks, background control,
  and recovery instead of requiring protocol knowledge before first use.
- Detached run views now show profile, provider/model, duration, exit state,
  and bounded failure context; resume rechecks unresolved prerequisites unless
  the operator explicitly uses `--force`.
- Shell completion covers practical subcommands and options, and human status,
  demo, and authentication output now honor a consistent English/Chinese path.
- Research strategy checkpoints commit only the objects they create instead of
  sweeping unrelated eligible changes into the same scientific decision.
- Ollama readiness verifies both service and model, supports numbered choices,
  and records local inference as zero API cost while preserving token budgets.
- Autopilot profiles now represent distinct discovery and publication
  structures, including competitive predictions, information-value ranking,
  independent review boards, and hold gates.

### Fixed

- Append-only rollback now reverses checkpoints that introduced files,
  restores tracked state atomically on failure, preserves policy-excluded
  generated views, and marks stale DAG output with an exact refresh command.
- Demo checkpoints include code, environment, dependency, data, seed, and
  measurement provenance; generated views no longer block reproduction bundles.
- Structured start and research-strategy commands keep human launch failures
  out of JSON output.
- Provider and Doctor checks distinguish unreachable Ollama, missing Docker,
  stopped Docker, unconfigured clients, and unknown model pricing correctly.
- Cancellation preserves bounded-output truncation metadata and refuses to
  signal a process whose persisted identity or process group no longer matches.
- Detached `--detach` parsing is attached to `xscientist start` rather than the
  workspace setup command.
- Explicit missing workspace paths now fail in status, run, executor, and
  upgrade journeys instead of being reported as empty compatible workspaces.
- Interactive Ollama setup discovers locally installed models and normalizes
  common bare model names to unambiguous provider-prefixed IDs.
- Contested-claim guidance advances through plan, experiment, evidence,
  inference, review, and replacement instead of repeating the same plan.
- Copyable status, Doctor, executor, conformance, and run commands preserve the
  workspace selected by the caller.
- The HTTP service validates its work directory at startup and exposes a useful
  root discovery response instead of failing the first submitted job.
- Empty or corrupted feedback no longer appears healthy; traceability,
  replayability, and independent verification are reported separately.
- Generated workspace refreshes preserve canonical formatting so provider and
  model updates do not create formatting-only Research VCS changes.

## [0.1.2] - 2026-08-17

### Added

- A one-command, provider-free `xscientist demo` that creates a deterministic
  Research VCS history with a failed attempt, supporting and refuting evidence,
  an independent rejection, a contested claim, and an offline DAG for `$0`.
- A compact, read-only `xscientist status` view for scientific progress,
  automated run state, budget use, results, the evidence DAG, and the next
  valid action.
- `xscientist provider check` now reports credential presence, client
  availability, live-verification scope, and optional cost-enforcement
  readiness without making a paid API request.
- Stable doctor error codes and structured copyable remediation records while
  preserving the existing `next_actions` compatibility surface.

- An additive deep-research strategy profile with competitive hypothesis
  portfolios, discriminating predictions, deterministic expected-information-
  value experiment ranking, anomaly scans, periodic program reviews, causal
  mechanism models, structured evidence-quality audits, and transfer matrices.
- Fail-closed `causal` and `transferable` claim depths, plus a six-layer
  scientific DAG, content-hashed theory frontier, claim reasoning drill-down,
  and `research program` CLI/Python surfaces.
- Frontier-aware semantic working memory shared by Research VCS, ARA, and the
  cross-project learning layer. Context receipt v3 separates complete audit
  closure from bounded prompt input, preserves current evidence and failures,
  archives superseded history, seals token/usability verdicts, and prevents
  recursive context-snapshot growth. `research context --prompt` exposes the
  exact agent-facing view.
- A locked method-discovery protocol that isolates the target edit surface,
  fixes resources and non-target variables, binds strong baselines and sealed
  cross-condition evaluation, checks proxy-to-target ranking fidelity, and
  blocks `method_discovery` claims without a passing generalization synthesis.

- A guarded `xscientist start` journey that creates or reuses one workspace,
  validates credentials/login/isolation/paper tooling, and runs Autopilot in
  the same local Research VCS from a plain-language question.
- Project-wide concurrent LLM budget accounting plus a fail-closed data gate
  with content-hashed read-only empirical inputs or an explicit synthetic-data
  scientific boundary.
- First-class question/Autopilot/resume/data/budget controls in the Python SDK
  and HTTP project contract.
- Source-checkout executor builds install the exact local source revision;
  PEP 610 VCS installs pin the isolated executor to the same safe HTTPS commit;
  installed releases remain pinned to their matching PyPI version. Image
  labels expose which source mode was actually selected.
- Shared project budget ledgers now use native advisory locks on both POSIX and
  Windows, preserving cross-process token/cost caps for parallel workers.
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

### Changed

- Guarded starts default to the smaller `research` capability profile and only
  install ML or PDF-layout tooling when `--task ml-study`, `paper`, or
  `pdf-review` requires it.
- Cost-limited starts resolve model pricing before provider use, fail closed
  with `unknown_model_price`, and accept explicit input/output/cached-input
  prices for unbundled or local models.
- Generated executor Dockerfiles now default to the selected research/provider
  capabilities instead of silently pulling the ML and PDF-layout stacks.

### Fixed

- Missing `pdflatex` is a warning for general research and PDF review, while
  remaining a blocker for tasks that actually compile a paper.
- Distribution checks exercise the exact provider-free `demo → status` path
  from an isolated built wheel, and the compatibility CI matrix runs that
  journey on Linux, macOS, and Windows.
- Provider and start cost checks no longer allow an unknown model to become an
  implicit zero-cost estimate.

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

[Unreleased]: https://github.com/smileformylove/XScientist/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/smileformylove/XScientist/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/smileformylove/XScientist/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/smileformylove/XScientist/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/smileformylove/XScientist/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/smileformylove/XScientist/releases/tag/v0.1.0
