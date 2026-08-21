# XScientist onboarding and adoption audit

Last updated: 2026-08-19

This document measures how difficult XScientist is to adopt, separates
intentional scientific rigor from accidental product friction, and defines the
work needed to make the repository genuinely approachable.

## Executive assessment

XScientist has two very different first-run experiences:

1. **Local Research Git** is lightweight, offline, and now approachable for a
   curious user. A question, falsifiable hypothesis, and offline DAG can be
   created in a few commands without an API key.
2. **Autonomous research** remains a power-user workflow. It requires choosing
   a provider and model, configuring credentials, building or validating an
   isolated executor, understanding the empirical/synthetic data boundary,
   and accepting non-trivial runtime and cost.

The product is therefore easy to *inspect* but still moderately difficult to
*operate end to end*. This distinction should remain explicit in all public
communication.

## Current `main` usability implementation update

The P0–P2 product work identified below is now implemented on `main`:

- deterministic offline Autopilot fixtures and a public first-run benchmark;
- corruption-aware status, stable workspace identity, and executable doctor
  results in both human and JSON output;
- detached local runs plus HTTP job logs, graceful cancel, and linked resume;
- interactive provider/model/evidence selection with a smaller progressive
  root help surface;
- version-matched executor check/build/prepare/update commands;
- read-only upgrade/schema compatibility checks and shell completion;
- a standalone offline protocol conformance kit;
- disabled-by-default, fixed-shape, local-only usage counters;
- a MkDocs site, reproducible example gallery, and MLflow/DVC/notebook adapter
  cookbook.
- behaviorally distinct `balanced`, `discovery`, and `publication` fixtures;
- durable cross-process feedback with honest unknown/corruption states;
- separate traceability, replayability, and verification closure reporting;
- persistent readiness blockers and bounded scientific-strategy follow-ups.
- workspace-scoped provider environments, secret-free model/endpoint
  provenance, and structured live-probe capability states.
- a GitHub-like default review surface showing clean/pending history,
  trace/replay/verify checks, compact checkpoint diffs, and stale DAG views;
- rollback that preserves policy-excluded generated views while atomically
  reversing checkpoints that added, changed, or removed tracked research files;
- machine-readable adapter API discovery through `xscientist info --json`.

The remaining adoption work is empirical: collect opt-in aggregate funnel
counts from willing users, publish multiple real domain studies with their cost
and negative-result receipts, and validate median setup/runtime across clean
supported hosts. Those outcomes cannot be inferred from unit tests alone.

Difficulty scale: 1 is trivial; 5 requires expert knowledge or substantial
setup.

| Journey | Current difficulty | Why |
| --- | ---: | --- |
| Create and view a local scientific DAG | 1.5/5 | Python + Git, no provider, guided next step, no manual IDs |
| Review an existing Research Git repository | 2/5 | Familiar log/diff/branch verbs, but scientific object states still require learning |
| Inspect or continue from an existing ARA | 2.5/5 | Offline tools are strong; users still need to locate the correct ARA/node |
| Run an exploratory autonomous study | 4/5 | Provider, model, credentials, Docker, data choice, budget, and optional paper tools |
| Integrate another platform or agent | 3.5/5 | Versioned adapter contract exists, but examples and third-party fixtures are limited |
| Promote a claim to independently verified | 4.5/5 | Intentionally rigorous: evidence closure, provenance, independent authority, and reproduction are required |

The last row should not be made “easy” by weakening gates. Product work should
make the requirements understandable and help users satisfy them.

## Baseline problems found

Before the README redesign, the English README was 1,221 lines and roughly
6,700 words. The first Quick Start appeared after vision, overview, features,
interfaces, and repository layout. A first-time visitor had to solve several
problems before seeing value:

- infer whether XScientist was a paper generator, daemon, protocol, Research
  Git implementation, or all of them;
- distinguish free offline protocol operations from paid model-backed work;
- distinguish the stable PyPI release from unreleased `main` capabilities;
- choose among several overlapping setup, init, project, and research paths;
- read provider, login, preflight, Docker, output, and quality-gate details
  before seeing a successful artifact;
- manually carry immutable object IDs through a scientific lifecycle;
- understand why “full automation” still cannot confer independent scientific
  verification on its own output.

The problem was primarily information architecture, not a lack of capability.
The system already had strong diagnostics, safe local history, resumable
execution, semantic merge, structured artifacts, and honest scientific gates.

## README changes now implemented

The root README now follows a conversion path based on the questions a visitor
actually asks:

1. **What outcome do I get?** A short claim: autonomous research that is
   auditable, forkable, and reproducible.
2. **Can I try it safely?** A provider-free local demo appears before the full
   autonomous setup.
3. **Which path is mine?** A four-row chooser separates protocol exploration,
   full automation, review/reproduction, and integration.
4. **What is different?** The evidence DAG, exact context receipts, Research
   Git history, and scientific verification levels are shown before deep
   configuration.
5. **What can go wrong?** Alpha status, API cost, generated-code isolation,
   unverified machine claims, and no-auto-push defaults are visible early.
6. **Where is the detail?** Operational reference material is linked from a
   task-oriented documentation table instead of duplicated in the README.

This structure follows GitHub's guidance that a repository README should say
what the project does, why it is useful, how to start, where to get help, and
how to contribute, while moving longer documentation elsewhere:
[About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes).

The comparison set also suggests that scientific repositories convert better
when they show a short runnable path and a concrete output near the top. See
[PaperQA](https://github.com/Future-House/paper-qa),
[The AI Scientist](https://github.com/SakanaAI/AI-Scientist), and
[OpenHands](https://github.com/OpenHands/OpenHands). XScientist should copy the
clarity pattern, not their product positioning.

## First-run funnel

The desired funnel is:

```text
repository visit
    ↓  one sentence explains the outcome
choose: offline demo | autonomous study | inspect existing work | integrate
    ↓
one install command
    ↓
doctor returns either ready or one actionable remediation
    ↓
first visible artifact: offline DAG browser
    ↓
guided next scientific step
    ↓
optional provider, isolation, data, and budget setup
    ↓
resumable autonomous run with transparent progress
```

Every extra choice before the first visible DAG is a regression unless it
protects data, credentials, cost, or scientific validity.

## Remaining friction by persona

### Curious developer or student

What works:

- can install the core package and create a real repository without a model;
- sees a DAG immediately;
- receives an exploratory or confirmatory next step;
- can use `@latest:<kind>` selectors instead of copying IDs.

Remaining friction:

- installing `main` from Git is longer and less trustworthy than PyPI;
- the bundled fixtures are intentionally synthetic and do not replace a real
  domain study with authentic data, cost, and failure receipts;
- the README still lacks a short recorded product demonstration.

### Domain researcher without agent-infrastructure experience

What works:

- commands use hypotheses, plans, evidence, review, claims, and estimands;
- scientific gates explain why a claim is held or contested;
- local history does not require a GitHub account.

Remaining friction:

- Docker, model IDs, environment variables, and package extras are unfamiliar;
- “provider”, “ARA”, “CAS”, “context receipt”, and “closure” need progressive
  explanation in the CLI;
- a real study still requires the user to design data and validation that are
  scientifically appropriate for the domain.

### ML engineer running Autopilot

What works:

- one guarded `start` command;
- explicit data/synthetic boundary and project-wide budgets;
- resumable experiment search and structured output;
- fail-closed isolation and exact source revision in the executor.

Remaining friction:

- Docker image build and optional LaTeX/PDF dependencies dominate setup time;
- provider/model compatibility errors can surface before a user understands
  the model naming convention;
- the deterministic fixture validates product surfaces, not a real provider,
  Docker engine, or domain workload on every clean host.

### Agent or platform author

What works:

- JSON output, semantic selectors, typed research objects, versioned profiles,
  ARA, context packs, and adapter entry points;
- offline bundles and standard-oriented exports;
- a downstream agent can inspect and continue from exact history.

Remaining friction:

- the conformance kit and built-in adapter examples still need adoption by
  independent producers;
- third-party adapter examples are less prominent than the built-in implementation;
- third-party consumers still need to adopt the advertised adapter entry point
  and conformance fixtures before ecosystem compatibility is proven in use.

### Small CLI consistency gaps found during this audit

- lifecycle commands and `research blame` now accept selectors such as
  `@latest:hypothesis`; historical blame resolves the selector against the
  requested commit rather than the mutable working tree;
- manual DAG output may be placed inside a repository and is safely excluded
  from scientific staging, while Autopilot writes its view outside the working
  tree; the distinction should be surfaced in CLI output;
- the installed-package journey is reliable, but `python -m xscientist` from a
  source checkout stops resolving after a user changes into another workspace
  unless the package was installed first.

These are product consistency issues, not reasons to weaken immutable identity
or tracking policy.

## Optimization plan

### P0 — convert current capability into an honest first success

Delivery status (2026-08-19): items 1–5 are implemented on `main`. The
provider-free contested-evidence demo and deterministic Autopilot fixtures run
the exact `demo → status` journey without cost or network; the three profiles
now produce different scientific structures rather than different labels over
one DAG. Release publishing remains a release-operation concern, so the README
continues to distinguish the stable package from source installation.

1. **Publish the next release.** Put guided start, the unified DAG, context
   receipts, branch maintenance, and protocol v2 on PyPI so the first command
   is `pip install xscientist`, not a VCS install.
2. **Add `xscientist demo`.** Install or generate a tiny provider-free sample
   containing support, refutation, a failed attempt, review, a context receipt,
   and a contested claim; open the DAG with one command.
3. **Test README commands in CI.** Run the exact offline quick start from a
   built wheel on Linux, macOS, and Windows, and validate every local link.
4. **Make doctor remediation executable.** Each failed check should return one
   copyable command and a stable JSON error category.
5. **Ship one deterministic autonomous fixture.** A stub provider + tiny
   executor should exercise the complete lifecycle without cost or network.

Exit criteria:

- median time to first offline DAG below 5 minutes on a clean supported host;
- no manual object-ID copying before the first experiment plan;
- 100% pass rate for the README offline journey in release CI;
- all setup failures expose one remediation and preserve resumability.

### P1 — reduce autonomous setup and make value visible

1. **Provider wizard.** Present installed/ready providers and validate a chosen
   model before creating the workspace.
2. **Executor bootstrap.** Cache a signed version-matched image or offer a
   documented remote/container runtime adapter while preserving isolation.
3. **Demo artifact gallery.** Publish small ARAs with browser screenshots,
   research questions, cost/runtime receipts, negative results, and exact
   reproduction commands.
4. **Hosted docs.** Add search, a CLI reference generated from parsers, protocol
   schema pages, and version switching.
5. **Progressive terminology.** Human output should lead with plain language;
   stable protocol terms and object IDs should remain visible in a secondary
   detail layer and JSON.

Exit criteria:

- setup-to-ready under 15 minutes excluding downloads and model execution;
- a new user can explain the difference between traceable, replayable, and
  verified after the first DAG view;
- at least one reproducible example for every supported Autopilot profile;
- documented Windows, macOS, and Linux paths for the full executor.

### P2 — grow an ecosystem, not only a repository

1. **Protocol conformance kit.** Publish fixtures, a validator badge, and a
   compatibility matrix for third-party ARA/Research VCS producers.
2. **Adapter cookbook.** Provide complete MLflow, DVC, notebook, ELN, and
   instrument examples with provenance and context receipts.
3. **Public benchmark.** Measure task completion, evidence closure,
   reproducibility, cost, and failure preservation instead of only manuscript
   scores.
4. **Community examples.** Label `good first issue`, publish an adapter request
   template, and feature independently reproduced studies.
5. **Optional privacy-preserving funnel metrics.** If telemetry is ever added,
   it must be opt-in, aggregate, documented, and contain no research payload,
   credentials, paths, or object summaries.

Exit criteria:

- at least two independent protocol consumers or adapters;
- external reproduction of a published sample ARA;
- release notes include migration, evidence semantics, and rollback guidance;
- community contributions can be validated without running a paid model.

## Repository popularity strategy

Popularity should be treated as a consequence of demonstrated usefulness and
trust, not badge accumulation.

### Discovery

- keep the title and description outcome-oriented and searchable;
- use focused GitHub topics such as `autonomous-research`, `research-agents`,
  `scientific-reproducibility`, `provenance`, and `research-software`;
- publish release notes and short technical demonstrations that link to an
  exact reproducible artifact;
- keep the system paper, docs, and repository cross-linked.

### Conversion

- show a 30–60 second recording of question → branch → contested evidence →
  DAG before adding more prose;
- make the provider-free demo the default call to action;
- show expected output after every quick-start block;
- keep stable and development installation commands visibly separate;
- maintain a task-oriented documentation index and searchable troubleshooting.

### Retention

- publish a small example study on a predictable cadence;
- preserve negative results and failed runs so users see realistic operation;
- maintain a public roadmap tied to issues and release milestones;
- respond to first-run issues with reproducible environment receipts;
- make contribution paths possible without API keys or expensive compute.

### Trust

- report benchmark conditions, cost, failures, and unsupported domains;
- never call an internally generated review “independent”;
- keep the no-auto-push, secret-redaction, and generated-code isolation
  defaults visible;
- version the protocol separately from marketing or package versions;
- make every showcased claim traceable to a public ARA or Research Git bundle.

## Metrics to review each release

| Metric | Target |
| --- | --- |
| Time to first offline DAG | < 5 minutes |
| Commands from install to first DAG | ≤ 5 |
| Manual immutable IDs in beginner path | 0 |
| README offline journey CI | 100% on supported core OS/Python matrix |
| Broken local links in README/docs index | 0 |
| Doctor failures with copyable remediation | 100% |
| Public examples with cost/runtime/environment receipts | 100% |
| Public claims linked to inspectable evidence artifacts | 100% |
| Paid model required to contribute protocol/docs/tests | No |

These metrics should be reported from reproducible tests or opt-in user
studies. They should not be inferred from stars, downloads, or model-generated
success narratives.

## Summary

XScientist's scientific rigor is an adoption advantage when its states and
gates are visible. The adoption risk comes from operational choices arriving
before value. The correct strategy is therefore:

1. give every visitor a provider-free DAG in minutes;
2. make autonomous setup one guarded, resumable path;
3. show exact context, evidence, failures, and verification status;
4. publish reproducible examples instead of broad capability claims;
5. grow adapters and conformance around the protocol.

That preserves the hard scientific boundary while making the system usable by
people who did not build it.
