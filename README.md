<p align="center">
  <img src="https://raw.githubusercontent.com/smileformylove/XScientist/main/docs/xscientist-evidence-mark.png" width="220" alt="XScientist evidence-path mark">
</p>

<h1 align="center">XScientist</h1>

<p align="center"><strong>From one idea to a Git-like research history: inspectable, reproducible, and reversible.</strong></p>

<p align="center">
  Bring one idea—even if you do not know models or API keys. XScientist helps
  test it without hiding uncertainty, failed attempts, or contrary evidence.
</p>

<p align="center">
  <a href="https://pypi.org/project/xscientist/"><img src="https://img.shields.io/pypi/v/xscientist.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/xscientist/"><img src="https://img.shields.io/pypi/pyversions/xscientist.svg" alt="Python versions"></a>
  <a href="https://github.com/smileformylove/XScientist/actions/workflows/smoke.yml"><img src="https://github.com/smileformylove/XScientist/actions/workflows/smoke.yml/badge.svg?branch=main" alt="Smoke checks"></a>
  <a href="https://github.com/smileformylove/XScientist/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0 license"></a>
  <a href="https://arxiv.org/abs/2607.12301"><img src="https://img.shields.io/badge/arXiv-2607.12301-b31b1b.svg" alt="arXiv paper"></a>
</p>

<p align="center">
  <a href="#start-with-your-own-idea--no-api-key">Quick start</a> ·
  <a href="#run-an-autonomous-study">Autonomous study</a> ·
  <a href="#inspect-and-reproduce">Audit</a> ·
  <a href="#installation">Install</a> ·
  <a href="https://github.com/smileformylove/XScientist/tree/main/docs">Docs</a> ·
  <a href="https://zhuanlan.zhihu.com/p/2027818800238666075">Build notes (中文)</a> ·
  <a href="https://github.com/smileformylove/XScientist/blob/main/docs/README.zh.md">中文</a>
</p>

XScientist is a local-first research system and an open scientific protocol. It
can explore competing explanations, choose informative experiments, execute
them behind an isolation boundary, criticize its own results, and preserve the
whole path as typed, machine-readable research objects. A completed run is
never presented as a verified scientific claim unless its evidence and review
gates actually pass.

> [!IMPORTANT]
> XScientist is alpha research software, not an oracle. Autonomous runs may use
> paid models. Generated code requires the configured isolated executor.
> Machine-generated claims remain unverified until their evidence and
> independent review gates are complete.

This README describes the published `0.1.3` release. Pin the package version or
a source commit when an experiment must remain exactly reproducible.

## Choose the shortest path

| Your starting point | Run first | Provider or cost | Immediate result |
| --- | --- | --- | --- |
| An idea, but no model or API key | `xscientist explore ./my-study` | None | A local, versioned, falsifiable research start |
| You want to see the system before using your idea | `xscientist demo ./first-study --autopilot --open` | None; `$0.00` | A complete but deliberately contested evidence history |
| You have a local Ollama model | `xscientist provider list` | Local compute; no hosted key | Detected models and the next setup command |
| You have a hosted-model key | `xscientist start ./my-study` | May incur provider cost | A guarded autonomous study in the same history |

If you are unsure, start with `explore`. It records what you know and leaves
unknown fields honestly incomplete.

## Start with your own idea — no API key

Requirements: Python 3.10+ and Git. No API key, model, Docker, or network call
is needed after installation.

```bash
python -m pip install \
  "xscientist==0.1.3"
xscientist explore ./my-study
```

The guided flow uses ordinary questions instead of provider or protocol terms:

- What idea do you want to investigate?
- What observable change do you expect?
- What result would make you change your mind?
- What fair comparison or test could you run first?

You may stop after the first answer and run the same command later. XScientist
versions the exact state as `idea saved`, `falsifiable`, or `planned`; it never
fills a blank with invented science. This path uses no provider, makes no model
call, executes no generated code, and creates no evidence or conclusion.

For a scripted start, the same path is explicit:

```bash
xscientist explore ./my-study \
  --idea "Does daily walking improve sleep quality?" \
  --expect "Daily walking improves a preregistered sleep score." \
  --disprove "The score is unchanged or worse." \
  --test "Compare walking and usual-activity periods." \
  --non-interactive
```

The workspace is understandable without reading internal logs:

- `question.md` is the human-readable research framing;
- `research.yaml` records local policy and workspace identity;
- `.xscientist/objects/` and `checkpoints/` preserve typed decisions and history;
- the local Git repository has no remote and never pushes itself.

Use `status` and `history` to inspect these records; new users should not need
to edit the internal object store directly.

To see what a complete but contested evidence history looks like, run the
bundled `$0.00` example:

```bash
xscientist demo ./first-study --autopilot --open
xscientist status ./first-study
```

The demo intentionally ends with “more evidence needed”: held-out evidence
challenges an over-broad claim. Preserving that conflict is a successful
scientific outcome, not a software failure.

Use `xscientist status ./first-study --verbose` only when you need branch,
pipeline, token, or background-run details. Use `--json` for automation.

## Run an autonomous study

Offline guidance can structure user-supplied reasoning, but it cannot honestly
invent domain knowledge, data, or findings. For AI-assisted exploration, add a
model only after the research question is safely recorded. The same workspace
can be upgraded without replacing its history.

First discover usable models. This works before a workspace exists and detects
a running local Ollama service before suggesting hosted services.

```bash
xscientist provider list
```

### Local model

[Install Ollama](https://ollama.com/download), download a local model, and make
sure its local service is running. The desktop app starts the service; a
headless setup can use `ollama serve`. No hosted API key is needed. The current
[official CLI reference](https://docs.ollama.com/cli) uses `ollama pull` to
download and `ollama ls` to list local models:

```bash
ollama pull gemma3
ollama ls

python -m pip install \
  "xscientist[research,openai-compatible]==0.1.3"
xscientist provider list
xscientist start ./my-study
```

The interactive flow asks only for missing choices: question, provider/model,
evidence source, local research identity, and optional budget. If one usable
provider is detected, it is selected automatically. For an `explore` workspace,
the saved question is reused and existing research files are preserved. A local
model removes hosted API cost, but it still uses your machine's compute and
does not remove Docker isolation requirements for generated experiment code.

### Hosted model

Install the research runtime plus one provider client:

```bash
python -m pip install \
  "xscientist[research,openai]==0.1.3"
export OPENAI_API_KEY="..."
xscientist start ./hosted-study
```

Available client extras are `openai`, `anthropic`, `zhipu`, `bedrock`,
`vertex`, and `openai-compatible`. The last covers local Ollama and compatible
services such as DeepSeek, Gemini, OpenRouter, and custom endpoints.

For any OpenAI-compatible service, configure the endpoint explicitly. `custom`
is a friendly alias for the generic `openai_compat` provider; the URL and key
stay in the workspace's permission-restricted, Git-ignored `.env` file:

```bash
python -m pip install "xscientist[research,openai-compatible]"
export OPENAI_COMPAT_API_KEY="..."
xscientist provider add custom \
  --model gpt-5.6-luna \
  --base-url "https://your-compatible-service.example/v1" \
  --non-interactive
xscientist provider test custom --json
```

`provider test` makes one explicit minimal request and compares the model sent
to the model reported by the endpoint. A mismatch (for example a gateway
silently selecting a smaller model) is reported as unverified; the response
content is never stored by the test.

For scripts and CI, make every consequential choice explicit:

```bash
xscientist start ./ood-study \
  --question "Why does retrieval-guided reflection fail out of distribution?" \
  --provider openai \
  --model openai/gpt-4.1 \
  --user YOUR_NAME \
  --autopilot discovery \
  --data-dir ./data \
  --max-cost-usd 10 \
  --non-interactive
```

Use `--allow-synthetic-data` instead of `--data-dir` only for an explicitly
exploratory study. Input data is content-hashed and mounted read-only. Unknown
model pricing fails closed when a cost limit is active.

### Isolation and readiness

Generated experiment code never runs silently in the host Python process. A
model-backed experiment needs Docker and a version-matched executor:

```bash
xscientist executor prepare --workspace ./ood-study
xscientist provider check --workspace ./ood-study --max-cost-usd 10
xscientist doctor --workspace ./ood-study --deep
```

The commands distinguish missing clients, credentials, local models, Docker
CLI, Docker daemon, and executor-image mismatches. They print ordered repair
commands without making a paid provider request. If you explicitly want one
minimal remote verification, opt in separately:

```bash
xscientist provider check --workspace ./ood-study --live --timeout 30 --json
```

`--live` may incur provider cost and reports transport/model identity only;
response content is never recorded. The default check remains configuration
only.

### Long-running studies

```bash
xscientist start ./ood-study \
  --question "Where does the mechanism break?" \
  --allow-synthetic-data --max-cost-usd 10 --detach

xscientist runs list --workspace ./ood-study
xscientist runs watch RUN_ID --workspace ./ood-study
xscientist runs logs RUN_ID --workspace ./ood-study --tail 100
xscientist runs cancel RUN_ID --workspace ./ood-study
xscientist runs resume RUN_ID --workspace ./ood-study
```

`xscientist status ./ood-study` shows a failed or active background run before
lower-priority scientific follow-ups. A failed state returns a non-zero exit
code while preserving a complete JSON report for automation.

## What remains autonomous

The simple entry point does not reduce the research loop. Depending on the
selected profile, XScientist can:

1. propose rival and null hypotheses instead of defending the first idea;
2. lock predictions and rank experiments by expected information value;
3. execute bounded experiments and retain failed or negative attempts;
4. scan anomalies, contradictions, evidence quality, and transfer boundaries;
5. run independent review roles, repair bounded defects, and stop at hard gates;
6. package the paper, evidence DAG, provenance, and exact continuation context.

Autonomy does not bypass scientific authority. XScientist does not silently
invent missing user answers, label synthetic data as empirical, run generated
code on the host, promote an unreviewed claim, publish research, or push a
workspace remote.

Profiles expose one meaningful trade-off:

| Profile | Use it for | Emphasis |
| --- | --- | --- |
| `balanced` | A first end-to-end study | Bounded search and standard review |
| `discovery` | Mechanism and boundary finding | Rival hypotheses, refutation, branch diversity |
| `publication` | A manuscript candidate | Independent reviews and stricter hold gates |

Deep strategy commands remain available under `xscientist research`, but new
users do not need to learn them before the first result. See the
[deep-research protocol](https://github.com/smileformylove/XScientist/blob/main/docs/DEEP_RESEARCH_PROTOCOL.md)
and [method-discovery protocol](https://github.com/smileformylove/XScientist/blob/main/docs/METHOD_DISCOVERY_PROTOCOL.md).
For literature-to-open-problem discovery, the [FAR-inspired opportunity
funnel](docs/OPPORTUNITY_FUNNEL.md) records every candidate, explicit negative
outcome, independent judgment, and allocation assumption without turning an
external paper's counts into a local score. This protocol is an XScientist
integration inspired by FAR, not a reproduction of FAR's combinatorics pilot.

## Inspect and reproduce

Research Git versions scientific objects rather than asking users to infer
meaning from a folder of logs. Git is the current local storage adapter; no
GitHub account or remote is required, and XScientist never pushes research by
itself.

If you know GitHub, the mental model is deliberately familiar:

| GitHub | XScientist |
| --- | --- |
| Repository | One local research workspace |
| Commit and activity | Hash-checked checkpoint and `history list` |
| Files changed | Scientific `history diff`, including claim/object changes |
| Branch and pull request | Competing research line and semantic merge preview |
| Required checks | `trace → replay → verify` scientific gates |
| Revert and Actions artifacts | Append-only rollback, reproducible run, and bundle |

```bash
xscientist status ./first-study
xscientist history list ./first-study
xscientist history show ./first-study --commit HEAD
xscientist history diff ./first-study
xscientist audit ./first-study --level trace
xscientist audit ./first-study --level replay
xscientist audit ./first-study --level verify
```

Audit answers three different questions and never conflates them:

- `trace`: can every claim be traced to recorded evidence and decisions?
- `replay`: are code, data, environment, seed, and command sufficient to rerun it?
- `verify`: was the result independently checked under the required gates?

These levels form a one-way ladder: a recorded claim may be traceable without
being replayable, and replayable without being independently verified. A
blocked audit is an actionable scientific gap, not necessarily a software
failure.

### Paper quality status

The writing pass separates a readable manuscript from a verified result. A
`quality_gate_passed` result requires a locked preregistration, completed
confirmatory records for every registered task, independent seeds, persisted
result artifacts, numeric candidate-versus-baseline comparisons with
uncertainty, deterministic hashes, a task → metric → claim path, and a
clean-room verification report covering every required criterion. Prose,
figures, or an LLM score cannot substitute for missing evidence.

Until that chain is complete, XScientist labels the output
`exploratory_draft` or `manuscript_draft`; it does not call it
`submission_ready`. Result JSON also includes
`scientific_evidence_failures` and short `scientific_evidence_next_actions`, so
a blocked run tells you what to fix next instead of silently lowering a score.
See the [research integrity contract](docs/RESEARCH_INTEGRITY.md) for the exact
record fields and replay requirements.

Save a meaningful manual change before trying a risky alternative. Rollback is
preview-only unless `--apply` is explicit. Applying it appends a reversal
checkpoint: it never deletes or rewrites the original result.

```bash
xscientist history save ./first-study -m "record corrected measurement rule"
xscientist history rollback ./first-study --commit HEAD
# Review the target, impact, blockers, and generated apply command first.
xscientist history rollback ./first-study --commit HEAD --apply
```

Unsaved tracked, staged, selected, or research-eligible changes and the first
checkpoint block rollback. Policy-excluded generated views are preserved and do
not block it; after a reversal, `status` marks an older DAG as stale and prints
the exact refresh command. Reverting an older checkpoint can still conflict
with newer work, in which case `--apply` stops without discarding current
history.

For reproduction, bundles, object inspection, context snapshots, deep diffs,
and branches, use the advanced protocol surface:

```bash
xscientist research reproduce HEAD --repo ./first-study --execute --record \
  --reproduces @latest:claim --verifier human:REPRODUCER

xscientist research bundle --repo ./first-study --dest ./study-backup
xscientist research export --repo ./first-study --dest ./exchange
```

A generated DAG is a disposable view, not scientific source data. Regenerating
it does not dirty a research checkpoint or prevent a bundle. Eligible research
changes, tracked edits, or staged changes still block bundling until reviewed.

### Process benchmark comparison (offline and reproducible)

The [linked WeChat article](https://mp.weixin.qq.com/s/pRPBg5RE1a6jWdO8LdP89A)
points to [AutoResearchEval](https://arxiv.org/abs/2608.14905): a six-stage, artifact-aware
diagnostic benchmark with 100 tasks and 800 trajectories. XScientist does not
claim to reproduce its model score: the official rollout service and annotated
trajectories are external. Instead, the repository includes an explicit,
zero-cost conformance pilot that checks task framing and the evidence exposed by
one local workspace:

```bash
# Optional, explicit one-time export/download from the official dataset page;
# save one JSON/JSONL task manifest locally. The pilot itself stays offline.
# (The published dataset layout may evolve; do not hard-code a remote path.)

xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl \
  --workspace ./first-study \
  --limit 20 --kind open-ended --json
```

The pilot never downloads data, reads gold conclusions, calls a provider, or
executes a model rollout. It reports `official_comparable: false` and keeps
three measurements separate: task-contract validity, A–F artifact coverage, and
XScientist's `trace → replay → verify` plus metacognitive repair signals. See
[the benchmark protocol](docs/BENCHMARKS.md) for the exact boundary and the
[official task dataset](https://huggingface.co/datasets/PrentisAI/AutoResearchEval).
The benchmark-driven completion status and explicit blockers are in the
[optimization status](docs/OPTIMIZATION_ROADMAP.md); it contains no dated
delivery plan or unverified completion promise.

The report also contains a bounded `diagnostics` backlog. `P0` means a fair
quality claim is blocked, `P1` is evidence/lifecycle debt, and `P2` is an
exploration or usability improvement. `stage_coverage` is explicitly a
structural measure (`score_semantics: structural_stage_coverage_only`), never a
scientific quality score; even 83.3% coverage keeps
`quality_claim_allowed: false`.

For workspaces, the report includes a read-only `evidence_index` covering the
allowlisted Research VCS, ARA/CAS, and generated-view surfaces. It records
bounded counts and aggregate SHA-256 digests, with an explicit `digest_scope`
(`observed_files` or `bounded_prefix`), truncation, and read-error fields, but
never filenames, paths, or raw payloads. The same
report exposes `workspace.exploration` when an ARA exploration graph exists;
missing graphs are `unavailable`, not zero failed or unattempted candidates.
Its `ara_contract` record counts manifests, locks, graphs, and verify reports;
`fsck_run` and `bundle_created` stay false in this redacted index: the benchmark
does not attest that an external `fsck` or bundle command was run. Retain and
verify those command outputs separately when a full audit package is required.
The index also exposes `walk_entries_observed`, `walk_truncated`, and
`source_count_complete`; when a scan is truncated, source counts describe a
bounded prefix and are not complete totals. Exploration is versioned as
`xscientist.exploration-audit.v1`; malformed nodes are surfaced as unknown/read
errors rather than counted as successful or failed work.
Use `xscientist benchmark verify --report <report.json> --json` to validate a
saved report offline. Its `reproducibility.fingerprint` excludes timestamps and
runtime noise while binding the manifest, task slice, workspace head, and
bounded source totals.

Feedback self-evolution uses the same conservative semantics: `health_score`
is an `observational_heuristic`, not a scientific-quality or causal-effect
score. `independence_status: "independence_unverified"` records an evaluator
link without proving evaluator independence; paired observations remain
traceability signals only. `causal_claim_allowed` and
`promotion_signal_allowed` stay `false` until a fixed independent evolution
gate is recorded, so feedback cannot silently label its own change as an
improvement. The persisted history is also bounded and JSON-portable: oversized
files, deep/cyclic metric trees, and non-finite values are rejected or surfaced
as load errors rather than silently merged.

#### Evidence and ARA retention boundary

The pilot is read-only. It does not create a trajectory, copy the task
manifest, or silently write an ARA. The Python API returns the report in
memory; the CLI persists the report only when `--output` or stdout redirection
is explicitly used. Any
Research VCS objects, checkpoints, Git refs, ARA directories, or CAS payloads
already present in the workspace remain in their original locations, but the
benchmark report is a bounded, redacted index—not a full evidence archive.

For a safer one-command summary export, use the explicit atomic `--output`
option. It writes the redacted report and diagnostics, but never raw prompts,
model responses, ARA files, or CAS payloads:

```bash
xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl --workspace ./first-study \
  --limit 20 --kind open-ended --json \
  --output ./benchmark-evidence/autoresearch-report.json
```

| Source | What remains on disk | What the pilot report contains |
| --- | --- | --- |
| Task manifest | The caller's original JSON/JSONL file | SHA-256, counts, and redacted contract failures; no gold/task prose |
| Research VCS / typed evidence | `.xscientist/objects/`, `checkpoints/`, Git history, and local pointers | Bounded artifact/decision rows, hashes, signals, source totals, and truncation flags; payloads omitted |
| ARA / CAS | Existing `ara/` roots and `.ara-store/`/local CAS remain untouched | Closure and binding summary only; no automatic full ARA snapshot or payload copy |
| ARFT coverage | Nothing is written by `build_arft_coverage()` | Embedded structural summary; `save_arft_coverage()` is an explicit write |

To preserve a complete review package, opt in explicitly and treat the result
as potentially sensitive:

```bash
# Persist the bounded benchmark report itself.
xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl --workspace ./first-study \
  --limit 20 --kind open-ended --json > benchmark-report.json

# Verify checkpoint, ARA-manifest, pointer, and CAS bindings.
xscientist research fsck --repo ./first-study

# Full ARA audit bundle (includes every non-GC ARA file).
xscientist ara bundle --ara ./first-study/ara/<run> \
  --dest ./benchmark-evidence/ara-audit.tar.gz --profile audit

# Research VCS interoperability export; payloads require an explicit flag.
xscientist research export --repo ./first-study --ref HEAD \
  --dest ./benchmark-evidence/research-export --include-payloads
```

Inspect and redact these bundles before sharing: they may contain prompts,
tool output, datasets, or model responses. `--show-process` and
`workspace.process` intentionally remain summaries and never claim to contain
all raw evidence.

One local run on 2026-08-21 (macOS, Python 3.13, bundled balanced demo) produced:

| Measurement | Result | Interpretation |
| --- | ---: | --- |
| Open-ended task contracts (first 20) | 20/20 | Manifest framing is structurally valid; no gold was used |
| Optimization task contracts (first 20) | 20/20 | Same structural check on the separate task family |
| Demo six-stage coverage | 5/6 (83.3%) | Retrieval artifacts are intentionally absent from the offline fixture |
| Demo closure | `trace` pass · `replay` pass · `verify` blocked | A held-out conflict and missing independent reproduction remain visible |
| Demo metacognitive status | `contained` · 2 issues · 0 shipped | The gate holds the conclusion instead of hiding review debt |
| Demo process trail | 3 commits · 1 branch · 16 typed artifacts | Intermediate objects and checkpoint boundaries remain inspectable; no hidden transcript is exported |
| Branch conformance fixture | 2 branches · 3 commits · per-commit branch membership | Divergence is visible; fairness stays `NOT VERIFIED` until budget/evaluator/base are evidenced |
| Network / provider / model cost | none / none / $0 | This is a conformance measurement, not an autonomous-agent score |

The table is a baseline for improving the harness and evidence contracts; it
must not be compared numerically with published model leaderboard values. In
the JSON report, `stage_coverage` counts stages meeting the minimum evidence
bar; each stage also exposes `complete` for the stricter all-criteria result.
Review debt without an explicit hold/reject gate is reported as `open`, never
silently upgraded to `contained`.

This historical table is a checked-in summary, not a claim that its raw task
manifests, ARA files, or reports are stored in this repository. Rerun the
commands above with `--output` and the explicit evidence-export commands when
a reproducible bundle is required.

For orientation, the paper's headline measurements and this pilot sit on
different layers:

| Layer | AutoResearchEval paper | XScientist local pilot |
| --- | --- | --- |
| Scale | 100 tasks, 800 model/harness trajectories | 20 open-ended + 20 optimization manifest rows checked; 0 rollouts |
| Diagnosis | Artifact-aware judge; κ 0.75 (pattern) / 0.83 (root cause) | No judge; typed-artifact coverage and closure only |
| Metacognitive signal | F.4 in 660/800 analyses (82.5%) | Bundled demo: 2 unresolved issues, `contained`, 0 shipped; not the same statistic |
| Cost / comparability | External rollout/evaluation budget | $0, `official_comparable: false` |

The paper figures are reported for context, not as a score that this repository
claims to match; see the [paper](https://arxiv.org/abs/2608.14905) for its
artifact-aware judge and full trajectory protocol.

#### Compare the other systems in the talk (without inventing a ranking)

The attached Expo Talk names systems that operate at different layers: full
research agents (ScientistOne, AI Scientist v2, AutoResearchClaw, DeepScientist,
AI-Researcher), adaptive search components (AdaEvolve, EvoX, MARS), a review
component (ScholarPeer), a paper-writing component (PaperOrchestra), and a
figure component (PaperBanana). FAR (Find–Attempt–Recommend) is an adjacent
primary-source discovery/allocation reference, while MLE-STAR and DS-STAR are
adjacent primary-source execution references added for coverage; these three
are not claimed to be named in the attached 107-page talk.
The report also keeps talk-only references (Deep Researcher Agent and the AST
role diagram) visible without pretending they have a matched benchmark. A
figure or writing score is not an end-to-end discovery score, so the project
keeps these scopes separate.
FAR's reported expert/judge review is not a recruited human task-performance
arm, and its combinatorics counts are not local XScientist measurements.
Context-only mentions and future concepts (for example ScientistTwo) remain
listed with their slide number in `talk_inventory` rather than being promoted
to evaluated competitors.

Generate the source-audited matrix locally:

```bash
# No network, provider, external rollout, or cross-system score aggregation.
xscientist benchmark systems --json > system-comparison.json

# Add the bounded Git-like process view for one local workspace.
xscientist benchmark systems --workspace ./first-study --show-process
```

See the [English comparison](docs/SYSTEM_COMPARISON.md) and
[中文对比](docs/SYSTEM_COMPARISON.zh.md). Each row records its primary paper or
official repository, the benchmark layer it actually measures, and an explicit
status (`reported_primary`, `local_observed`, `scoped_component`, or
`not_measured_here`). The report hard-codes
`official_comparable: false`, `score_claim_allowed: false`, and
`quality_claim_allowed: false`; external numbers are never copied into
`workspace.score`. Its `rollout_scope` and `cost_scope` are explicitly
`this_audit_only`, while historical trajectory cost remains `unobserved`.
With `--workspace`, branch topology, intermediate artifact counts, fairness
blockers, and `artifact_scope: current_checkout_only` remain visible without
exporting prompts or hidden free-form reasoning.
The report also records the attached 107-page talk's filename and SHA-256, so a
future audit can tell exactly which slide source was used.

The fair next experiment is a registered matched rollout: same task slice,
starting artifact, model/backbone, hardware, budget, evaluator, retry rule,
seed count, and canonical rerun. Until that exists, this is a capability and
evidence comparison—not a claim that XScientist beats any system or person.

#### Can this be compared with people?

Yes, but the current pilot does not yet produce a human-vs-agent scientific
score. A defensible human arm must use the same task manifest and slice,
starting artifact, tools/data/network policy, wall-clock and cost budget,
output format, verifier/evaluator, and number of attempts. Randomize task
order, pre-register the stopping rule, use more than one participant/run, and
report uncertainty rather than a single best result.

The same process contract can then record human checkpoints, evidence,
failures, repairs, and gates without collecting private free-form thoughts.
Comparable measures should be the evaluator's final score (when the official
verifier is available), artifact-aware process diagnosis, time/cost, evidence
completeness, auditability, and failure/recovery coverage. Until those controls
and a real human trajectory set exist, this repository must keep
`official_comparable: false`; it can compare process observability and
usability, not claim that XScientist beats or matches researchers.

#### External human baselines (source-audited)

We also maintain a [source-audited inventory of public human baselines](docs/HUMAN_BASELINES.md),
updated 2026-08-23. It separates real participant runs from leaderboard/SOTA
references, expert validation, human judge calibration, and human+agent
workflow studies. The strongest directly measured rows include RE-Bench (61
experts, 71 attempts), PaperBench (8 ML PhDs on a four-paper subset), and
DiscoveryWorld (11 scientists on 16 tasks). For a biology-specific reference,
BAISBench v1 reports a human arm on its own frozen 198-question/31-dataset
release; the later v2 changes the task and only plots the aggregate human score,
so the inventory deliberately does not transfer or approximate it. DSBench is
listed separately as a small, incompletely documented sample rather than an
expert baseline. Every score is reported only with its original task slice and
budget; these numbers are not pooled into a “human average” or pasted into the
XScientist report.
For retrieval and research-engineering context it also records BrowseComp,
BrowseComp-V³ (including its published human process score), VeriWeb, Mind2Web
2, WebArena, and MLRC-Bench. A separately labelled human ideation study covers
research-idea generation only. Adjacent GPQA, GAIA, and H-ARC human/annotator
reference measurements remain outside the scientific-research comparison.
Mind2Web 2 is a 30-task
human subset of a 130-task suite; WebArena samples 170 templated intents with
five CS graduate participants. Neither number is a human score for XScientist.
ScholarPeer’s existing human reviews and PaperOrchestra’s 11-researcher
side-by-side judgments are retained as judge-calibration/reference evidence,
not as human task-performance arms.

For a compact, source-scoped comparison (not a leaderboard), the directly
reported figures are:

| External human arm | Reported result | Scope that must stay attached |
| --- | ---: | --- |
| RE-Bench | 82% non-zero; 24% matched/exceeded the strong reference | 61 experts, 71 attempts, 7 ML research-engineering environments, 8h |
| PaperBench | 41.4% human best@3 after 48h | 3-paper subset of the human study; paper reproduction, not open research |
| DiscoveryWorld | completion 0.66; knowledge 0.55 | 11 MSc/PhD scientists, 16 simulated-world tasks, 1h/task |
| [Research ideation study](https://arxiv.org/abs/2409.04109) | Human ideas: novelty 4.86 ± 1.26; feasibility 6.53 ± 1.50; overall 4.69 ± 1.16 | 49 NLP idea writers, one proposal each in a 10-day window; ideation-only, not end-to-end research |
| [PaperQA2 / LitQA2](https://arxiv.org/abs/2409.13740) | Human precision 73.8% ± 9.6%; accuracy 67.7% ± 11.9% | 9 biology/science PhD or PhD-student evaluators; literature QA only, roughly one week per quiz |
| [VeriWeb](https://arxiv.org/abs/2508.04026) | Human completion L1→L5: 47% / 40% / 15% / 6% / 1%; full success 0% under 12 min/task | 5 annotators, 10 random tasks per level; web information seeking, not scientific-code execution |
| BAISBench v1 | BAIS-SD 0.762; CellTypist 0.437 ± 0.014 | Frozen v1: 198 questions/31 datasets; do not transfer to v2 |
| BrowseComp | 29.2% solved; 86.4% agreement conditional on solved | 1,255 attempted of 1,266 questions, human trainers, 2h cap; 29.2% is solve rate, not accuracy |
| Mind2Web 2 | partial 0.79; success 0.54; Pass@3 0.83 (cross-participant) | Random Subset-30 of 130 long-horizon web tasks; 7 participants, 3 different people per task |

These rows are external measurements with different tasks, tools, metrics, and
budgets. They state the design requirements for a matched arm, not numbers
that can be substituted into `workspace.score`.

For the linked AutoResearchEval paper, the honest status is
`human_task_performance_baseline: not_reported_in_audited_source`: its human
work is trajectory annotation and judge calibration, not a task-performance
arm. XScientist itself currently has zero human runs and zero model rollouts,
so it reports no human-vs-agent scientific score. “Not reported” is preserved
as a first-class result rather than replaced with zero or an invented estimate.
The JSON report makes this machine-checkable with
`human_baseline.status: "not_reported"`, `matched_arm: false`, and `score: null`.
The same record reports `local_runs: 0` and `external_scores_injected: false`.
Its `evidence_retention` field also states, machine-readably, that the pilot
does not copy raw trajectories, ARA snapshots, or CAS payloads; complete audit
bundles require the explicit export commands documented above.

To inspect the git-like process rather than only the endpoint, add
`--show-process` to the pilot command:

```bash
xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl \
  --workspace ./first-study \
  --limit 20 --kind open-ended --show-process
```

The JSON `workspace.process` section
contains bounded commits, branches, parent/checkpoint counts, typed intermediate
artifact IDs/hashes, relation types, failure/recovery signals, and a fairness
contract tied to the manifest SHA-256. It deliberately excludes prompts,
completions, held-out conclusions, and free-form payloads; it is an
artifact-backed reasoning trail, not hidden chain-of-thought. Commit membership
is retained for each visible branch, but artifact rows are explicitly scoped to
the current checkout (`artifact_scope: current_checkout_only`); the pilot does
not fabricate per-branch artifact outcomes. Branch comparison
is only called fair when the report can verify the same task manifest, budget,
evaluator, and base; otherwise the corresponding field stays unverified.
Shareable output also replaces free-form branch names and commit subjects with
stable aliases/digests, so Git metadata cannot become a covert gold or local
text channel. The process payload is versioned as
`xscientist.process-audit.v1`; its JSON schema validates both an available
Research VCS workspace and an explicit unavailable/empty state.

To challenge a conclusion without erasing its history:

```bash
xscientist research branch challenge/boundary --repo ./first-study --switch
xscientist research plan @latest:hypothesis --repo ./first-study \
  "Search for a counterexample" \
  --test "A reproducible failure refutes the current mechanism"
xscientist research switch main --repo ./first-study
xscientist research merge challenge/boundary --repo ./first-study --preview
```

## A small surface over inspectable layers

```mermaid
flowchart TB
  U["explore · start · status"] --> O["Autonomous research loop"]
  O --> E["Isolated experiments and providers"]
  O --> R["Typed Research Git history"]
  E --> D["Evidence DAG and ARA artifacts"]
  R --> D
  D --> A["audit · history · reproduce"]
```

The everyday surface stays small: `explore`, `start`, `status`, `audit`, and
`history`. Readiness repair lives under `doctor`, detached execution under
`runs`, and the complete scientific protocol under `research`. The first table
in this README is the only decision tree a new user needs.

The public orchestration surface lives in `xscientist/`, the experiment
workflow in `ai_scientist/`, and versioned schemas in
`ai_scientist/protocol/`. See [Architecture](https://github.com/smileformylove/XScientist/blob/main/docs/ARCHITECTURE.md).

## Installation

| Channel | Command |
| --- | --- |
| Published 0.1.3 | `python -m pip install "xscientist==0.1.3"` |
| Development `main` | `python -m pip install "xscientist @ git+https://github.com/smileformylove/XScientist.git@main"` |
| Contributor | `python -m pip install -e ".[research,openai,dev]" -c requirements/constraints-ci.txt` |

Pin a commit rather than `main` for an exactly repeatable experiment.

Install optional capabilities only when a study needs them:

| Extra | Purpose |
| --- | --- |
| `research` | End-to-end autonomous research runtime |
| provider extra | Exactly one model client or compatible route |
| `plot`, `pdf`, `pdf-layout`, `ml` | Specialist experiment capabilities |
| `service` | FastAPI/Uvicorn service |
| `trust` | Optional signing primitives |
| `full` | Backward-compatible all-in-one environment |

Core protocol and CLI support Python 3.10–3.13. Autonomous execution also
depends on the selected provider, Docker, and the study's experiment stack.

## Outputs and boundaries

An autonomous project keeps configuration, ideas, experiments, papers, logs,
and ARA handoff artifacts separate. The exact layout is documented in
[Output directories](https://github.com/smileformylove/XScientist/blob/main/docs/guides/OUTPUT_DIRECTORIES.md).

| Boundary | Default |
| --- | --- |
| Generated code | Isolated executor; strict setups fail closed |
| Experiment network | Disabled in strict isolation |
| Secrets | Private env, Git ignore, and redacted diagnostics |
| Remote publication | Never automatic |
| Claims | Draft until evidence and independent gates qualify them |
| Negative results | Preserved as first-class history |
| Self-evolution | Shadow → sealed evaluation → canary → signed promotion |

For sensitive domains, use XScientist as research infrastructure—not as a
substitute for domain experts, ethics review, or regulated validation.

## SDK and documentation

```python
from xscientist import ProjectRequest, XScientist

client = XScientist(output_root="./research-output")
result = client.run_project(
    ProjectRequest(
        project="retrieval-study",
        question="When does retrieval-guided reflection fail?",
        autopilot="discovery",
        allow_synthetic_data=True,
        max_cost_usd=10,
    )
)
print(result.returncode)
```

| Need | Guide |
| --- | --- |
| First project and recovery | [Getting started](https://github.com/smileformylove/XScientist/blob/main/docs/GETTING_STARTED.md) · [Long-running guide](https://github.com/smileformylove/XScientist/blob/main/docs/LONG_RUNNING_GUIDE.md) |
| Research history and protocol | [Local Research Git](https://github.com/smileformylove/XScientist/blob/main/docs/LOCAL_RESEARCH_GIT.md) · [Protocol v2](https://github.com/smileformylove/XScientist/blob/main/docs/RESEARCH_PROTOCOL_V2.md) |
| Integrity and scientific strategy | [Research integrity](https://github.com/smileformylove/XScientist/blob/main/docs/RESEARCH_INTEGRITY.md) · [Science constitution](https://github.com/smileformylove/XScientist/blob/main/docs/SCIENCE_CONSTITUTION.md) |
| Current limitations and audit | [2026 project audit](https://github.com/smileformylove/XScientist/blob/main/docs/PROJECT_AUDIT_2026-08.md) · [Onboarding audit](https://github.com/smileformylove/XScientist/blob/main/docs/ONBOARDING_AUDIT.md) |
| SDK, HTTP API, and adapters | [SDK/API](https://github.com/smileformylove/XScientist/blob/main/docs/guides/SDK_AND_API.md) · [DAG/adapters](https://github.com/smileformylove/XScientist/blob/main/docs/RESEARCH_DAG_AND_ADAPTERS.md) |
| Configuration and operations | [Configuration](https://github.com/smileformylove/XScientist/blob/main/docs/CONFIG_REFERENCE.md) · [Operations](https://github.com/smileformylove/XScientist/blob/main/docs/OPERATIONS_CHECKLIST.md) |

Run `xscientist --help` for the small everyday command set and
`xscientist research --help` for the complete scientific protocol surface.

## Project status

XScientist is under active alpha development. Contributions should include a
test, preserve protocol/schema compatibility, and avoid weakening provenance,
isolation, cost, or scientific gates. Read [CONTRIBUTING.md](https://github.com/smileformylove/XScientist/blob/main/.github/CONTRIBUTING.md)
and [CHANGELOG.md](https://github.com/smileformylove/XScientist/blob/main/CHANGELOG.md).

Paper: [XScientist: Towards an AI-Driven Scientific Research Ecosystem](https://arxiv.org/abs/2607.12301).

Apache-2.0 licensed. See [LICENSE](https://github.com/smileformylove/XScientist/blob/main/LICENSE).
