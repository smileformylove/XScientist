# Benchmarks and process audits

The public benchmark measures a usability property: can a clean local process
go from an empty directory to inspectable, scientifically honest status without
credentials, network access, or model cost?

```bash
xscientist benchmark first-run --json
xscientist benchmark first-run --max-seconds 30
xscientist benchmark first-run --json --output ./benchmark-evidence/first-run.json
```

It creates the deterministic Autopilot fixture, builds the Research VCS DAG,
reads status, records duration and structural counts, and deletes its temporary
workspace. `--workspace DIR` retains a named workspace for inspection.
The JSON is validated by `load_schema("first_run_benchmark")`; `--output`
persists the redacted result with an atomic write and never copies raw research
payloads.

This is not a model-quality benchmark. It must not be used to claim scientific
performance, autonomous discovery quality, or provider speed. Those require
separate registered datasets, budgets, evidence, and evaluation authority.

## Cross-system capability matrix

The talk accompanying this project names several systems that are not
interchangeable benchmarks: some are end-to-end discovery agents, while
AdaEvolve/EvoX/MARS, ScholarPeer, PaperOrchestra, and PaperBanana focus on
search, review, writing, and figures respectively. MLE-STAR and DS-STAR are
adjacent primary-source execution references added for coverage; they are not
claimed to be named in the attached 107-page talk. Generate the source-audited,
non-ranking matrix with:

```bash
xscientist benchmark systems --json > system-comparison.json
xscientist benchmark systems --workspace ./first-study --show-process
```

This command is provider-free and does not fetch or execute any external
system. It records primary-source links, talk-only provenance, capability
scope, and an optional redacted XScientist process view. Its report fixes
`official_comparable: false`, `score_claim_allowed: false`, and
`quality_claim_allowed: false`. The attached 107-page talk is retained by
filename and SHA-256 in `source_manifest`; it is a scope-discovery source, not
a substitute for a primary paper or matched run. Read [the full comparison](SYSTEM_COMPARISON.md)
for the dimensions, fair-run requirements, and explicit non-claims.

## AutoResearchEval-inspired pilot

The supplied article discusses [AutoResearchEval](https://arxiv.org/abs/2608.14905):
100 frontier-science tasks, six lifecycle stages, 800 trajectories, and an
artifact-aware ARFT diagnosis. The official rollout harness and annotated
trajectories are not part of this repository, so XScientist exposes a small,
offline conformance pilot instead. It reads a task JSON/JSONL file, ignores gold
answer fields, and audits a local workspace read-only.

Task manifests are published separately in the
[official dataset](https://huggingface.co/datasets/PrentisAI/AutoResearchEval)
(70 open-ended discovery tasks and 30 target-anchored optimization tasks). The
[official repository](https://github.com/PrentisAI/AutoResearchEval) is useful
for taxonomy and judge tooling, but does not contain the full rollout service.

```bash
# Export/download one JSON/JSONL manifest from the official dataset page and
# save it locally (the remote file layout may evolve; the pilot itself is offline).

xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl \
  --workspace ./first-study \
  --limit 20 \
  --kind open-ended \
  --json
```

Use the analogous `optimization/tasks.jsonl` file with `--kind optimization`
for the target-anchored subset. The manifest includes held-out fields; the
pilot validates only task framing and never emits those fields.

The report contains three deliberately separate measurements:

| Field | Meaning | Comparable to the paper's score? |
| --- | --- | --- |
| `tasks.valid_task_contracts` | The local manifest has the required task framing fields | No |
| `workspace.stage_coverage` / `stage_score` | Typed evidence coverage for A–F; `covered` is the minimum two-criterion bar and `complete` means every listed criterion passed; empty folders do not pass | No |
| `workspace.closure.levels` and `workspace.metacognition` | `trace → replay → verify` and acknowledged-issue containment/repair | No; these are XScientist governance signals |
| `workspace.process` | Bounded commits, branch topology, typed intermediate artifacts, failure/recovery signals, and fairness boundaries | No; it is the inspectable process layer |
| `human_baseline` | Explicit local human-arm status; external inventory is never substituted | No; `not_reported` is preserved until a matched local arm exists |

The pilot also emits a bounded `diagnostics` backlog. It is an action list,
not a score: `P0` blocks a fair quality claim, `P1` is evidence or lifecycle
debt, and `P2` is an exploration/usability improvement. Typical findings are
`QUALITY.NO_MATCHED_ROLLOUT`, `FAIRNESS.BRANCH_CONTRACT_UNVERIFIED`,
`AUDIT.BOUNDED_VIEW`, and `FEEDBACK.CONTAINED_REVIEW_DEBT`. Every item includes
a fixed verification condition so a later run can show whether the gap closed.

The stage percentage is explicitly `structural_stage_coverage_only`, never a
scientific quality score. A run showing 83.3% coverage still has
`quality_claim_allowed: false` until an evaluator-backed, repeated rollout is
available.

The top-level `human_baseline` record is intentionally explicit:
`status: "not_reported"`, `matched_arm: false`, `score: null`, `local_runs: 0`,
and `external_scores_injected: false` for the local pilot. External rows in
[the source-audited inventory](HUMAN_BASELINES.md) are context only; they are
never silently substituted into this field.

The machine-readable `evidence_retention` record makes the storage boundary
explicit: the API/CLI pilot writes no raw trajectory, ARA snapshot, or CAS
payload, and the process view is a bounded redacted index. A caller may retain
the JSON with `--output` or by redirecting stdout and may create a complete, sensitive audit
package only with the explicit `fsck`, `ara bundle`, and payload-export
commands below.

Each report also records a SHA-256 of the supplied manifest and redacted
row-level contract failures, so two runs can be compared without copying task
answers or disclosing the local path. It records the XScientist version and
coarse Python/OS runtime as context, not as a claim of cross-machine
performance equivalence.

Use `--show-process` for a concise terminal timeline in addition to the JSON:

```bash
xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl \
  --workspace ./first-study \
  --limit 20 --kind open-ended --show-process
```

To retain the redacted report without relying on shell redirection, use the
explicit atomic output option. It saves the summary and diagnostics only; it
does not copy prompts, model responses, ARA files, or CAS payloads:

```bash
xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl --workspace ./first-study \
  --limit 20 --kind open-ended --json \
  --output ./benchmark-evidence/autoresearch-report.json
```

### Evidence and ARA retention

The pilot is read-only. It does not create a trajectory, copy the task
manifest, or write an ARA. The API returns a report in memory; callers can use
`persist_benchmark_report()` or the CLI `--output`/`--json` path to persist that
summary. Existing Research VCS
objects, checkpoints, Git refs, ARA roots, and CAS payloads remain in the
workspace, while `workspace.process` and the stage report expose only bounded,
redacted metadata. The report is therefore an audit index, not a complete raw
evidence archive. `build_arft_coverage()` is also read-only; call
`save_arft_coverage()` explicitly if its structural report should become a
pipeline artifact.

For a complete, potentially sensitive review package, export it explicitly:

```bash
xscientist research fsck --repo ./first-study
xscientist ara bundle --ara ./first-study/ara/<run> \
  --dest ./benchmark-evidence/ara-audit.tar.gz --profile audit
xscientist research export --repo ./first-study --ref HEAD \
  --dest ./benchmark-evidence/research-export --include-payloads
```

Inspect and redact the resulting archives before sharing. They may contain
prompts, tool output, datasets, or model responses; the shareable process
report intentionally never includes those raw payloads.

For a two-line conformance fixture, the human surface is intentionally
process-shaped rather than score-shaped:

```text
Process: 3 visible commits, 2/2 branches, 2 typed artifacts
  branch alternative-1 (diverged_or_behind, 3 commits)
  branch current (current, 2 commits)
  commit … init: checkpoint:init [alternative-1,current]
  commit … ideation: checkpoint:ideation [alternative-1,current]
  commit … experiment: checkpoint:experiment [alternative-1]
  Fair branch comparison: NOT VERIFIED (unverified: same_task_slice, same_budget, same_evaluator, same_base)
```

The ellipses stand for short commit hashes. This fixture is a conformance
example, not a scientific result or a claim that the two lines received equal
model budgets.

### Human comparison

A human arm is possible, but this pilot does not currently claim a human-vs-agent
scientific score. A matched study must hold constant the task manifest and
slice, starting artifact, tools/data/network policy, wall-clock and cost
budget, output schema, verifier/evaluator, and attempt count. Task order and
stopping rules should be randomized or pre-registered; use multiple people or
runs and report uncertainty. The same process contract can capture
artifact-backed human checkpoints and decisions without collecting private
free-form thought. Until a real human trajectory set and the official
evaluator are available, compare only process observability/usability and keep
`official_comparable: false`.

The process section exposes checkpoint IDs, short commit IDs, stage/state,
parent counts, branch relation (`current`, `same_head`, or
`diverged_or_behind`), artifact IDs/hashes, relation types, and structured
signals such as `falsifier`, `counterevidence`, `provenance`, and
`independence`. It intentionally omits prompts, completions, task conclusions,
and free-form payloads. `fairness` records the manifest hash, fixed task slice,
gold exclusion, zero network/provider/model use, and which stronger conditions
(same budget/evaluator/base branch) remain unverified. This is the project's
observable reasoning trail—not hidden chain-of-thought. Commit membership is
shown for every visible branch, while artifact rows are explicitly marked
`artifact_scope: current_checkout_only`; per-branch artifact outcomes are not
invented from a single checkout.

The shareable process view treats branch names, commit subjects, task payloads,
and issue prose as untrusted free text. It emits stable aliases/digests and
typed enums or boolean signals instead of those strings. Branch comparison is
eligible only when the same manifest/task slice, fork base, budget, and
evaluator are all verified; otherwise the report names the unverified checks
and refuses to turn branch visibility into a fairness claim.

The process payload is versioned as `xscientist.process-audit.v1` and can be
validated offline with `load_schema("process_audit")`; both available and
unavailable workspaces use the same top-level shape. This makes a missing local
repository an explicit state rather than a schema-less empty result.

The complete conformance report is likewise validated by
`load_schema("autoresearch_conformance")`. The schema fixes the redaction and
non-comparability invariants (`official_comparable: false`,
`quality_claim_allowed: false`, zero rollouts, and no raw payloads).

`official_comparable` is always `false`, `gold_fields_used` is always `false`,
and the pilot reports zero rollouts, provider calls, network use, and model
cost for the audit itself; historical trajectory cost remains unobserved. A
held/rejected gate is reported as contained review debt; it is not
mislabelled as the paper's F.4 “shipped despite awareness” failure. The pilot
therefore measures observability and governance, not autonomous scientific
ability.

### External human-baseline inventory

The companion [human-baseline inventory](HUMAN_BASELINES.md) records public
sources as-of 2026-08-22. It does not pretend that every paper mentioning
“human” contains a human performance arm. `measured_human` is reserved for
people who actually ran the stated tasks; public leaderboards and prior SOTA
are marked as proxies; expert labeling, judge calibration, and human-verified
ground truth are separate classes. The linked AutoResearchEval paper is
explicitly `not_reported` for human task performance: its human contribution
is annotation/calibration only.

The direct measured rows (RE-Bench, PaperBench, DiscoveryWorld, BAISBench,
BrowseComp, BrowseComp-V³, VeriWeb, Mind2Web 2, WebArena, and a sampled DSBench
study), plus the separate research-ideation arm,
are useful external context, but their tasks, budgets, tools, and metrics differ
from this pilot. They must not be copied into
`workspace.score` or combined into a cross-benchmark human average. The local
pilot still has zero human runs and zero model rollouts; the only honest local
status is no human-vs-agent scientific score. The inventory also includes a
CORE-Bench follow-up showing a human-agent time uplift, plus MLRC-Bench and
ResearchGym reference artifacts, all clearly labelled as process studies or
proxies rather than autonomous human capability baselines.
Adjacent GPQA/GAIA/H-ARC human/annotator reference measurements are listed in the inventory for breadth,
but are explicitly outside the scientific-research comparison.

If review issues exist without an explicit corrective or hold/reject gate, the
report uses `metacognition.status: "open"`; it never infers containment merely
from the absence of a shipped artifact. On a Research VCS-only workspace, the
embedded ARFT summary is `not_applicable` with
`source: "research_vcs_typed_objects"` until a legacy-contract adapter is
available. An empty workspace is reported separately as `not_initialized`.

## AutoResearchEval / ARFT observability audit

The project can also emit an offline, artifact-only coverage report for the
six lifecycle stages and the 45-pattern AutoResearch Failure Taxonomy (ARFT):

```python
from ai_scientist.utils.arft_coverage import build_arft_coverage, save_arft_coverage

report = build_arft_coverage("/path/to/workspace")  # read-only
save_arft_coverage("/path/to/workspace")             # writes arft_coverage.json
```

This report answers “do our local artifacts expose enough evidence to audit
this pattern?” with `covered`, `partial`, or `unassessed`; it does not infer
that a failure happened and is not comparable to the published benchmark.
Malformed JSON/JSONL contracts are retained as redacted `input_errors` instead
of silently becoming an empty evidence set, so an auditor can distinguish
“missing signal” from “unreadable artifact”.
The official AutoResearchEval-style evaluation needs complete trajectories,
task manifests, evaluator outputs, repeated runs, and an independent judge.

CI may use a generous threshold to detect severe first-run regressions. Local
runtime is descriptive and should not be compared across unreported hardware.
