# Benchmarks and process audits

The public benchmark measures a usability property: can a clean local process
go from an empty directory to inspectable, scientifically honest status without
credentials, network access, or model cost?

```bash
xscientist benchmark first-run --json
xscientist benchmark first-run --max-seconds 30
```

It creates the deterministic Autopilot fixture, builds the Research VCS DAG,
reads status, records duration and structural counts, and deletes its temporary
workspace. `--workspace DIR` retains a named workspace for inspection.

This is not a model-quality benchmark. It must not be used to claim scientific
performance, autonomous discovery quality, or provider speed. Those require
separate registered datasets, budgets, evidence, and evaluation authority.

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

For a two-line conformance fixture, the human surface is intentionally
process-shaped rather than score-shaped:

```text
Process: 3 visible commits, 2/2 branches, 2 typed artifacts
  branch alternative-1 (diverged_or_behind, 3 commits)
  branch current (current, 2 commits)
  commit … experiment: checkpoint:experiment [alternative-1]
  commit … ideation: checkpoint:ideation [alternative-1,current]
  Fair branch comparison: NOT VERIFIED (unverified: same_task_slice, same_budget, same_evaluator, same_base)
```

The ellipses stand for short commit hashes. This fixture is a conformance
example, not a scientific result or a claim that the two lines received equal
model budgets.

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

`official_comparable` is always `false`, `gold_fields_used` is always `false`,
and the pilot reports zero rollouts, provider calls, network use, and model
cost. A held/rejected gate is reported as contained review debt; it is not
mislabelled as the paper's F.4 “shipped despite awareness” failure. The pilot
therefore measures observability and governance, not autonomous scientific
ability.

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
