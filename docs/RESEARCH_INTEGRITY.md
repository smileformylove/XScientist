# Research Integrity Protocol

XScientist separates **exploratory search** from **confirmatory evidence**.
Exploration may branch, fail, and change direction. A result cannot be promoted
to a manuscript claim until it passes the immutable verification path below.

## Evidence lifecycle

1. `idea_cards.json` records the proposed hypothesis and falsifiers.
2. `research_plan.json` declares exploratory tasks and the claim-promotion policy.
3. `preregistration.json` fixes the hypotheses, datasets, metrics, baselines,
   statistical family, stopping rule, controls, and holdout split hashes.
4. `experiment_registry.jsonl` records exploratory, confirmatory, and
   independent-reproduction runs separately.
5. Research Git stores the corresponding typed `experiment_attempt` objects,
   their origin checkpoints, registry-row bindings, and immutable dispositions
   for unsuccessful attempts.
6. `verification_report.json` checks protocol fidelity, split hashes,
   deterministic metric provenance, seed coverage, blind holdout access,
   verifier independence, clean-room execution, reproduction linkage, and the
   live structured-trajectory closure.
7. `research review` is a local advisory surface: locally declared identities,
   including a requested `pass`, always produce an effective hold gate and
   cannot mint a verified claim.
8. Only a host-recomputed report with `status: verified`,
   `claim_promotion_allowed: true`, and a separate externally trusted Ed25519
   authority receipt may authorize confirmatory manuscript claims.

The preregistration content hash excludes timestamps and mutable deviation
records, but covers the scientific protocol. Editing a locked hypothesis,
outcome, analysis rule, or data policy invalidates `registration_hash`.

## Submission-preparation evidence gate

The high-quality writing pass treats these lifecycle artifacts and the live
Research Git trajectory closure as one deterministic gate. It requires a
locked preregistration, completed registry records for every registered task,
at least the preregistered seed count, artifact-bound numeric results with
inferential statistics, content-addressed metric lineage, a
task-to-metric-to-claim path in `claim_evidence_graph.json`, and a verified
clean-room report. Missing evidence is reported as a blocker; it cannot be
compensated for by prose quality, keywords, figures, or an LLM review score.
Passing this gate describes preparation under the recorded local contract; it
does not estimate or promise conference acceptance.

Numbers are extracted from experiment result artifacts only. When a registry is
present, publication-level numeric matching accepts only `completed` or
`verified` records with `study_phase: confirmatory`; cached values and
exploratory runs remain diagnostic data. Every detected manuscript claim must
also resolve to a task → metric → claim path. A `\claimref[claim=<claim_id>]{<node_id>}`
marker can carry the paper claim ID and the ARA exploration-node ID together;
the writer must never invent either ID. Numbers that appear for the first time
in the manuscript remain narrative text and cannot establish a confirmed
result. Runs without the gate are labelled `exploratory_draft` or
`manuscript_draft`, never `submission_ready`.

## Locking a top-venue confirmation campaign

Planning initially writes a draft. For a NeurIPS/ICML campaign, first resolve
every placeholder dataset, metric and baseline, prepare the read-only empirical
snapshot in `00_config/data_manifest.json`, and hash every task's final split.
Then use the host-owned transition instead of letting a planning or execution
model authorize its own evidence:

```bash
xscientist research confirm \
  --paper-dir PAPER_DIR \
  --registered-by recorder:RESEARCHER \
  --split task_0=sha256:<64-hex> \
  --split task_1=sha256:<64-hex> \
  --split task_2=sha256:<64-hex>
```

`--registered-by` is self-reported recorder provenance only. It records who or
what invoked the host transition; it does not authenticate a person, grant a
`human:` principal, establish evaluator independence, or confer verifier
authority. Independent authority is established only by the separately signed,
externally trusted verifier receipt.

The command verifies the current empirical manifest and Research VCS state,
locks the generated multi-task primary/ablation/robustness plan, commits the
registration transition, writes `confirmatory_queue.json`, and checkpoints the
paper mirrors. The queue gives one auditable command per task. Post-lock changes
remain visible as deviations, but the current protocol has no independently
authenticated pre-unblinding deviation-approval object. A mutable
`approved_before_unblinding` flag therefore cannot authorize publication.
Lower-level Python locking remains useful for non-publication protocols, but it
does not by itself create the host/VCS attestation required by the NeurIPS/ICML
gate.

## Bind every registry row to the Research Git trajectory

The structured trajectory is the publication-facing projection of the typed
Research Git/Research VCS object and checkpoint model, not an optional text log.
For every confirmatory and independent-reproduction registry row, bind the
exact immutable row to its matching typed `experiment_attempt`:

```bash
xscientist research trajectory-bind \
  --paper-dir PAPER_DIR \
  --record-id REGISTRY_RECORD_ID \
  --attempt ATTEMPT_OBJECT_ID
```

The host verifies the full contract—task, phase, preregistration, frozen state,
data snapshot, protocol, evidence role, producer, terminal state, configuration
and result artifacts—and records the attempt's content hash, origin commit, and
checkpoint hash. One registry row and one attempt may participate in exactly
one binding.

Failed, timed-out, and cancelled attempts remain in both histories. Each needs
one immutable publication disposition; a preserved terminal negative is the
minimal example:

```bash
xscientist research attempt-disposition \
  --paper-dir PAPER_DIR \
  --record-id REGISTRY_RECORD_ID \
  --disposition terminal_negative \
  --reason "The attempt hit its preregistered terminal failure condition; artifacts are retained." \
  --negative-result-artifact PATH/TO/RESULT.json \
  --negative-result-evidence EVIDENCE_OBJECT_ID
```

`technical_failure_retried` additionally requires `--retry-record-id` naming a
completed, same-task, trajectory-bound retry. Only that disposition and a
`terminal_negative` with failure class `scientific_negative_result`, non-empty
matching attempt/registry artifact hashes, a bounded host-rehashed repository
file, and a metric-bearing evidence assessment derived from that exact attempt
resolve the local trajectory blocker. These bindings are covered by the
trajectory hash and therefore by the final external Ed25519 authority receipt;
a self-reported preservation boolean is never evidence. `approved_deviation`
and `excluded_with_reason` retain the decision for audit, but remain publication
blockers. Running attempts must first reach a terminal state. Missing,
duplicate, extra, hidden, unbound, or non-resolving attempts are hard blockers
for publication readiness.

## Confirmatory experiment records

Confirmatory records must add these fields to the normal experiment registry:

- `study_phase: confirmatory`
- `preregistration_id`
- `protocol_fidelity_hash` equal to the canonical hash of the locked outcome,
  analysis plan, data policy, and controls for that task
- `dataset_split_hash`
- `data_manifest_hash` and `data_snapshot_id`, matching the locked empirical
  snapshot in the preregistration and host manifest
- `configuration_hash`, recomputed from the exact structured run configuration
- for ablation/robustness, a hash-valid `transformation_manifest` that binds the
  primary configuration, resulting configuration and actual changed factors
- `metric_provenance: deterministic_verified`
- `evaluator_input_hash` and `evaluator_result_hash`
- `verification_recomputed: true`
- `verification_command` (or a reproducible `verification_method`)
- `verification_metric_hash` equal to the canonical `result_summary` hash
- `verification_output_hash` equal to bytes read from a concrete result artifact
  inside the study root (a hash-only self-report is never sufficient)
- `artifacts.input`/`artifacts.result` (or the evaluator-specific aliases) whose
  bytes match `evaluator_input_hash`/`evaluator_result_hash`
- `holdout_access: verifier_only`
- `producer_id`

The clean-room report must contain every required criterion, with no omitted or
failed criterion, and must list exactly the completed confirmatory and
reproduction `record_id`s. It also binds the same empirical manifest and
snapshot IDs. The publication gate rebuilds this report from the current
registry and compares the result; editing a report by hand is not a substitute
for rerunning verification. A final external Ed25519 authority receipt binds
that report, its producers, the current registry/evidence/manuscript hashes and
the same empirical snapshot through a trust root outside the workspace.
Duplicate/empty task IDs or record IDs, unregistered tasks, generic directories
without a result file, and self-reported status text are not evidence.

Independent reproduction records additionally use
`independent_reproduction: true`, `replicates_record_id`, `verifier_id`, and
`clean_room: true`.

## Execution isolation

Packaged BFTS configurations now use `backend: auto` with
`require_isolation: true`. Docker and the pinned executor image must be
available; otherwise autonomous execution fails closed. A local developer may
explicitly opt into `backend: process` with `require_isolation: false`, but
those runs are non-isolated and should not be used as confirmatory evidence.

## Operating rule

Treat `verified` as an evidence state, not a writing score. LLM reviewer
approval, attractive plots, or a high novelty score cannot override a failed
integrity criterion.
