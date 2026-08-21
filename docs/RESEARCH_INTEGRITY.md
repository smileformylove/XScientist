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
5. `verification_report.json` checks protocol fidelity, split hashes,
   deterministic metric provenance, seed coverage, blind holdout access,
   verifier independence, clean-room execution, and reproduction linkage.
6. Only a report with `status: verified` and
   `claim_promotion_allowed: true` may authorize confirmatory manuscript claims.

The preregistration content hash excludes timestamps and mutable deviation
records, but covers the scientific protocol. Editing a locked hypothesis,
outcome, analysis rule, or data policy invalidates `registration_hash`.

## Publication quality gate

The high-quality writing pass now treats the four artifacts above as one
deterministic gate. It requires a locked preregistration, completed registry
records for every registered task, at least the preregistered seed count,
artifact-bound numeric results with inferential statistics, content-addressed
metric lineage, a task-to-metric-to-claim path in `claim_evidence_graph.json`,
and a verified clean-room report. Missing evidence is reported as a blocker;
it cannot be compensated for by prose quality, keywords, figures, or an LLM
review score.

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

## Locking a preregistration

Planning initially writes a draft. Resolve placeholder datasets, metrics, and
baselines, hash every task's final data split, then lock it:

```python
from ai_scientist.utils.research_integrity import (
    lock_preregistration,
    save_preregistration,
)

locked = lock_preregistration(
    draft,
    split_hashes={"task_0": "sha256:<64 hex characters>"},
    registered_by="planning-agent-v2",
)
save_preregistration(project_root, locked, producer="planning-agent-v2")
```

Post-lock changes must be recorded as deviations. A deviation is not accepted
by the verification gate unless it was approved before holdout unblinding.

## Confirmatory experiment records

Confirmatory records must add these fields to the normal experiment registry:

- `study_phase: confirmatory`
- `preregistration_id`
- `protocol_fidelity_hash` equal to the canonical hash of the locked outcome,
  analysis plan, data policy, and controls for that task
- `dataset_split_hash`
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
reproduction `record_id`s. The publication gate rebuilds this report from the
current registry and compares the result; editing a report by hand is not a
substitute for rerunning verification.
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
