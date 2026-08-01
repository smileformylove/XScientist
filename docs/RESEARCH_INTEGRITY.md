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
- `dataset_split_hash`
- `metric_provenance: deterministic_verified`
- `evaluator_input_hash` and `evaluator_result_hash`
- `holdout_access: verifier_only`
- `producer_id`

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
