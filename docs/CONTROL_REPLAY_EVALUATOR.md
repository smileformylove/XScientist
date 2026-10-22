# Realized-Trace Metric Recomputer

`ai_scientist.utils.control_evaluator` is a bounded host-side artifact
recomputer for the matched latent/non-latent protocol v2.  Despite the
historical module name, it is **not** an environment replay or environment
verifier: it consumes submitter-provided action/reward traces and does not
simulate transitions, inspect hidden state, or establish that rewards came
from a declared world.

## Current contract

The `xscientist.control-replay.v1` JSON payload contains fixed-horizon traces
for one or more arms and explicitly aligned oracle and zero-data references.
The report currently recomputes only:

* `true_environment_value`: mean undiscounted realized reward per episode;
* `oracle_normalized_regret`: `(V_oracle - V_arm) / max(V_oracle - V_zero_data, 1e-6)`.

It also emits the diagnostic
`diagnostics.aligned_step_reward_gap`.  This is algebraically
`(V_reference - V_arm) / H` for aligned equal-horizon traces.  Its per-action
bins are descriptive bookkeeping, not a counterfactual action-value
estimate.  The protocol's `action_occupancy_weighted_regret` is explicitly
listed as unsupported; no three-endpoint control-regret binding is claimed.

## Fail-closed and provenance boundary

Unknown fields, duplicate JSON keys, empty/duplicate IDs, identical oracle
and zero-data reference IDs, mismatched episode order, invalid horizon or
action domains, non-finite rewards, and non-finite aggregates are rejected.
The parser and object validator enforce 32 MiB encoded input, 100,000
container items, depth 16, and 4,096 characters per string.  Optional
`expected_horizon`, `expected_actions`, and `expected_protocol_id` bindings
are type-checked and compared exactly.  Canonical JSON binds the input and
report with SHA-256.  The report includes the evaluator source hash and
declares `verification_scope: artifact_internal_consistency`,
`ground_truth_authority: submitter_asserted_unverified`,
`environment_provenance_verified: false`, `reference_optimality_verified:
false`, `comparison_ready: false`, and all experiment/training/claim
authority flags false.

These controls establish deterministic artifact recomputation, not signed
provenance.  They do not provide a signed provenance receipt, verify actual
environment execution, prove oracle optimality, establish independent
holdout access, compare implementations, or authorize a scientific run.

## Correction to the first pass

The first-pass implementation and audit text overstated coverage by using
the replay label and by presenting all three protocol control endpoints as
implemented.  This was corrected before integration: the current artifact
is a trace-level measurement utility, the occupancy regret endpoint is
unsupported, and the remaining environment/reward provenance and actual
replay obligations stay open.  Protocol v2 therefore remains
execution-blocked pending its independent data, runner, comparator, review,
signer, and admission requirements.
