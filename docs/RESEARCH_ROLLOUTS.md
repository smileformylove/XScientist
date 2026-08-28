# Research-policy rollouts

Faraday's useful architectural lesson is not a parameter-count comparison. It
is that a research policy should choose experiments and delegate implementation
to a stronger coding tool, while a task-specific evaluator checks the result.
XScientist now records that contract as a metadata-only `research_rollout`
object.

## What is recorded

- a hash-bound task and split (`train`, `test`, `holdout`, or `external`);
- an optional harness/resource/evaluator boundary (`comparison_boundary`) for
  fair tool-swap checks beyond the task hash;
- a five-dimension rubric: result fidelity, claim support, implementation
  fidelity, resource efficiency, and scientific integrity;
- tool calls with provider/model fingerprints, input/output hashes, budgets,
  decision type, and outcome — never prompts, stdout, credentials, or raw
  responses;
- turn metadata and observational post-hoc positive reward-delta credit;
- a deterministic `strategy_budget_summary` with decision ownership, contiguous
  budget accounting, failure/recovery observations, and a next-action hint;
- zero or more evaluator samples with per-dimension scores, mean, and
  disagreement;
- explicit `quality_claim_allowed=false` and `causal_claim_allowed=false`.

The evaluator summary is a measurement record, not a ground-truth claim. A
separate independent gate is still required before promoting a scientific
claim. Missing reward traces remain missing; the builder never imputes a
credit assignment.

## What the Faraday result changes for this project

The [Faraday paper](https://arxiv.org/abs/2608.13331) is a trained outer
research policy (Qwen3.6-27B) that delegates coding to a frontier tool and is
post-trained on the authors' Replica figure-replication tasks. XScientist is a
protocol and local audit SDK: it does not contain Faraday weights, run the
Replica harness, call Codex, or report the paper's score. The useful boundary
is therefore the *contract*, not a claimed model comparison:

| Faraday ingredient | XScientist implementation boundary |
| --- | --- |
| policy chooses the next research action | `tool_delegations` plus `strategy_budget_summary` record role, decision, sequence, budget, and recovery without storing prompts or stdout |
| coding agent executes the experiment | successful `coding_executor` calls must expose an output hash; the audit binds that artifact to the evaluator evidence |
| task-specific five-dimension judge | locked rubric and per-sample evaluation rows; scores remain observational and judge-reference-only |
| multi-sample/turn credit | bounded evaluator samples and post-hoc turn credit are recorded; no causal attribution or RL update is inferred |
| human-calibrated, independent checking | the evaluator signs a receipt binding its principal, all observed producer identities, and inspected artifact hashes; `rollout-audit` verifies it against a local trust store |

This makes a missing strategy step, unaccounted budget, unreviewed executor
artifact, or absent independent evaluator a visible blocker rather than an
implicit success. It is an audit/training-data contract inspired by Faraday,
not a Replica importer, solver, three-judge reproduction, or local quality
result.

## CLI

Create a JSON file such as `episode.json` containing `task_id`, a full
`task_hash`, `time_budget_seconds`, and optional `tool_delegations`, `turns`,
and `evaluations`, then record it with:

```bash
xscientist research rollout episode.json \
  --repo ./first-study --json > rollout.json
```

The command is offline and idempotent. It stores only a redacted, content-
addressed Research VCS object and creates the usual experiment checkpoint. Its
JSON wrapper includes the canonical payload under `rollout`, so the captured
file can be passed directly to `rollout-audit` without extracting or rewriting
the payload.

To audit a saved raw payload (or the JSON wrapper printed by the command),
provide the hashes known to the local evidence index:

```bash
xscientist research rollout-audit rollout.json \
  --evidence-hash sha256:... \
  --trust-store trust-store.json --json
```

The audit is fail-closed for completed episodes. It checks schema and all
content hashes, task/rubric binding, budget boundaries, the strategy summary,
successful executor-artifact binding, and an actor-disjoint evaluator receipt.
It emits only bounded blockers/warnings and always keeps
`quality_claim_allowed=false` and `causal_claim_allowed=false`. Without an
evidence resolver, hash syntax can be checked but artifact existence is not
verified. Without a trusted evaluator attestation, a caller-declared
`identity_verified=true` remains observational. Either omission prevents a
completed rollout from becoming verification-eligible.

For a `completed` rollout, budget and recovery are audit gates rather than
dashboard hints. Every recorded call needs a before/after budget boundary, the
first boundary must account for the declared budget, adjacent calls must form a
contiguous chain, and the result must remain within the declared boundary. A
failed or timed-out call marked `follow_up_required=true` must be followed by a
successful `repair`/`delegate` response or an explicit terminal `stop`. A
failed repair attempt does not count as recovery; absent a later successful
response or stop, the completed rollout remains blocked. A `stop` followed by
any later tool call is non-terminal and also blocks verification.

## Signed evaluator receipts

Use the public builders to create the exact canonical binding, sign it with the
existing attestation protocol, and wrap the envelope in a receipt:

```python
from ai_scientist.protocol.attestation import sign_attestation
from xscientist import (
    INDEPENDENCE_ATTESTATION_PURPOSE,
    build_independence_attestation_payload,
    build_independence_receipt,
)

binding = build_independence_attestation_payload(
    evaluator_id="judge-independent",
    evaluator_identity="human:reviewer-42",
    target_hashes=[executor_output_hash],
    producer_actor_ids=all_rollout_producer_ids,
)
attestation = sign_attestation(
    binding,
    purpose=INDEPENDENCE_ATTESTATION_PURPOSE,
    identity="human:reviewer-42",
    key_id="reviewer-42-ed25519",
    algorithm="ed25519",
    key=private_key_pem,
)
receipt = build_independence_receipt(
    evaluator_id="judge-independent",
    evaluator_identity="human:reviewer-42",
    target_hashes=[executor_output_hash],
    producer_actor_ids=all_rollout_producer_ids,
    attestation=attestation,
)
```

The local trust store is keyed by `key_id` and supplies the expected identity,
algorithm, public key, and optional revocation state. Prefer Ed25519 for shared
or durable records. HMAC is supported for local workflows, but an HMAC trust
store contains a secret and must not be committed or printed. The receipt
builder checks shape and content binding only; trust and optional freshness are
checked by `rollout-audit` (`--max-attestation-age-seconds` when required).

## Tool swaps and comparison boundaries

`assess_tool_swap_compatibility(reference, candidate)` checks task hash, rubric
hash, split, and time budget. If both reports provide `comparison_boundary`, it
also checks the harness, resource fingerprint, evaluator protocol, starting
artifact, network policy, and seed policy. An eligible boundary is only a
prerequisite for a controlled comparison; it does not say that one model or
tool is better.

Use strict mode when eligibility must include the rollout audits themselves:

```python
from xscientist import assess_tool_swap_compatibility

comparison = assess_tool_swap_compatibility(
    reference,
    candidate,
    strict=True,
    audit_evidence_hashes=all_reference_and_candidate_evidence_hashes,
    audit_trust_store=local_trust_store,
    max_attestation_age_seconds=3600,  # optional
)
```

`audit_evidence_hashes` is the union resolver for both rollouts, and
`audit_trust_store` must verify both evaluator receipts. Strict mode fails
closed when either input is absent, when either rollout is not
verification-ready, or when the rollout/tool signature did not actually
change. Passing this check still grants neither a quality nor a causal claim.

The current implementation does not reproduce Faraday's RL training,
three-judge protocol, coding-agent provider, benchmark task set, or reported
scores. The design is an XScientist audit/training-data contract inspired by
the [Faraday paper](https://arxiv.org/abs/2608.13331), not a local replication.
