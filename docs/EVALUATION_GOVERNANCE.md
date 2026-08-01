# Independent Scientific Evaluation Governance

Scientific claims must not be promoted by the same authority that produced
them. XScientist therefore separates research, verification, benchmark
custody, and final approval, then binds every decision to immutable content
hashes.

## Authority model

An `evaluation_charter.json` locks four mutually exclusive roles:

- `researcher`: produces the candidate claim or theory;
- `verifier`: evaluates frozen candidates; at least two identities are needed;
- `benchmark_custodian`: controls tasks and answer keys without exposing them;
- `approver`: a `human:` identity that authorizes high-confidence promotion.

The charter inherits the project's science constitution. Its policy is
code-anchored, so changing thresholds, hard gates, role assignments, or policy
hashes invalidates the artifact.

## Evaluation layers

Promotion to `robust` or `canonical` requires the same candidate hash to pass:

1. **Sealed** evaluation with opaque task hashes and a custodian attestation.
2. **Prospective** evaluation frozen before a future observation, with a
   resolution attestation and an enforced not-before timestamp.
3. **External** evaluation under an independently hashed protocol and external
   custody.

Public benchmarks may guide exploration, but cannot authorize promotion.
Benchmark manifests contain hashes and custody metadata only; raw tasks and
answers are forbidden.

## Fixed decision criteria

Every layer applies integrity, safety, and reproducibility hard gates plus
fixed thresholds for objective quality, worst-domain quality, reproducibility,
false-discovery rate, calibration error, and information gain. The final
decision also requires:

- all required layers to pass;
- at least two distinct verifier identities;
- an external verifier distinct from internal verifiers;
- producer, verifier, custodian, and approver separation;
- explicit epistemic node IDs covered by the approval.

Runs and decisions are reconstructed during validation. Rewriting a result,
criterion, threshold, scope, or verdict after the fact remains invalid even if
an attacker recomputes the outer hash.

## Epistemic promotion gate

`advance_epistemic_node(..., to_status="robust")` and promotion to `canonical`
require an approved evaluation report scoped to that node. Stage standards
apply the same check whenever the graph already contains robust or canonical
knowledge. Early speculative work remains lightweight; the heavier protocol is
activated only at consequential promotion boundaries.

## Artifacts

- `evaluation_charter.json`: locked authority and policy definition;
- `evaluation_benchmarks.jsonl`: append-only opaque benchmark manifests;
- `evaluation_report.json`: hash-bound runs, criteria, decision, and scope.

The construction and validation API is in
`ai_scientist.utils.evaluation_governance`.
