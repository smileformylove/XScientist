# XScientist Science Constitution

The science constitution is the non-negotiable policy boundary for autonomous
research. Projects inherit the same code-anchored core policy. Project-specific
constraints may tighten that policy, but autonomous workflows cannot weaken or
replace it.

## Priority order

```text
truth -> safety -> novelty -> impact -> throughput
```

The core principles require separation of exploration and confirmation,
traceable evidence, permanent negative results, independent authority,
explicit uncertainty, risk-bounded autonomy, and immutable audit history.
Awards, citations, publication counts, reviewer scores, and self-evaluation
scores are prohibited as direct optimization objectives.

## Integrity model

`science_constitution.json` contains the exact core policy and its canonical
SHA-256 hash. Validation compares the stored policy to the version anchored in
the installed XScientist code, so recomputing a hash after weakening a rule
does not make the artifact valid.

The following assets are protected from autonomous mutation:

- the science constitution and epistemic history;
- raw evidence and sealed benchmarks;
- evaluation hard gates and identity rules;
- safety and authorization boundaries.

## Amendments

An autonomous process may append an amendment proposal, but the proposal has
`automatic_application_allowed: false` and does not alter the locked core.
Actual amendment requires a new policy version, a public rationale, an impact
assessment, external audit, and at least two independent human approvers.

```python
from ai_scientist.utils.science_constitution import (
    build_science_constitution,
    propose_science_constitution_amendment,
)

constitution = build_science_constitution(project_name="grand-discovery")
proposal = propose_science_constitution_amendment(
    constitution,
    proposed_by="human:principal-investigator",
    rationale="Add a stricter replication requirement.",
    impact_assessment="No existing principle is weakened.",
    proposed_changes={"minimum_external_labs": 2},
)
```

The returned artifact contains a proposal only. XScientist intentionally has
no autonomous `apply_amendment` operation.
