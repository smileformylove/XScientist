# Opportunity funnel (FAR-inspired)

XScientist now exposes a domain-agnostic, auditable analogue of the
Find → Attempt → Recommend cascade described in [FAR](https://arxiv.org/abs/2608.16977).
The FAR paper and [its research repository](https://github.com/zeyu-zheng/FAR)
remain external references; XScientist does not claim to have rerun their
combinatorics pilot or reproduced its counts.

## Contract

The funnel is recorded with existing Research VCS kinds and explicit
`protocol_kind` values, so old semantic-profile digests stay compatible:

1. `save_research_direction` records a human- or investigator-specified
   research direction (`research_goal`).
2. `save_opportunity_pool` records the complete candidate set (`question`).
   Every candidate remains visible even when its source status is unknown or
   its allocation probability is missing.
   The pool is a bounded, caller-supplied extraction; XScientist does not claim
   to implement FAR's corpus-wide Label/Extract/Check importer.
3. `save_opportunity_attempt` records one of `known`, `new`, `fix`, or `none`.
   A negative result is still a durable attempt; it is never silently dropped.
   The API records an external runner's result and evidence; it does not invoke
   FAR's solver or import its work directories automatically.
4. `save_opportunity_judgment` records a declared provenance-disjoint evaluator
   decision (`pass`, `fail`, or `known`). The receipt says actor disjointness
   was checked, not that identity or scientific correctness was verified.
5. `save_opportunity_grade` records `known`, `minor`, or `substantial` as a
   completed review. It does not promote a claim or create a publication score.
6. `save_opportunity_allocation` is fail-closed: it accepts only a complete
   pool whose every candidate has `source_status=open`. The pure ranking
   helper can still inspect provisional rows, but it never hides them.

Judging and grading have explicit stage gates. A judgment normally targets an
attempt with outcome `new`; a grade normally follows a `pass` or `known`
judgment. A retrospective exception requires `--allow-stage-override` and a
non-empty `--override-reason`, which is included in the immutable hash.

The allocation layer accepts continuous `[0, 1]` `difficulty`, `importance`,
`expected_success_probability`, `expected_artifact_probability`, and
`expected_importance` fields. Its default
`probability_semantics=conditional_artifact_given_success` treats the first
probability as accepted-attempt probability and the second as the conditional
publishable probability. If the conditional factor is omitted, a neutral
`1.0` is used only as an explicit, recorded assumption. With
`probability_semantics=joint_artifact_probability`,
`expected_artifact_probability` is already the joint probability and is never
multiplied by success again; if it is missing, the row is unselected.
The default
`calibration_status=declared_inputs_not_calibrated` is intentional: the module
does not fit a success model, leak post-attempt outcomes into pre-attempt
ranking, or transfer FAR's domain-specific AUC/results to another field.

`source_object_ids` may bind candidates to local `search_receipt`,
`source_snapshot`, `passage_evidence`, or `question` objects. External URLs or
labels can remain in `source_refs`, but then `lineage_complete` is false. Raw
responses, credentials, oversized text, and deeply nested reference objects are
rejected before persistence. Attempts, judgments, and grades can likewise bind
canonical `evidence_object_ids`; those bindings become `derived_from` relations
instead of an uncheckable prose citation. `source_status=open` means that no
credible resolution was found in the recorded search evidence at that time; it
is not a proof of global novelty or an assertion that humans have not solved it.
The current contract records one judgment per review object and flags duplicate
judgments as incomplete; it does not implement FAR's three-judge all-pass rule.
Consequently, a local `pass` is not numerically comparable to FAR's reported
pilot counts.
For audit completeness, an attempt may also be recorded retrospectively for a
pool row whose status is not `open`; such a row cannot enter a locked allocation
plan, and this broader recording surface is not the same as FAR's open-only
attempt stage.

## Python API

```python
from xscientist import (
    ResearchRepository,
    save_research_direction,
    save_opportunity_pool,
    save_opportunity_attempt,
    save_opportunity_judgment,
    save_opportunity_grade,
    save_opportunity_allocation,
    inspect_opportunity_funnel,
)

direction = save_research_direction(
    repo,
    direction_id="mechanism-search-v1",
    statement="Which mechanism explains the held-out anomaly?",
    objective="Produce a falsifiable and reproducible result.",
)
pool = save_opportunity_pool(
    repo,
    direction_id=direction["object"].object_id,
    candidates=[
        {
            "candidate_id": "q-001",
            "question": "Does the proposed mechanism survive a held-out test?",
            "source_object_ids": ["rso-0123456789abcdef"],
            "source_status": "open",
            "expected_success_probability": 0.4,
            "expected_importance": 0.8,
        }
    ],
)
```

The CLI exposes the same surface under `xscientist research opportunity`:

```text
xscientist research opportunity direction DIRECTION STATEMENT OBJECTIVE
xscientist research opportunity pool DIRECTION_ID candidates.json
xscientist research opportunity attempt POOL_ID CANDIDATE_ID none "No result"
xscientist research opportunity inspect POOL_ID --json
```

Use `--no-commit` for a batch and create one explicit checkpoint after review;
the normal API defaults to the repository's append-only checkpoint behavior.

## Scope boundary

This is a process and allocation contract, not a benchmark score. It does not
claim autonomous mathematical discovery, human parity, publication readiness,
or superiority over FAR or any other system. A candidate becomes a hypothesis
or experiment design only through an explicit downstream user decision and the
existing XScientist gates.
