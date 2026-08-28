# Belief-context projections

XScientist borrows one important idea from the
[Belief Context Graph (BCG)](https://github.com/bigai-nlco/belief-context-graph):
an agent needs more than retrieval. It needs explicit context about support,
challenge, provenance, temporal validity, and uncertainty before it acts.

The implementation boundary is deliberately different. XScientist does not
create a second mutable memory database or inject a probabilistic belief graph
into a model prompt. It derives a bounded, deterministic, metadata-only
projection from one immutable Research VCS source closure. Research VCS remains
the single source of truth.

## What was borrowed, and what was not

| Concern | BCG inspiration | XScientist contract |
| --- | --- | --- |
| Agent context | Make beliefs, evidence, conflict, and time explicit | Derive a read-only decision view from Research VCS objects at one ref |
| State | Maintain belief-oriented graph context | Emit deterministic ordinal evidence states |
| Provenance | Link a belief to its originating context | Bind every signal to Research Object IDs, content hashes, relations, and source-family hashes |
| Uncertainty | Use confidence to decide whether action should be blocked | Use named blockers and conservative decision postures; never emit a calibrated probability |
| Lifecycle | Track changing beliefs over time | Evaluate `valid_until`, invalidation, retraction, and supersession at a declared logical time |
| Agent integration | Supply a belief snapshot as model context | Include a hash-bound projection in the existing Research VCS context snapshot |

XScientist does **not** reproduce BCG's graph-construction backends, confidence
formula, HTTP service, reference agent, prompt-injection strategy, benchmark
harness, or benchmark results. BCG's published accuracy and token-cost numbers
do not transfer to XScientist. Any performance claim for XScientist still
requires a separately declared harness, model, dataset, resource boundary, and
local result artifact.

## Research VCS is the single source of truth

`build_belief_context_projection(...)` accepts Research Objects already present
in the exact context closure and produces a derived view. It does not write a
belief row, update a confidence field, or maintain hidden state. Rebuilding the
same canonical closure with the same logical time and limits produces the same
`projection_hash`, regardless of input order.

The projection binds:

- canonical target and source-object IDs;
- every source object's content hash through `source_closure_hash`;
- the logical `as_of` boundary and how it was selected;
- the support, challenge, lineage, invalidation, and supersession relations;
- graph limits, truncation, cycles, conflicts, blockers, and warnings;
- the complete projection through `projection_hash`.

The projection may guide a decision, but it cannot promote a scientific claim.
The top-level and per-target `scientific_promotion_allowed` values are always
`false`; `quality_claim_allowed` and `causal_claim_allowed` are also always
`false`. Existing Research VCS closure, independent evaluation, and promotion
gates retain authority. An ordinal state is therefore context for a later
gate, never a sufficient or sole promotion gate by itself.

## Ordinal semantics, not calibrated confidence

The fixed semantics identifier is:

```text
deterministic_ordinal_evidence_state_not_calibrated_probability
```

The states are ordered decision labels, not probabilities, Bayesian posteriors,
or measures of scientific truth:

| State | Observed condition | Default posture |
| --- | --- | --- |
| `unassessed` | No active supporting or challenging signal is observed | `collect_discriminating_evidence` |
| `supported` | Active support exists, but fewer than two distinct source families are observed | Usually `seek_independent_review` |
| `corroborated` | Active support resolves to at least two distinct source families | `review_with_scientific_gate` when independent authority is also observed |
| `contested` | Active support and active challenge both exist | `investigate_conflict` |
| `challenged` | Active challenge exists, or the target is in a terminal negative state | `collect_discriminating_evidence` |
| `stale` | Support exists but is no longer active, or the target has expired | `refresh_evidence` |
| `superseded` | The target is superseded or invalidated | No action based on the old target |

`corroborated` does not mean verified. It records a source-family count under a
deterministic rule. Independent authority is checked separately and is observed
only for `human` or `independent_evaluator` actors. Even a corroborated target
remains subject to the scientific gate.

## Same-origin deduplication

Several passages derived from one paper are not several independent sources.
The projection follows lineage relations such as `depends_on`, `derived_from`,
`quotes`, `observes`, and `tested_by` to source roots, then counts unique
source-family hashes.

Support itself remains type- and relation-bound. Explicit `supports`,
`qualified_supports`, `replicates`, and `reproduces` relations are support
signals. In addition, a `claim` that `depends_on` an `evidence`,
`passage_evidence`, `inference`, or `evidence_synthesis` object receives a
derived `depends_on_evidence` support binding, so the normal claim/evidence
shape is not lost. A generic `depends_on` edge to any other object, or an
arbitrary lineage relation, is **not** silently reclassified as support.

For `source_snapshot` objects, the first declared canonical identity among DOI,
PMID, arXiv ID, URL, and source content hash is used. When no source snapshot is
reachable, a declared producer actor can provide a fallback family identity.
Raw identifiers are not copied into the projection; only deterministic hashes
are emitted.

This prevents repeated passages, summaries, or derived objects from inflating
`independent_support_source_count`. It does not prove legal, institutional, or
experimental independence. Missing source roots remain visible through
`independence_observed=false`, and truncated lineage becomes an action blocker.

## Temporal validity, invalidation, and conflict

Use a timezone-aware `--as-of` value when the decision must be replayed at an
explicit boundary. Without it, the projection uses the latest valid
`created_at` timestamp in the observed source closure. If neither is available,
the logical time is marked `unavailable`; it is never replaced with the wall
clock.

An explicit historical boundary is a true cutoff: evidence created after
`as_of` is marked `not_yet_observed` and cannot support or challenge the target.
Retraction, invalidation, and supersession objects created after that boundary
are also ignored for that historical projection, so a later event cannot
rewrite the earlier decision context. A target created after the boundary
makes the projection incomplete rather than being treated as if it existed.

Signals can declare `valid_until`. Expired signals and signals invalidated by a
retraction, withdrawal, invalidation, or superseding relation cannot provide
active support. A malformed temporal declaration is marked `invalid` and is
also excluded from active evidence. Invalidation is followed through bounded
lineage. An expired target becomes `stale`; an invalidated or superseded target
becomes `superseded`.

Support and challenge are retained together. They are not averaged away. A
target with both becomes `contested`, receives an unresolved, deterministic
`conflict_id`, and is directed to `investigate_conflict`. The conflict remains
visible until the underlying Research VCS record changes.

These are fail-closed decision rules: stale, superseded, challenged, contested,
or unassessed targets receive action blockers, while scientific promotion
remains disabled for every state.

## Cycles and bounded computation

Public projection work is bounded by hard limits:

- at most 1,024 nodes;
- at most 8,192 observed relations;
- at most 8 lineage hops when resolving a source family.

Callers may choose lower positive `max_nodes` and `max_relations` values, but
cannot exceed the hard maxima. Node or relation truncation, an endpoint outside
the observed graph, a duplicate or invalid object identity, or a lineage cycle
makes the projection incomplete. In that case `complete=false`,
`decision_context_usable=false`, and the containing Research VCS context is also
blocked.

The CLI `--budget` controls the human-readable semantic working set in the
containing context snapshot. It never trims the immutable IDs and hashes that
form the hard source closure. If required decision semantics cannot fit, the
context fails closed instead of silently dropping inconvenient evidence.

## Audit boundary

`audit_belief_context_projection(...)` verifies the public contract without
returning source statements or raw evidence payloads. It checks policy and
semantics identifiers, canonical targets, completeness consistency, immutable
non-promotion flags, and the projection hash. Output is bounded to stable issue
codes and has its own `audit_hash`.

An audit result with `verification_allowed=true` means that the projection
artifact passed these integrity checks. It does **not** mean that a belief is
true, that source independence was externally proven, or that a scientific
claim may be promoted. The audit result repeats all three non-claim flags as
`false`.

## CLI

Build a projection for one or more Research Object selectors:

```bash
xscientist research belief @latest:hypothesis \
  --repo ./first-study \
  --ref HEAD \
  --as-of 2026-08-28T00:00:00+00:00 \
  --budget 4000 \
  --json > belief.json
```

`--ref` defaults to `WORKTREE`. Use a committed ref for exact historical
replay. The command exits successfully only when the projection is complete.
Without `--json`, it prints the projection hash, logical time, completeness,
conflict count, state, and next posture for each target.

Audit a raw projection, a Research Context containing `belief_context`, or a
JSON object containing such a context:

```bash
xscientist research belief-audit belief.json --json
```

The audit command exits with status `0` only when the artifact passes the
projection contract. It deliberately emits no source text.

## Scope and limitations

This feature is a BCG-inspired scientific decision-context projection, not an
installation or fork of BCG. It currently does not:

- learn belief updates or run graph-construction models;
- estimate calibrated probabilities or causal effects;
- prove that two source families are truly independent;
- resolve scientific conflicts automatically;
- replace Research VCS evidence closure, independent review, or promotion;
- reproduce or inherit any BCG benchmark number.

The intended outcome is narrower and auditable: make the evidence state that
an agent is about to rely on explicit, deterministic, bounded, and unable to
silently grant itself scientific authority.
