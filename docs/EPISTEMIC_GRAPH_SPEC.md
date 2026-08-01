# Epistemic Graph Specification

The epistemic graph is XScientist's cumulative scientific technology tree. It
stores knowledge-bearing entities rather than treating papers as the primary
unit of scientific memory.

## Node types

`question`, `concept`, `theory`, `hypothesis`, `prediction`, `protocol`,
`observation`, `claim`, `theorem`, `refutation`, `replication`, and `artifact`.

Every node has immutable content, a content hash, provenance, applicability,
uncertainty, parent IDs, and an initial `speculative` state. Theory,
hypothesis, prediction, and claim nodes require explicit falsifiers.

## Relations

The graph supports relations such as `supports`, `refutes`, `depends_on`,
`generalizes`, `contradicts`, `replicates`, `tested_by`, and `formalizes`.
Edges are content-addressed and must reference existing nodes.

## Evidence lifecycle

```text
speculative -> grounded -> preregistered -> tested
            -> replicated -> robust -> canonical
```

At any relevant stage, new evidence may move an item to `contested`, `failed`,
`refuted`, or `superseded`. Failed, refuted, and superseded nodes are terminal:
a revised idea must be added as a new descendant instead of rewriting history.

Nodes never change status in place. Every transition is an append-only event
containing its actor, reason, evidence references, previous event hash, and
event hash. Higher states require progressively stronger evidence references.
The graph itself is bound to the project's locked science constitution.

## Project initialization

New project, experiment, batch, and paper roots automatically receive:

- `science_constitution.json`;
- `epistemic_graph.json` seeded with question and hypothesis nodes;
- manifest bindings and stage-standard checks.

The graph complements `hypothesis_archive.json`: the archive optimizes
discovery diversity, while the epistemic graph governs cumulative scientific
status and evidence history.
