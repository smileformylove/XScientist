# XScientist Research Version Control Specification

Status: draft v1

## 1. Scope

Research Version Control (Research VCS) is XScientist's native state and
history protocol for autonomous research. Git, GitHub, object storage, and
archive files are optional storage or transport backends. They do not define
the public semantics of Research VCS.

An agent using Research VCS MUST be able to inspect, branch, record, compare,
merge, reproduce, and evaluate research without invoking a backend-specific
command. A research commit records a scientific state transition, not merely
a filesystem snapshot.

## 2. Normative invariants

1. Every durable research object MUST have a canonical content hash.
2. Durable objects and commits MUST be immutable. Corrections create a new
   `revert` or `supersede` transition; history is never rewritten.
3. Failed, timed-out, cancelled, negative, contradictory, and rejected work
   MUST remain representable in the commit graph.
4. Confirmatory evidence MUST reference a preregistration that was locked
   before the confirmatory execution began.
5. Every promoted claim MUST reference evidence and an independent evaluation
   decision.
6. The agent proposing a candidate MUST NOT change the evaluator, hidden
   benchmark, promotion threshold, or previously recorded result used to
   evaluate that candidate.
7. Repository writes MUST be atomic, idempotent, concurrency-safe, and
   recoverable after interruption.
8. Serialization, hashing, ordering, graph traversal, and deterministic gate
   decisions MUST be reproducible for identical inputs.
9. Garbage collection MAY remove unreachable caches. It MUST NOT remove a
   reachable research object, failure record, evaluation, or provenance link.
10. Secrets and machine-local paths MUST NOT enter portable objects or
    persistent traces.
11. A new review, promotion gate, or agent evaluation MUST bind the exact
    evidence/context/memory snapshot it consumed. Prompt-budget trimming MUST
    NOT remove identities from that snapshot's hard closure.

## 3. Logical layers

Research VCS separates six logical layers:

| Layer | Responsibility |
|---|---|
| Working state | Mutable agent workspace derived from one commit and branch |
| Semantic index | Explicit set of proposed scientific changes |
| Object store | Immutable, content-addressed research objects |
| Commit graph | Atomic scientific transitions with one or more parents |
| Ref store | Branches and immutable tags |
| Policy engine | Validation, conflict, evaluation, and promotion decisions |

Backends implement persistence for these layers. Backend-specific identifiers
MUST NOT replace Research VCS object, commit, or ref identifiers in the public
API.

## 4. Research objects

The initial object kinds are:

- `question`
- `hypothesis`
- `preregistration`
- `research_plan`
- `experiment_attempt`
- `metric`
- `evidence`
- `claim`
- `review`
- `gate_decision`
- `manuscript`
- `reproduction`
- `agent_candidate`
- `agent_evaluation`
- `context_snapshot`

Every object contains a versioned schema identifier, object kind, lifecycle
state, payload, typed relations, actor receipt, provenance receipt, and content
hash. Relations include `depends_on`, `supports`, `refutes`, `supersedes`,
`reproduces`, `contradicts`, `derived_from`, `evaluates`, and `promotes`.

Scientific payload schemas MAY be kind-specific, but an implementation MUST
validate a payload before it can enter the durable object store.

A `context_snapshot` is a first-class scientific object, not a transient prompt
cache. It records the selected ref/commit, target objects, the complete sorted
set of source object IDs and content hashes, negative outcomes, prior decisions,
external memory hashes, options considered, rejection reasons, constraints,
selection-policy hash, omissions, and a canonical `context_hash`. Token budgets
MAY trim summaries only. They MUST NOT trim source IDs, hashes, failed attempts,
contradictions, or decision alternatives. Memory can inform a decision but does
not gain evaluator or verification authority merely by being remembered.

## 5. Research commits

A research commit contains:

- a canonical Research VCS commit identifier;
- zero or more parent commit identifiers;
- branch and operation (`commit`, `merge`, `revert`, `promotion`, or
  `migration`);
- intent, stage, status, and summary;
- added, superseded, and removed object references;
- explicit failed and negative outcome references;
- agent, model, provider, prompt, tool, and evaluator receipts;
- environment, dependency, dataset, and random-seed receipts;
- token, cost, time, and compute budget receipts when available;
- validation and gate decisions;
- reproduction instructions;
- a canonical content hash.

The first implementation MAY store commits as Git commits plus protocol
objects. Research VCS identifiers and semantics remain authoritative.

## 6. Refs and branch roles

The reference implementation initializes `main` as the default local ref.
Deployments MAY designate an immutable tag or a policy-controlled ref as
`stable`; a branch name alone never grants scientific authority. Implementations
support the following
logical branch roles:

| Role | Meaning |
|---|---|
| `explore` | Candidate mechanisms and inexpensive falsification |
| `confirm` | Preregistered confirmatory work |
| `challenge` | Counter-hypotheses and adversarial tests |
| `reproduce` | Independent reproduction |
| `repair` | Reviewer-directed correction |
| `evolve` | Candidate changes to agent behavior |

Branch names are labels, not scientific authority. Only a promotion decision
can advance stable knowledge. A merge retains every scientific parent.

## 7. Operations

The reference implementation exposes backend-independent operations equivalent to:

- `init`, `status`, `stage`, `commit` and `show`;
- `branch` (create/list/delete/rename), `fork`, `switch` and `branches`;
- `log`, `tree`, `diff`, `blame`, `restore` and `revert`;
- `merge`, semantic conflict preflight and `tag`;
- `fsck`, `audit`, `reproduce`, standards `export`, offline `bundle` and bundle restore.
- `context` for read-only compilation or explicit recording of an exact
  decision/evidence/memory snapshot.

`bisect`, cross-repository `clone`, policy-level promotion, and safe semantic
garbage collection remain protocol operations for a later conformance level;
they MUST NOT be advertised as implemented commands until shipped.

The reference implementation also exposes a payload-free `audit` view with
three explicit sufficiency targets over the effective claim frontier. Explicit
`supersedes` relations, the `superseded` state, and immutable draft-to-verified
promotion determine which historical claims remain active. `trace` requires the claim/evidence/attempt
and planning chain; `replay` adds immutable code, data, environment, and
measurement identities; `verify` adds a passing gate and a verified
reproduction receipt. Storage integrity (`fsck`) and scientific sufficiency
(`audit`) are separate verdicts. `verify` proves local, hash-bound protocol
closure; it is not by itself a third-party attestation or claim of scientific
truth. Signed attestations are a separate trust layer and do not change the
meaning of the closure verdict.

Each operation MUST have a structured result, stable error category, explicit
mutation summary, and deterministic exit status. Retrying an operation with
the same idempotency key MUST NOT create duplicate scientific transitions.

## 8. Semantic diff and merge

Semantic diff compares scientific meaning rather than only text. At minimum it
reports changes to hypotheses and falsifiers, preregistration fields, data
split hashes, metrics, baselines, stopping rules, claims, evidence, uncertainty,
gate decisions, agent/evaluator receipts, environment, and budget.

A merge MUST stop for an unresolved conflict when, at minimum:

- the same claim is both supported and refuted;
- data split hashes or primary metric definitions differ;
- a locked preregistration was changed;
- evaluator or rubric versions are incompatible;
- reproduction direction or material effect disagrees;
- evidence coverage is insufficient after combining claims; or
- an agent candidate introduces a gated regression.

Conflict resolution is itself a durable object and commit. It cannot silently
choose one file version.

The `preserve_as_contested` resolution MAY be used only when the complete
blocking set consists of opposed evidence. It MUST retain both evidence sets,
write a deterministic `hold` gate with promotion disabled, bind that gate to a
stable conflict identifier, and keep the target on the contested frontier.
Backend conflicts, incompatible locked preregistrations, metric-definition
conflicts, and ungated agent candidates MUST NOT use this resolution.

A long-term technology-tree view MUST union immutable objects reachable from
all local research-line heads by content identity. Shared ancestry is de-duped,
line membership is retained, and missing targets or relation cycles are
reported. A privacy-preserving tree view MUST NOT emit object payloads.

## 9. Authority model

| Actor | Authority |
|---|---|
| Research agent | Propose objects, branches, experiments, and commits |
| Recorder | Validate and atomically persist accepted transitions |
| Independent evaluator | Read candidate state and write a separate evaluation |
| Deterministic gate | Enforce fixed promotion and integrity rules |
| Human approver | Optional approval at policy-defined high-impact boundaries |

A research agent MUST NOT write an evaluation that is treated as independent
for its own candidate. Evaluator inputs and versions are bound to the decision.
For decisions created under the context-snapshot policy, the bound snapshot
MUST be complete and its context/source/memory hashes MUST recompute. Legacy
unbound decisions remain readable but MUST be diagnosed as weaker evidence.

## 10. Scientific lifecycle

The standard lifecycle is:

`question -> ideation -> preregistration_draft -> preregistration_locked ->
experiment_attempt -> evidence -> independent_review -> claim_promotion ->
manuscript -> release`.

Exploratory work may proceed without a locked preregistration, but its claims
remain candidate claims. Confirmatory and submission-grade promotion fails
closed without the required locked state and independent evidence.

Before mutating history, an autonomous agent SHOULD create a deterministic,
read-only version-control decision trace. Material terminal or milestone state
requires a checkpoint; a competing hypothesis, incompatible method,
contradictory interpretation, independent replication, or agent candidate
requires a separate research line; a merge requires a clean working state and
semantic preflight. The trace records policy version, inputs, current head,
reasons, recommended operations, and a stable decision identifier. Producing
the trace itself MUST NOT mutate the repository.

Before an independent review, deterministic promotion gate, or agent-evolution
evaluation is recorded, the producer MUST compile or record its exact decision
context. The decision binds that snapshot through a typed `depends_on`
relation with role `decision_context` and repeats its `context_hash`. A user or
agent can compile the same view at an older ref; selector resolution and object
loading MUST occur against that ref, never against the current worktree.

## 11. Agent self-evolution

Scientific history and agent-evolution history are separate commit namespaces
with explicit cross-references. Every scientific commit identifies the agent
version that produced it.

An `evolve` candidate records its mechanism, changed prompts/policies/tools,
fixed evaluator and benchmark references, paired results, ablations, failure
cases, and promotion decision. Candidate code cannot write to its evaluator or
benchmark namespace. Rejected candidates remain negative knowledge. Rollback
creates a new evolution commit.

## 12. Compatibility and migration

Existing `xscientist.research-repository.v1` repositories and research
checkpoints remain readable. Migration creates a new migration commit and MUST
NOT edit an existing commit, checkpoint, object hash, or branch ancestry.

The existing `checkpoint` operation remains a compatibility alias for a
Research VCS commit until a future major version. Ordinary Git inspection MAY
remain available for operators, but agents and public clients do not depend on
it.

## 13. Out of scope for the core

The following are optional adapters and are not prerequisites for Research
VCS: GitHub integration, automatic push, hosted collaboration, DOI issuance,
and cloud object storage. The core MUST work offline and MUST NOT perform a
network or paid-model call merely to inspect or validate repository state.
