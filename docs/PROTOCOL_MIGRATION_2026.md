# Protocol compatibility notes (2026 hardening)

This release is additive. Existing `ara.v1` directories and
`xscientist.research-object.v1` objects remain readable and keep their original
hashes. Migration never rewrites committed history.

## ARA node identity

New nodes declare `ara.node-identity.v2`. Version 2 binds a digest of the full
source, ContextPack hashes, LLM-call hashes, execution/container/dependency
identity, evaluator bindings, tools, datasets, and seeds when present. Nodes
without `identity_profile` are verified with the legacy v1 algorithm.

The v1 node-ID grammar is now enforced consistently at schema and filesystem
boundaries: `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`. This is an intentional
security tightening, not a silent compatibility conversion. An older artifact
whose IDs contain separators, whitespace, Unicode, or path syntax must be
explicitly migrated and re-bound before export, fork, or replay.

Portable manifests declare `ara.portable.v1` and use relative path references.
Validation now reports cumulative `index`, `trace`, `replay`, and `verify`
levels. Requesting a level does not upgrade an artifact; it fails when the
required receipts are absent.

Canonical `ara/<run>/nodes/<node-id>/metrics.json` companions are now a built-in
Research VCS checkpoint path, including for repositories whose existing
`research.yaml` predates this release. The node ID must be one portable path
component, the normal checkpoint file-size and secret policies still apply,
and deeper lookalike paths are not admitted implicitly. Historical commits are
not rewritten: an older snapshot without a companion remains readable but its
Research DAG detail view can use only fields already stored in the graph.

## Research Object identity and evidence

The short `rso-<16 hex>` identifier remains the convenient local selector.
New objects also carry a full 256-bit `qualified_id` for federation and export.
Legacy objects without that field remain valid.

Literature provenance uses the immutable chain `search_plan -> search_receipt
-> source_snapshot -> passage_evidence -> claim`. Claim applicability is a
structured, hashed envelope; legacy free-text scope is normalized into its
description field. Missing scope remains conservative during merge conflict
analysis.

## Trust and exchange

The existing local attestation stays supported. Federated consumers can use an
in-toto Statement v1 inside a DSSE envelope with HMAC or Ed25519 verification
and a configured signature threshold. No transparency log or remote
publication is implied unless an external adapter supplies and verifies it.

Standards export now adds Process Run RO-Crate, OpenLineage, Croissant 1.1, and
Nanopublication JSON-LD alongside the existing PROV, CWL, DVC, and MLflow
views. Export is local, atomic, and metadata-only unless payload inclusion is
explicitly requested.

## Research strategy v2

The original `https://xscientist.io/profiles/research-strategy/v1` descriptor
is frozen and remains a registered validator. New strategy records use
`research-strategy/v2`; existing object bytes, hashes, commits, and bundles are
not rewritten.

Version 2 adds `posterior_update` and binds each priority row to an immutable
`experiment_design` plus one locked prediction per portfolio hypothesis. A
selected attempt consumes the priority/design, and a posterior binds the
attempt, observation, evidence, prior, and declared likelihoods. Version 2 also
adds provenance-disjoint independent-assessor receipts, intervention-lineage
receipts for validated mechanisms, and disjoint evidence/attempt/dataset gates
for transfer-ready matrices.

To upgrade an active v1 program, append v2 objects from the current effective
frontier and use `supersedes` where one v2 object replaces a v1 decision. Do not
edit v1 JSON. A historical v1 priority remains inspectable but is not treated
as satisfying v2 executable-closure gates.

## External experiment admission

Repositories configured with `experiment_admission.mode: required` now accept
execution authority only from a dedicated, committed
`xscientist.experiment-admission.v1` gate produced by the Ed25519 ingest path.
Private or legacy `gate_decision` objects without a lossless signed receipt and
host high-water event remain readable as history but cannot authorize a run.
There is no automatic authority migration: prepare and externally sign a new
exact payload for the current plan and execution scope.

The trust store must live outside the repository, the shared Git common
directory, and every linked worktree. Consumers re-check the persisted
signature against that current store, so key revocation intentionally makes an
older gate unusable. Exact scope matching may expose older workflows that
relied on omitted fields as wildcards; sign every runtime admission dimension,
or stop injecting metadata that is not an execution identity. In particular,
Research VCS gate/start commits are not execution `code_commit` values.

Latest-decision state is an append-only, repository-wide hash chain under the
Git common directory and survives checkout changes. It now applies across every
branch and linked worktree: a later overlapping denial blocks stale checkouts
and mixed execution batches, while an authorization that was never published as
high-water is not auto-discovered. Valid pending transitions block all
worktrees until exact recovery.

Each subject also owns a hidden `refs/xscientist/admission/<subject-hash>` ref.
Do not prune, rewrite, mirror-filter, or namespace-remap these refs: they retain
the authoritative checkpoint after branch deletion and Git GC. Existing
repositories repair a missing exact ref during idempotent high-water recovery;
a conflicting ref fails closed. Backup and mirror policies must include this
namespace.

Required-admission runners now distinguish a committed start from execution.
They commit the complete start batch first, then publish a durable single-use
claim immediately before entering the worker boundary. The claim is the
linearization point. Tests or integrations that synthesize terminal results
must explicitly simulate that claim; terminal writes no longer create one on
the caller's behalf.

Once claimed, an outcome remains recordable even if a later denial, signer
revocation, or trust-policy change occurs. Such an outcome is marked as admitted
at start, reports its current policy status, and is quarantined from claim
promotion. This preserves experiment truth without converting stale authority
into current authority.

Terminal reservations bind one start and receipt to one exact attempt identity.
New v2 reservations additionally freeze a canonically hashed terminal-policy
snapshot. Exact retries restore that snapshot before rebuilding object identity,
so a denial or signer revocation published after reservation cannot make the
same result unrecoverable or silently rewrite it as a different object. Existing
v1 markers remain readable, but they can recover across a policy change only if
the regenerated attempt identity happens to remain exact; no missing historical
policy state is inferred. Exact retries recover reservation-only and
uncommitted-object crash states, and an already committed retry must resolve to
its original terminal-only checkpoint. Conflicting retries fail closed.
Deployments should run Research VCS fsck before release; malformed runtime
markers, retention-ref divergence, snapshot/object divergence, or invalid
terminal introductions are integrity errors, while structurally valid in-flight
claims, pending transitions, and recoverable uncommitted terminal state are
reported as incomplete-state warnings.

Production deployments must make the Git common directory executor-unwritable.
A committed gate whose state publication was interrupted fails closed;
re-ingesting the identical signed payload repairs the state without a second
gate commit.

Normative upstream references:

- [Process Run Crate 0.5](https://www.researchobject.org/workflow-run-crate/profiles/process_run_crate/)
- [OpenLineage API](https://openlineage.io/apidocs/openapi/)
- [Croissant 1.1](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html)
- [Nanopublication guidelines](https://nanopub.net/guidelines/working_draft/)
- [in-toto attestation specifications](https://in-toto.io/docs/specs/)
- [DSSE specification](https://github.com/secure-systems-lab/dsse)
