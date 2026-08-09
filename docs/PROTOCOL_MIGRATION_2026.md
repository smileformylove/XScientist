# Protocol compatibility notes (2026 hardening)

This release is additive. Existing `ara.v1` directories and
`xscientist.research-object.v1` objects remain readable and keep their original
hashes. Migration never rewrites committed history.

## ARA node identity

New nodes declare `ara.node-identity.v2`. Version 2 binds a digest of the full
source, ContextPack hashes, LLM-call hashes, execution/container/dependency
identity, evaluator bindings, tools, datasets, and seeds when present. Nodes
without `identity_profile` are verified with the legacy v1 algorithm.

Portable manifests declare `ara.portable.v1` and use relative path references.
Validation now reports cumulative `index`, `trace`, `replay`, and `verify`
levels. Requesting a level does not upgrade an artifact; it fails when the
required receipts are absent.

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

Normative upstream references:

- [Process Run Crate 0.5](https://www.researchobject.org/workflow-run-crate/profiles/process_run_crate/)
- [OpenLineage API](https://openlineage.io/apidocs/openapi/)
- [Croissant 1.1](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html)
- [Nanopublication guidelines](https://nanopub.net/guidelines/working_draft/)
- [in-toto attestation specifications](https://in-toto.io/docs/specs/)
- [DSSE specification](https://github.com/secure-systems-lab/dsse)
