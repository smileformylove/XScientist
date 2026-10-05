# Exuvia research exchange

XScientist can prepare a public, inspectable research package for discussion on
Exuvia. This is an opt-in exchange workflow; it does not grant Exuvia access to
the local workspace and it does not publish anything automatically.

## Current integration status

The repository already exposes the right local boundary for this use case:

- `xscientist research export` writes hash-bound RO-Crate and PROV-style
  metadata for a committed Research VCS ref.
- `xscientist research dag --metadata-only` produces an offline view of the
  claim/evidence graph without copying scientific payloads.
- `xscientist research adapter sync` is an explicit, versioned extension point
  for a future platform adapter. Adapter results are redacted, hash-bound, and
  recorded as receipts.

There is no built-in Exuvia client in this repository. The Exuvia guide and API
contract must be reachable and reviewed before an adapter is implemented. Until
that happens, use the manual, offline workflow below. Do not invent endpoint
paths, request fields, authentication schemes, or publication semantics from an
outreach email alone.

## Prepare a shareable package

Run these commands against a committed Research VCS workspace:

```bash
xscientist research fsck --repo ./my-study
xscientist research export \
  --repo ./my-study \
  --ref HEAD \
  --dest ./exuvia-export \
  --format ro-crate \
  --format prov-json
xscientist research dag \
  --repo ./my-study \
  --ref HEAD \
  --output ./exuvia-dag \
  --metadata-only
```

The default export is metadata-only. It contains object identities, hashes,
relations, checkpoint provenance, and actor attribution, but not experiment
payloads. Add `--include-payloads` only after a human has checked every file and
confirmed that it is authorized for public release. Review the generated
directory and remove anything that is not intended for a public post; the
export is an exchange artifact, not a disclosure decision.

For the already-public gravitational-network manuscript, link the exact
[submitted PDF at commit `6b2b875`](https://github.com/smileformylove/XScientist/blob/6b2b875843dd095b7cba47280339ba705734fb9f/example/icml_submitted_gravitation_paper.pdf)
and commit rather than presenting an unverified local reproduction. A
discussion that reconciles an energy-cascade figure with numerical text should
represent both states explicitly:

1. keep the original claim and its source/evidence bindings immutable;
2. add a typed correction or challenge that `supersedes` or qualifies the
   original claim; and
3. export the resulting history so agents can see the original result, the
   correction, and the attribution of each contribution.

This preserves the published result while making a correction inspectable. A
figure, a number in prose, or an agent comment alone is not an independent
verification.

## Human publication gate

Before sharing a package on Exuvia, a human owner must confirm all of the
following:

- the Exuvia terms, retention, and public visibility are acceptable;
- the paper, repository commit, figures, and metadata are authorized for
  publication;
- no credentials, private data, patient data, prompts, or hidden reasoning are
  included;
- human setup, review, and later contributions have the requested attribution;
- the post describes XScientist output as an inspectable artifact, not as an
  independently reproduced or verified result unless the corresponding gates
  actually passed.

The current XScientist boundary remains **remote publication: never automatic**.
When Exuvia publishes a supported API contract, a future adapter should be
implemented as a separate package using the
`xscientist.research_adapters` entry-point group. It must keep credentials in
the caller's environment or secret store, never persist tokens in receipts, and
fail closed when the remote response cannot be bound to the exported hash.

Relevant Exuvia links from the invitation:

- Agent guide: <https://exuvia-two.vercel.app/llms.txt>
- API documentation: <https://exuvia-two.vercel.app/api/docs>
