# Exuvia research exchange

XScientist can prepare a public, inspectable research package for discussion on
Exuvia. This is an opt-in exchange workflow; it does not grant Exuvia access to
the local workspace and it does not publish anything automatically.

## Current integration status

As of **2026-10-05**, the [XScientist agent](https://exuvia-two.vercel.app/agents/de98e043-e077-4262-a0cc-da97eb549537)
is registered, human-claimed, and enabled on Exuvia. Its Research Post,
[A Gravitational Field Theory for Deep Networks](https://exuvia-two.vercel.app/posts/e7154a9e-f555-4220-b359-d01df2f0404a),
was published on that date through an explicit, operator-authorized API action.
The post links to the unchanged original GitHub PDF pinned below; it does not
host a separately uploaded PDF. Sharing the manuscript is not conference
acceptance or independent verification, and this publication involved no new
experiments, research-code runs, or independent reproduction.

The original PDF credits XScientist as its AI author. Jixiang Luo authorized
the sharing and claimed the agent; Codex assisted with the summary, provenance
framing, and API publication. The post invites discussion of claim-to-evidence
provenance and reconciliation of energy-cascade figures with numerical text.

The repository already exposes the right local boundary for this use case:

- `xscientist research export` writes hash-bound RO-Crate and PROV-style
  metadata for a committed Research VCS ref.
- `xscientist research dag --metadata-only` produces an offline view of the
  claim/evidence graph without copying scientific payloads.
- `xscientist research adapter sync` is an explicit, versioned extension point
  for a future platform adapter. Adapter results are redacted, hash-bound, and
  recorded as receipts.

There is no built-in Exuvia client or background publishing loop in this
repository. The Exuvia guide and API contract were reviewed for the publication
above; this does not implement a reusable adapter. The workflow below prepares
an offline package only. Any future adapter must review the then-current API
and authentication contract rather than infer it from the outreach email.

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
Published Exuvia posts may be retained and currently have no agent-facing
deletion route; anonymity is not guaranteed.

A future adapter should be implemented as a separate package using the
`xscientist.research_adapters` entry-point group. It must keep credentials in
the caller's environment or secret store, never persist tokens in receipts, and
fail closed when the remote response cannot be bound to the exported hash.

Relevant Exuvia links from the invitation:

- Agent guide: <https://exuvia-two.vercel.app/llms.txt>
- API documentation: <https://exuvia-two.vercel.app/api/docs>
