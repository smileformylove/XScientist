# Research Protocol v2: Profiles, Arguments, and Replayable Context

Research Protocol v2 is an additive upgrade to the v1 Research Object envelope.
Existing objects and commits remain valid. Newly created objects bind a
content-addressed Semantic Profile so domain extensions can be stored without
weakening the verification authority model.

## Stable envelope, extensible scientific vocabulary

The core envelope remains immutable and content addressed. `kind` is a safe
local storage token; `semantic_profile.uri` supplies the globally meaningful
namespace. The bound profile contains:

- an absolute profile URI and version;
- the object kinds and relation types it declares;
- a canonical `schema_digest` over that declaration.

Built-in profiles cover the original Research VCS objects, epistemic arguments,
autonomous-research decisions, and the additive deep-research strategy profile.
An object using an unknown profile remains
storable, hashable, exportable, and inspectable. It cannot reach `verified`
until a trusted validator for that exact URI/version/digest is installed.
Profile extensions never grant actor authority and cannot redefine closure,
independence, or promotion rules.

The strategy profile adds competitive hypothesis portfolios, discriminating
predictions, information-value experiment ranking, anomaly/review objects,
mechanism models, evidence-quality assessments, and boundary/transfer matrices.
It deliberately reuses the stable relation vocabulary, preserving historical
profile digests. See [Deep Research Strategy Protocol](DEEP_RESEARCH_PROTOCOL.md).

Use `research record --profile-file PROFILE.json` for a domain extension.
Absolute-URI relation types use `TYPE=TARGET[:ROLE]` because URIs contain
colons. Built-in relations retain `TYPE:TARGET[:ROLE]`.

## Scientific argument graph

Research VCS distinguishes an artifact lineage from the argument made from it:

```text
source snapshot -> exact passage ----+
                                      +-> effect estimate -> inference -> claim
experiment -> evidence --------------+       |              |
                                          estimand      warrant/assumption/method
```

First-class types include `inference`, `warrant`, `assumption`, `method`,
`estimand`, `effect_estimate`, `protocol_deviation`, `sensitivity_analysis`,
`risk_of_bias`, `evidence_synthesis`, and `challenge`. Relevant relations
include `has_premise`, `uses_method`, `under_assumption`,
`addresses_estimand`, `has_effect_estimate`, and `challenges_inference`.

The beginner-facing path does not require JSON:

```bash
xscientist research estimand "one-week recall" \
  --population "eligible adults" \
  --intervention "spaced practice" --comparator "massed practice" \
  --summary-measure "mean difference"

xscientist research effect @latest:estimand 0.31 \
  --metric mean_difference --lower 0.12 --upper 0.50 \
  --from @latest:evidence

xscientist research infer "Spaced practice improves one-week recall." \
  --premise @latest:effect_estimate \
  --warrant "The interval excludes a non-positive effect under the recorded design."

xscientist research claim "Spaced practice improves one-week recall." \
  --evidence @latest:inference --population "eligible adults"
```

`infer` creates a separate immutable warrant automatically. Closure walks
through inference and effect-estimate nodes to the underlying evidence,
attempt, plan, source, and context.

## Retrieval and memory receipts v2/v3

Literature and decision-context retrieval now record more than the final
selection. A receipt binds the request, algorithm, complete candidate set,
ranks, score semantics, selection reasons, transformation lineage, corpus or
Git snapshot, omissions, and all component hashes. Secrets, credentials, and
raw provider bodies remain forbidden.

Literature receipts accept optional replay metadata:

```bash
xscientist research literature receipt @latest:search_plan \
  --provider OpenAlex --query "QUERY" --results candidates.json \
  --query-rewrite "EXPANDED QUERY" --retriever openalex-rest-v1 \
  --reranker local-cross-encoder-v2 \
  --corpus-snapshot-hash sha256:... --page 1
```

An incomplete or truncated result set must use `--incomplete`. Context summary
budgets may omit readable views, but never omit source IDs, hashes, candidates,
negative knowledge, or prior decisions.

Literature receipts remain on v2. Research decision contexts and ARA
ContextPacks now emit retrieval receipt v3. The v3 receipt keeps the complete
audit candidate set while binding a separate semantic working set with:

- an effective-frontier status that demotes, but does not delete, superseded
  history;
- task relevance, DAG distance, recency, and authority score components;
- required semantic lanes for decisive evidence, active contradiction,
  failure/do-not-repeat memory, open questions, and prior decisions;
- a conservative post-render token estimate and `decision_usable` verdict;
- a compact previous-context hash chain instead of recursive snapshot copying.

If a required lane cannot fit the declared budget, the pack is incomplete and
promotion must stop. ContextPack receipts v1/v2 and Research Context receipt v2
remain readable; newly compiled packs use v3.

## Exact passages and source updates

Every new `passage_evidence` contains a W3C Web Annotation-compatible
`TextQuoteSelector`; optional prefix, suffix, and text positions make the quote
relocatable after reflow:

```bash
xscientist research literature passage @latest:source_snapshot \
  "EXACT QUOTE" --locator "page=7;section=Results" \
  --prefix "TEXT BEFORE" --suffix "TEXT AFTER" --start 2401 --end 2510
```

Corrections, withdrawals, and retractions append immutable events instead of
editing a source snapshot:

```bash
xscientist research literature update @latest:source_snapshot \
  --status retracted --type retraction \
  --provider Crossref-Retraction-Watch \
  --checked-at 2026-08-09T12:00:00Z --notice-id NOTICE_ID
```

Closure fails closed when a used source is invalidated. Historical commits
still show what was known before that update.

## Autonomous-research contract

The autonomous profile defines `research_goal`, `action_proposal`,
`experiment_design`, `resource_budget`, `stopping_decision`, `novelty_check`,
`evaluation_blinding`, `context_robustness`, and `human_escalation`.
Context-bearing decisions must link to an exact `context_snapshot`. These
objects are records of what an agent proposed or decided; they are not proof
that the proposal is scientifically correct. Independent review, deterministic
gates, reproduction, and the existing self-evolution isolation rules still
control verification and promotion.

`research start` now creates a locked `research_goal` between the question and
falsifiable hypothesis. This gives both users and agents a stable objective,
success condition, and authority policy from the first commit.

## Compatibility rules

- v1 core objects without `semantic_profile` remain valid.
- New objects always bind a built-in or explicitly supplied profile.
- Context receipt validators retain legacy compatibility; new Research Context
  and ARA ContextPack retrieval receipts use v3, while literature uses v2.
- Direct evidence-to-claim links remain traceable, but verified legacy claims
  receive an `claim_inference_unmodeled` warning.
- Claims without `depth_level` remain `descriptive`. New verified `causal`
  claims require a validated evidence-bound mechanism plus an independent
  strong/moderate quality assessment; `transferable` also requires a passing
  transfer matrix. Draft hypotheses about deeper claims remain recordable.
- Source status checks are advisory until an invalidating update exists; an
  explicit retraction/withdrawal is a closure blocker.
- Exporters retain native qualified IDs and Semantic Profile metadata. Formats
  remain projections; a green Research DAG node is not a universal truth badge.
