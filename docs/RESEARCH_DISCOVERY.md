# Open-Ended Research Discovery

XScientist treats innovation as a population search problem rather than a
single prompt followed by repeated self-editing.

## Hypothesis archive

Every seeded project writes `hypothesis_archive.json`. Its nodes are
append-only and contain:

- immutable content hash and hypothesis ID;
- parent hypotheses and generation operator;
- mechanism, falsifiers, candidate datasets, metrics, and baselines;
- literature queries and retrieved evidence;
- novelty, feasibility, falsifiability, information gain, impact, evidence
  grounding, and safety scores;
- separate Elo ratings from auditable pairwise comparisons, leaving hypothesis
  node content immutable.

The archive builds a lexical proximity graph for dependency-free operation,
clusters near-duplicates into discovery niches, computes a multidimensional
Pareto front, and selects strong representatives across niches. Embedding-based
proximity can be added later without changing the artifact contract.

Supported generation operators are `analogy`, `combination`, `contradiction`,
`boundary_condition`, `simplification`, `failure_driven`, `mechanism`, and
`out_of_distribution`. New hypotheses preserve their parents; evolution never
silently overwrites an earlier scientific direction.

## Literature gate

`FinalizeIdea` is rejected until Semantic Scholar returns evidence. Search
metadata is attached to the idea and propagated into its hypothesis node.
Literature ordering balances API relevance, recency, and citations so emerging
work is not hidden by citation-only sorting.

This is a minimum grounding gate, not a proof of novelty. A confirmatory claim
still needs the independent protocol in [RESEARCH_INTEGRITY.md](RESEARCH_INTEGRITY.md).

## Idea judging

Invalid or unavailable LLM judgments receive `total_score: 0`,
`ranking_eligible: false`, and `trust_tier: untrusted_fallback`. Keyword count
and text length are never used as semantic novelty evidence.

Pass a comma-separated model portfolio to `--idea-rank-model` to enable
independent judging:

```bash
xscientist project demo \
  --rank-ideas \
  --idea-rank-model "provider/model-a,provider/model-b"
```

The ranker uses the median per dimension, reports judge disagreement, and
requires at least two valid judgments when multiple models are configured.
Historical acceptance priors are withheld if the primary judgment is
untrusted.

## Socratic challenge before execution

`research_plan.json` does not let the favored hypothesis monopolize the
experiment budget. Every plan now carries a `socratic_challenge` with:

- null-effect, substitute-mechanism, measurement-artifact, and scope-boundary
  rivals, plus any researcher-supplied alternative hypotheses;
- paired baseline, single-mechanism ablation, negative-control, and boundary
  probes that distinguish those rivals;
- an uncertainty contract that forbids treating self-scores as evidence and
  requires an evidence-linked posterior update;
- explicit rival-hypothesis decisions in experiment outputs and the
  claim-evidence graph.

This moves adversarial reasoning before expensive execution and before the
paper narrative hardens. The design is informed by the causal questioning and
falsification loop in [Socratic agents for autonomous scientific
discovery](https://arxiv.org/abs/2606.26722), while keeping the progressive
experiment-manager search used by [AI Scientist
v2](https://arxiv.org/abs/2504.08066). Negative and boundary results remain
first-class outputs rather than failed attempts to hide.
