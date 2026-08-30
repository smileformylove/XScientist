# Research and code lineage

This document distinguishes modified code lineage from design inspiration,
evaluation references and newly assessed convergent work. A paper or repository
listed here does not make XScientist's implementation equivalent to that source
and does not transfer its benchmark results.

## Submission-preparation contract

XScientist improves a top-level run's preparedness against named local
submission gates. It does not estimate an acceptance probability or promise a
paper, acceptance, novelty, correctness or scientific truth. A run may
honestly finish with a blocked package, an inconclusive result or a negative
result.

Its Git-like semantics rest on an externalized structured trajectory: typed
hypotheses, actions, tool executions, evidence, failures, recoveries and gates
form content-addressed state transitions. Research VCS versions that explicit
trail; it does not treat a hidden reasoning transcript as scientific history.
For publication qualification, this Research Git object/checkpoint projection
is a hard gate: every confirmatory or reproduction registry row must bind one
typed attempt and origin checkpoint, and every unsuccessful attempt must carry
one immutable disposition.

For NeurIPS/ICML targeting, local qualification requires a host-verified phase
boundary:

```text
adaptive exploration
  -> freeze hypothesis, method, code/Research VCS, memory, protocol, split,
     metric, empirical snapshot and evaluator specification
  -> fresh confirmatory primary/ablation/robustness execution with
     post-freeze adaptation disabled and hash-bound configuration changes
  -> one-to-one registry-row / typed-attempt / origin-checkpoint closure,
     retaining an explicit disposition for every unsuccessful attempt
  -> genuinely independent reproduction and review
  -> external Ed25519 verifier authority over the final report and producers
  -> byte-verified official venue/year template
  -> claim-evidence closure
  -> submission_package_ready or blocked
```

Passing permits only the statement that the package passed XScientist's named
local submission-preparation gates under the recorded manifest and content
hashes. “NeurIPS/ICML level”, “publishable”, “scientifically correct” and
“likely accepted” still require external experts and independent reproduction.

## Code lineage

| Primary source | Relationship | What is carried forward | Boundary |
| --- | --- | --- | --- |
| [AI-Scientist-v2 paper](https://arxiv.org/abs/2504.08066) · [audited Apache-era snapshot](https://github.com/SakanaAI/AI-Scientist-v2/tree/defddb8174905aac3bf4f7de7650e4cbf2ac353c) | Modified code lineage | Original `ai_scientist` ideation, experiment-tree, plotting, review and writing runtime, now substantially hardened and extended | Modified derivative code, not merely inspiration. No benchmark parity is claimed. Upstream later changed its license; future imports require a new review |
| [AIDE audited snapshot](https://github.com/WecoAI/aideml/tree/a4d58d94ad2035b7b458b5677c26a55e66ea8ca0) | Transitive modified code lineage through AI-Scientist-v2 | Interpreter, journal, model backends, metric/response utilities, serialization and tree visualization foundations | MIT notice is retained; no AIDE result is claimed |
| [AI-Scientist](https://github.com/SakanaAI/AI-Scientist) | Architectural predecessor | `classic_pipeline` and the template-oriented ideation → experiment → write-up → review flow | This audit established no additional direct import beyond the AI-Scientist-v2 lineage |

Affected path families, notices and unresolved acquisition-provenance fields
are recorded in
[THIRD_PARTY_NOTICES.md](https://github.com/smileformylove/XScientist/blob/main/THIRD_PARTY_NOTICES.md)
and
[`provenance/upstream_sources.json`](https://github.com/smileformylove/XScientist/blob/main/provenance/upstream_sources.json).

## Research design and evaluation lineage

| Primary source | Relationship | XScientist implementation or lesson | Boundary |
| --- | --- | --- | --- |
| [autoresearch](https://github.com/karpathy/autoresearch) | Design inspiration | `program_driven` workflow, explicit research program, budgets, stopping and acceptance rules | No vendored code or result reproduction |
| [awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing) | Design inspiration | `writing_studio`, evidence-to-writing, figure/caption and manuscript repair | No vendored code |
| [DeepReviewer-v2](https://github.com/ResearAI/DeepReviewer-v2) | Design inspiration | `review_board`, multi-role objections, repair ownership and reviewer-facing hardening | No vendored code or score equivalence |
| [GEPA](https://github.com/gepa-ai/gepa) | Algorithmic inspiration | Pareto manuscript candidate pool and per-issue repair trajectories | Independent implementation, not the GEPA optimizer |
| [Faraday / Replica](https://arxiv.org/abs/2608.13331) | Research-agent architecture inspiration | Auditable policy rollouts and separation of research judgment, model execution and deterministic host gates | No weights, RL training, Replica tasks, coding harness or reported scores |
| [FAR](https://arxiv.org/abs/2608.16977) · [code](https://github.com/zeyu-zheng/FAR) | Discovery and allocation inspiration | Auditable find → attempt → recommend opportunity funnel | No prompts, corpus, outputs or code imported; no mathematical-discovery result reproduced |
| [Belief Context Graph](https://github.com/bigai-nlco/belief-context-graph) | Belief-context inspiration | Bounded belief projection over immutable research objects, with conflict, time and provenance | Independent implementation; ordinal state is not calibrated confidence |
| [AutoResearchEval](https://arxiv.org/abs/2608.14905) · [code](https://github.com/PrentisAI/AutoResearchEval) | Evaluation reference | Offline artifact/process coverage and explicit local conformance | No official rollout, annotated trajectory or judge score; no cross-system score comparison |
| [MLS-Bench](https://arxiv.org/abs/2605.08678) · [code](https://github.com/Imbernoulli/MLS-Bench) | Method-discovery reference | Gates that separate local engineering gains from transferable methods | No benchmark code, tasks or scores included |
| [Reflexion](https://arxiv.org/abs/2303.11366), [AI co-scientist](https://arxiv.org/abs/2502.18864), [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954), [Red Queen Gödel Machine](https://arxiv.org/abs/2606.26294), [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | Self-evolution synthesis | L0/L1/L2 adaptation, fixed evaluators within an epoch, shadow candidates, sealed evaluation, canary and rollback gates | Model self-scores remain advisory and cannot authorize promotion |
| [Recuris paper](https://arxiv.org/abs/2608.24876) · [code](https://github.com/Gen-Verse/Recuris) · [audited commit](https://github.com/Gen-Verse/Recuris/tree/a0479b27a2d08b7fbf2607acf1841a06b121ee91) | Newly assessed, convergent design reference | State-grounded `M=(E,W,ρ,C)`, structured step evidence and paired held-out admission motivated the host-owned exploration/confirmation freeze and typed primary/ablation/robustness portfolio | No Recuris code or benchmark imported. Component-local repair and activation-fingerprint scoring are not claimed as implemented. Older XScientist memory/evolution features predate the paper and are not retroactively described as Recuris-derived |

Recuris improves long-horizon execution reliability; it does not replace the
novelty review, strong baselines, ablations, statistical robustness,
reproducibility, independent review or claim-evidence closure required for a
scientific submission.
