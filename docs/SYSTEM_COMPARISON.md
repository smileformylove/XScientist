# Research-system comparison (source-audited, not a leaderboard)

This page compares XScientist with the systems named in the attached
conference talk, using the primary papers and official repositories available
on 2026-08-23. The talk is a discovery source, not an experimental protocol:
details that appear only in the talk are labelled `talk_reported` and are not
treated as independently reproduced results.

The machine-readable version is available without network access:

```bash
xscientist benchmark systems --json > system-comparison.json
xscientist benchmark systems --workspace ./first-study --show-process
```

The command performs no provider call, network request, external rollout, or
score aggregation. With `--workspace`, it adds the same bounded, redacted
process view used by the AutoResearchEval-inspired pilot, so commits, branch
topology, intermediate artifact counts, and fairness status can be inspected
next to the qualitative matrix.
The machine-readable local fields label rollout and cost as
`this_audit_only`; any historical trajectory cost is `unobserved`, not zero.

## How to read the evidence labels

| Label | Meaning | What it does not mean |
| --- | --- | --- |
| `reported_primary` | The cited paper or official project describes/evaluates the capability. | XScientist reproduced it, or the claim is universally valid. |
| `reported_talk` | The attached talk reports the detail. | A peer-reviewed or independently audited result. |
| `local_observed` | An XScientist local artifact exposes the signal. | A model-quality or scientific-novelty score. |
| `local_structural_only` | The local pilot checked a contract or evidence shape. | An autonomous rollout. |
| `scoped_component` | The system covers one layer of the stack. | End-to-end research ability. |
| `not_measured_here` | No local experiment was run. | Failure, weakness, or absence in the external system. |
| `not_in_scope` | The source defines a narrower component. | A claim that the system can never do the task. |

The matrix deliberately has no rank column. A writing benchmark, a figure
benchmark, an MLE search benchmark, an artifact-integrity audit, and an
end-to-end discovery benchmark answer different questions.

Every JSON row also carries `human_evidence`. In this audit its `score` is
always `null` and `same_condition` is always `false`: `human_reference_proxy`
means a historical/reference result, `human_SOTA_reference` means a published
human-designed SOTA/reference result, and `human_judgment_calibration` means people
rated or calibrated an evaluator, and `human_agent_process` means an
intervention/uplift study. `not_reported` is retained when no task-performance
arm was found. This prevents a human-authored target paper or judge score from
silently becoming a human baseline.

The talk-only architectural references are retained as `reported_talk` rows
when they are named as systems (Deep Researcher Agent and the Autonomous
Science Team framework). Adjacent examples that are not independent systems
(the CAST sample paper, Paper Assistant Tool integration, benchmark arms, and
AutoResearchEval itself) are listed in the JSON `talk_inventory` instead of
being turned into fake competitors. Context-only mentions and future concepts
such as ScientistTwo are listed separately with their slide number and no
benchmark row.

Human comparison is a separate evidence layer. See the [source-audited human
baseline inventory](HUMAN_BASELINES.md): it distinguishes recruited participant
runs (for example RE-Bench, PaperBench, and DiscoveryWorld) from published
human-SOTA references, expert validation, and human+agent process studies.
Those rows are not injected into this matrix or into `workspace.score`.

MLE-STAR and DS-STAR are included below as adjacent primary-source execution
references. FAR (Find–Attempt–Recommend) is likewise included as an adjacent
primary-source discovery/allocation reference: it is not claimed to be named
in the attached 107-page talk, so its `talk_slides` list is empty in the JSON
report. These adjacent rows are useful for coverage, but none is a local rerun.

## What is being compared

| System | Layer / input-output boundary | Main idea visible in the source | Evaluation anchor (reported, not rerun here) |
| --- | --- | --- | --- |
| **XScientist** | Git-like evidence and workflow substrate; local pilot starts from a task contract or workspace | Typed research objects, append-only checkpoints, branch-aware process audit, `trace → replay → verify`, explicit fairness gates | First-run usability and AutoResearchEval-inspired structural conformance; **0 rollouts, no scientific score** |
| **Deep Researcher Agent (talk reference)** | Broad research-assistant / deep-research reference | Product-level deep research and test-time diffusion are mentioned in the talk | Talk-only reference; no matched task slice or local rerun |
| **Autonomous Science Team (AST) framework** | Talk-level role decomposition | Generator → implementor → paper writer → paper reviewer | Architectural slide only; no independent score or runnable protocol identified |
| **Science One / ScientistOne** | End-to-end: problem investigation → discovery → paper/claim verification | Chain-of-Evidence built during production plus a four-check integrity audit | ADRS (5 tasks), 75-paper CoE audit, MLE-Bench and Parameter Golf; primary paper reports 0/337 phantom references and 12/12 score verification, but XScientist has not reproduced it (ADRS “human” is a published reference, not a newly recruited arm) |
| **Sakana AI Scientist v2** | End-to-end idea → experiments → analysis → manuscript | Progressive best-first experiment tree search and VLM review loop | Three autonomous ICLR-workshop manuscripts (with human theme/idea setup and selection around the runs); not a matched XScientist run |
| **AutoResearchClaw** | End-to-end 23-stage pipeline | Multi-agent debate, self-healing `Pivot/Refine`, read-only result reporting, targeted HITL, cross-run lessons | ARC-Bench experiment/end-to-end modes and ablations; pin paper/repository commit before comparing |
| **DeepScientist** | Long-horizon goal-oriented discovery | Hierarchical hypothesize → verify → analyze and cumulative Findings Memory | Large, GPU-intensive progressive-discovery study; reported scale is not comparable to the provider-free pilot |
| **AI-Researcher** | End-to-end survey → algorithm implementation → publication-ready paper | Specialized survey, coding, and writing agents with code-validate-refine | Scientist-Bench guided and open-ended tasks; source reports an agent benchmark, not a human-run baseline |
| **FAR (Find–Attempt–Recommend)** | Literature → open-problem pool → candidate attempts → judged/graded discoveries | Find unresolved problems from a research direction, attempt every apparently well-posed candidate, then allocate expert review | Combinatorics pilot reports 4,717 conjectures attempted, 1,050 claimed `NEW`, 598 judged `PASS`, and 77 graded publishable; **reported primary result, not rerun here and no human task-performance arm** |
| **MARS** | MLE-focused execution/search component | Budget-aware MCTS, Design–Decompose–Implement, comparative reflective memory | MLE-Bench and cross-branch lesson transfer; not a complete literature-to-paper system |
| **AdaEvolve** | Adaptive evolutionary optimization component | Progress-aware adaptation of exploration intensity and resource allocation | ADRS / open-ended optimization anchors; component search, not a full research pipeline |
| **EvoX (Meta-Evolution)** | Meta-evolution search-strategy component | Co-evolves candidate solutions and the strategy that selects/mutates them | ADRS / broad optimization anchors; exact task/evaluator revision must be pinned |
| **MLE-STAR** | MLE search and targeted code refinement | Web search, ablation-guided code-block selection, inner/outer refinement | MLE-Bench Lite; no paper-writing or claim-grounding contract |
| **DS-STAR** | Heterogeneous data-science planning/execution component | Data-file analysis, verifier, sequential plan refinement | DABStep, KramaBench, DA-Code; not literature-to-paper discovery |
| **ScholarPeer** | Review-only component | Live literature context, historian/baseline-scout roles, skeptical multi-aspect verification | ScholarEval (DeepReview-Bench + AgentReview, with a DeepReview-13K subset); does not execute the underlying experiment |
| **PaperOrchestra** | Writing-only component; supplied raw materials → LaTeX | Literature synthesis, section planning, visual generation, paper assembly | PaperWritingBench (200 paper-derived raw-material cases; dataset/release must be pinned and verified separately); cannot validate supplied experiments |
| **PaperBanana / PaperVizAgent** | Figure-only component; context/reference → diagrams or plots | Retriever/planner/stylist/renderer/critic refinement loop | PaperBananaBench (292 methodology diagrams); figure quality is not discovery quality; paper `Human=50` is an evaluator reference scale, not a human drawing score |

Primary sources: [ScientistOne](https://arxiv.org/abs/2605.26340), [AI
Scientist-v2](https://arxiv.org/abs/2504.08066),
[AutoResearchClaw](https://arxiv.org/abs/2605.20025),
[DeepScientist](https://arxiv.org/abs/2509.26603),
[AI-Researcher](https://arxiv.org/abs/2505.18705),
[FAR](https://arxiv.org/abs/2608.16977) ([official repository](https://github.com/zeyu-zheng/FAR)),
[MARS](https://arxiv.org/abs/2602.02660),
[AdaEvolve](https://arxiv.org/abs/2602.20133),
[EvoX](https://arxiv.org/abs/2602.23413),
[MLE-STAR](https://arxiv.org/abs/2506.15692),
[DS-STAR](https://arxiv.org/abs/2509.21825),
[ScholarPeer](https://arxiv.org/abs/2601.22638),
[PaperOrchestra](https://arxiv.org/abs/2604.05018), and
[PaperBanana](https://arxiv.org/abs/2601.23265). Official or author repositories
are linked where available; absence is explicit (for example, a turnkey
ScholarPeer or ScientistOne system repository was not identified).
Where a Google Research repository is linked, it is treated as an author/research
code release; a repository's “not an officially supported Google product”
disclaimer is not converted into a product-support claim.

For FAR, the funnel counts and allocation findings are copied only as
source-reported anchors. The paper's manually checked subset is author-selected
and is not a 100% accuracy estimate; the expert/judge stages do not constitute a
human performance arm. It is an arXiv preprint rather than an independently
audited cross-system benchmark. XScientist has not executed the FAR repository
or its combinatorics corpus here.

## Capability matrix

The following is a compact reading of the machine-readable `capabilities`
field. “Reported” means the source describes the mechanism; “local” means the
XScientist pilot can expose an artifact or contract. It is not a score.

| Dimension | XScientist today | End-to-end systems in the talk | Focused systems in the talk and adjacent references |
| --- | --- | --- | --- |
| Problem framing | Local typed question/plan and explicit falsifier checks; no autonomous benchmark discovery in the pilot | Usually human-supplied research problem; ScientistOne and ARC describe investigator/scoping stages | FAR starts from a human-specified research direction and builds an open-problem pool; MARS/MLE-STAR/DS-STAR accept task tuples or data files; review/writing/figure systems start later |
| Literature | Provenance/contract surfaces exist, but no provider-backed retrieval in the pilot | ScientistOne and ARC explicitly ground/retrieve literature; AI-Researcher has a collector/filter stage | ScholarPeer makes live context retrieval its central contribution; PaperOrchestra consumes supplied material |
| Exploration | Branch and checkpoint metadata are observable; no model-generated search trajectory is claimed | AI Scientist v2 uses an experiment tree; ScientistOne uses parallel explore/exploit; DeepScientist uses persistent hierarchical search | FAR enumerates and attempts literature-derived candidates; MARS uses resource-aware MCTS; AdaEvolve/EvoX adapt evolutionary search; component systems do not claim full discovery search |
| Execution | Typed attempts, receipts, failures, and closure can be audited; no rollout score | All five end-to-end systems report execution, with different budgets and environments | MARS/MLE-STAR/DS-STAR focus execution/resource planning; review/writing/figure systems are downstream |
| Claims | Closure, review debt, provenance and gates are local observables; no quality inference | ScientistOne makes claim evidence chains explicit; ARC reports verified result reporting; others require artifact-level audit to compare | ScholarPeer critiques claims; PaperOrchestra/PaperBanana can improve presentation but do not prove truth |
| Writing/visuals | Output generation is not what the current pilot measures | End-to-end systems include writing; quality protocols differ | PaperOrchestra and PaperBanana isolate writing/figure quality and should be benchmarked separately |
| Feedback/evolution | Append-only history, repair/gate signals, and self-evolution contracts are inspectable | ARC and DeepScientist explicitly report persistent lessons; AI Scientist v2 is primarily tree-search | FAR reports allocation analysis but not a persistent self-evolution arm; MARS explicitly studies cross-branch lesson transfer |
| Process/fairness | Strongest current local signal: branch membership, bounded timeline, source totals, fairness checks, redaction | External papers report budgets/branches differently; no automatic claim that their histories are Git-like | Component benchmarks need their own matched harness |
| Reproduction | `fsck`, bundles, CAS/ARA pointers, and `trace → replay → verify` contracts | External systems publish varying logs/code; claims require their own rerun protocol | Paper/figure outputs need separate artifact and evaluator checks |

## What a fair experiment would require

To compare XScientist with any one end-to-end system, freeze all of the
following in a manifest and record the digest in every trajectory:

1. the same task statement, starting artifact, data snapshot, and permitted
   network/tool policy;
2. the same backbone/model version, prompt/adaptation commit, hardware,
   wall-clock/turn/GPU budget, cost ceiling, and seed count;
3. the same evaluator, direction, tolerance, retry rule, and stopping rule;
4. the same output contract (code, logs, claims, paper, and evidence package);
5. independent canonical reruns of submitted code, plus an artifact-aware
   audit of citations, numbers, method-code alignment, and specification
   violations;
6. process measures: branch count and fork base, failed/completed attempts,
   repairs, human interventions, time/cost, evidence completeness, and
   reviewer uncertainty.

The `workspace.process.fairness` object refuses `eligible` until task slice,
budget, evaluator, and base are evidenced. A visible branch is therefore not
silently treated as a fair experimental arm. The report also keeps
`artifact_scope: current_checkout_only` when it cannot prove per-branch
artifacts.

## What XScientist can learn from the comparison

This is an integration map, not a claim that one system subsumes the others:

- adopt ScientistOne's claim-level evidence-chain vocabulary as a bridge from
  typed research objects to paper claims;
- combine AutoResearchClaw's explicit failure-to-repair/evolution records with
  XScientist's append-only history and fairness gates;
- use DeepScientist/MARS-style findings memory and resource-aware planning, but
  store the lesson provenance and credit assignment as typed objects;
- adopt FAR's explicit research-direction → candidate-pool → attempt → judge →
  grade funnel, preserving every candidate outcome (including `NONE`, `KNOWN`,
  and failed/invalid status) instead of counting only successful discoveries;
- expose FAR-style expected-yield and expected-importance allocation as an
  optional policy, with calibration metadata and no claim that its reported
  combinatorics rates transfer to another domain;
- expose AdaEvolve/EvoX-style search-policy adaptation as an optional optimizer
  adapter, while retaining the candidate/strategy lineage and evaluator digest;
- use MLE-STAR/DS-STAR as execution-layer adapters for code/data tasks, keeping
  their task scores separate from claim and paper quality;
- connect ScholarPeer as an adversarial review adapter, retaining its external
  search receipts rather than copying prose into a claim;
- connect PaperOrchestra and PaperBanana as downstream writers/renderers whose
  inputs are verified artifacts, never the authority for experiment truth;
- add matched adapters for ADRS, ARC-Bench, Scientist-Bench, MLE-Bench,
  PaperWritingBench, and PaperBananaBench only after their versions, evaluators,
  and licensing/data boundaries are pinned.

## Explicit non-claims

At the current repository state we do **not** claim that XScientist:

- beats a human, ScientistOne, AI Scientist, AutoResearchClaw, DeepScientist,
  AI-Researcher, FAR, MARS, ScholarPeer, PaperOrchestra, or PaperBanana;
- has reproduced any number in the talk or in the external papers;
- has completed an autonomous provider-backed trajectory merely because its
  local process audit is complete;
- can turn a review/figure/writing score into a scientific-discovery score.

The honest current result is a process/evidence result: the project can make
intermediate decisions, branches, closure levels, and fairness blockers
inspectable without exporting prompts or hidden free-form reasoning. The next
step toward a scientific comparison is a registered, matched rollout—not a
larger unqualified table.
