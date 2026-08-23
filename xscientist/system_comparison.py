"""A source-audited, non-ranking comparison of research-agent systems.

The comparison is intentionally qualitative.  A capability reported by a
paper is not treated as a local reproduction, and a component benchmark is
never merged into an end-to-end research score.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ._version import __version__

SYSTEM_COMPARISON_SCHEMA = "xscientist.system-comparison.v1"

COMPARISON_DIMENSIONS: tuple[dict[str, str], ...] = (
    {"id": "framing", "label": "Problem framing / benchmark discovery"},
    {"id": "literature", "label": "Literature retrieval and provenance"},
    {"id": "exploration", "label": "Ideation, branches, and exploration"},
    {"id": "execution", "label": "Implementation and experiment execution"},
    {"id": "claims", "label": "Analysis and claim grounding"},
    {"id": "writing", "label": "Paper writing and visual output"},
    {"id": "review", "label": "Self-verification / adversarial review"},
    {"id": "evolution", "label": "Feedback, memory, and self-evolution"},
    {"id": "process", "label": "Intermediate process and fair comparison"},
    {"id": "reproduction", "label": "Reproduction and artifact export"},
)

CAPABILITY_STATUSES = frozenset(
    {
        "local_observed",
        "local_structural_only",
        "reported_primary",
        "reported_talk",
        "scoped_component",
        "not_measured_here",
        "not_in_scope",
    }
)

_ATTACHED_TALK_FILENAME = (
    "3.Expo Talk-05 The End-to-End AI Scientist - Automating Discovery and the "
    "Research Pipeline.pdf"
)
_ATTACHED_TALK_SHA256 = (
    "996ee0ebabbddd0bedfbf6c8702ebc81e41768b6a44f777a490fdd59bbc46404"
)


def _attached_talk_source(slide: int | str) -> dict[str, str]:
    """Return a stable, path-free locator for the reviewed talk slide."""

    return _source(
        f"Attached Expo Talk (slide {slide})",
        f"attachment://{_ATTACHED_TALK_FILENAME}#slide-{slide}",
        "attached_talk",
    )


def _source(title: str, url: str, kind: str = "primary_paper") -> dict[str, str]:
    return {"title": title, "url": url, "kind": kind}


def _row(
    system_id: str,
    name: str,
    scope: str,
    sources: list[dict[str, str]],
    capabilities: dict[str, str],
    anchors: list[str],
    distinction: str,
    *,
    talk_slides: list[int] | None = None,
    limitations: list[str] | None = None,
    source_status: str = "reported_primary",
    comparison_status: str = "not_measured_here",
) -> dict[str, Any]:
    dimension_ids = {item["id"] for item in COMPARISON_DIMENSIONS}
    unknown = set(capabilities) - dimension_ids
    missing = dimension_ids - set(capabilities)
    if unknown or missing:
        raise ValueError(
            f"{system_id}: capability dimensions unknown={sorted(unknown)} "
            f"missing={sorted(missing)}"
        )
    if any(value not in CAPABILITY_STATUSES for value in capabilities.values()):
        raise ValueError(f"{system_id}: unsupported capability status")
    return {
        "id": system_id,
        "name": name,
        "scope": scope,
        "source_status": source_status,
        "comparison_status": comparison_status,
        "sources": sources,
        "talk_slides": list(talk_slides or []),
        "capabilities": dict(capabilities),
        "benchmark_anchors": list(anchors),
        "distinction": distinction,
        "limitations": list(limitations or []),
    }


_E2E = {
    "framing": "reported_primary",
    "literature": "reported_primary",
    "exploration": "reported_primary",
    "execution": "reported_primary",
    "claims": "reported_primary",
    "writing": "reported_primary",
    "review": "reported_primary",
    "evolution": "not_measured_here",
    "process": "not_measured_here",
    "reproduction": "reported_primary",
}


_SYSTEMS: tuple[dict[str, Any], ...] = (
    _row(
        "xscientist",
        "XScientist",
        "Git-like evidence/provenance substrate and workflow surface",
        [
            _source(
                "XScientist repository",
                "https://github.com/smileformylove/XScientist",
                "local_repository",
            ),
            _source("Local benchmark protocol", "docs/BENCHMARKS.md", "local_protocol"),
        ],
        {
            "framing": "local_observed",
            "literature": "local_structural_only",
            "exploration": "local_structural_only",
            "execution": "local_structural_only",
            "claims": "local_observed",
            "writing": "not_measured_here",
            "review": "local_observed",
            "evolution": "local_observed",
            "process": "local_observed",
            "reproduction": "local_observed",
        },
        [
            "first-run usability benchmark",
            "AutoResearchEval-inspired offline conformance pilot",
            "trace → replay → verify closure and process-audit schema",
        ],
        "The current public pilot exposes typed artifacts, checkpoints, branch metadata, "
        "fairness gates, and bounded redacted process evidence; it runs zero model rollouts.",
        source_status="local_observed",
        comparison_status="local_structural_only",
        limitations=[
            "No matched external-system rollout or autonomous paper-quality score is claimed.",
            "A single checkout is not inferred to contain per-branch outcome artifacts.",
        ],
    ),
    _row(
        "deep_researcher_agent",
        "Google Deep Researcher Agent (talk reference)",
        "Broad research-assistant / deep-research product reference",
        [_attached_talk_source(7)],
        {
            "framing": "reported_talk",
            "literature": "reported_talk",
            "exploration": "reported_talk",
            "execution": "not_measured_here",
            "claims": "reported_talk",
            "writing": "reported_talk",
            "review": "reported_talk",
            "evolution": "reported_talk",
            "process": "not_measured_here",
            "reproduction": "not_measured_here",
        },
        ["Talk slide: Deep Researcher Agent / test-time diffusion"],
        "Included so the talk's broad product-level reference is not silently omitted; it is not treated as a reproducible peer system.",
        talk_slides=[7, 8],
        limitations=[
            "The attached talk is the only source recorded for this row in this audit.",
            "No standardized task slice, artifact package, or matched local rollout was identified here.",
        ],
        source_status="reported_talk",
    ),
    _row(
        "autonomous_science_team",
        "Autonomous Science Team (AST) framework",
        "Talk-level multi-agent role decomposition for research",
        [_attached_talk_source(57)],
        {
            "framing": "reported_talk",
            "literature": "reported_talk",
            "exploration": "reported_talk",
            "execution": "reported_talk",
            "claims": "reported_talk",
            "writing": "reported_talk",
            "review": "reported_talk",
            "evolution": "not_measured_here",
            "process": "reported_talk",
            "reproduction": "not_measured_here",
        },
        ["Talk slide: Generator → Implementor → Paper Writer → Paper Reviewer"],
        "Recorded as an architectural idea from the talk, not promoted to a separately evaluated system without a primary protocol.",
        talk_slides=[57],
        limitations=[
            "No independent paper, task manifest, score, or runnable repository was identified for this slide-level framework.",
            "The role diagram must not be mistaken for evidence of autonomous scientific quality.",
        ],
        source_status="reported_talk",
    ),
    _row(
        "scientistone",
        "Science One / ScientistOne",
        "End-to-end autonomous research with Chain-of-Evidence",
        [
            _source("ScientistOne paper", "https://arxiv.org/abs/2605.26340"),
            _source(
                "Science One official overview",
                "https://research.google/blog/science-one-framework-a-verifiable-autonomous-research-framework-via-chain-of-evidence/",
                "official_blog",
            ),
            _source(
                "ScientistOne generated artifacts",
                "https://github.com/scientist-one/generated-artifacts",
                "official_artifacts",
            ),
            _source(
                "ScientistOne project site",
                "https://scientist-one.github.io/",
                "official_project",
            ),
        ],
        {**_E2E, "process": "reported_primary"},
        [
            "ADRS five-task systems-optimization evaluation",
            "CoE Integrity Audit over 75 papers from five systems",
            "MLE-Bench and Parameter Golf generalization",
        ],
        "Builds evidence chains during literature grounding, parallel discovery, and claim-grounded writing, "
        "then applies four integrity checks to paper/code/reference artifacts.",
        talk_slides=[11, 21, 22, 25, 28, 35, 40, 45, 52, 54],
        limitations=[
            "Primary experiments emphasize deterministic systems-optimization evaluators.",
            "The paper leaves open-ended-domain verification and deeper citation entailment as future work.",
            "The ADRS human column is a published human-designed reference, not a newly recruited matched human arm.",
            "The public project organization exposes artifacts and documentation; a turnkey full-system repository was not confirmed.",
        ],
    ),
    _row(
        "ai_scientist_v2",
        "Sakana AI Scientist v2",
        "End-to-end agentic discovery with best-first experiment tree search",
        [
            _source("AI Scientist-v2 paper", "https://arxiv.org/abs/2504.08066"),
            _source(
                "SakanaAI AI-Scientist-v2",
                "https://github.com/SakanaAI/AI-Scientist-v2",
                "official_repository",
            ),
        ],
        {
            **_E2E,
            "literature": "not_measured_here",
            "evolution": "not_measured_here",
            "process": "not_measured_here",
        },
        [
            "Three autonomous manuscripts in an ICLR workshop setting",
            "Progressive agentic tree search",
        ],
        "A strong end-to-end baseline whose central abstraction is an experiment tree and a separate write-up pipeline.",
        talk_slides=[37, 43, 52, 91],
        limitations=[
            "The workshop result is not a matched score on XScientist's pilot.",
            "The paper describes human theme/idea selection around the autonomous runs; it is not a zero-human study design.",
            "The source does not establish the Git-like branch/fairness contract used here.",
        ],
    ),
    _row(
        "autoresearchclaw",
        "AutoResearchClaw",
        "End-to-end multi-agent pipeline with self-healing, HITL, and cross-run lessons",
        [
            _source("AutoResearchClaw paper", "https://arxiv.org/abs/2605.20025"),
            _source(
                "AutoResearchClaw repository",
                "https://github.com/aiming-lab/AutoResearchClaw",
                "official_repository",
            ),
        ],
        {**_E2E, "process": "reported_primary"},
        [
            "ARC-Bench experiment-stage and end-to-end modes",
            "Seven intervention regimes and mechanism ablations",
        ],
        "Makes debate, self-healing, verification, targeted human intervention, and persistent lessons first-class.",
        talk_slides=[37, 43, 52, 91],
        limitations=[
            "ARC-Bench has not been executed by XScientist here; version and budget must be pinned.",
            "HITL ablations are workflow evidence, not a human-only capability baseline.",
        ],
    ),
    _row(
        "deepscientist",
        "DeepScientist",
        "Long-horizon goal-oriented discovery with findings memory",
        [
            _source("DeepScientist paper", "https://arxiv.org/abs/2509.26603"),
            _source(
                "DeepScientist repository",
                "https://github.com/ResearAI/DeepScientist",
                "official_repository",
            ),
        ],
        {**_E2E, "process": "not_measured_here"},
        [
            "Month-long progressive discovery experiments",
            "Hierarchical hypothesize → verify → analyze loop",
            "Findings Memory / Bayesian-optimization framing",
        ],
        "Optimizes for persistent exploration and accumulated findings rather than only one short paper.",
        talk_slides=[37, 43, 52, 54, 91],
        limitations=[
            "The reported GPU scale is not comparable to the provider-free local pilot.",
            "Human-designed SOTA references are not recruited human runs.",
        ],
    ),
    _row(
        "ai_researcher",
        "AI-Researcher",
        "Orchestrated multi-agent literature → code → manuscript pipeline",
        [
            _source("AI-Researcher paper", "https://arxiv.org/abs/2505.18705"),
            _source(
                "HKUDS AI-Researcher",
                "https://github.com/hkuds/ai-researcher",
                "official_repository",
            ),
        ],
        {**_E2E, "evolution": "not_measured_here", "process": "not_measured_here"},
        [
            "Scientist-Bench guided-innovation and open-ended tasks",
            "Code-validate-refine loop",
        ],
        "Uses role-specialized survey, coding, and writing agents across a broad end-to-end pipeline.",
        talk_slides=[37, 43, 52, 91],
        limitations=[
            "Published benchmark results are not a matched run against this repository.",
            "Role decomposition alone is not evidence of claim-level provenance or branch fairness.",
        ],
    ),
    _row(
        "far",
        "FAR (Find–Attempt–Recommend)",
        "Literature-to-open-problem discovery and resource-allocation cascade",
        [
            _source(
                "The Problem Is the Problem: Towards Scalable Mathematical Discovery",
                "https://arxiv.org/abs/2608.16977",
            ),
            _source(
                "FAR official repository",
                "https://github.com/zeyu-zheng/FAR",
                "official_repository",
            ),
        ],
        {
            "framing": "reported_primary",
            "literature": "reported_primary",
            "exploration": "reported_primary",
            "execution": "reported_primary",
            "claims": "reported_primary",
            "writing": "not_in_scope",
            "review": "reported_primary",
            "evolution": "not_measured_here",
            "process": "reported_primary",
            "reproduction": "reported_primary",
        },
        [
            "Find–Attempt–Recommend cascade over a literature-derived open-problem pool",
            "Combinatorics pilot: 51,110 mathematics papers → 5,245 combinatorics papers",
            "Find stage: 6,453 candidate conjectures/open problems from 2,742 papers → 4,717 apparently well-posed and still-open conjectures",
            "4,717 conjectures attempted → 1,050 claimed NEW → 598 judged PASS → 77 graded publishable outcomes",
            "Difficulty/importance-guided allocation analysis (reported study; not rerun here)",
        ],
        "Starts from a human-specified research direction, extracts and checks open problems, "
        "attempts every apparently well-posed candidate, then independently judges and grades "
        "claimed discoveries before a small expert-review queue.",
        limitations=[
            "This is a domain-specific combinatorics pilot and a discovery/allocation cascade, not a standardized end-to-end benchmark.",
            "The cited source is an arXiv preprint; its reported funnel outcomes are not an independently audited cross-system benchmark.",
            "XScientist has not run FAR; all counts and outcomes remain reported_primary and comparison_status=not_measured_here.",
            "No same-condition recruited human task-performance arm is reported; expert review and judging are not a human baseline.",
            "The paper's 15 manually checked outcomes were author-selected and are not a benchmark accuracy estimate.",
            "Full reproduction requires the source corpus, API/tool configuration, and a pinned repository revision; the public repository is not a turnkey pilot bundle.",
        ],
        source_status="reported_primary",
        comparison_status="not_measured_here",
    ),
    _row(
        "adaevolve",
        "AdaEvolve",
        "Adaptive LLM-driven evolutionary optimization component",
        [
            _source("AdaEvolve paper", "https://arxiv.org/abs/2602.20133"),
            _source(
                "SkyDiscover implementation",
                "https://github.com/skydiscover-ai/skydiscover",
                "author_repository",
            ),
        ],
        {
            "framing": "scoped_component",
            "literature": "not_in_scope",
            "exploration": "reported_primary",
            "execution": "reported_primary",
            "claims": "scoped_component",
            "writing": "not_in_scope",
            "review": "scoped_component",
            "evolution": "reported_primary",
            "process": "reported_primary",
            "reproduction": "reported_primary",
        },
        ["ADRS systems-optimization tasks", "185 open-ended optimization tasks"],
        "Adapts exploration intensity and resource allocation from accumulated improvement signals; it is an optimization/search method, not a literature-to-paper pipeline.",
        talk_slides=[40, 41, 42],
        limitations=[
            "The talk's ADRS table is a reported external comparison, not a local rerun.",
            "A component optimization score cannot be compared directly with an end-to-end research score.",
        ],
    ),
    _row(
        "evox_meta",
        "EvoX (Meta-Evolution)",
        "Meta-evolution search-strategy component",
        [
            _source(
                "EvoX: Meta-Evolution for Automated Discovery",
                "https://arxiv.org/abs/2602.23413",
            ),
            _source(
                "SkyDiscover implementation",
                "https://github.com/skydiscover-ai/skydiscover",
                "author_repository",
            ),
        ],
        {
            "framing": "scoped_component",
            "literature": "not_in_scope",
            "exploration": "reported_primary",
            "execution": "reported_primary",
            "claims": "scoped_component",
            "writing": "not_in_scope",
            "review": "scoped_component",
            "evolution": "reported_primary",
            "process": "reported_primary",
            "reproduction": "reported_primary",
        },
        ["ADRS systems-optimization tasks", "Nearly 200 optimization tasks"],
        "Co-evolves candidate solutions and the strategy that selects and mutates them; this is a search primitive that can sit inside a research system.",
        talk_slides=[40, 41, 42],
        limitations=[
            "The source does not establish an end-to-end scientific framing, writing, or evidence contract.",
            "The talk's numbers remain reported references until the exact task/evaluator revision is pinned.",
        ],
    ),
    _row(
        "mars",
        "MARS",
        "MLE-focused modular search, resource-aware planning, and reflective memory",
        [
            _source("MARS paper", "https://arxiv.org/abs/2602.02660"),
            _source(
                "MARS repository",
                "https://github.com/jfc43/MARS",
                "official_repository",
            ),
        ],
        {
            "framing": "scoped_component",
            "literature": "not_in_scope",
            "exploration": "reported_primary",
            "execution": "reported_primary",
            "claims": "scoped_component",
            "writing": "not_in_scope",
            "review": "scoped_component",
            "evolution": "reported_primary",
            "process": "reported_primary",
            "reproduction": "reported_primary",
        },
        [
            "MLE-Bench",
            "Cost-constrained MCTS",
            "Design–Decompose–Implement",
            "Cross-branch lesson transfer",
        ],
        "Targets the MLE bottleneck: resource-aware planning, modular repositories, and credit assignment.",
        talk_slides=[77, 78, 84, 85, 87],
        limitations=[
            "It is not presented as a complete literature-to-paper evidence chain.",
            "MLE-Bench results are not directly comparable to XScientist's structural pilot.",
        ],
    ),
    _row(
        "mle_star",
        "MLE-STAR",
        "Machine-learning-engineering search and targeted refinement component",
        [
            _source("MLE-STAR paper", "https://arxiv.org/abs/2506.15692"),
            _source(
                "Google Research publication page",
                "https://research.google/pubs/mle-star-machine-learning-engineering-agent-via-search-and-targeted-refinement/",
                "official_page",
            ),
        ],
        {
            "framing": "scoped_component",
            "literature": "reported_primary",
            "exploration": "reported_primary",
            "execution": "reported_primary",
            "claims": "scoped_component",
            "writing": "not_in_scope",
            "review": "scoped_component",
            "evolution": "reported_primary",
            "process": "reported_primary",
            "reproduction": "reported_primary",
        },
        ["MLE-Bench Lite", "Web search plus targeted code-block refinement"],
        "Starts from retrieved task-specific solutions, uses ablations to select high-impact code blocks, and refines them in an inner/outer loop.",
        # MLE-STAR is included as a useful adjacent execution-layer reference,
        # but it is not named in the attached 107-page talk.  Keep the source
        # boundary explicit instead of inventing slide numbers.
        talk_slides=[],
        limitations=[
            "It targets model-development execution, not autonomous problem discovery or paper verification.",
            "MLE-Bench results require the original model, hardware, data, and evaluator protocol.",
        ],
    ),
    _row(
        "ds_star",
        "DS-STAR",
        "Data-science analysis and planning component",
        [
            _source("DS-STAR paper", "https://arxiv.org/abs/2509.21825"),
            _source(
                "DS-STAR repository",
                "https://github.com/google-research/ds-star",
                "official_repository",
            ),
        ],
        {
            "framing": "scoped_component",
            "literature": "not_in_scope",
            "exploration": "reported_primary",
            "execution": "reported_primary",
            "claims": "reported_primary",
            "writing": "not_in_scope",
            "review": "reported_primary",
            "evolution": "reported_primary",
            "process": "reported_primary",
            "reproduction": "reported_primary",
        },
        ["DABStep", "KramaBench", "DA-Code"],
        "Analyzes heterogeneous data files, executes a simple plan, and iteratively refines it when a verifier finds the plan insufficient.",
        # DS-STAR is likewise a primary-source adjacent comparison, not a
        # claim about content found in the attached talk.
        talk_slides=[],
        limitations=[
            "It is a data-science task solver, not a literature-grounded discovery-to-paper system.",
            "Open-ended verifier judgments and benchmark revisions must be pinned for a fair rerun.",
        ],
    ),
    _row(
        "scholarpeer",
        "ScholarPeer",
        "Search-enabled multi-agent peer review",
        [
            _source("ScholarPeer paper", "https://arxiv.org/abs/2601.22638"),
            _source(
                "Google Research overview",
                "https://research.google/blog/improving-the-academic-workflow-introducing-two-ai-agents-for-better-figures-and-peer-review/",
                "official_blog",
            ),
        ],
        {
            "framing": "scoped_component",
            "literature": "reported_primary",
            "exploration": "not_in_scope",
            "execution": "not_in_scope",
            "claims": "reported_primary",
            "writing": "not_in_scope",
            "review": "reported_primary",
            "evolution": "not_measured_here",
            "process": "reported_primary",
            "reproduction": "scoped_component",
        },
        [
            "ScholarEval (DeepReview-Bench + AgentReview; also a DeepReview-13K subset)",
            "Context acquisition and active verification",
        ],
        "A review-stage component for adversarial critique, historical context, and missing-baseline search—not execution.",
        talk_slides=[58, 67, 68, 69, 70, 71],
        limitations=[
            "Review quality is not an end-to-end discovery score.",
            "Automated review remains a proxy for expert judgment.",
            "ScholarEval's human-review/H-Max values are reference labels and normalization, not a newly recruited matched reviewer arm.",
            "An official end-to-end ScholarPeer repository was not identified in the source audit.",
        ],
    ),
    _row(
        "paperorchestra",
        "PaperOrchestra",
        "Multi-agent synthesis of research materials into a paper",
        [
            _source("PaperOrchestra paper", "https://arxiv.org/abs/2604.05018"),
            _source(
                "PaperOrchestra repository",
                "https://github.com/google-research/paper-orchestra",
                "official_repository",
            ),
            _source(
                "PaperWritingBench dataset",
                "https://huggingface.co/datasets/yiwen-song/PaperWritingBench",
                "author_dataset",
            ),
        ],
        {
            "framing": "not_in_scope",
            "literature": "reported_primary",
            "exploration": "not_in_scope",
            "execution": "not_in_scope",
            "claims": "scoped_component",
            "writing": "reported_primary",
            "review": "reported_primary",
            "evolution": "not_in_scope",
            "process": "scoped_component",
            "reproduction": "scoped_component",
        },
        [
            "PaperWritingBench: raw materials from 200 top-tier papers",
            "Human side-by-side writing evaluation",
        ],
        "Starts after experiments: its unit of automation is manuscript synthesis, literature review, and visuals.",
        talk_slides=[88, 94, 95, 98, 99, 101, 102],
        limitations=[
            "It cannot establish that supplied experiments were valid.",
            "The benchmark is writing-focused, not an autonomous research benchmark.",
            "The repository notes that benchmark data availability is separate and must be pinned/verified.",
        ],
    ),
    _row(
        "paperbanana",
        "PaperBanana",
        "Agentic academic illustration and plot generation",
        [
            _source("PaperBanana paper", "https://arxiv.org/abs/2601.23265"),
            _source(
                "Google Research PaperVizAgent",
                "https://github.com/google-research/papervizagent",
                "official_repository",
            ),
            _source(
                "PaperBanana author repository",
                "https://github.com/dwzhu-pku/PaperBanana",
                "author_repository",
            ),
            _source(
                "PaperBananaBench dataset",
                "https://huggingface.co/datasets/dwzhu/PaperBananaBench",
                "author_dataset",
            ),
        ],
        {
            "framing": "not_in_scope",
            "literature": "scoped_component",
            "exploration": "not_in_scope",
            "execution": "not_in_scope",
            "claims": "scoped_component",
            "writing": "reported_primary",
            "review": "reported_primary",
            "evolution": "not_measured_here",
            "process": "scoped_component",
            "reproduction": "scoped_component",
        },
        [
            "PaperBananaBench: 292 methodology-diagram cases",
            "Faithfulness, conciseness, readability, aesthetics",
        ],
        "A visual-output component with retrieval, planning, rendering, and critique; it complements a research trajectory.",
        talk_slides=[72, 74, 75, 76],
        limitations=[
            "Figure quality cannot stand in for scientific validity or discovery quality.",
            "The figure benchmark is not comparable to an end-to-end research score.",
            "External dataset splits and evaluator mapping must be pinned before claiming full reproduction.",
            "The paper's Human=50 convention is an evaluator reference/tie scale, not a measured human drawing score.",
        ],
    ),
)

# Human evidence is deliberately a first-class, machine-readable field.  A
# missing participant arm is represented as ``not_reported`` rather than being
# inferred from a paper's human-authored reference solution or reviewer scores.
_HUMAN_EVIDENCE_DEFAULT = {
    "status": "not_reported",
    "same_condition": False,
    "score": None,
    "source": "docs/HUMAN_BASELINES.md",
    "note": "No same-condition recruited human task-performance arm was used here.",
}
_HUMAN_EVIDENCE_BY_ID: dict[str, dict[str, Any]] = {
    "far": {
        "status": "not_reported",
        "note": "No same-condition recruited human task-performance arm is reported; expert judging/review is not a human baseline.",
    },
    "scientistone": {
        "status": "human_SOTA_reference",
        "note": "ADRS human column is a published human-designed reference, not a new participant run.",
    },
    "ai_scientist_v2": {
        "status": "human_judgment_calibration",
        "note": "Workshop paper review and human setup choices are not a matched human capability arm.",
    },
    "autoresearchclaw": {
        "status": "human_agent_process",
        "note": "HITL/intervention ablations are process evidence, not human-only performance.",
    },
    "deepscientist": {
        "status": "human_SOTA_reference",
        "note": "Human-designed SOTA and reviewer references are not recruited same-condition runs.",
    },
    "ai_researcher": {
        "status": "human_reference_proxy",
        "note": "Human-authored target papers are references; a matched human Scientist-Bench run is not reported.",
    },
    "adaevolve": {
        "status": "human_SOTA_reference",
        "note": "ADRS human values are historical/reference results, not a new participant arm.",
    },
    "evox_meta": {
        "status": "human_SOTA_reference",
        "note": "ADRS human values are historical/reference results, not a new participant arm.",
    },
    "mle_star": {
        "status": "human_reference_proxy",
        "note": "MLE-Bench public leaderboard results are heterogeneous historical references, not matched runs.",
    },
    "scholarpeer": {
        "status": "human_judgment_calibration",
        "note": "Existing human reviews and expert judge calibration are not newly recruited same-task reviewers.",
    },
    "paperorchestra": {
        "status": "human_judgment_calibration",
        "note": "Eleven researchers provide side-by-side preferences; they do not independently write the benchmark cases.",
    },
    "paperbanana": {
        "status": "human_judgment_calibration",
        "note": "Human/LLM visual judgments calibrate the evaluator; no matched human illustration arm is reported.",
    },
}

_HUMAN_EVIDENCE_STATUSES = frozenset(
    {
        "not_reported",
        "human_reference_proxy",
        "human_SOTA_reference",
        "human_judgment_calibration",
        "human_agent_process",
    }
)

for _system in _SYSTEMS:
    _evidence = deepcopy(_HUMAN_EVIDENCE_DEFAULT)
    _evidence.update(_HUMAN_EVIDENCE_BY_ID.get(_system["id"], {}))
    _system["human_evidence"] = _evidence


def _local_process_observation(workspace: str | Path | None) -> dict[str, Any]:
    """Include only a redacted local process summary, never a score."""

    base = {
        "workspace_supplied": workspace is not None,
        "path_disclosed": False,
        "rollouts": 0,
        "rollout_scope": "this_audit_only",
        "provider_calls": 0,
        "network_used": False,
        "model_cost_usd": 0.0,
        "cost_scope": "this_audit_only",
        "historical_trajectory_cost": "unobserved",
        "score": None,
    }
    if workspace is None:
        return {"status": "not_requested", "process": None, **base}
    try:
        root = Path(workspace).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return {
            "status": "unavailable",
            "process": None,
            **base,
        }
    if not root.is_dir():
        raise ValueError("workspace does not exist or is not a directory")
    try:
        from .process_audit import build_process_summary

        process = build_process_summary(root, gold_fields_used=False)
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        UnicodeError,
        OverflowError,
        RuntimeError,
        RecursionError,
        MemoryError,
    ) as exc:
        from .process_audit import _unavailable_summary

        process = _unavailable_summary(
            task_manifest_sha256=None,
            task_count=None,
            task_filter="all",
            task_limit=None,
            gold_fields_used=False,
            errors=[type(exc).__name__],
            limits={
                "max_branches": 32,
                "max_commits": 32,
                "max_artifacts": 96,
                "max_decisions": 32,
            },
        )
    return {"status": "local_process_audit", "process": process, **base}


def build_system_comparison(workspace: str | Path | None = None) -> dict[str, Any]:
    """Build a provider-free, non-ranking capability comparison.

    External systems are represented by published scope and benchmark anchors;
    they are not executed or assigned a score.  With ``workspace`` the report
    embeds XScientist's existing redacted process audit so branch and
    intermediate-artifact visibility can be inspected beside the matrix.
    """

    return {
        "schema": SYSTEM_COMPARISON_SCHEMA,
        "version": __version__,
        "ok": True,
        "comparison_mode": "qualitative_source_audit",
        "official_comparable": False,
        "score_claim_allowed": False,
        "quality_claim_allowed": False,
        "network_used": False,
        "provider_used": False,
        "external_rollouts": 0,
        "source_manifest": {
            "attached_talk": {
                "filename": _ATTACHED_TALK_FILENAME,
                "sha256": _ATTACHED_TALK_SHA256,
                "page_count": 107,
                "role": "scope_discovery_only; talk-only claims remain reported_talk",
            }
        },
        "talk_inventory": {
            "included_as_rows": [
                "ScientistOne",
                "AI Scientist v2",
                "AutoResearchClaw",
                "DeepScientist",
                "AI-Researcher",
                "MARS",
                "AdaEvolve",
                "EvoX",
                "ScholarPeer",
                "PaperOrchestra",
                "PaperBanana",
                "Deep Researcher Agent",
                "Autonomous Science Team (AST) framework",
            ],
            "adjacent_references_not_ranked": [
                "CAST example paper",
                "Google Paper Assistant Tool (PAT) integration",
                "SingleAgent and Human (GD/GT) benchmark arms",
                "AutoResearchEval benchmark itself",
                "FAR (adjacent discovery/allocation reference; not named in attached talk)",
                "MLE-STAR (adjacent execution-layer reference; not named in attached talk)",
                "DS-STAR (adjacent data-science reference; not named in attached talk)",
            ],
            "context_only_mentions": [
                "Agentic AI for Forecasting (slide 7; no standalone protocol identified)",
                "Agentic AI for Data Analytics (slide 7; no standalone protocol identified)",
                "Agent Memory System / MemoryBank (slide 7; no standalone protocol identified)",
                "ScientistTwo (slide 105; future concept, not an evaluated system)",
            ],
            "note": "A named talk item is either represented as a scoped row or explicitly listed as an adjacent reference; neither list is a score leaderboard.",
        },
        "source_policy": {
            "primary_source": "reported capability, not local reproduction",
            "attached_talk": "scope discovery; talk-only details remain talk_reported",
            "local_observed": "reserved for artifacts produced by the local harness",
            "not_measured_here": "no inference of superiority, failure, or absence",
            "repository_label": "author/research-code release; not a product-support claim",
            "google_repository_disclaimer": "research-code release only; not an officially supported product",
            "dataset_label": "author-linked artifact; split and revision must be pinned",
        },
        "dimensions": deepcopy(list(COMPARISON_DIMENSIONS)),
        "systems": deepcopy(list(_SYSTEMS)),
        "xscientist_local": _local_process_observation(workspace),
        "interpretation": (
            "Rows cover different layers of the research stack. Component benchmarks "
            "must not be ranked against end-to-end discovery benchmarks, and local "
            "structural observations must not be converted into model-quality scores."
        ),
    }


__all__ = [
    "CAPABILITY_STATUSES",
    "COMPARISON_DIMENSIONS",
    "SYSTEM_COMPARISON_SCHEMA",
    "build_system_comparison",
]
