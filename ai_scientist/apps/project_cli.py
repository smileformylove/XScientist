"""Command-line contract for running one XScientist research project."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from ai_scientist.config.venues import TARGET_VENUES


def _model_default(role: str, fallback: str) -> str:
    role_name = f"AI_SCIENTIST_MODEL_{role.upper()}"
    return (
        os.environ.get(role_name)
        or os.environ.get("AI_SCIENTIST_DEFAULT_MODEL")
        or fallback
    )


def build_parser(
    *,
    default_output_root: str,
    default_writing_profile: str,
    writing_profiles: Sequence[str],
    workflow_modes: Sequence[str],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XScientist project runner - process multiple papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Start from a plain-language question with guarded Autopilot:
   xscientist project my_project --question "Why does this mechanism fail out of distribution?" --autopilot discovery

2. Generate three ideas and process them in parallel:
   python -m xscientist project my_project --topic topic.md --num-ideas 3 --parallel

3. Process two existing ideas in parallel:
   python -m xscientist project my_project --ideas ideas.json --idea-indices 0,1 --parallel

4. Run two bounded improvement rounds:
   python -m xscientist project my_project --topic topic.md --improvement-rounds 2
        """,
    )

    parser.add_argument(
        "project_dir",
        type=str,
        help="project directory (relative paths resolve below --output-root/projects)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=default_output_root,
        help="shared output root used when project_dir is relative",
    )

    parser.add_argument("--topic", type=str, help="topic description file")
    parser.add_argument(
        "--question",
        type=str,
        help="plain-language research question (mutually exclusive with --topic/--ideas)",
    )
    parser.add_argument("--ideas", type=str, help="existing ideas JSON file")
    parser.add_argument(
        "--autopilot",
        nargs="?",
        const="balanced",
        choices=["balanced", "discovery", "publication"],
        default=None,
        help=(
            "Guarded automation preset: balanced controls cost, discovery increases "
            "rival/refutation pressure, and publication strengthens review gates."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume safely from progress.json and the latest valid BFTS checkpoint",
    )
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="read-only experiment inputs; Autopilot hashes every file before model calls",
    )
    data_group.add_argument(
        "--allow-synthetic-data",
        action="store_true",
        help="allow synthetic/computational data; results remain exploratory and unverified",
    )
    parser.add_argument(
        "--max-project-tokens",
        type=int,
        default=None,
        help="lower project-wide LLM token ceiling shared by all stages",
    )
    parser.add_argument(
        "--max-project-hours",
        type=float,
        default=None,
        help="lower project-wide LLM wall-clock ceiling",
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        help="hard project-wide USD limit; unknown model prices fail closed",
    )
    parser.add_argument(
        "--model-ideation",
        type=str,
        default=_model_default("IDEATION", "glm-4-flash"),
    )
    parser.add_argument(
        "--num-ideas", type=int, default=3, help="number of ideas to generate"
    )
    parser.add_argument("--num-reflections", type=int, default=5)

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="process multiple ideas concurrently",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="number of parallel workers",
    )
    parser.add_argument(
        "--idea-indices",
        type=str,
        help="comma-separated idea indices, for example 0,1,2",
    )
    parser.add_argument(
        "--rank-ideas", action="store_true", help="rank ideas before selecting them"
    )
    parser.add_argument(
        "--top-k-ideas",
        type=int,
        default=None,
        help="process only the top K ranked ideas",
    )
    parser.add_argument(
        "--idea-rank-model",
        type=str,
        default=None,
        help="idea-ranking model; comma-separate independent reviewer models",
    )
    parser.add_argument("--submission-mode", action="store_true")
    parser.add_argument("--fallback-ranked-ideas", action="store_true")
    parser.add_argument("--breakthrough-mode", action="store_true")
    parser.add_argument(
        "--workflow-mode",
        type=str,
        choices=list(workflow_modes),
        default="adaptive",
        help="research orchestration mode, from classic templates to agentic/review-board flows",
    )
    parser.add_argument(
        "--override-strict-fallbacks",
        action="store_true",
        help="allow recorded fallbacks that strict publication/high-quality modes normally block",
    )

    parser.add_argument(
        "--seed-from-ara",
        type=str,
        default=None,
        help=(
            "Directory produced by `run_ara_fork.py fork`, or an ARA root used "
            "with --seed-node-id. The first BFTS draft reuses its code without an LLM."
        ),
    )
    parser.add_argument(
        "--seed-node-id",
        type=str,
        default=None,
        help="node_id to seed when --seed-from-ara points to an ARA root",
    )

    parser.add_argument(
        "--improvement-rounds",
        type=int,
        default=1,
        help="bounded reflection/improvement rounds per paper",
    )

    parser.add_argument("--skip-ideation", action="store_true")
    parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help=(
            "skip the entire per-idea experiment -> plot -> writeup -> review "
            "pipeline; this run does not generate a paper"
        ),
    )
    parser.add_argument(
        "--bfts-config",
        type=str,
        default="bfts_config.yaml",
        help="BFTS config controlling search depth, seeds, parallelism, and timeouts",
    )

    parser.add_argument(
        "--model-agg-plots",
        type=str,
        default=_model_default("AGG_PLOTS", "glm-4-flash"),
    )
    parser.add_argument(
        "--model-writeup",
        type=str,
        default=_model_default("WRITEUP", "glm-4-plus"),
    )
    parser.add_argument(
        "--model-writeup-small",
        type=str,
        default=_model_default("WRITEUP_SMALL", "glm-4-air"),
    )
    parser.add_argument(
        "--model-citation",
        type=str,
        default=_model_default("CITATION", "glm-4-air"),
    )
    parser.add_argument(
        "--model-review",
        type=str,
        default=_model_default("REVIEW", "glm-4-plus"),
    )

    parser.add_argument("--num-cite-rounds", type=int, default=15)
    parser.add_argument("--writeup-retries", type=int, default=3)
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb", "journal", "extended"],
    )
    parser.add_argument("--review-reflections", type=int, default=1)
    parser.add_argument("--review-ensemble", type=int, default=1)
    parser.add_argument("--review-fewshot", type=int, default=1)
    parser.add_argument("--review-temperature", type=float, default=0.75)
    parser.add_argument(
        "--review-strategy",
        type=str,
        choices=[
            "standard",
            "fast",
            "depth",
            *TARGET_VENUES,
        ],
        default=None,
    )
    parser.add_argument("--high-quality-mode", action="store_true")
    parser.add_argument(
        "--quality-preset",
        choices=["balanced", "high", "publishable"],
        default="balanced",
    )
    parser.add_argument("--quality-model", type=str, default=None)
    parser.add_argument(
        "--target-venue",
        type=str,
        choices=list(TARGET_VENUES),
        default=None,
        help=(
            "Venue-specific template and quality policy; this target does not "
            "predict or guarantee acceptance."
        ),
    )
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--rigor-threshold", type=float, default=None)
    parser.add_argument("--quality-rewrite-rounds", type=int, default=None)
    parser.add_argument("--autonomous-quality-followup-rounds", type=int, default=0)
    parser.add_argument("--min-submission-priority", type=float, default=None)
    parser.add_argument("--max-submission-blockers", type=int, default=None)
    parser.add_argument("--require-quality-gate", action="store_true")
    parser.add_argument(
        "--integrity-forensics",
        dest="integrity_forensics",
        action="store_true",
        default=None,
        help="enable deterministic final-manuscript integrity forensics",
    )
    parser.add_argument(
        "--no-integrity-forensics",
        dest="integrity_forensics",
        action="store_false",
        help="disable deterministic final-manuscript integrity forensics",
    )
    parser.add_argument("--auto-adjust-paper-type", action="store_true")
    parser.add_argument(
        "--writing-profile",
        type=str,
        choices=list(writing_profiles),
        default=default_writing_profile,
        help="writing prompt profile controlling constraints and reflection checks",
    )
    parser.add_argument(
        "--writing-audit-rounds",
        type=int,
        default=0,
        help="additional structured writing-audit rounds during reflection",
    )
    parser.add_argument(
        "--strict-writing-guardrails",
        action="store_true",
        help="fail the final draft when critical citation or section gaps remain",
    )
    parser.add_argument(
        "--guardrail-repair-rounds",
        type=int,
        default=1,
        help="automatic repair rounds before strict writing guard failure",
    )
    parser.add_argument(
        "--disable-hostile-critic",
        action="store_true",
        help="benchmark ablation: disable the independent hostile-critic channel",
    )
    parser.add_argument(
        "--disable-owner-aware-repair",
        action="store_true",
        help="benchmark ablation: disable owner-aware reviewer repair routing",
    )
    parser.add_argument(
        "--research-vcs",
        "--research-git",
        dest="research_git",
        choices=["off", "local"],
        default="local",
        help=(
            "Enable native Research VCS (Git is a replaceable storage backend); "
            "local by default, server-free, and never pushed automatically."
        ),
    )
    parser.add_argument(
        "--checkpoint-policy",
        "--git-checkpoint-policy",
        dest="git_checkpoint_policy",
        choices=["manual", "stage", "milestone"],
        default="milestone",
        help=(
            "Research checkpoint policy: milestone records key scientific states; "
            "stage records every requested phase."
        ),
    )
    parser.add_argument(
        "--research-vcs-strict",
        "--research-git-strict",
        dest="research_git_strict",
        action="store_true",
        help="stop on Research VCS init/checkpoint failure instead of warning and preserving outputs",
    )
    parser.add_argument("--git-user-name", default=None)
    parser.add_argument("--git-user-email", default=None)
    return parser
