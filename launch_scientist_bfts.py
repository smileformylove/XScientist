import json
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from ai_scientist.utils.launcher_workflow import (
    prepare_idea_artifacts,
    run_experiment_phase,
    run_review_phase,
    run_writeup_phase,
)
from ai_scientist.utils.launcher_cli import normalize_common_launcher_args
from ai_scientist.utils.pipeline_helpers import (
    cleanup_child_processes,
    find_best_pdf_path,
    get_available_gpus,
    save_token_tracker as persist_token_tracker,
)
from ai_scientist.utils.workflow_selection import (
    resolve_paper_type_for_venue,
    select_ranked_idea_candidates,
)
from ai_scientist.utils.fallback_audit import (
    format_strict_fallback_error,
    record_ranking_fallbacks,
    should_enforce_strict_fallbacks,
)
from ai_scientist.utils.auth_session import require_login
from ai_scientist.utils.runtime_bootstrap import (
    format_project_relative_path,
    initialize_runtime,
    require_model_credentials,
    resolve_writing_profile_env,
)
from ai_scientist.writing_prompt_profiles import (
    DEFAULT_WRITING_PROFILE,
    list_writing_profiles,
)
from ai_scientist.utils.workflow_modes import list_workflow_modes

# 导入路径配置
from ai_scientist.config.paths import (
    get_experiment_dir,
)


def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def save_token_tracker(idea_dir):
    persist_token_tracker(idea_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb", "journal", "extended"],
        help="Type of writeup to generate (normal=8 page, icbinb=4 page, journal=12 page, extended=2 page)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ideas/i_cant_believe_its_not_better.json",
        help="Path to a JSON file containing pregenerated ideas",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="If set, load a Python file with same name as ideas file but .py extension",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument("--auto-best-idea", action="store_true", help="Rank loaded ideas and run the highest-scoring one")
    parser.add_argument("--idea-rank-model", type=str, default=None, help="Model used to rank loaded ideas")
    parser.add_argument("--fallback-ranked-ideas", action="store_true", help="If the best idea fails the quality gate, try the next ranked idea")
    parser.add_argument("--auto-adjust-paper-type", action="store_true", help="Automatically switch to a better paper type for the target venue")
    parser.add_argument("--submission-mode", action="store_true", help="Enable a full submission-grade preset")
    parser.add_argument("--breakthrough-mode", action="store_true", help="Bias the system toward major-problem, high-impact submissions")
    parser.add_argument(
        "--workflow-mode",
        type=str,
        choices=list_workflow_modes(),
        default="adaptive",
        help="Research orchestration mode.",
    )
    parser.add_argument("--max-ranked-candidates", type=int, default=None, help="Maximum ranked ideas to try when fallback is enabled")
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="If set, add a HF dataset reference to the idea",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts to try",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID, used to distinguish same idea in different attempts in parallel runs",
    )
    parser.add_argument(
        "--bfts-config",
        type=str,
        default="bfts_config.yaml",
        help="Path to the BFTS experiment config YAML (controls search depth, seeds, workers, timeouts).",
    )
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="o3-mini-2025-01-31",
        help="Model to use for plot aggregation",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="o1-preview-2024-09-12",
        help="Model to use for writeup",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for citation gathering",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds to perform",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="gpt-4o-2024-05-13",
        help="Smaller model to use for writeup",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="gpt-4o-2024-11-20",
        help="Model to use for review main text and captions",
    )
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="If set, skip the writeup process",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="If set, skip the review process",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="If set, ignore completed workflow state and rerun all stages",
    )
    parser.add_argument(
        "--high-quality-mode",
        action="store_true",
        help="Enable stronger review, quality gating, and targeted rewrite passes",
    )
    parser.add_argument(
        "--quality-preset",
        choices=["balanced", "high", "publishable"],
        default="balanced",
        help="Preset for high-quality paper generation",
    )
    parser.add_argument("--quality-model", type=str, default=None, help="Model used for quality evaluation")
    parser.add_argument("--target-venue", type=str, choices=["neurips", "iclr", "cvpr", "journal", "nature"], default=None)
    parser.add_argument("--quality-threshold", type=float, default=None, help="Minimum target quality score")
    parser.add_argument("--rigor-threshold", type=float, default=None, help="Minimum target rigor score")
    parser.add_argument("--quality-rewrite-rounds", type=int, default=None, help="Maximum targeted rewrite rounds")
    parser.add_argument("--autonomous-quality-followup-rounds", type=int, default=0, help="Maximum autonomous quality follow-up rounds when submission bar is not met")
    parser.add_argument("--min-submission-priority", type=float, default=None, help="Minimum submission priority score to accept a draft")
    parser.add_argument("--max-submission-blockers", type=int, default=None, help="Maximum blocker count allowed for an accepted draft")
    parser.add_argument(
        "--require-quality-gate",
        action="store_true",
        help="Fail the run when high-quality mode does not pass the quality gate",
    )
    parser.add_argument("--review-reflections", type=int, default=1, help="Reflection rounds for textual review")
    parser.add_argument("--review-ensemble", type=int, default=1, help="How many review samples to ensemble")
    parser.add_argument("--review-fewshot", type=int, default=1, help="Few-shot review exemplars")
    parser.add_argument("--review-temperature", type=float, default=0.75, help="Temperature for review generation")
    parser.add_argument(
        "--review-strategy",
        choices=["standard", "fast", "depth", "neurips", "iclr", "cvpr", "journal", "nature"],
        default=None,
        help="Review strategy preset",
    )
    parser.add_argument(
        "--writing-profile",
        type=str,
        choices=list_writing_profiles(),
        default=resolve_writing_profile_env(
            invalid_profile_logger=lambda exc, raw: print(
                "Warning: invalid AI_SCIENTIST_WRITING_PROFILE="
                f"{raw!r}; falling back to {DEFAULT_WRITING_PROFILE}"
            )
        ),
        help="Prompt writing profile used to guide style and self-checks.",
    )
    parser.add_argument(
        "--writing-audit-rounds",
        type=int,
        default=0,
        help="Number of structured writing audit rounds injected into reflection loop.",
    )
    parser.add_argument(
        "--strict-writing-guardrails",
        action="store_true",
        help="Fail writeup when final citation/section guardrails are not satisfied.",
    )
    parser.add_argument(
        "--guardrail-repair-rounds",
        type=int,
        default=1,
        help="Automatic repair rounds attempted before strict guardrail failure.",
    )
    parser.add_argument(
        "--override-strict-fallbacks",
        action="store_true",
        help="Disable strict fallback blocking while still recording fallback debt.",
    )
    return parser.parse_args()


def find_pdf_path_for_review(idea_dir):
    return find_best_pdf_path(idea_dir, prefer_reflections=True)


def _collect_requested_models(args) -> list[str]:
    candidates = [
        args.model_agg_plots,
        args.model_writeup,
        args.model_citation,
        args.model_writeup_small,
        args.model_review,
        args.idea_rank_model,
        args.quality_model,
    ]
    models: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        model = str(value or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        models.append(model)
    return models


if __name__ == "__main__":
    require_login("科学家启动入口(launch_scientist_bfts)")

    args = parse_arguments()
    normalize_common_launcher_args(
        args,
        invalid_profile_logger=lambda exc: print(
            f"Warning: invalid writing profile: {exc}; falling back to {DEFAULT_WRITING_PROFILE}"
        ),
    )
    runtime = initialize_runtime(
        source_file=__file__,
        ensure_dirs=True,
        apply_cache=True,
    )
    print(f"Set AI_SCIENTIST_ROOT to {runtime.project_root}")
    research_root = runtime.research_root

    # Check available GPUs and adjust parallel processes if necessary
    available_gpus = get_available_gpus()
    print(f"Using GPUs: {available_gpus}")
    print(f"Writing profile: {args.writing_profile}")
    print(f"Workflow mode: {args.workflow_mode}")
    print(f"Writing audit rounds: {args.writing_audit_rounds}")
    strict_fallbacks = should_enforce_strict_fallbacks(
        args.workflow_mode,
        submission_mode=bool(args.submission_mode),
        high_quality_mode=bool(args.high_quality_mode),
        target_venue=args.target_venue,
    )
    if args.override_strict_fallbacks and strict_fallbacks:
        print(
            "Warning: override-strict-fallbacks enabled; fallback events will be recorded but not block this run."
        )
        strict_fallbacks = False
    elif strict_fallbacks:
        print(
            "Strict fallback policy active: ranking or quality fallbacks will stop this launcher run."
        )
    print(f"Strict writing guardrails: {args.strict_writing_guardrails}")
    print(f"Guardrail repair rounds: {args.guardrail_repair_rounds}")
    require_model_credentials(_collect_requested_models(args))

    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
        print(f"Loaded {len(ideas)} pregenerated ideas from {args.load_ideas}")

    candidate_indices, rankings = select_ranked_idea_candidates(
        ideas,
        ranking_enabled=args.auto_best_idea,
        ranking_model=args.idea_rank_model or args.model_writeup,
        target_venue=args.target_venue,
        prioritize_breakthrough=args.breakthrough_mode,
        research_root=research_root,
        ranking_output_path=Path(args.load_ideas).with_suffix(".ranking.json"),
        default_indices=[args.idea_idx],
        fallback_to_ranked=args.fallback_ranked_ideas,
        limit=(
            args.max_ranked_candidates
            if (args.fallback_ranked_ideas or not args.auto_best_idea)
            else None
        ),
    )
    if rankings and candidate_indices:
        args.idea_idx = candidate_indices[0]
        print(f"Auto-selected best idea index: {args.idea_idx} ({rankings[0].get('idea_name')}); ranking_score={rankings[0].get('ranking_score')}")

    args.writeup_type = resolve_paper_type_for_venue(
        args.writeup_type,
        args.target_venue,
        auto_adjust=args.auto_adjust_paper_type,
        warning_template="Warning: writeup_type '{paper_type}' may be a weak fit for target venue '{target_venue}'",
        adjusted_template="Auto-adjusted writeup_type to '{adjusted}'",
    )

    final_exit_code = 1
    for candidate_idx in candidate_indices:
        idea = ideas[candidate_idx]

        idea_dir = str(get_experiment_dir(idea['Name'], args.attempt_id))
        print(f"Results will be saved in {idea_dir}")
        print(
            "(Relative to project root: "
            f"{format_project_relative_path(idea_dir, project_root=runtime.project_root)})"
        )
        os.makedirs(idea_dir, exist_ok=True)
        ranking_event = record_ranking_fallbacks(
            idea_dir,
            rankings,
            producer="launch_scientist_bfts.idea_ranking",
            strict=strict_fallbacks,
        )
        if strict_fallbacks and ranking_event:
            print(
                format_strict_fallback_error(
                    ranking_event,
                    workflow_mode=args.workflow_mode,
                    stage_hint="idea_ranking",
                )
            )
            final_exit_code = 1
            break

        idea, idea_path_json = prepare_idea_artifacts(
            ideas,
            candidate_idx,
            args.load_ideas,
            idea_dir,
            load_code=args.load_code,
            add_dataset_ref=args.add_dataset_ref,
            echo_added_code=True,
        )

        run_experiment_phase(
            idea_dir,
            idea_path_json,
            args.model_agg_plots,
            config_path=args.bfts_config,
            resume=not args.force_rerun,
        )

        exit_code = 0
        writeup_result = {"success": True}

        if not args.skip_writeup:
            writeup_result = run_writeup_phase(
                idea_dir,
                writeup_type=args.writeup_type,
                writeup_retries=args.writeup_retries,
                num_cite_rounds=args.num_cite_rounds,
                model_citation=args.model_citation,
                model_writeup_small=args.model_writeup_small,
                model_writeup=args.model_writeup,
                high_quality_mode=args.high_quality_mode,
                quality_preset=args.quality_preset,
                quality_model=args.quality_model,
                target_venue=args.target_venue,
                quality_threshold=args.quality_threshold,
                rigor_threshold=args.rigor_threshold,
                max_quality_rewrites=args.quality_rewrite_rounds,
                autonomous_quality_followup_rounds=args.autonomous_quality_followup_rounds,
                require_quality_gate=args.require_quality_gate,
                min_submission_priority=args.min_submission_priority,
                max_submission_blockers=args.max_submission_blockers,
                writing_profile=args.writing_profile,
                writing_audit_rounds=args.writing_audit_rounds,
                strict_guardrails=args.strict_writing_guardrails,
                guardrail_repair_rounds=args.guardrail_repair_rounds,
                workflow_mode=args.workflow_mode,
                submission_mode=args.submission_mode,
                strict_fallbacks=strict_fallbacks,
                research_root=research_root,
                resume=not args.force_rerun,
            )
            if not writeup_result.get("success"):
                print(
                    "Writeup process did not complete successfully: "
                    + str(writeup_result.get("failure_reason") or "unknown")
                )
                exit_code = 1

        if (
            not args.skip_review
            and not args.skip_writeup
            and writeup_result.get("success")
        ):
            review_result = run_review_phase(
                idea_dir,
                model_review=args.model_review,
                paper_type=args.writeup_type,
                target_venue=args.target_venue,
                text_filename="review_text.txt",
                image_filename="review_img_cap_ref.json",
                text_mode="text_json",
                review_reflections=args.review_reflections,
                review_fewshot=args.review_fewshot,
                review_ensemble=args.review_ensemble,
                review_temperature=args.review_temperature,
                review_strategy=args.review_strategy,
                high_quality_mode=args.high_quality_mode,
                research_root=research_root,
                workflow_mode=args.workflow_mode,
                submission_mode=args.submission_mode,
                require_quality_gate=args.require_quality_gate,
                min_submission_priority=args.min_submission_priority,
                max_submission_blockers=args.max_submission_blockers,
                resume=not args.force_rerun,
            )
            if review_result["found"]:
                print("Paper found at: ", review_result["pdf_path"])
                print("Paper review completed.")
                submission_acceptance = review_result.get("submission_acceptance") or {}
                if (
                    args.high_quality_mode
                    and submission_acceptance
                    and submission_acceptance.get("accepted") is False
                ):
                    print(
                        "Final submission gate not met after review: "
                        + "; ".join(submission_acceptance.get("reasons", []))
                    )
                    exit_code = 1

        print("Start cleaning up processes")
        cleanup_child_processes(
            include_orphans=True,
            workspace_roots=[runtime.project_root, research_root],
        )

        final_exit_code = exit_code
        if exit_code == 0:
            break
        if not args.fallback_ranked_ideas:
            break
        print("Trying next ranked idea due to unsuccessful run...")

    # Finally, terminate the current process
    # current_process.send_signal(signal.SIGTERM)
    # try:
    #     current_process.wait(timeout=3)
    # except psutil.TimeoutExpired:
    #     current_process.kill()

    # exit the program
    sys.exit(final_exit_code)
