#!/usr/bin/env python3
"""
AI Scientist-v2 启动脚本 - 智谱API优化版
使用智谱AI模型运行完整的科研流程
"""
import json
import argparse
import os
import sys
from datetime import datetime
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
    """保存token使用统计"""
    persist_token_tracker(idea_dir)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="运行AI科学家实验 - 智谱API版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认智谱模型配置
  python launch_scientist_zhipu.py --load_ideas "ai_scientist/ideas/my_research_topic.json"

  # 自定义模型选择
  python launch_scientist_zhipu.py \\
    --load_ideas "ai_scientist/ideas/my_research_topic.json" \\
    --model_writeup glm-4-plus \\
    --model_citation glm-4-air \\
    --model_review glm-4-plus \\
    --num_cite_rounds 15

  # 跳过写作和审查,仅运行实验
  python launch_scientist_zhipu.py \\
    --load_ideas "ai_scientist/ideas/my_research_topic.json" \\
    --skip_writeup \\
    --skip_review
        """
    )

    # 基本参数
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb", "journal", "extended"],
        help="论文类型: normal=8页标准论文, icbinb=4页workshop论文, journal=12页期刊论文, extended=2页扩展摘要 (默认: icbinb)",
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        default="ai_scientist/ideas/i_cant_believe_its_not_better.json",
        help="预生成的想法JSON文件路径",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="如果设置,加载与想法文件同名的Python代码文件",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="要运行的想法索引 (默认: 0)",
    )
    parser.add_argument("--auto-best-idea", action="store_true", help="对加载的 ideas 排序并自动选择最高分")
    parser.add_argument("--idea-rank-model", type=str, default=None, help="用于 idea 排序的模型")
    parser.add_argument("--fallback-ranked-ideas", action="store_true", help="若最佳 idea 未通过质量门槛，则尝试下一个高分 idea")
    parser.add_argument("--auto-adjust-paper-type", action="store_true", help="自动切换到更适合目标 venue 的 paper type")
    parser.add_argument("--submission-mode", action="store_true", help="启用完整的投稿级 preset")
    parser.add_argument("--breakthrough-mode", action="store_true", help="偏向重大问题和高影响力投稿")
    parser.add_argument(
        "--workflow-mode",
        type=str,
        choices=list_workflow_modes(),
        default="adaptive",
        help="研究编排模式",
    )
    parser.add_argument("--max-ranked-candidates", type=int, default=None, help="fallback 模式下最多尝试多少个高分 idea")
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="如果设置,添加Hugging Face数据集引用到想法中",
    )

    # 重试和尝试参数
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="论文写作重试次数 (默认: 3)",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="尝试ID,用于区分同一想法的并行运行 (默认: 0)",
    )
    parser.add_argument(
        "--bfts-config",
        type=str,
        default="bfts_config.yaml",
        help="BFTS实验配置文件路径 (控制搜索深度、随机种子、并行度、超时等)",
    )

    # 智谱模型配置 (默认使用性价比高的模型组合)
    parser.add_argument(
        "--model_agg_plots",
        type=str,
        default="glm-4-flash",
        help="图表聚合使用的模型 (默认: glm-4-flash)",
    )
    parser.add_argument(
        "--model_writeup",
        type=str,
        default="glm-4-plus",
        help="论文写作使用的主要模型 (默认: glm-4-plus)",
    )
    parser.add_argument(
        "--model_citation",
        type=str,
        default="glm-4-air",
        help="文献检索使用的模型 (默认: glm-4-air)",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=15,
        help="文献检索轮数 (默认: 15,智谱API建议降低以节省成本)",
    )
    parser.add_argument(
        "--model_writeup_small",
        type=str,
        default="glm-4-air",
        help="论文写作使用的辅助模型 (默认: glm-4-air)",
    )
    parser.add_argument(
        "--model_review",
        type=str,
        default="glm-4-plus",
        help="论文审查使用的模型 (默认: glm-4-plus)",
    )

    # 跳过选项
    parser.add_argument(
        "--skip_writeup",
        action="store_true",
        help="如果设置,跳过论文写作过程",
    )
    parser.add_argument(
        "--skip_review",
        action="store_true",
        help="如果设置,跳过论文审查过程",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="如果设置,忽略已完成状态并重新执行所有阶段",
    )
    parser.add_argument(
        "--high-quality-mode",
        action="store_true",
        help="启用更强审稿、质量门控和定向重写",
    )
    parser.add_argument(
        "--quality-preset",
        choices=["balanced", "high", "publishable"],
        default="balanced",
        help="高质量论文生成预设",
    )
    parser.add_argument("--quality-model", type=str, default=None, help="质量评估使用的模型")
    parser.add_argument("--target-venue", type=str, choices=["neurips", "iclr", "cvpr", "journal", "nature"], default=None)
    parser.add_argument("--quality-threshold", type=float, default=None, help="目标最低质量分")
    parser.add_argument("--rigor-threshold", type=float, default=None, help="目标最低严谨性分")
    parser.add_argument("--quality-rewrite-rounds", type=int, default=None, help="定向重写最大轮数")
    parser.add_argument("--autonomous-quality-followup-rounds", type=int, default=0, help="投稿标准未达标时自动补跑更强质量 follow-up 的最大轮数")
    parser.add_argument("--min-submission-priority", type=float, default=None, help="接受稿件所需的最低投稿优先级")
    parser.add_argument("--max-submission-blockers", type=int, default=None, help="接受稿件所允许的最大 blocker 数")
    parser.add_argument(
        "--require-quality-gate",
        action="store_true",
        help="如果设置,高质量模式未通过质量门槛时视为失败",
    )
    parser.add_argument("--review-reflections", type=int, default=1, help="文本审稿反思轮数")
    parser.add_argument("--review-ensemble", type=int, default=1, help="审稿 ensemble 数量")
    parser.add_argument("--review-fewshot", type=int, default=1, help="审稿 few-shot 示例数")
    parser.add_argument("--review-temperature", type=float, default=0.75, help="审稿温度")
    parser.add_argument(
        "--review-strategy",
        choices=["standard", "fast", "depth", "neurips", "iclr", "cvpr", "journal", "nature"],
        default=None,
        help="审稿策略预设",
    )
    parser.add_argument(
        "--writing-profile",
        type=str,
        choices=list_writing_profiles(),
        default=resolve_writing_profile_env(
            invalid_profile_logger=lambda exc, raw: print(
                "⚠️  忽略无效的 AI_SCIENTIST_WRITING_PROFILE="
                f"{raw!r}，回退为 {DEFAULT_WRITING_PROFILE}"
            )
        ),
        help="写作提示词 profile",
    )
    parser.add_argument(
        "--writing-audit-rounds",
        type=int,
        default=0,
        help="写作反思阶段追加的结构化写作审计轮数",
    )
    parser.add_argument(
        "--strict-writing-guardrails",
        action="store_true",
        help="启用严格写作守护：若最终引用/章节检查不通过则判定写作失败",
    )
    parser.add_argument(
        "--guardrail-repair-rounds",
        type=int,
        default=1,
        help="严格写作守护失败前自动尝试的修复轮数",
    )
    parser.add_argument(
        "--override-strict-fallbacks",
        action="store_true",
        help="禁用严格兜底拦截，但继续记录 fallback debt。",
    )

    return parser.parse_args()


def find_pdf_path_for_review(idea_dir):
    """查找需要审查的PDF文件"""
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
    require_login("科学家启动入口(launch_scientist_zhipu)")

    args = parse_arguments()
    normalize_common_launcher_args(
        args,
        invalid_profile_logger=lambda exc: print(
            f"⚠️  无效 writing profile: {exc}，回退为 {DEFAULT_WRITING_PROFILE}"
        ),
    )
    runtime = initialize_runtime(
        source_file=__file__,
        ensure_dirs=True,
        apply_cache=True,
    )
    print(f"设置 AI_SCIENTIST_ROOT 为 {runtime.project_root}")
    research_root = runtime.research_root

    # 设置研究输出目录
    research_dir = str(research_root)
    print(f"输出目录: {research_dir}")

    # 打印智谱API配置信息
    print("=" * 80)
    print("AI Scientist-v2 - 智谱API版本")
    print("=" * 80)
    print(f"论文写作模型: {args.model_writeup}")
    print(f"文献检索模型: {args.model_citation}")
    print(f"论文审查模型: {args.model_review}")
    print(f"图表聚合模型: {args.model_agg_plots}")
    print(f"文献检索轮数: {args.num_cite_rounds}")
    print(f"编排模式: {args.workflow_mode}")
    print(f"写作Profile: {args.writing_profile}")
    print(f"写作审计轮数: {args.writing_audit_rounds}")
    print(f"严格写作守护: {args.strict_writing_guardrails}")
    print(f"守护自动修复轮数: {args.guardrail_repair_rounds}")
    strict_fallbacks = should_enforce_strict_fallbacks(
        args.workflow_mode,
        submission_mode=bool(args.submission_mode),
        high_quality_mode=bool(args.high_quality_mode),
        target_venue=args.target_venue,
    )
    if args.override_strict_fallbacks and strict_fallbacks:
        print("⚠️  override-strict-fallbacks 已启用：继续记录 fallback，但本次不阻断。")
        strict_fallbacks = False
    elif strict_fallbacks:
        print("🛡️  严格兜底策略已启用：idea ranking 或质量 fallback 会直接终止本次运行。")
    print("=" * 80)

    # 检查可用的GPU并根据需要调整并行进程
    available_gpus = get_available_gpus()
    print(f"使用GPU: {available_gpus}")

    require_model_credentials(_collect_requested_models(args))

    # 加载想法
    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
        print(f"从 {args.load_ideas} 加载了 {len(ideas)} 个预生成的想法")

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
        print(f"🏅 自动选择最佳 idea: index={args.idea_idx}, name={rankings[0].get('idea_name')}, ranking_score={rankings[0].get('ranking_score')}")

    args.writeup_type = resolve_paper_type_for_venue(
        args.writeup_type,
        args.target_venue,
        auto_adjust=args.auto_adjust_paper_type,
        warning_template="⚠️  警告: writeup_type '{paper_type}' 与目标 venue '{target_venue}' 可能不够匹配",
        adjusted_template="✅ 自动调整 writeup_type -> {adjusted}",
    )

    final_exit_code = 1
    for candidate_idx in candidate_indices:
        if candidate_idx >= len(ideas):
            continue

        idea = ideas[candidate_idx]
        print(f"选择想法 #{candidate_idx}: {idea.get('Name', 'Unknown')}")

        idea_dir = str(get_experiment_dir(idea['Name'], args.attempt_id))
        print(f"结果将保存在 {idea_dir}")
        print(
            "(相对于项目根目录: "
            f"{format_project_relative_path(idea_dir, project_root=runtime.project_root)})"
        )
        os.makedirs(idea_dir, exist_ok=True)
        ranking_event = record_ranking_fallbacks(
            idea_dir,
            rankings,
            producer="launch_scientist_zhipu.idea_ranking",
            strict=strict_fallbacks,
        )
        if strict_fallbacks and ranking_event:
            print(
                "❌ "
                + format_strict_fallback_error(
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
        )

        print("\n" + "=" * 80)
        print("🔬 开始实验阶段...")
        print("=" * 80)
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
            print("\n" + "=" * 80)
            print("✍️  开始论文写作阶段...")
            print("=" * 80)

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
                    "❌ 论文写作未成功完成: "
                    + str(writeup_result.get("failure_reason") or "unknown")
                )
                exit_code = 1

        if (
            not args.skip_review
            and not args.skip_writeup
            and writeup_result.get("success")
        ):
            print("\n" + "=" * 80)
            print("🔍 开始论文审查阶段...")
            print("=" * 80)

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
                print(f"📄 找到论文: {review_result['pdf_path']}")
                print("✅ 论文审查完成")
                submission_acceptance = review_result.get("submission_acceptance") or {}
                if (
                    args.high_quality_mode
                    and submission_acceptance
                    and submission_acceptance.get("accepted") is False
                ):
                    print(
                        "❌ 最终投稿门禁未通过: "
                        + "; ".join(submission_acceptance.get("reasons", []))
                    )
                    exit_code = 1
            else:
                print(f"⚠️  未找到论文文件: {review_result['pdf_path']}")

        print("\n" + "=" * 80)
        print("🧹 清理进程...")
        print("=" * 80)
        cleanup_child_processes(
            include_orphans=True,
            workspace_roots=[runtime.project_root, research_root],
        )

        final_exit_code = exit_code
        if exit_code == 0:
            print("\n" + "=" * 80)
            print("🎉 AI Scientist-v2 运行完成!")
            print(f"📁 结果保存在: {idea_dir}")
            print("=" * 80)
            break
        if not args.fallback_ranked_ideas:
            break
        print("🔁 当前 idea 未达标，尝试下一个高分 idea...")

    sys.exit(final_exit_code)
