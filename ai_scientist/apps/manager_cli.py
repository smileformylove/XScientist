"""Command-line interface for the XScientist research manager."""

from __future__ import annotations

import argparse
import sys
from collections import Counter


def main(
    argv=None,
    *,
    manager_cls,
    require_login_fn,
    resolve_output_path_fn,
    run_index_path_fn,
    format_size_fn,
):
    ResearchManager = manager_cls
    require_login = require_login_fn
    resolve_output_path = resolve_output_path_fn
    run_index_path = run_index_path_fn
    format_size = format_size_fn
    parser = argparse.ArgumentParser(
        description="XScientist research management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 列出所有批次:
   python research_manager.py list-batches

2. 查看批次详情:
   python research_manager.py batch-summary 20240101_120000

3. 列出所有论文:
   python research_manager.py list-papers

4. 列出特定类型论文:
   python research_manager.py list-papers --type normal

5. 搜索论文:
   python research_manager.py search-papers "transformer"

6. 清理旧文件 (30天前):
   python research_manager.py cleanup --days 30

7. 预览将要清理的文件:
   python research_manager.py cleanup --days 30 --dry-run

8. 重建结果索引:
   python research_manager.py rebuild-index

9. 按质量生成投稿 shortlist:
   python research_manager.py shortlist --top 5 --require-gate

10. 查看 Nature-style readiness benchmark:
   python research_manager.py readiness-benchmark --venue nature --top 10

11. 查看 source lineage 看板:
   python research_manager.py source-board --top 20

12. 查看 source mix 建议:
   python research_manager.py source-mix --desired-policy program_driven

13. 查看 reviewer repair queue:
   python research_manager.py repair-board --top 20

14. 查看 self-evolution 看板:
   python research_manager.py evolution-board --top 20
        """,
    )

    parser.add_argument(
        "--research-dir",
        type=str,
        default=str(resolve_output_path()),
        help="研究目录路径",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # list-batches 命令
    subparsers.add_parser("list-batches", help="列出所有批次")

    # batch-summary 命令
    batch_summary_parser = subparsers.add_parser("batch-summary", help="查看批次摘要")
    batch_summary_parser.add_argument("batch_name", help="批次名称 (不含batch_前缀)")

    # list-papers 命令
    list_papers_parser = subparsers.add_parser("list-papers", help="列出所有论文")
    list_papers_parser.add_argument(
        "--type", choices=["icbinb", "normal", "journal", "extended"], help="按类型过滤"
    )
    list_papers_parser.add_argument(
        "--detailed", action="store_true", help="显示详细信息"
    )
    list_papers_parser.add_argument(
        "--sort", choices=["modified", "quality"], default="modified", help="排序方式"
    )

    # paper-details 命令
    details_parser = subparsers.add_parser("paper-details", help="查看论文详细信息")
    details_parser.add_argument("folder", help="论文文件夹名称")

    # list-ideas 命令
    subparsers.add_parser("list-ideas", help="列出所有想法")

    # search-papers 命令
    search_parser = subparsers.add_parser("search-papers", help="搜索论文")
    search_parser.add_argument("query", help="搜索关键词")
    search_parser.add_argument(
        "--type", choices=["icbinb", "normal", "journal", "extended"], help="按类型过滤"
    )

    # cleanup 命令
    cleanup_parser = subparsers.add_parser("cleanup", help="清理旧文件")
    cleanup_parser.add_argument("--days", type=int, default=30, help="天数阈值")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="仅显示，不删除")

    # stats 命令
    subparsers.add_parser("stats", help="显示统计信息")

    # rebuild-index 命令
    subparsers.add_parser("rebuild-index", help="重建输出索引")

    # shortlist 命令
    shortlist_parser = subparsers.add_parser(
        "shortlist", help="按质量筛选最值得投稿的论文"
    )
    shortlist_parser.add_argument(
        "--type", choices=["icbinb", "normal", "journal", "extended"], help="按类型过滤"
    )
    shortlist_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        help="按目标 venue 过滤",
    )
    shortlist_parser.add_argument(
        "--require-gate", action="store_true", help="只保留通过质量门槛的稿件"
    )
    shortlist_parser.add_argument(
        "--require-ready",
        action="store_true",
        help="只保留 submission readiness 为 ready 的稿件",
    )
    shortlist_parser.add_argument(
        "--min-breakthrough", type=float, default=None, help="最小 breakthrough score"
    )
    shortlist_parser.add_argument(
        "--min-priority",
        type=float,
        default=None,
        help="最小 submission priority score",
    )
    shortlist_parser.add_argument(
        "--max-blockers", type=int, default=None, help="最多允许的 blocker 数"
    )
    shortlist_parser.add_argument(
        "--min-rewrite-gain",
        type=float,
        default=None,
        help="最小 rewrite priority gain",
    )
    shortlist_parser.add_argument(
        "--max-fallbacks",
        type=int,
        default=None,
        help="最多允许的 fallback 事件数",
    )
    shortlist_parser.add_argument(
        "--max-strict-fallbacks",
        type=int,
        default=0,
        help="最多允许的 strict fallback 事件数，默认不接受 strict fallback",
    )
    shortlist_parser.add_argument(
        "--max-blocked-stages",
        type=int,
        default=0,
        help="最多允许的 blocked stage standards 数，默认不接受 blocked stage",
    )
    shortlist_parser.add_argument(
        "--max-missing-stages",
        type=int,
        default=None,
        help="最多允许的 missing stage standards 数",
    )
    shortlist_parser.add_argument(
        "--max-attention-stages",
        type=int,
        default=None,
        help="最多允许的 needs_attention stage standards 数",
    )
    shortlist_parser.add_argument(
        "--min-stage-score",
        type=float,
        default=None,
        help="最小 stage standards overall score",
    )
    shortlist_parser.add_argument(
        "--max-self-evolution-required-failures",
        type=int,
        default=0,
        help="最多允许的 self-evolution required failure 数，默认不接受 required failure",
    )
    shortlist_parser.add_argument(
        "--min-self-evolution-score",
        type=float,
        default=None,
        help="最小 self-evolution score",
    )
    shortlist_parser.add_argument(
        "--allow-blocked-self-evolution",
        action="store_true",
        help="允许 self-evolution status 为 blocked 的稿件进入 shortlist",
    )
    shortlist_parser.add_argument(
        "--max-blocked-processes",
        type=int,
        default=0,
        help="最多允许的 blocked process_alignment 过程数，默认不接受 blocked process",
    )
    shortlist_parser.add_argument(
        "--min-process-alignment-score",
        type=float,
        default=None,
        help="最小 process alignment overall score",
    )
    shortlist_parser.add_argument("--top", type=int, default=5, help="返回前 N 篇")
    shortlist_parser.add_argument(
        "--export", type=str, help="导出 Markdown shortlist 文件"
    )

    # submission-board 命令
    board_parser = subparsers.add_parser(
        "submission-board", help="按 venue 查看当前最值得投稿的论文"
    )
    board_parser.add_argument(
        "--top", type=int, default=3, help="每个 venue 显示前 N 篇"
    )
    board_parser.add_argument(
        "--min-breakthrough",
        type=float,
        default=None,
        help="只显示突破潜力高于阈值的稿件",
    )
    board_parser.add_argument(
        "--min-priority",
        type=float,
        default=None,
        help="只显示投稿优先级高于阈值的稿件",
    )
    board_parser.add_argument(
        "--max-blockers",
        type=int,
        default=None,
        help="只显示 blocker 数不超过阈值的稿件",
    )
    board_parser.add_argument(
        "--min-rewrite-gain",
        type=float,
        default=None,
        help="只显示 rewrite priority gain 高于阈值的稿件",
    )
    board_parser.add_argument(
        "--require-gate", action="store_true", help="只显示通过质量门槛的稿件"
    )
    board_parser.add_argument(
        "--max-fallbacks",
        type=int,
        default=None,
        help="最多允许的 fallback 事件数",
    )
    board_parser.add_argument(
        "--max-strict-fallbacks",
        type=int,
        default=0,
        help="最多允许的 strict fallback 事件数，默认不接受 strict fallback",
    )
    board_parser.add_argument(
        "--max-blocked-stages",
        type=int,
        default=0,
        help="最多允许的 blocked stage standards 数，默认不接受 blocked stage",
    )
    board_parser.add_argument(
        "--max-missing-stages",
        type=int,
        default=None,
        help="最多允许的 missing stage standards 数",
    )
    board_parser.add_argument(
        "--max-attention-stages",
        type=int,
        default=None,
        help="最多允许的 needs_attention stage standards 数",
    )
    board_parser.add_argument(
        "--min-stage-score",
        type=float,
        default=None,
        help="最小 stage standards overall score",
    )
    board_parser.add_argument(
        "--max-self-evolution-required-failures",
        type=int,
        default=0,
        help="最多允许的 self-evolution required failure 数，默认不接受 required failure",
    )
    board_parser.add_argument(
        "--min-self-evolution-score",
        type=float,
        default=None,
        help="最小 self-evolution score",
    )
    board_parser.add_argument(
        "--allow-blocked-self-evolution",
        action="store_true",
        help="允许 self-evolution status 为 blocked 的稿件进入投稿看板",
    )
    board_parser.add_argument(
        "--max-blocked-processes",
        type=int,
        default=0,
        help="最多允许的 blocked process_alignment 过程数，默认不接受 blocked process",
    )
    board_parser.add_argument(
        "--min-process-alignment-score",
        type=float,
        default=None,
        help="最小 process alignment overall score",
    )
    board_parser.add_argument("--export", type=str, help="导出 Markdown board 文件")

    # rewrite-board 命令
    rewrite_board_parser = subparsers.add_parser(
        "rewrite-board", help="查看最值得继续重写优化的论文"
    )
    rewrite_board_parser.add_argument(
        "--type", choices=["icbinb", "normal", "journal", "extended"], help="按类型过滤"
    )
    rewrite_board_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        help="按目标 venue 过滤",
    )
    rewrite_board_parser.add_argument(
        "--min-priority",
        type=float,
        default=None,
        help="只显示投稿优先级高于阈值的稿件",
    )
    rewrite_board_parser.add_argument(
        "--min-rewrite-gain",
        type=float,
        default=None,
        help="只显示 rewrite priority gain 高于阈值的稿件",
    )
    rewrite_board_parser.add_argument(
        "--max-blockers",
        type=int,
        default=None,
        help="只显示 blocker 数不超过阈值的稿件",
    )
    rewrite_board_parser.add_argument(
        "--require-gate", action="store_true", help="只显示通过质量门槛的稿件"
    )
    rewrite_board_parser.add_argument(
        "--include-ready", action="store_true", help="包含已 ready 的稿件"
    )
    rewrite_board_parser.add_argument("--top", type=int, default=10, help="返回前 N 篇")
    rewrite_board_parser.add_argument(
        "--export", type=str, help="导出 Markdown board 文件"
    )

    repair_board_parser = subparsers.add_parser(
        "repair-board", help="查看 reviewer 反馈转成的结构化修复任务"
    )
    repair_board_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        help="按目标 venue 过滤",
    )
    repair_board_parser.add_argument(
        "--priority-tier",
        choices=["p0", "p1", "p2"],
        help="只显示指定优先级 tier 的修复任务",
    )
    repair_board_parser.add_argument(
        "--only-ready",
        action="store_true",
        help="只显示已经具备动作和验证计划的修复任务",
    )
    repair_board_parser.add_argument(
        "--top", type=int, default=20, help="返回前 N 个 repair tasks"
    )
    repair_board_parser.add_argument(
        "--export", type=str, help="导出 Markdown repair board 文件"
    )

    evolution_board_parser = subparsers.add_parser(
        "evolution-board",
        help="查看 reviewer 修复闭环沉淀出的 self-evolution 结果",
    )
    evolution_board_parser.add_argument(
        "--status",
        choices=["ready", "needs_attention", "blocked"],
        help="按 self-evolution 状态过滤",
    )
    evolution_board_parser.add_argument(
        "--top", type=int, default=20, help="返回前 N 个 self-evolution 项目"
    )

    # submission-dossier 命令
    dossier_parser = subparsers.add_parser(
        "submission-dossier", help="导出单篇论文的投稿材料包"
    )
    dossier_parser.add_argument("folder", help="论文文件夹名称")
    dossier_parser.add_argument("output_dir", help="导出目录")

    benchmark_parser = subparsers.add_parser(
        "readiness-benchmark",
        help="汇总当前研究产出的投稿 readiness benchmark",
    )
    benchmark_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        default="nature",
        help="按目标 venue 的门槛做基准评估",
    )
    benchmark_parser.add_argument(
        "--top", type=int, default=10, help="展示前 N 篇 benchmark 结果"
    )
    benchmark_parser.add_argument(
        "--max-entries",
        type=int,
        default=200,
        help="最多扫描多少篇历史结果",
    )
    benchmark_parser.add_argument(
        "--include-other-venues",
        action="store_true",
        help="包含 target venue 不匹配的稿件作为横向参考",
    )
    benchmark_parser.add_argument(
        "--export", type=str, help="导出 Markdown benchmark 文件"
    )

    pipeline_status_parser = subparsers.add_parser(
        "pipeline-status",
        help="查看 contracts 驱动的项目流水线状态",
    )
    pipeline_status_parser.add_argument(
        "--top", type=int, default=20, help="显示前 N 个项目"
    )

    stage_standards_parser = subparsers.add_parser(
        "stage-standards",
        help="查看每个流程阶段的结构化评估标准与得分",
    )
    stage_standards_parser.add_argument(
        "--top", type=int, default=60, help="显示前 N 条阶段记录"
    )
    stage_standards_parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="按阶段过滤，如 ideation / planning / experiment / figure / manuscript / review",
    )
    stage_standards_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="按阶段状态过滤，如 ready / blocked / needs_attention / missing",
    )

    process_board_parser = subparsers.add_parser(
        "process-board",
        help="查看点对点对标开源参考的过程级对齐状态",
    )
    process_board_parser.add_argument(
        "--top", type=int, default=80, help="显示前 N 条过程记录"
    )
    process_board_parser.add_argument(
        "--process",
        type=str,
        default=None,
        help="按过程过滤，如 ideation / program / exploration / experiment / figure / writing / review / evolution / packaging",
    )
    process_board_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="按过程状态过滤，如 ready / blocked / needs_attention / missing",
    )

    fallback_board_parser = subparsers.add_parser(
        "fallback-board",
        help="查看 pipeline fallback 事件与兜底债务",
    )
    fallback_board_parser.add_argument(
        "--top", type=int, default=30, help="显示前 N 个项目"
    )
    fallback_board_parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="按 fallback stage 过滤",
    )

    idea_board_parser = subparsers.add_parser(
        "idea-board",
        help="查看结构化 idea card 看板",
    )
    idea_board_parser.add_argument(
        "--top", type=int, default=30, help="显示前 N 条 idea"
    )
    idea_board_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="按 idea status 过滤",
    )

    experiment_board_parser = subparsers.add_parser(
        "experiment-board",
        help="查看 experiment registry 看板",
    )
    experiment_board_parser.add_argument(
        "--top", type=int, default=50, help="显示前 N 条实验记录"
    )
    experiment_board_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="按实验状态过滤",
    )

    figure_board_parser = subparsers.add_parser(
        "figure-board",
        help="查看 figure spec 看板",
    )
    figure_board_parser.add_argument(
        "--top", type=int, default=50, help="显示前 N 条 figure 记录"
    )
    figure_board_parser.add_argument(
        "--ready-only",
        action="store_true",
        help="只显示 ready 的 figure",
    )

    source_board_parser = subparsers.add_parser(
        "source-board",
        help="查看 source lineage 与 planning 看板",
    )
    source_board_parser.add_argument(
        "--top", type=int, default=30, help="显示前 N 个 source"
    )
    source_board_parser.add_argument(
        "--archetype",
        type=str,
        default=None,
        help="按 source archetype 过滤",
    )

    source_mix_parser = subparsers.add_parser(
        "source-mix",
        help="查看 source mix 与下一批研究倾斜建议",
    )
    source_mix_parser.add_argument(
        "--desired-policy",
        type=str,
        default=None,
        help="按当前期望 workflow / execution policy 给建议",
    )
    source_mix_parser.add_argument(
        "--top", type=int, default=50, help="最多纳入多少个 source 做分析"
    )

    source_next_batch_parser = subparsers.add_parser(
        "source-next-batch",
        help="生成下一批 source 组合与节奏建议",
    )
    source_next_batch_parser.add_argument(
        "--desired-policy",
        type=str,
        default=None,
        help="按当前期望 workflow / execution policy 生成组合建议",
    )
    source_next_batch_parser.add_argument(
        "--top", type=int, default=50, help="最多纳入多少个 source 做分析"
    )
    source_next_batch_parser.add_argument(
        "--max-slots",
        type=int,
        default=3,
        help="最多输出多少个下一批组合 lane",
    )

    trend_parser = subparsers.add_parser(
        "benchmark-trends",
        help="按日期查看 readiness benchmark 趋势",
    )
    trend_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        default="nature",
        help="趋势统计的目标 venue",
    )
    trend_parser.add_argument(
        "--max-entries",
        type=int,
        default=200,
        help="最多扫描多少篇历史结果",
    )

    args = parser.parse_args(argv)
    require_login("研究管理操作(research_manager)")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = ResearchManager(args.research_dir)

    if args.command == "list-batches":
        batches = manager.list_batches()
        print(f"\n共有 {len(batches)} 个批次:\n")
        for batch in batches:
            completed = len(batch["progress"].get("papers_completed", []))
            failed = len(batch["progress"].get("papers_failed", []))
            print(f"📁 {batch['name']}")
            print(f"   创建时间: {batch['created_at']}")
            print(f"   状态: {completed} 完成, {failed} 失败")
            print()

    elif args.command == "batch-summary":
        summary = manager.get_batch_summary(args.batch_name)
        if not summary:
            print(f"❌ 未找到批次: {args.batch_name}")
            sys.exit(1)

        print(f"\n批次: {summary['batch_name']}")
        print(f"路径: {summary['path']}")
        source_provenance = (
            summary.get("report", {}).get("source_provenance")
            or summary.get("progress", {}).get("source_provenance")
            or {}
        )
        if source_provenance:
            print(
                "来源: "
                f"{source_provenance.get('source_name') or source_provenance.get('source_key')} | "
                f"workflow={source_provenance.get('source_workflow_mode')} | "
                f"archetype={source_provenance.get('source_archetype')} | "
                f"profile={source_provenance.get('source_batch_profile')}"
            )
        print(f"\n论文 ({len(summary['papers'])} 个):")
        for paper in summary["papers"]:
            print(f"  📄 {paper['name']} ({paper['type']})")
            print(f"     {paper['path']}")

        if summary["report"]:
            stats = summary["report"].get("statistics", {})
            print(f"\n统计:")
            print(f"  总计: {stats.get('total_papers', 0)}")
            print(f"  成功: {stats.get('completed', 0)}")
            print(f"  失败: {stats.get('failed', 0)}")

            quality_summary = summary["report"].get("quality_summary", {})
            if quality_summary:
                if quality_summary.get("avg_quality_score") is not None:
                    print(
                        f"  平均质量分: {quality_summary.get('avg_quality_score'):.2f}"
                    )
                if quality_summary.get("avg_rigor_score") is not None:
                    print(f"  平均严谨性: {quality_summary.get('avg_rigor_score'):.2f}")
                print(f"  质量门槛通过: {quality_summary.get('gate_passed', 0)}")
                print(f"  质量门槛未过: {quality_summary.get('gate_failed', 0)}")
                top_papers = quality_summary.get("top_papers", [])
                if top_papers:
                    print(f"\n推荐优先查看:")
                    for paper in top_papers[:3]:
                        print(
                            f"  - idea #{paper.get('idea_idx')} [{paper.get('paper_type')}] "
                            f"quality={paper.get('quality_score')} rigor={paper.get('rigor_score')} "
                            f"gate={paper.get('quality_gate_passed')}"
                        )

        failure_summary = summary.get("failure_summary", {})
        if failure_summary.get("total", 0):
            print(f"\n失败诊断:")
            for stage, count in sorted(failure_summary.get("by_stage", {}).items()):
                print(f"  {stage}: {count}")
            for sample in failure_summary.get("samples", []):
                print(
                    f"  - idea #{sample.get('idea_idx')} [{sample.get('paper_type')}] "
                    f"stage={sample.get('stage')} error={sample.get('error', '')}"
                )

    elif args.command == "list-papers":
        papers = manager.list_papers(args.type, args.sort)
        print(f"\n共有 {len(papers)} 篇论文:\n")
        for paper in papers:
            print(f"📄 {paper['name']}")
            print(f"   类型: {paper['type']}")
            print(f"   文件夹: {paper['folder']}")
            print(f"   大小: {format_size(paper['size'])}")
            print(f"   创建时间: {paper['created_at']}")
            if paper.get("latest_stage"):
                print(f"   阶段: {paper['latest_stage']}")
            if paper.get("has_reviews") is not None:
                print(f"   审查: {'yes' if paper['has_reviews'] else 'no'}")
            if isinstance(paper.get("quality_score"), (int, float)):
                print(f"   质量分: {paper['quality_score']:.2f}")
            if isinstance(paper.get("rigor_score"), (int, float)):
                print(f"   严谨性: {paper['rigor_score']:.2f}")
            if isinstance(paper.get("claim_support_score"), (int, float)):
                print(f"   论证支撑: {paper['claim_support_score']:.2f}")
            if isinstance(paper.get("claim_alignment_score"), (int, float)):
                print(f"   论断对齐: {paper['claim_alignment_score']:.2f}")
            if isinstance(paper.get("numeric_coverage_score"), (int, float)):
                print(f"   数值覆盖: {paper['numeric_coverage_score']:.2f}")
            if isinstance(paper.get("breakthrough_score"), (int, float)):
                print(f"   突破潜力: {paper['breakthrough_score']:.2f}")
            if paper.get("claims_detected") is not None:
                print(f"   claims: {paper['claims_detected']}")
            if paper.get("unsupported_claims_count") is not None:
                print(f"   unsupported claims: {paper['unsupported_claims_count']}")
            if paper.get("suggested_claim_rewrites_count") is not None:
                print(
                    f"   suggested claim rewrites: {paper['suggested_claim_rewrites_count']}"
                )
            if paper.get("num_figures") is not None:
                print(
                    f"   figures/tables: {paper['num_figures']}/{paper.get('num_tables', 0)}"
                )
            if isinstance(paper.get("evidence_density_score"), (int, float)):
                print(f"   证据密度: {paper['evidence_density_score']:.2f}")
            if paper.get("key_results_count") is not None:
                print(f"   key results: {paper['key_results_count']}")
            if paper.get("structured_results_count") is not None:
                print(f"   structured results: {paper['structured_results_count']}")
            if paper.get("contribution_count") is not None:
                print(f"   contributions: {paper['contribution_count']}")
            if paper.get("target_venue"):
                print(f"   目标 venue: {paper['target_venue']}")
            if paper.get("submission_status"):
                print(f"   投稿准备度: {paper['submission_status']}")
            if isinstance(paper.get("submission_priority_score"), (int, float)):
                print(
                    f"   投稿优先级: {paper['submission_priority_score']:.2f} ({paper.get('submission_priority_tier')})"
                )
            if isinstance(paper.get("blocker_count"), int):
                print(f"   blocker 数: {paper['blocker_count']}")
            if isinstance(paper.get("experiment_todo_count"), int):
                print(
                    f"   experiment TODO: total={paper.get('experiment_todo_count')} p0={paper.get('experiment_todo_p0_count')}"
                )
            if isinstance(paper.get("experiment_todo_closure_rate"), (int, float)):
                print(
                    f"   experiment TODO closure: total={paper.get('experiment_todo_closure_rate'):.2f} p0={paper.get('experiment_todo_p0_closure_rate')}"
                )
            if isinstance(paper.get("experiment_todo_unresolved_count"), int):
                print(
                    f"   experiment TODO unresolved/closed: {paper.get('experiment_todo_unresolved_count')}/{paper.get('experiment_todo_closed_count')}"
                )
            if paper.get("experiment_todo_top_action"):
                print(f"   experiment TODO top: {paper['experiment_todo_top_action']}")
            if isinstance(paper.get("rewrite_priority_gain_total"), (int, float)):
                print(f"   rewrite 增益: {paper['rewrite_priority_gain_total']:.2f}")
            if paper.get("experiment_todo_file") and args.detailed:
                print(f"   experiment TODO file: {paper['experiment_todo_file']}")
            if paper.get("experiment_todo_progress_file") and args.detailed:
                print(
                    f"   experiment TODO progress: {paper['experiment_todo_progress_file']}"
                )
            if paper.get("submission_package_file") and args.detailed:
                print(f"   submission package: {paper['submission_package_file']}")
            if paper.get("narrative_map_file") and args.detailed:
                print(f"   narrative map: {paper['narrative_map_file']}")
            if paper.get("result_story_file") and args.detailed:
                print(f"   result story: {paper['result_story_file']}")
            if paper.get("contribution_map_file") and args.detailed:
                print(f"   contribution map: {paper['contribution_map_file']}")
            if paper.get("editor_pitch_file") and args.detailed:
                print(f"   editor pitch: {paper['editor_pitch_file']}")
            if paper.get("rebuttal_package_file") and args.detailed:
                print(f"   rebuttal package: {paper['rebuttal_package_file']}")
            if paper.get("risk_register_file") and args.detailed:
                print(f"   risk register: {paper['risk_register_file']}")
            if paper.get("cover_letter_file") and args.detailed:
                print(f"   cover letter: {paper['cover_letter_file']}")
            if paper.get("abstract_polish_file") and args.detailed:
                print(f"   abstract polish: {paper['abstract_polish_file']}")
            if paper.get("impact_brief_file") and args.detailed:
                print(f"   impact brief: {paper['impact_brief_file']}")
            if paper.get("contribution_bullets_file") and args.detailed:
                print(f"   contribution bullets: {paper['contribution_bullets_file']}")
            if paper.get("strongest_claims_file") and args.detailed:
                print(f"   strongest claims: {paper['strongest_claims_file']}")
            if paper.get("submission_manifest_file") and args.detailed:
                print(f"   submission manifest: {paper['submission_manifest_file']}")
            if paper.get("submission_dashboard_file") and args.detailed:
                print(f"   submission dashboard: {paper['submission_dashboard_file']}")
            if paper.get("risk_language_plan_file") and args.detailed:
                print(f"   risk language plan: {paper['risk_language_plan_file']}")
            if paper.get("claim_softening_plan_file") and args.detailed:
                print(f"   claim softening plan: {paper['claim_softening_plan_file']}")
            if paper.get("rewrite_effectiveness_file") and args.detailed:
                print(
                    f"   rewrite effectiveness: {paper['rewrite_effectiveness_file']}"
                )
            if paper.get("rewrite_trace_summary_file") and args.detailed:
                print(
                    f"   rewrite trace summary: {paper['rewrite_trace_summary_file']}"
                )
            if isinstance(paper.get("rewrite_round_count"), int):
                print(f"   rewrite rounds: {paper['rewrite_round_count']}")
            if isinstance(paper.get("rewrite_priority_gain_total"), (int, float)):
                print(
                    f"   rewrite priority gain: {paper['rewrite_priority_gain_total']:.2f}"
                )
            if isinstance(paper.get("rewrite_best_round_priority_delta"), (int, float)):
                print(
                    f"   best rewrite round delta: {paper['rewrite_best_round_priority_delta']:.2f}"
                )
            if paper.get("rewrite_top_section"):
                print(f"   top rewrite section: {paper['rewrite_top_section']}")
            if paper.get("quality_rewrite_applied") is not None:
                print(
                    f"   高质量重写: {'yes' if paper['quality_rewrite_applied'] else 'no'}"
                )
            if paper.get("quality_gate_passed") is not None:
                print(
                    f"   质量门槛: {'pass' if paper['quality_gate_passed'] else 'fail'}"
                )
            if paper.get("quality_status"):
                print(f"   质量状态: {paper['quality_status']}")
            if args.detailed:
                print(f"   路径: {paper['path']}")
            print()

    elif args.command == "paper-details":
        details = manager.get_paper_details(args.folder)
        if not details:
            print(f"❌ 未找到论文文件夹: {args.folder}")
            sys.exit(1)

        print(f"\n论文详细信息: {details['folder']}")
        print(f"路径: {details['path']}\n")

        if details["idea"]:
            print("📝 想法信息:")
            print(f"   名称: {details['idea'].get('Name', 'N/A')}")
            print(f"   标题: {details['idea'].get('Title', 'N/A')}")
            print(f"   摘要: {details['idea'].get('Abstract', 'N/A')[:200]}...")
            print()

        print("📁 文件列表:")
        for file_info in details["files"]:
            print(f"   {file_info['name']} ({format_size(file_info['size'])})")

        if details["reviews"]:
            print(f"\n🔍 审查记录 ({len(details['reviews'])} 轮):")
            for review in details["reviews"]:
                print(f"   {review['round']}")

        if details.get("quality"):
            quality = details["quality"]
            print(f"\n🏁 高质量摘要:")
            print(f"   target venue: {quality.get('target_venue')}")
            print(f"   quality after: {quality.get('quality_score_after')}")
            print(f"   rigor after: {quality.get('rigor_score_after')}")
            print(f"   claim support after: {quality.get('claim_support_after')}")
            print(f"   claim alignment after: {quality.get('claim_alignment_after')}")
            print(f"   numeric coverage after: {quality.get('numeric_coverage_after')}")
            print(f"   breakthrough score: {quality.get('breakthrough_score')}")
            print(f"   evidence density: {quality.get('evidence_density_score')}")
            print(
                f"   submission priority: {quality.get('submission_priority_score')} ({quality.get('submission_priority_tier')})"
            )
            print(f"   gate passed: {quality.get('quality_gate_passed')}")
            readiness = quality.get("submission_readiness", {})
            if readiness:
                print(f"   readiness: {readiness.get('status')}")
                for blocker in readiness.get("blockers", [])[:5]:
                    print(f"     - {blocker}")
                categories = readiness.get("categories", {})
                if categories:
                    print(f"   blocker categories: {categories}")
            unsupported = quality.get("unsupported_claims_count")
            if unsupported is not None:
                print(f"   unsupported claims: {unsupported}")
            claim_rewrites = quality.get("suggested_claim_rewrites_count")
            if claim_rewrites is not None:
                print(f"   suggested claim rewrites: {claim_rewrites}")
            revision_actions = quality.get("revision_actions") or []
            if revision_actions:
                print("   revision actions:")
                for item in revision_actions[:4]:
                    print(
                        f"     - [{item.get('priority')}] {item.get('focus')}: {item.get('action')}"
                    )

        if details.get("evidence_pack"):
            evidence = details["evidence_pack"]
            print(f"\n📎 Strongest Results:")
            for item in evidence.get("strongest_results", [])[:3]:
                print(
                    f"   - {item.get('type')}:{item.get('label')} "
                    f"refs={item.get('ref_count')} caption={item.get('caption', '')[:120]}"
                )

        if details.get("key_results"):
            print(f"\n🔢 Key Numerical Results:")
            for value in details["key_results"].get("values", [])[:10]:
                print(f"   - {value}")

        if details.get("claim_alignment"):
            print(f"\n🔗 Claim Alignment:")
            for item in details["claim_alignment"].get("claims", [])[:5]:
                if item.get("suggested_rewrite"):
                    print(f"   - claim: {item.get('claim')[:120]}")
                    print(f"     suggestion: {item.get('suggested_rewrite')}")

        if details.get("contribution_map"):
            print(f"\n🧭 Contribution Map:")
            for item in details["contribution_map"].get("contributions", [])[:3]:
                print(f"   - {item.get('title')}: {item.get('claim')[:120]}")
                print(
                    f"     evidence: {', '.join(item.get('evidence_labels', [])) or 'n/a'}"
                )
                print(
                    f"     key results: {', '.join(item.get('key_results', [])) or 'n/a'}"
                )

        if details.get("submission_dashboard"):
            print("\n🧪 Submission Dashboard:\n")
            print(details["submission_dashboard"][:1200])

        if details.get("editor_pitch"):
            print("\n📝 Editor Pitch:\n")
            print(details["editor_pitch"][:1200])

        if details.get("narrative_map"):
            print("\n🗺️ Narrative Map:\n")
            print(details["narrative_map"][:1200])

        if details.get("result_story"):
            print("\n📚 Result Story:\n")
            print(details["result_story"][:1200])

        if details.get("impact_brief"):
            print("\n🌍 Impact Brief:\n")
            print(details["impact_brief"][:1200])

        if details.get("contribution_bullets"):
            print("\n📌 Contribution Bullets:\n")
            print(details["contribution_bullets"][:1200])

        if details.get("strongest_claims"):
            print("\n💥 Strongest Claims:\n")
            print(details["strongest_claims"][:1200])

        if details.get("risk_register"):
            print("\n⚠️ Risk Register:\n")
            print(details["risk_register"][:1200])

        if details.get("cover_letter"):
            print("\n📨 Cover Letter:\n")
            print(details["cover_letter"][:1200])

        if details.get("abstract_polish"):
            print("\n✍️ Abstract Polish:\n")
            print(details["abstract_polish"][:1200])

        if details.get("rebuttal_package"):
            print("\n🛡️ Rebuttal Package:\n")
            print(details["rebuttal_package"][:1200])

        if details.get("risk_language_plan"):
            print("\n🧷 Risk Language Plan:\n")
            print(details["risk_language_plan"][:1200])

        if details.get("claim_softening_plan"):
            print("\n🪶 Claim Softening Plan:\n")
            print(details["claim_softening_plan"][:1200])

        if details.get("rewrite_effectiveness"):
            print("\n📈 Rewrite Effectiveness:\n")
            print(details["rewrite_effectiveness"][:1200])

        if details.get("rewrite_trace_summary"):
            trace = details["rewrite_trace_summary"]
            print("\n🧮 Rewrite Trace Summary:\n")
            print(f"  rounds: {trace.get('round_count')}")
            print(f"  total priority gain: {trace.get('priority_gain_total')}")
            print(
                f"  avg priority gain / round: {trace.get('avg_priority_gain_per_round')}"
            )
            best_round = trace.get("best_round") or {}
            if best_round:
                print(
                    f"  best round: {best_round.get('round')} (delta={best_round.get('priority_delta')}, quality={best_round.get('quality_delta')})"
                )
            if trace.get("top_frontmatter_style"):
                print(f"  top frontmatter style: {trace.get('top_frontmatter_style')}")
            if trace.get("top_section_style"):
                print(f"  top section style: {trace.get('top_section_style')}")
            if trace.get("top_section"):
                print(f"  top section: {trace.get('top_section')}")
    elif args.command == "list-ideas":
        ideas = manager.list_ideas()
        print(f"\n共有 {len(ideas)} 个想法:\n")
        for idea in ideas:
            print(f"💡 {idea['name']}")
            if idea["title"]:
                print(f"   标题: {idea['title']}")
            print(f"   来源: {idea['source']}")
            print()

    elif args.command == "search-papers":
        papers = manager.search_papers(args.query, args.type)
        print(f"\n找到 {len(papers)} 篇匹配 '{args.query}' 的论文:\n")
        for paper in papers:
            print(f"📄 {paper['name']} ({paper['type']})")
            print(f"   {paper['path']}")
            print()

    elif args.command == "cleanup":
        manager.cleanup_old_files(args.days, args.dry_run)

    elif args.command == "stats":
        batches = manager.list_batches()
        papers = manager.list_papers()
        ideas = manager.list_ideas()
        index_summary = manager.get_index_summary()
        failure_stage_counts = Counter()
        quality_scores = []
        claim_support_scores = []
        priority_scores = []
        rewrite_gains = []
        rewrite_top_sections = Counter()
        rewrite_top_frontmatter_styles = Counter()
        priority_tiers = Counter()
        quality_gate_pass = 0
        quality_gate_fail = 0
        for batch in batches:
            for failed in batch.get("progress", {}).get("papers_failed", []):
                failure_stage_counts[failed.get("stage") or "unknown"] += 1
        for paper in papers:
            if isinstance(paper.get("quality_score"), (int, float)):
                quality_scores.append(paper["quality_score"])
            if isinstance(paper.get("claim_support_score"), (int, float)):
                claim_support_scores.append(paper["claim_support_score"])
            if isinstance(paper.get("submission_priority_score"), (int, float)):
                priority_scores.append(paper["submission_priority_score"])
            if paper.get("submission_priority_tier"):
                priority_tiers[paper["submission_priority_tier"]] += 1
            if isinstance(paper.get("rewrite_priority_gain_total"), (int, float)):
                rewrite_gains.append(paper["rewrite_priority_gain_total"])
            if paper.get("rewrite_top_section"):
                rewrite_top_sections[paper["rewrite_top_section"]] += 1
            if paper.get("rewrite_top_frontmatter_style"):
                rewrite_top_frontmatter_styles[
                    paper["rewrite_top_frontmatter_style"]
                ] += 1
            if paper.get("quality_gate_passed") is True:
                quality_gate_pass += 1
            elif paper.get("quality_gate_passed") is False:
                quality_gate_fail += 1

        print(f"\n📊 研究目录统计")
        print(f"=" * 50)
        print(f"目录: {args.research_dir}")
        print()
        print(f"批次: {len(batches)}")
        print(f"论文: {len(papers)}")
        print(f"想法: {len(ideas)}")
        print(f"索引条目: {index_summary['entries']}")
        print()

        # 按类型统计论文
        paper_types = {}
        for paper in papers:
            paper_type = paper["type"]
            paper_types[paper_type] = paper_types.get(paper_type, 0) + 1

        print("按类型统计:")
        for paper_type, count in sorted(paper_types.items()):
            print(f"  {paper_type}: {count}")

        if index_summary["generated_at"]:
            print()
            print(f"索引更新时间: {index_summary['generated_at']}")
            for category, count in sorted(index_summary["by_category"].items()):
                print(f"  index/{category}: {count}")

        if quality_scores:
            print()
            print(f"平均质量分: {sum(quality_scores) / len(quality_scores):.2f}")
            print(f"最高质量分: {max(quality_scores):.2f}")
            if claim_support_scores:
                print(
                    f"平均论证支撑: {sum(claim_support_scores) / len(claim_support_scores):.2f}"
                )
            print(f"质量门槛通过: {quality_gate_pass}")
            print(f"质量门槛未过: {quality_gate_fail}")
            if priority_scores:
                print(
                    f"平均投稿优先级: {sum(priority_scores) / len(priority_scores):.2f}"
                )
                if rewrite_gains:
                    print(
                        f"平均 rewrite 增益: {sum(rewrite_gains) / len(rewrite_gains):.2f}"
                    )
                for tier, count in sorted(priority_tiers.items()):
                    print(f"  priority/{tier}: {count}")
                if rewrite_top_sections:
                    print("  热门重写章节:")
                    for section, count in rewrite_top_sections.most_common(5):
                        print(f"    - {section}: {count}")
                if rewrite_top_frontmatter_styles:
                    print("  热门 frontmatter 风格:")
                    for style, count in rewrite_top_frontmatter_styles.most_common(5):
                        print(f"    - {style}: {count}")

        if failure_stage_counts:
            print()
            print("失败阶段摘要:")
            for stage, count in sorted(failure_stage_counts.items()):
                print(f"  {stage}: {count}")

    elif args.command == "rebuild-index":
        index = manager.rebuild_index()
        print(f"✅ 索引已重建: {run_index_path(args.research_dir)}")
        print(f"条目数: {len(index.get('entries', {}))}")

    elif args.command == "shortlist":
        shortlist = manager.shortlist_papers(
            paper_type=args.type,
            target_venue=args.venue,
            require_gate=args.require_gate,
            require_ready=args.require_ready,
            min_breakthrough=args.min_breakthrough,
            min_priority=args.min_priority,
            max_blockers=args.max_blockers,
            min_rewrite_gain=args.min_rewrite_gain,
            max_fallbacks=args.max_fallbacks,
            max_strict_fallbacks=args.max_strict_fallbacks,
            max_blocked_stages=args.max_blocked_stages,
            max_missing_stages=args.max_missing_stages,
            max_attention_stages=args.max_attention_stages,
            min_stage_score=args.min_stage_score,
            max_self_evolution_required_failures=args.max_self_evolution_required_failures,
            min_self_evolution_score=args.min_self_evolution_score,
            allow_blocked_self_evolution=args.allow_blocked_self_evolution,
            max_blocked_processes=args.max_blocked_processes,
            min_process_alignment_score=args.min_process_alignment_score,
            top_n=args.top,
        )
        print(f"\n🎯 投稿 shortlist ({len(shortlist)} 篇):\n")
        for paper in shortlist:
            print(f"📄 {paper['name']}")
            print(f"   类型: {paper['type']}")
            print(f"   venue: {paper.get('target_venue')}")
            print(
                f"   投稿优先级: {paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')})"
            )
            print(f"   rewrite 增益: {paper.get('rewrite_priority_gain_total')}")
            print(f"   blocker 数: {paper.get('blocker_count')}")
            print(
                f"   阶段标准: score={paper.get('stage_overall_score')} blocked={paper.get('blocked_stage_count')} "
                f"attention={paper.get('needs_attention_stage_count')} missing={paper.get('missing_stage_count')}"
            )
            print(
                f"   reviewer修复: resolution={paper.get('review_resolution_rate')} active={paper.get('review_active_issue_count')} "
                f"persistent={paper.get('review_persistent_issue_count')} checks={paper.get('review_verification_count')}"
            )
            print(
                f"   reviewer绑定: coverage={paper.get('review_target_binding_coverage')} active_coverage={paper.get('review_active_binding_coverage')} "
                f"unbound={paper.get('review_unbound_issue_count')}"
            )
            print(
                f"   reviewer修复队列: queue={paper.get('review_repair_queue_count')} ready={paper.get('review_repair_ready_count')} "
                f"ready_coverage={paper.get('review_repair_ready_coverage')} verification_ready={paper.get('review_repair_verification_ready_count')}"
            )
            print(
                f"   self-evolution: status={paper.get('self_evolution_status')} score={paper.get('self_evolution_score')} "
                f"required_failures={paper.get('self_evolution_required_failure_count')} lane={paper.get('self_evolution_dominant_lane')} "
                f"role={paper.get('self_evolution_dominant_role')}"
            )
            if paper.get("self_evolution_top_risks"):
                print(
                    "   self-evolution风险: "
                    + ", ".join(paper.get("self_evolution_top_risks") or [])
                )
            print(
                f"   process-alignment: score={paper.get('process_alignment_overall_score')} "
                f"blocked={paper.get('process_alignment_blocked_process_count')} attention={paper.get('process_alignment_attention_process_count')} "
                f"missing={paper.get('process_alignment_missing_process_count')}"
            )
            if paper.get("process_alignment_top_risks"):
                print(
                    "   process风险: "
                    + ", ".join(paper.get("process_alignment_top_risks") or [])
                )
            print(
                f"   fallback: total={paper.get('fallback_count')} strict={paper.get('strict_fallback_count')}"
            )
            print(f"   质量分: {paper.get('quality_score')}")
            print(f"   严谨性: {paper.get('rigor_score')}")
            print(f"   论证支撑: {paper.get('claim_support_score')}")
            print(f"   门槛: {paper.get('quality_gate_passed')}")
            print(f"   路径: {paper['path']}")
            print()

        if args.export:
            export_path = manager.export_shortlist_markdown(shortlist, args.export)
            print(f"📝 已导出 shortlist: {export_path}")

    elif args.command == "submission-board":
        board = manager.submission_board(
            args.top,
            args.min_breakthrough,
            args.min_priority,
            args.max_blockers,
            args.min_rewrite_gain,
            args.require_gate,
            args.max_fallbacks,
            args.max_strict_fallbacks,
            args.max_blocked_stages,
            args.max_missing_stages,
            args.max_attention_stages,
            args.min_stage_score,
            args.max_self_evolution_required_failures,
            args.min_self_evolution_score,
            args.allow_blocked_self_evolution,
            args.max_blocked_processes,
            args.min_process_alignment_score,
        )
        print("\n🗂️ Submission Board\n")
        for venue, papers in sorted(board.items()):
            print(f"## {venue}")
            for paper in papers:
                print(
                    f"- {paper['name']} | priority={paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')}) | "
                    f"rewrite_gain={paper.get('rewrite_priority_gain_total')} | blockers={paper.get('blocker_count')} | "
                    f"stage_score={paper.get('stage_overall_score')} blocked_stages={paper.get('blocked_stage_count')} "
                    f"attention_stages={paper.get('needs_attention_stage_count')} missing_stages={paper.get('missing_stage_count')} | "
                    f"self_evolution={paper.get('self_evolution_status')}:{paper.get('self_evolution_score')} "
                    f"required_failures={paper.get('self_evolution_required_failure_count')} | "
                    f"process_alignment={paper.get('process_alignment_overall_score')} "
                    f"blocked_processes={paper.get('process_alignment_blocked_process_count')} | "
                    f"review_resolution={paper.get('review_resolution_rate')} review_binding={paper.get('review_target_binding_coverage')} "
                    f"repair_ready={paper.get('review_repair_ready_coverage')} "
                    f"active_review_issues={paper.get('review_active_issue_count')} persistent_review_issues={paper.get('review_persistent_issue_count')} | "
                    f"fallbacks={paper.get('fallback_count')} strict={paper.get('strict_fallback_count')} | breakthrough={paper.get('breakthrough_score')} | "
                    f"rigor={paper.get('rigor_score')} | gate={paper.get('quality_gate_passed')}"
                )
            print()

        if args.export:
            export_path = manager.export_submission_board_markdown(board, args.export)
            print(f"📝 已导出 submission board: {export_path}")

    elif args.command == "rewrite-board":
        papers = manager.rewrite_board(
            top_n=args.top,
            paper_type=args.type,
            target_venue=args.venue,
            min_priority=args.min_priority,
            min_rewrite_gain=args.min_rewrite_gain,
            max_blockers=args.max_blockers,
            require_gate=args.require_gate,
            include_ready=args.include_ready,
        )
        print("\n🛠️ Rewrite Board\n")
        for paper in papers:
            print(
                f"- {paper['name']} | priority={paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')}) | "
                f"rewrite_gain={paper.get('rewrite_priority_gain_total')} | best_round={paper.get('rewrite_best_round_priority_delta')} | "
                f"top_section={paper.get('rewrite_top_section')} | blockers={paper.get('blocker_count')} | "
                f"todo={paper.get('experiment_todo_count')} p0={paper.get('experiment_todo_p0_count')} closure={paper.get('experiment_todo_closure_rate')} | "
                f"repair_ready={paper.get('review_repair_ready_coverage')} binding={paper.get('review_active_binding_coverage')}"
            )
            print(f"  next: {paper.get('suggested_next_step')}")
            print()

        if args.export:
            export_path = manager.export_rewrite_board_markdown(papers, args.export)
            print(f"📝 已导出 rewrite board: {export_path}")
    elif args.command == "repair-board":
        rows = manager.repair_board(
            top_n=args.top,
            target_venue=args.venue,
            priority_tier=args.priority_tier,
            only_ready=args.only_ready,
        )
        print("\n🧩 Repair Board\n")
        for row in rows:
            print(
                f"- {row.get('name')} | {row.get('priority_tier')} {row.get('status')} | "
                f"target={row.get('primary_target_type')}:{row.get('primary_target_id')} | "
                f"role={row.get('role')} | issue={row.get('issue_text')}"
            )
            print(f"  actions: {' | '.join(row.get('repair_actions') or []) or 'none'}")
            print(
                f"  verification: {' | '.join(row.get('verification_checks') or []) or 'none'}"
            )
            print()

        if args.export:
            export_path = manager.export_repair_board_markdown(rows, args.export)
            print(f"📝 已导出 repair board: {export_path}")
    elif args.command == "evolution-board":
        rows = manager.evolution_board(
            top_n=args.top,
            status=args.status,
        )
        print("\n🧠 Self-Evolution Board\n")
        for row in rows:
            print(
                f"- {row.get('name')} | status={row.get('status')} | score={row.get('score')} | "
                f"lane={row.get('dominant_lane')} | role={row.get('dominant_role')} | "
                f"lessons={row.get('lesson_count')}"
            )
            if row.get("stage_risks"):
                print(f"  stage_risks={row.get('stage_risks')[:3]}")
            if row.get("top_lessons"):
                print(
                    "  lessons="
                    + " | ".join(
                        str(item.get("recommended_action") or "")
                        for item in row.get("top_lessons") or []
                        if str(item.get("recommended_action") or "").strip()
                    )
                )
            if row.get("next_cycle_defaults"):
                print(f"  next_cycle_defaults={row.get('next_cycle_defaults')}")
            print()
    elif args.command == "submission-dossier":
        result = manager.export_submission_dossier(args.folder, args.output_dir)
        if result["status"] != "success":
            print(f"❌ {result['reason']}")
            sys.exit(1)
        print(f"📦 投稿材料已导出到: {result['output_dir']}")
        print(f"📋 Manifest: {result['manifest']}")
    elif args.command == "readiness-benchmark":
        benchmark = manager.readiness_benchmark(
            target_venue=args.venue,
            max_entries=args.max_entries,
            top_n=args.top,
            include_other_venues=args.include_other_venues,
        )
        summary = benchmark.get("summary", {})
        print("\n🏁 Readiness Benchmark\n")
        print(f"Target venue: {benchmark.get('target_venue')}")
        print(f"Research dir: {benchmark.get('research_root')}")
        print(f"Entries: {summary.get('entries')}")
        print(f"Venue match: {summary.get('venue_match_count')}")
        print(f"Ready: {summary.get('ready_count')}")
        print(f"Gate passed: {summary.get('gate_pass_count')}")
        print(f"Avg benchmark score: {summary.get('avg_benchmark_score')}")
        print(f"Avg submission priority: {summary.get('avg_submission_priority')}")
        print(f"Avg blocker count: {summary.get('avg_blocker_count')}")
        print(
            "Avg process alignment score: "
            f"{summary.get('avg_process_alignment_score')}"
        )
        print(
            "Avg blocked process alignment count: "
            f"{summary.get('avg_process_alignment_blocked_count')}"
        )
        print(f"Avg self-evolution score: {summary.get('avg_self_evolution_score')}")
        print(
            "Avg self-evolution required failures: "
            f"{summary.get('avg_self_evolution_required_failure_count')}"
        )

        top_gaps = summary.get("top_gap_dimensions") or {}
        if top_gaps:
            print("\nTop gap dimensions:")
            for name, count in top_gaps.items():
                print(f"  - {name}: {count}")

        top_categories = summary.get("top_blocker_categories") or {}
        if top_categories:
            print("\nTop blocker categories:")
            for name, count in top_categories.items():
                print(f"  - {name}: {count}")

        top_process_risks = summary.get("top_process_alignment_risks") or {}
        if top_process_risks:
            print("\nTop process alignment risks:")
            for name, count in top_process_risks.items():
                print(f"  - {name}: {count}")

        top_evolution_risks = summary.get("top_self_evolution_risks") or {}
        if top_evolution_risks:
            print("\nTop self-evolution risks:")
            for name, count in top_evolution_risks.items():
                print(f"  - {name}: {count}")

        ranked = benchmark.get("ranked_papers") or []
        if ranked:
            print("\nTop papers:")
            for paper in ranked:
                print(
                    f"- {paper.get('name')} | benchmark={paper.get('benchmark_score')} | "
                    f"status={paper.get('submission_status')} | gate={paper.get('quality_gate_passed')} | "
                    f"priority={paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')}) | "
                    f"blockers={paper.get('blocker_count')} | "
                    f"process_alignment={paper.get('process_alignment_overall_score')} "
                    f"blocked={paper.get('process_alignment_blocked_process_count')} | "
                    f"self_evolution={paper.get('self_evolution_status')}:{paper.get('self_evolution_score')} "
                    f"required_failures={paper.get('self_evolution_required_failure_count')} | "
                    f"venue={paper.get('paper_target_venue')} | "
                    f"match={paper.get('venue_match')}"
                )
                failing_metrics = paper.get("failing_metrics") or []
                if failing_metrics:
                    print(
                        "  gaps: "
                        + ", ".join(
                            f"{item.get('name')}({item.get('gap')})"
                            for item in failing_metrics[:3]
                        )
                    )
                if paper.get("top_blockers"):
                    print("  blockers: " + " | ".join(paper["top_blockers"]))
                if paper.get("recommendation"):
                    print(f"  next: {paper.get('recommendation')}")
                print()

        if args.export:
            export_path = manager.export_readiness_benchmark(benchmark, args.export)
            print(f"📝 已导出 readiness benchmark: {export_path}")
    elif args.command == "pipeline-status":
        rows = manager.pipeline_status(top_n=args.top)
        print("\n🧩 Pipeline Status\n")
        for row in rows:
            print(
                f"- {row['project']} | ready={row.get('ready_count')}/{row.get('artifact_total')} | "
                f"stage_score={row.get('stage_overall_score')} | "
                f"process_alignment={row.get('process_alignment_overall_score')} blocked_processes={row.get('process_alignment_blocked_process_count')} | "
                f"evolution={row.get('self_evolution_status')}:{row.get('self_evolution_score')} "
                f"review_resolution={row.get('review_resolution_rate')} review_binding={row.get('review_target_binding_coverage')} "
                f"repair_ready={row.get('review_repair_ready_coverage')} "
                f"persistent_review_issues={row.get('review_persistent_issue_count')} | "
                f"fallbacks={row.get('fallback_count')} strict={row.get('strict_fallback_count')} | "
                f"template={row.get('template_profile')} ({row.get('template_capability')})"
            )
            print(
                f"  blocked={row.get('blocked_artifacts')} failed={row.get('failed_artifacts')} missing={row.get('missing_artifacts')}"
            )
            print(
                f"  standard_blocked={row.get('blocked_standard_stages')} attention={row.get('attention_standard_stages')} missing={row.get('missing_standard_stages')}"
            )
            if row.get("top_standard_risks"):
                print(f"  top_risks={row.get('top_standard_risks')}")
            if row.get("process_alignment_top_risks"):
                print(f"  process_risks={row.get('process_alignment_top_risks')}")
            if row.get("warnings"):
                print(f"  warnings={row.get('warnings')[:3]}")
            print()
    elif args.command == "stage-standards":
        rows = manager.stage_standards_board(
            top_n=args.top,
            stage=args.stage,
            status=args.status,
        )
        print("\n📏 Stage Standards\n")
        for row in rows:
            print(
                f"- {row.get('project')} | stage={row.get('stage')} | status={row.get('status')} | "
                f"score={row.get('score')} | passed={row.get('passed_criteria_count')}/{row.get('criteria_count')}"
            )
            if row.get("required_failures"):
                print(f"  required_failures={row.get('required_failures')}")
            if row.get("missing_reason"):
                print(f"  missing_reason={row.get('missing_reason')}")
            if row.get("signals"):
                print(f"  signals={row.get('signals')}")
            print()
    elif args.command == "process-board":
        rows = manager.process_board(
            top_n=args.top,
            process=args.process,
            status=args.status,
        )
        print("\n🧭 Process Alignment Board\n")
        for row in rows:
            print(
                f"- {row.get('project')} | process={row.get('process')} | status={row.get('status')} | "
                f"score={row.get('score')} | passed={row.get('passed_criteria_count')}/{row.get('criteria_count')}"
            )
            if row.get("references"):
                print(f"  refs={row.get('references')}")
            if row.get("required_failures"):
                print(f"  required_failures={row.get('required_failures')}")
            if row.get("risks"):
                print(f"  risks={row.get('risks')}")
            if row.get("missing_reason"):
                print(f"  missing_reason={row.get('missing_reason')}")
            if row.get("signals"):
                print(f"  signals={row.get('signals')}")
            print()
    elif args.command == "fallback-board":
        rows = manager.fallback_board(top_n=args.top, stage=args.stage)
        print("\n🧯 Fallback Board\n")
        for row in rows:
            print(
                f"- {row.get('project')} | workflow={row.get('workflow_mode')} | "
                f"fallbacks={row.get('fallback_count')} strict={row.get('strict_fallback_count')}"
            )
            print(f"  stages={row.get('stage_counts')} kinds={row.get('kind_counts')}")
            print(
                f"  latest={row.get('latest_stage')} / {row.get('latest_kind')} | reason={row.get('latest_reason')}"
            )
            print(
                f"  metadata={row.get('latest_metadata')} recorded_at={row.get('latest_recorded_at')}"
            )
            print()
    elif args.command == "idea-board":
        rows = manager.idea_board(top_n=args.top, status=args.status)
        print("\n💡 Idea Board\n")
        for row in rows:
            print(
                f"- {row.get('project')}::{row.get('idea_id')} | status={row.get('status')} | "
                f"venue={row.get('target_venue')} | risk={row.get('compute_risk')}"
            )
            print(
                f"  datasets={row.get('datasets')} metrics={row.get('metrics')} baselines={row.get('baselines')}"
            )
            print(f"  mve={row.get('minimum_viable_experiment')}")
            print()
    elif args.command == "experiment-board":
        rows = manager.experiment_board(top_n=args.top, status=args.status)
        print("\n🧪 Experiment Board\n")
        for row in rows:
            summary = row.get("result_summary") or {}
            print(
                f"- {row.get('project')}::{row.get('task_id')} | status={row.get('status')} | "
                f"dataset={row.get('dataset')} | metric={row.get('metric')} | storyline={row.get('entered_storyline')}"
            )
            if summary:
                print(
                    f"  best_metric={summary.get('metric_name')} mean={summary.get('metric_mean')} warnings={summary.get('warnings')}"
                )
            if row.get("error_type") or row.get("error_message"):
                print(f"  error={row.get('error_type')}: {row.get('error_message')}")
            print()
    elif args.command == "figure-board":
        rows = manager.figure_board(
            top_n=args.top,
            include_blocked=not args.ready_only,
        )
        print("\n📈 Figure Board\n")
        for row in rows:
            print(
                f"- {row.get('project')}::{row.get('figure_id')} | status={row.get('status')} | "
                f"claim={row.get('claim_id')} | type={row.get('figure_type')} | slot={row.get('paper_slot')}"
            )
            print(
                f"  data_files={row.get('data_files')} source_records={row.get('source_records')} blocking={row.get('blocking_reasons')}"
            )
            print()
    elif args.command == "source-board":
        rows = manager.source_board(top_n=args.top, archetype=args.archetype)
        print("\n🧭 Source Board\n")
        for row in rows:
            print(
                f"- {row.get('source_name') or row.get('source_key')} | type={row.get('source_type')} | "
                f"archetype={row.get('source_archetype')} | profile={row.get('source_batch_profile')} | "
                f"workflow={row.get('source_workflow_mode')}"
            )
            print(
                f"  runs={row.get('run_count')} ready={row.get('ready_count')} gate={row.get('gate_pass_count')} "
                f"avg_quality={row.get('avg_quality_score')} avg_priority={row.get('avg_submission_priority')} "
                f"avg_fallback={row.get('avg_fallback_count')} strict_fallback={row.get('avg_strict_fallback_count')} "
                f"venue={row.get('target_venue')}"
            )
            print(
                f"  source={row.get('source_value')} latest={row.get('latest_project')} "
                f"fallback_free_rate={row.get('fallback_free_rate')} updated={row.get('updated_at')}"
            )
            print()
    elif args.command == "source-mix":
        advisory = manager.source_mix_advisory(
            desired_policy=args.desired_policy,
            top_n=args.top,
        )
        summary = advisory.get("summary") or {}
        print("\n🧭 Source Mix\n")
        print(f"Desired policy: {advisory.get('desired_policy') or 'n/a'}")
        print(f"Source count: {summary.get('source_count')}")
        print(f"Dominant archetype: {summary.get('dominant_archetype')}")
        print(f"Dominant workflow: {summary.get('dominant_workflow_mode')}")
        print(f"Archetype counts: {summary.get('archetype_counts')}")
        print(f"Workflow counts: {summary.get('workflow_mode_counts')}")
        print(f"Batch profile counts: {summary.get('batch_profile_counts')}")
        if advisory.get("top_sources"):
            print("\nTop sources:")
            for row in advisory.get("top_sources") or []:
                print(
                    f"- {row.get('source_name') or row.get('source_key')} | archetype={row.get('source_archetype')} | "
                    f"workflow={row.get('source_workflow_mode')} | ready={row.get('ready_count')} | "
                    f"gate={row.get('gate_pass_count')} | avg_priority={row.get('avg_submission_priority')}"
                )
        if advisory.get("recommendations"):
            print("\nRecommendations:")
            for item in advisory.get("recommendations") or []:
                print(f"- [{item.get('tier')}] {item.get('recommendation')}")
    elif args.command == "source-next-batch":
        advisory = manager.source_next_batch_advisory(
            desired_policy=args.desired_policy,
            top_n=args.top,
            max_slots=args.max_slots,
        )
        summary = advisory.get("summary") or {}
        cadence = advisory.get("cadence") or {}
        print("\n🧭 Next Batch Source Mix\n")
        print(f"Desired policy: {advisory.get('desired_policy') or 'n/a'}")
        print(f"Source count: {summary.get('source_count')}")
        print(f"Dominant archetype: {summary.get('dominant_archetype')}")
        print(f"Dominant workflow: {summary.get('dominant_workflow_mode')}")
        print(f"Cadence: {cadence.get('label')} | {cadence.get('reason')}")
        if advisory.get("slots"):
            print("\nSlots:")
            for slot in advisory.get("slots") or []:
                print(
                    f"- {slot.get('lane')} | {slot.get('source')} | archetype={slot.get('source_archetype')} | "
                    f"workflow={slot.get('source_workflow_mode')} | profile={slot.get('source_batch_profile')} | "
                    f"share={slot.get('share')}"
                )
                print(
                    f"  ready={slot.get('ready_count')} gate={slot.get('gate_pass_count')} "
                    f"avg_priority={slot.get('avg_submission_priority')} score={slot.get('batch_score')}"
                )
                print(f"  focus={slot.get('focus')}")
                print(f"  rationale={slot.get('rationale')}")
                print()
        if advisory.get("recommendations"):
            print("Recommendations:")
            for item in advisory.get("recommendations") or []:
                print(f"- [{item.get('tier')}] {item.get('recommendation')}")
    elif args.command == "benchmark-trends":
        trends = manager.benchmark_trends(
            target_venue=args.venue,
            max_entries=args.max_entries,
        )
        print("\n📊 Benchmark Trends\n")
        print(f"Target venue: {trends.get('target_venue')}")
        print(f"Summary: {trends.get('summary')}")
        for row in trends.get("timeline") or []:
            print(
                f"- {row.get('date')} | count={row.get('count')} | ready={row.get('ready_count')} | "
                f"gate={row.get('gate_pass_count')} | avg_benchmark={row.get('avg_benchmark_score')} | "
                f"avg_priority={row.get('avg_submission_priority')}"
            )
