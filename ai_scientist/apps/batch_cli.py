"""Command-line contract for continuous XScientist paper generation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser(
    *,
    default_research_dir: str,
    default_writing_profile: str,
    writing_profiles: Sequence[str],
    workflow_modes: Sequence[str],
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XScientist continuous paper generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 生成想法并为所有类型生成论文:
   python -m xscientist batch \\
     --topic my_topic.md \\
     --num-ideas 5 \\
     --all-types

2. 仅生成workshop论文:
   python -m xscientist batch \\
     --topic my_topic.md \\
     --paper-types icbinb

3. 从已有想法生成:
   python -m xscientist batch \\
     --ideas existing_ideas.json \\
     --paper-types normal journal

4. 并行处理:
   python -m xscientist batch \\
     --topic my_topic.md \\
     --all-types \\
     --num-workers 2
        """,
    )

    parser.add_argument(
        "--research-dir",
        type=str,
        default=default_research_dir,
        help="研究输出目录（默认仓库平级输出目录，可通过 RESEARCH_OUTPUT_DIR 覆盖）",
    )
    parser.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="批次名称 (默认为时间戳)",
    )

    parser.add_argument("--topic", type=str, help="主题描述文件")
    parser.add_argument("--ideas", type=str, help="已有想法JSON文件")
    parser.add_argument("--num-ideas", type=int, default=5, help="生成的想法数量")
    parser.add_argument("--num-reflections", type=int, default=5)

    parser.add_argument(
        "--paper-types",
        type=str,
        nargs="+",
        default=["icbinb"],
        choices=["icbinb", "normal", "journal", "extended"],
        help="要生成的论文类型",
    )
    parser.add_argument(
        "--all-types",
        action="store_true",
        help="生成所有类型的论文",
    )

    parser.add_argument(
        "--idea-indices",
        type=str,
        help="要处理的想法索引 (逗号分隔)，如: 0,1,2",
    )
    parser.add_argument(
        "--submission-mode", action="store_true", help="启用完整的投稿级 preset"
    )
    parser.add_argument(
        "--breakthrough-mode", action="store_true", help="偏向重大问题和高影响力投稿"
    )
    parser.add_argument(
        "--rank-ideas",
        action="store_true",
        help="先对 ideas 排序，再优先选择更适合高质量论文的想法",
    )
    parser.add_argument(
        "--top-k-ideas",
        type=int,
        default=None,
        help="只处理评分最高的前 K 个 idea（需配合 --rank-ideas）",
    )
    parser.add_argument(
        "--idea-rank-model",
        type=str,
        default=None,
        help="用于想法排序的模型；可用逗号分隔多个独立评审模型",
    )

    parser.add_argument("--num-workers", type=int, default=1, help="并行worker数量")

    parser.add_argument(
        "--improvement-rounds",
        type=int,
        default=1,
        help="自动改进轮数 (0=禁用自动改进)",
    )
    parser.add_argument(
        "--improvement-preset",
        type=str,
        choices=["quick_paper", "standard_paper", "high_quality", "journal_submission"],
        default=None,
        help="使用预设的改进策略",
    )
    parser.add_argument(
        "--min-improvement-threshold",
        type=float,
        default=0.5,
        help="最小改进阈值 (低于此值将停止迭代)",
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
        help="高质量生成预设",
    )
    parser.add_argument(
        "--quality-model", type=str, default=None, help="质量评估使用的模型"
    )
    parser.add_argument(
        "--target-venue",
        type=str,
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        default=None,
        help="目标投稿 venue",
    )
    parser.add_argument(
        "--auto-adjust-paper-type",
        action="store_true",
        help="根据目标 venue 自动调整 paper type",
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=None, help="目标最低质量分"
    )
    parser.add_argument(
        "--rigor-threshold", type=float, default=None, help="目标最低严谨性分"
    )
    parser.add_argument(
        "--quality-rewrite-rounds", type=int, default=None, help="定向重写最大轮数"
    )
    parser.add_argument(
        "--autonomous-quality-followup-rounds",
        type=int,
        default=0,
        help="高质量模式未达提交标准时自动补跑 follow-up 的最大轮数",
    )
    parser.add_argument(
        "--min-submission-priority",
        type=float,
        default=None,
        help="接受稿件所需的最低投稿优先级",
    )
    parser.add_argument(
        "--max-submission-blockers",
        type=int,
        default=None,
        help="接受稿件所允许的最大 blocker 数",
    )
    parser.add_argument(
        "--require-quality-gate",
        action="store_true",
        help="高质量模式下，未通过质量门槛则视为失败",
    )
    parser.add_argument(
        "--integrity-forensics",
        dest="integrity_forensics",
        action="store_true",
        default=None,
        help="启用最终稿 deterministic integrity forensics 检查。",
    )
    parser.add_argument(
        "--no-integrity-forensics",
        dest="integrity_forensics",
        action="store_false",
        help="禁用最终稿 deterministic integrity forensics 检查。",
    )
    parser.add_argument(
        "--review-reflections", type=int, default=1, help="审稿反思轮数"
    )
    parser.add_argument(
        "--review-ensemble", type=int, default=1, help="审稿 ensemble 数量"
    )
    parser.add_argument(
        "--review-fewshot", type=int, default=1, help="审稿 few-shot 示例数"
    )
    parser.add_argument(
        "--review-temperature", type=float, default=0.75, help="审稿温度"
    )
    parser.add_argument(
        "--review-strategy",
        type=str,
        choices=[
            "standard",
            "fast",
            "depth",
            "neurips",
            "iclr",
            "cvpr",
            "journal",
            "nature",
        ],
        default=None,
        help="审稿策略预设",
    )
    parser.add_argument(
        "--writing-profile",
        type=str,
        choices=list(writing_profiles),
        default=default_writing_profile,
        help="写作提示词 profile（受实战 prompt 模板库启发）",
    )
    parser.add_argument(
        "--writing-audit-rounds",
        type=int,
        default=0,
        help="在写作反思阶段额外执行结构化写作审计轮数",
    )
    parser.add_argument(
        "--strict-writing-guardrails",
        action="store_true",
        help="启用严格写作守护：最终稿若存在关键引用/章节缺口则判定失败",
    )
    parser.add_argument(
        "--guardrail-repair-rounds",
        type=int,
        default=1,
        help="严格写作守护失败前自动尝试的修复轮数",
    )
    parser.add_argument(
        "--workflow-mode",
        type=str,
        choices=list(workflow_modes),
        default="adaptive",
        help="研究编排模式：兼容经典模板流、agentic tree、program-driven、writing-studio、review-board。",
    )
    parser.add_argument(
        "--override-strict-fallbacks",
        action="store_true",
        help="禁用默认的严格兜底拦截（高质量/投稿/程序驱动/评审模式下默认禁止 fallback）。",
    )

    parser.add_argument("--model-ideation", type=str, default="glm-4-flash")
    parser.add_argument("--model-agg-plots", type=str, default="glm-4-flash")
    parser.add_argument("--model-writeup", type=str, default="glm-4-plus")
    parser.add_argument("--model-writeup-small", type=str, default="glm-4-air")
    parser.add_argument("--model-citation", type=str, default="glm-4-air")
    parser.add_argument("--model-review", type=str, default="glm-4-plus")

    parser.add_argument("--num-cite-rounds", type=int, default=15)
    parser.add_argument("--writeup-retries", type=int, default=3)
    parser.add_argument(
        "--bfts-config",
        type=str,
        default="bfts_config.yaml",
        help="BFTS实验配置文件路径 (控制搜索深度、seed、并行度、超时等)",
    )
    return parser
