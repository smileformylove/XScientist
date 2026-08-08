"""Command-line contract for running one XScientist research project."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence


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
示例用法:

1. 从自然语言问题开箱运行（自动预检、恢复、洞见和科研 DAG）:
   xscientist project my_project --question "为什么这个机制在分布外失效？" --autopilot discovery

2. 生成3个想法并并行写成论文:
   python -m xscientist project my_project --topic topic.md --num-ideas 3 --parallel

3. 并行处理已有想法的前2个:
   python -m xscientist project my_project --ideas ideas.json --idea-indices 0,1 --parallel

4. 自动改进2轮:
   python -m xscientist project my_project --topic topic.md --improvement-rounds 2
        """,
    )

    parser.add_argument(
        "project_dir",
        type=str,
        help="项目目录路径（相对路径会解析到 --output-root/projects 下）",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=default_output_root,
        help="当 project_dir 为相对路径时，作为统一输出根目录",
    )

    parser.add_argument("--topic", type=str, help="主题描述文件")
    parser.add_argument(
        "--question",
        type=str,
        help="直接输入自然语言科研问题；无需预先创建 topic.md（与 --topic/--ideas 互斥）",
    )
    parser.add_argument("--ideas", type=str, help="已有想法JSON文件")
    parser.add_argument(
        "--autopilot",
        nargs="?",
        const="balanced",
        choices=["balanced", "discovery", "publication"],
        default=None,
        help=(
            "开箱自动科研预设。balanced 控制成本，discovery 强化反证与探索，"
            "publication 强化多角色复核和投稿门禁。"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 04_logs/progress.json 和最近的 BFTS checkpoint 安全续跑",
    )
    parser.add_argument(
        "--model-ideation",
        type=str,
        default=_model_default("IDEATION", "glm-4-flash"),
    )
    parser.add_argument("--num-ideas", type=int, default=3, help="生成的想法数量")
    parser.add_argument("--num-reflections", type=int, default=5)

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用并行处理多个想法",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="并行worker数量",
    )
    parser.add_argument(
        "--idea-indices",
        type=str,
        help="要处理的想法索引 (逗号分隔)，如: 0,1,2",
    )
    parser.add_argument(
        "--rank-ideas", action="store_true", help="先对 idea 排序再选择"
    )
    parser.add_argument(
        "--top-k-ideas", type=int, default=None, help="只处理评分最高的前 K 个 idea"
    )
    parser.add_argument(
        "--idea-rank-model",
        type=str,
        default=None,
        help="用于 idea 排序的模型；可用逗号分隔多个独立评审模型",
    )
    parser.add_argument("--submission-mode", action="store_true")
    parser.add_argument("--fallback-ranked-ideas", action="store_true")
    parser.add_argument("--breakthrough-mode", action="store_true")
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
        help="禁用严格兜底拦截（默认投稿/高质量/程序驱动模式会在出现 fallback 时终止）。",
    )

    parser.add_argument(
        "--seed-from-ara",
        type=str,
        default=None,
        help=(
            "路径：一个 `run_ara_fork.py fork` 产生的目录，或一个 ARA 根目录（需配合 --seed-node-id）。"
            "首个 BFTS draft 会直接使用该目录中的 code，跳过 LLM。"
        ),
    )
    parser.add_argument(
        "--seed-node-id",
        type=str,
        default=None,
        help="当 --seed-from-ara 指向 ARA 根目录时，指定要作为种子的 node_id。",
    )

    parser.add_argument(
        "--improvement-rounds",
        type=int,
        default=1,
        help="每篇论文的反思改进轮数",
    )

    parser.add_argument("--skip-ideation", action="store_true")
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument(
        "--bfts-config",
        type=str,
        default="bfts_config.yaml",
        help="BFTS实验配置文件路径 (控制搜索深度、seed、并行度、超时等)",
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
            "neurips",
            "iclr",
            "cvpr",
            "journal",
            "nature",
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
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        default=None,
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
        help="启用最终稿 deterministic integrity forensics 检查。",
    )
    parser.add_argument(
        "--no-integrity-forensics",
        dest="integrity_forensics",
        action="store_false",
        help="禁用最终稿 deterministic integrity forensics 检查。",
    )
    parser.add_argument("--auto-adjust-paper-type", action="store_true")
    parser.add_argument(
        "--writing-profile",
        type=str,
        choices=list(writing_profiles),
        default=default_writing_profile,
        help="写作提示词 profile（影响写作约束与反思自检）",
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
        help="启用严格写作守护：最终稿若存在关键引用/章节缺口则判定失败",
    )
    parser.add_argument(
        "--guardrail-repair-rounds",
        type=int,
        default=1,
        help="严格写作守护失败前自动尝试的修复轮数",
    )
    parser.add_argument(
        "--disable-hostile-critic",
        action="store_true",
        help="基准 ablation: 关闭独立 hostile critic 通道。",
    )
    parser.add_argument(
        "--disable-owner-aware-repair",
        action="store_true",
        help="基准 ablation: 关闭 owner-aware reviewer repair routing。",
    )
    parser.add_argument(
        "--research-vcs",
        "--research-git",
        dest="research_git",
        choices=["off", "local"],
        default="local",
        help=(
            "启用 XScientist 原生科研版本库（Git 仅作为可替换存储后端）；"
            "默认本地启用、无需服务器且永不自动 push。"
        ),
    )
    parser.add_argument(
        "--checkpoint-policy",
        "--git-checkpoint-policy",
        dest="git_checkpoint_policy",
        choices=["manual", "stage", "milestone"],
        default="milestone",
        help=(
            "科研 checkpoint 策略。milestone 记录构思、实验、证据和论文等关键状态；"
            "stage 记录每个请求的阶段。"
        ),
    )
    parser.add_argument(
        "--research-vcs-strict",
        "--research-git-strict",
        dest="research_git_strict",
        action="store_true",
        help="科研版本库初始化或 checkpoint 失败时终止；默认警告并保留研究产物。",
    )
    parser.add_argument("--git-user-name", default=None)
    parser.add_argument("--git-user-email", default=None)
    return parser
