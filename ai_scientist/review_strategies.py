#!/usr/bin/env python3
"""
AI Scientist 多样化审查策略系统
支持不同会议/期刊的审查标准和深度
"""
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ReviewStrategy(str, Enum):
    """审查策略类型"""
    STANDARD = "standard"           # 标准综合审查
    FAST = "fast"                   # 快速检查
    DEPTH = "depth"                 # 深度审查
    NEURIPS = "neurips"             # NeurIPS格式
    ICLR = "iclr"                   # ICLR格式
    CVPR = "cvpr"                   # CVPR格式
    JOURNAL = "journal"             # 期刊审查
    NATURE = "nature"               # Nature风格审查


class ReviewFocus(str, Enum):
    """审查焦点"""
    QUALITY = "quality"             # 质量
    CLARITY = "clarity"             # 清晰度
    ORIGINALITY = "originality"     # 原创性
    SIGNIFICANCE = "significance"   # 重要性
    SOUNDNESS = "soundness"         # 严谨性
    ETHICS = "ethics"               # 伦理
    PRESENTATION = "presentation"   # 呈现
    REPRODUCIBILITY = "reproducibility"  # 可复现性


# 审查策略配置
REVIEW_STRATEGIES = {
    ReviewStrategy.STANDARD: {
        "name": "标准综合审查",
        "description": "平衡的审查，涵盖所有主要方面",
        "focus_areas": [
            ReviewFocus.QUALITY,
            ReviewFocus.CLARITY,
            ReviewFocus.SIGNIFICANCE,
            ReviewFocus.SOUNDNESS,
        ],
        "review_depth": "medium",
        "expected_time": "10-15分钟",
        "scoring_categories": [
            "Originality", "Quality", "Clarity",
            "Significance", "Soundness", "Overall"
        ],
    },

    ReviewStrategy.FAST: {
        "name": "快速检查",
        "description": "聚焦主要问题和明显缺陷",
        "focus_areas": [
            ReviewFocus.SOUNDNESS,
            ReviewFocus.CLARITY,
            ReviewFocus.QUALITY,
        ],
        "review_depth": "shallow",
        "expected_time": "5-8分钟",
        "scoring_categories": [
            "Soundness", "Clarity", "Overall"
        ],
        "major_issues_only": True,
    },

    ReviewStrategy.DEPTH: {
        "name": "深度审查",
        "description": "全面深入的审查，包含所有细节",
        "focus_areas": [
            ReviewFocus.QUALITY,
            ReviewFocus.CLARITY,
            ReviewFocus.ORIGINALITY,
            ReviewFocus.SIGNIFICANCE,
            ReviewFocus.SOUNDNESS,
            ReviewFocus.ETHICS,
            ReviewFocus.REPRODUCIBILITY,
            ReviewFocus.PRESENTATION,
        ],
        "review_depth": "deep",
        "expected_time": "20-30分钟",
        "scoring_categories": [
            "Originality", "Quality", "Clarity",
            "Significance", "Soundness", "Presentation",
            "Contribution", "Ethical Concerns", "Overall"
        ],
        "detailed_feedback": True,
    },

    ReviewStrategy.NEURIPS: {
        "name": "NeurIPS格式",
        "description": "按照NeurIPS会议标准审查",
        "focus_areas": [
            ReviewFocus.QUALITY,
            ReviewFocus.CLARITY,
            ReviewFocus.ORIGINALITY,
            ReviewFocus.SIGNIFICANCE,
        ],
        "review_depth": "medium",
        "expected_time": "15分钟",
        "scoring_categories": [
            "Originality", "Quality", "Clarity",
            "Significance", "Soundness", "Overall",
            "Reproducibility"
        ],
        "venue_specific": True,
        "format_template": "neurips",
    },

    ReviewStrategy.ICLR: {
        "name": "ICLR格式",
        "description": "按照ICLR会议标准审查",
        "focus_areas": [
            ReviewFocus.QUALITY,
            ReviewFocus.CLARITY,
            ReviewFocus.ORIGINALITY,
            ReviewFocus.SIGNIFICANCE,
        ],
        "review_depth": "medium",
        "expected_time": "15分钟",
        "scoring_categories": [
            "Originality", "Quality", "Clarity",
            "Significance", "Soundness", "Overall"
        ],
        "venue_specific": True,
        "format_template": "iclr",
    },

    ReviewStrategy.CVPR: {
        "name": "CVPR格式",
        "description": "按照CVPR会议标准审查",
        "focus_areas": [
            ReviewFocus.QUALITY,
            ReviewFocus.CLARITY,
            ReviewFocus.ORIGINALITY,
            ReviewFocus.SIGNIFICANCE,
            ReviewFocus.REPRODUCIBILITY,
        ],
        "review_depth": "medium",
        "expected_time": "15分钟",
        "scoring_categories": [
            "Originality", "Quality", "Clarity",
            "Significance", "Soundness", "Overall",
            "Reproducibility"
        ],
        "venue_specific": True,
        "format_template": "cvpr",
    },

    ReviewStrategy.JOURNAL: {
        "name": "期刊审查",
        "description": "按照期刊标准进行严格审查",
        "focus_areas": [
            ReviewFocus.QUALITY,
            ReviewFocus.CLARITY,
            ReviewFocus.ORIGINALITY,
            ReviewFocus.SIGNIFICANCE,
            ReviewFocus.SOUNDNESS,
            ReviewFocus.REPRODUCIBILITY,
            ReviewFocus.PRESENTATION,
        ],
        "review_depth": "deep",
        "expected_time": "25-35分钟",
        "scoring_categories": [
            "Originality", "Quality", "Clarity",
            "Significance", "Soundness", "Presentation",
            "Contribution", "Reproducibility", "Overall"
        ],
        "detailed_feedback": True,
        "strict_standards": True,
    },

    ReviewStrategy.NATURE: {
        "name": "Nature风格审查",
        "description": "强调重大问题、强证据链、广泛影响与克制 claim 的高标准审查",
        "focus_areas": [
            ReviewFocus.QUALITY,
            ReviewFocus.CLARITY,
            ReviewFocus.ORIGINALITY,
            ReviewFocus.SIGNIFICANCE,
            ReviewFocus.SOUNDNESS,
            ReviewFocus.REPRODUCIBILITY,
            ReviewFocus.PRESENTATION,
        ],
        "review_depth": "deep",
        "expected_time": "30-40分钟",
        "scoring_categories": [
            "Originality", "Quality", "Clarity",
            "Significance", "Soundness", "Presentation",
            "Contribution", "Reproducibility", "Overall"
        ],
        "detailed_feedback": True,
        "strict_standards": True,
        "venue_specific": True,
        "format_template": "nature",
    },
}


class ReviewStrategyManager:
    """审查策略管理器"""

    @staticmethod
    def get_strategy(strategy: ReviewStrategy) -> Dict:
        """
        获取指定策略的配置

        Args:
            strategy: 审查策略

        Returns:
            策略配置
        """
        return REVIEW_STRATEGIES.get(strategy, REVIEW_STRATEGIES[ReviewStrategy.STANDARD])

    @staticmethod
    def list_strategies() -> List[Dict]:
        """列出所有可用策略"""
        return [
            {
                "key": strategy.value,
                "name": config["name"],
                "description": config["description"],
                "depth": config["review_depth"],
                "time": config["expected_time"],
            }
            for strategy, config in REVIEW_STRATEGIES.items()
        ]

    @staticmethod
    def recommend_strategy(
        paper_type: str = "icbinb",
        time_constraint: Optional[str] = None,
        quality_requirement: str = "standard",
    ) -> ReviewStrategy:
        """
        推荐合适的审查策略

        Args:
            paper_type: 论文类型 (icbinb, normal, journal, extended)
            time_constraint: 时间约束 (fast, normal, unlimited)
            quality_requirement: 质量要求 (basic, standard, high)

        Returns:
            推荐的审查策略
        """
        # 根据论文类型推荐
        if paper_type == "extended":
            return ReviewStrategy.FAST
        elif paper_type == "journal":
            return ReviewStrategy.JOURNAL
        elif paper_type in ["normal", "icbinb"]:
            # 根据时间和质量要求进一步细化
            if time_constraint == "fast":
                return ReviewStrategy.FAST
            elif quality_requirement == "high":
                return ReviewStrategy.DEPTH
            else:
                return ReviewStrategy.STANDARD

        return ReviewStrategy.STANDARD

    @staticmethod
    def customize_strategy(
        base_strategy: ReviewStrategy,
        focus_areas: List[ReviewFocus],
        review_depth: str = "medium",
    ) -> Dict:
        """
        自定义审查策略

        Args:
            base_strategy: 基础策略
            focus_areas: 自定义焦点区域
            review_depth: 审查深度

        Returns:
            自定义策略配置
        """
        base_config = REVIEW_STRATEGIES[base_strategy].copy()

        custom_strategy = {
            **base_config,
            "focus_areas": focus_areas,
            "review_depth": review_depth,
            "customized": True,
        }

        # 根据深度调整评分类别
        if review_depth == "shallow":
            custom_strategy["scoring_categories"] = [
                "Quality", "Clarity", "Overall"
            ]
        elif review_depth == "deep":
            custom_strategy["scoring_categories"] = [
                "Originality", "Quality", "Clarity",
                "Significance", "Soundness", "Presentation",
                "Contribution", "Reproducibility", "Overall"
            ]

        return custom_strategy


def generate_review_instruction(
    strategy: ReviewStrategy,
    custom_instructions: Optional[str] = None,
) -> str:
    """
    生成符合策略的审查指令前缀。

    Args:
        strategy: 审查策略
        custom_instructions: 自定义指令

    Returns:
        审查指令
    """
    config = REVIEW_STRATEGIES[strategy]

    # 构建焦点区域描述
    focus_descriptions = {
        ReviewFocus.QUALITY: "研究方法和实验设计的质量",
        ReviewFocus.CLARITY: "论文结构和表达的清晰度",
        ReviewFocus.ORIGINALITY: "研究的创新性和新颖性",
        ReviewFocus.SIGNIFICANCE: "研究的重要性和影响力",
        ReviewFocus.SOUNDNESS: "技术严谨性和论证的逻辑性",
        ReviewFocus.ETHICS: "伦理考量和潜在风险",
        ReviewFocus.REPRODUCIBILITY: "结果的可复现性",
        ReviewFocus.PRESENTATION: "呈现方式和图表质量",
    }

    # 构建焦点说明
    focus_text = "\n".join([
        f"- {focus.value}: {focus_descriptions[focus]}"
        for focus in config["focus_areas"]
    ])

    # 构建评分说明
    scoring_text = ", ".join(config["scoring_categories"])

    # 深度说明
    depth_instructions = {
        "shallow": "请聚焦于主要问题和明显缺陷，快速评估。",
        "medium": "请提供平衡的审查，涵盖主要优缺点。",
        "deep": "请提供全面深入的审查，详细分析各个方面。",
    }

    return f"""
请按照{config["name"]}标准审查以下论文。

**审查策略**: {config["description"]}
**审查深度**: {config["review_depth"]}
**预计时间**: {config["expected_time"]}

**审查焦点**:
{focus_text}

**评分类别**: {scoring_text}

**审查要求**:
{depth_instructions[config["review_depth"]]}

请提供结构化的审查反馈，包括:
1. Summary (摘要)
2. Strengths (优点)
3. Weaknesses (缺点)
4. Questions (问题)
5. Limitations (局限性)
6. Ethical Concerns (伦理关切，如适用)

**评分**: 为每个类别打分(1-4分，Overall为1-10分)

{custom_instructions if custom_instructions else ""}

请以JSON格式返回审查结果。
"""


def generate_review_prompt(
    strategy: ReviewStrategy,
    paper_content: str,
    custom_instructions: Optional[str] = None,
) -> str:
    """生成包含论文内容的完整审查提示词。"""
    instruction = generate_review_instruction(strategy, custom_instructions)
    return f"""
{instruction}

**论文内容**:
{paper_content[:10000]}
"""


class SmartIterationController:
    """智能迭代控制器"""

    def __init__(
        self,
        min_rounds: int = 1,
        max_rounds: int = 5,
        improvement_threshold: float = 0.5,
        convergence_rounds: int = 2,
    ):
        """
        初始化智能迭代控制器

        Args:
            min_rounds: 最小迭代轮数
            max_rounds: 最大迭代轮数
            improvement_threshold: 改进阈值（低于此值认为收敛）
            convergence_rounds: 连续几轮改进低于阈值时停止
        """
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds
        self.improvement_threshold = improvement_threshold
        self.convergence_rounds = convergence_rounds

        # 迭代状态
        self.round_count = 0
        self.improvement_history = []
        self.convergence_count = 0

    def should_continue(
        self,
        current_improvement: float,
        review_scores: Dict,
    ) -> Tuple[bool, str, Dict]:
        """
        判断是否继续迭代

        Args:
            current_improvement: 当前轮改进值
            review_scores: 当前审查评分

        Returns:
            (是否继续, 原因, 决策详情)
        """
        self.round_count += 1
        self.improvement_history.append(current_improvement)

        decision_details = {
            "round": self.round_count,
            "current_improvement": current_improvement,
            "improvement_history": self.improvement_history.copy(),
        }

        # 检查是否达到最大轮数
        if self.round_count >= self.max_rounds:
            return False, "达到最大迭代轮数", decision_details

        # 检查最小轮数
        if self.round_count < self.min_rounds:
            return True, "未达到最小轮数，继续迭代", decision_details

        # 检查改进效果
        if current_improvement >= self.improvement_threshold:
            # 有显著改进，重置收敛计数
            self.convergence_count = 0
            return True, f"改进显著 ({current_improvement:.2f} >= {self.improvement_threshold})，继续迭代", decision_details

        # 改进较小，增加收敛计数
        self.convergence_count += 1

        if self.convergence_count >= self.convergence_rounds:
            return False, f"连续{self.convergence_rounds}轮改进低于阈值，已收敛", decision_details

        # 检查总体趋势
        if len(self.improvement_history) >= 3:
            recent_avg = sum(self.improvement_history[-3:]) / 3
            if recent_avg < self.improvement_threshold / 2:
                return False, "近期改进趋势较弱，停止迭代", decision_details

        # 检查评分是否已经很高
        overall_score = review_scores.get("Overall", 0)
        if overall_score >= 8.0:
            return False, f"总体评分已较高 ({overall_score:.1f}/10)，可以停止", decision_details

        return True, f"改进较小 ({current_improvement:.2f})，但可能继续优化", decision_details

    def get_summary(self) -> Dict:
        """获取迭代总结"""
        return {
            "total_rounds": self.round_count,
            "improvement_history": self.improvement_history,
            "average_improvement": sum(self.improvement_history) / len(self.improvement_history) if self.improvement_history else 0,
            "total_improvement": sum(self.improvement_history),
            "final_convergence_count": self.convergence_count,
        }


# 预设策略配置示例
PRESET_STRATEGIES = {
    "quick_paper": {
        "strategy": ReviewStrategy.FAST,
        "max_rounds": 2,
        "improvement_threshold": 0.3,
        "description": "快速论文优化，适合时间紧迫的情况",
    },
    "standard_paper": {
        "strategy": ReviewStrategy.STANDARD,
        "max_rounds": 3,
        "improvement_threshold": 0.5,
        "description": "标准论文优化，平衡质量和时间",
    },
    "high_quality": {
        "strategy": ReviewStrategy.DEPTH,
        "max_rounds": 5,
        "improvement_threshold": 0.3,
        "description": "高质量论文优化，追求最佳效果",
    },
    "journal_submission": {
        "strategy": ReviewStrategy.JOURNAL,
        "max_rounds": 4,
        "improvement_threshold": 0.4,
        "description": "期刊投稿优化，严格标准",
    },
}
