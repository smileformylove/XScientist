#!/usr/bin/env python3
"""
自适应学习引擎
根据历史经验自适应调整策略
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import re
import numpy as np

from ai_scientist.config.paths import OUTPUT_PATH
from ai_scientist.self_learning_knowledge_base import (
    SelfLearningKnowledgeBase,
    PatternAnalyzer,
)
from ai_scientist.llm import create_client, get_response_from_llm


class AdaptiveLearningEngine:
    """自适应学习引擎 - 根据经验调整策略"""

    def __init__(
        self,
        knowledge_base: SelfLearningKnowledgeBase = None,
        research_dir: str = str(OUTPUT_PATH),
    ):
        """
        初始化自适应学习引擎

        Args:
            knowledge_base: 知识库实例
            research_dir: 研究目录
        """
        if knowledge_base is None:
            self.kb = SelfLearningKnowledgeBase(research_dir)
        else:
            self.kb = knowledge_base

        self.analyzer = PatternAnalyzer(self.kb)
        self.adaptation_threshold = 0.7  # 适应阈值

    def _load_self_evolution_guidance(self) -> Dict:
        """加载跨项目 self-evolution playbook，并转成当前推荐可消费的摘要。"""
        playbook = {}
        if hasattr(self.kb, "get_self_evolution_playbook"):
            playbook = self.kb.get_self_evolution_playbook() or {}
        if not isinstance(playbook, dict) or not playbook:
            return {}
        defaults = [
            f"{item.get('stage')}: {item.get('action')}"
            for item in (playbook.get("top_agentic_defaults") or [])
            if isinstance(item, dict)
            and str(item.get("stage") or "").strip()
            and str(item.get("action") or "").strip()
        ]
        risks = [
            str(item.get("risk") or "").strip()
            for item in (playbook.get("top_recurring_risks") or [])
            if isinstance(item, dict) and str(item.get("risk") or "").strip()
        ]
        return {
            "project_count": int(playbook.get("project_count") or 0),
            "status_counts": playbook.get("status_counts") or {},
            "top_agentic_defaults": defaults[:5],
            "top_recurring_risks": risks[:5],
        }

    def recommend_strategy(
        self,
        idea: Dict,
        paper_type: str = "neurips",
        context: Dict = None,
    ) -> Dict:
        """
        推荐策略

        Args:
            idea: 研究想法
            paper_type: 论文类型
            context: 上下文信息

        Returns:
            推荐的策略
        """
        print("\n🧠 自适应学习引擎分析中...")

        # 查找相似论文
        similar_papers = self.kb.find_similar_papers(
            idea,
            paper_type,
            top_k=5,
        )

        # 分析成功因素
        success_factors = self.analyzer.analyze_success_factors()

        # 预测成功概率
        success_prob = self.analyzer.predict_success_probability({
            "idea": idea,
            "paper_type": paper_type,
        })

        # 获取有效策略
        effective_strategies = self.kb.get_effective_strategies(min_success_rate=0.7)

        # 获取分数阈值
        score_thresholds = self.kb.get_score_thresholds()
        self_evolution_guidance = self._load_self_evolution_guidance()

        # 构建推荐
        recommendation = {
            "paper_type": paper_type,
            "similar_papers": similar_papers,
            "success_probability": success_prob,
            "writing_strategy": self._recommend_writing_strategy(
                idea,
                paper_type,
                similar_papers,
                success_factors,
            ),
            "review_strategy": self._recommend_review_strategy(
                paper_type,
                similar_papers,
            ),
            "improvement_strategy": self._recommend_improvement_strategy(
                similar_papers,
                effective_strategies,
            ),
            "target_scores": self._recommend_target_scores(
                score_thresholds,
                paper_type,
            ),
            "common_pitfalls": self._identify_common_pitfalls(
                idea,
                paper_type,
            ),
            "confidence": self._calculate_confidence(similar_papers, success_factors),
            "self_evolution_guidance": self_evolution_guidance,
        }
        if self_evolution_guidance.get("top_agentic_defaults"):
            recommendation["improvement_strategy"]["agentic_defaults"] = list(
                self_evolution_guidance["top_agentic_defaults"]
            )
        if self_evolution_guidance.get("top_recurring_risks"):
            recommendation["common_pitfalls"] = list(
                dict.fromkeys(
                    list(recommendation["common_pitfalls"])
                    + list(self_evolution_guidance["top_recurring_risks"])
                )
            )

        # 保存推荐
        self._save_recommendation(recommendation, idea)

        return recommendation

    def _recommend_writing_strategy(
        self,
        idea: Dict,
        paper_type: str,
        similar_papers: List[Dict],
        success_factors: Dict,
    ) -> Dict:
        """推荐写作策略"""
        strategy = {
            "template": paper_type,
            "emphasis_sections": [],
            "writing_tips": [],
            "structure_suggestions": [],
        }

        # 基于相似论文的建议
        if similar_papers:
            # 分析成功论文的章节重点
            section_scores = defaultdict(list)

            for paper in similar_papers:
                for review in paper.get("reviews", []):
                    if "review" in review and "scores" in review["review"]:
                        for section, score in review["review"]["scores"].items():
                            if isinstance(score, (int, float)):
                                section_scores[section].append(score)

            # 找出得分最高的章节
            if section_scores:
                avg_section_scores = {
                    section: sum(scores) / len(scores)
                    for section, scores in section_scores.items()
                }

                top_sections = sorted(
                    avg_section_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:3]

                strategy["emphasis_sections"] = [s[0] for s in top_sections]
                strategy["writing_tips"].append(
                    f"重点优化以下章节: {', '.join(strategy['emphasis_sections'])}"
                )

        # 基于成功因素的建议
        if success_factors.get("common_traits"):
            traits = success_factors["common_traits"]

            if traits.get("top_paper_types"):
                strategy["writing_tips"].append(
                    f"参考最成功的论文类型"
                )

            if traits.get("average_scores"):
                avg_scores = traits["average_scores"]
                weak_sections = [
                    section for section, score in avg_scores.items()
                    if score < 4.0
                ]
                if weak_sections:
                    strategy["writing_tips"].append(
                        f"需要特别关注: {', '.join(weak_sections)}"
                    )

        return strategy

    def _recommend_review_strategy(
        self,
        paper_type: str,
        similar_papers: List[Dict],
    ) -> Dict:
        """推荐审查策略"""
        strategy = {
            "strategy_name": "standard",
            "depth": "medium",
            "focus_areas": [],
            "rounds": 2,
        }

        # 基于论文类型
        if paper_type == "neurips":
            strategy["strategy_name"] = "neurips"
            strategy["focus_areas"] = ["theory", "novelty", "clarity"]
        elif paper_type == "iclr":
            strategy["strategy_name"] = "iclr"
            strategy["focus_areas"] = ["representation", "analysis", "intuition"]
        elif paper_type == "cvpr":
            strategy["strategy_name"] = "cvpr"
            strategy["focus_areas"] = ["methodology", "experiments", "results"]
        elif paper_type == "journal":
            strategy["strategy_name"] = "journal"
            strategy["depth"] = "deep"
            strategy["rounds"] = 3

        # 基于相似论文调整
        if similar_papers:
            # 计算平均改进轮数
            avg_rounds = []
            for paper in similar_papers:
                rounds = len(paper.get("improvements", []))
                avg_rounds.append(rounds)

            if avg_rounds:
                strategy["rounds"] = max(2, int(np.mean(avg_rounds)))

        return strategy

    def _recommend_improvement_strategy(
        self,
        similar_papers: List[Dict],
        effective_strategies: List[Dict],
    ) -> Dict:
        """推荐改进策略"""
        strategy = {
            "preferred_strategies": [],
            "avoid_strategies": [],
            "max_rounds": 3,
            "improvement_threshold": 0.5,
        }

        # 使用有效策略
        if effective_strategies:
            top_strategies = effective_strategies[:5]
            strategy["preferred_strategies"] = [
                s["strategy"] for s in top_strategies
            ]

        # 基于相似论文
        if similar_papers:
            # 找出有效的改进
            successful_improvements = []

            for paper in similar_papers:
                if paper.get("outcome") in ["accepted", "minor_revision"]:
                    for imp in paper.get("improvements", []):
                        if imp.get("improvement_score", 0) > 0:
                            successful_improvements.append(imp)

            if successful_improvements:
                # 提取最有效的策略
                strategy_impact = defaultdict(list)

                for imp in successful_improvements:
                    strategy_name = imp.get("strategy", "unknown")
                    score = imp.get("improvement_score", 0)
                    strategy_impact[strategy_name].append(score)

                # 推荐高影响的策略
                avg_impact = {
                    s: sum(scores) / len(scores)
                    for s, scores in strategy_impact.items()
                }

                top_impact = sorted(avg_impact.items(), key=lambda x: x[1], reverse=True)[:3]

                strategy["preferred_strategies"] = [s[0] for s in top_impact]

        return strategy

    def _recommend_target_scores(
        self,
        score_thresholds: Dict,
        paper_type: str,
    ) -> Dict:
        """推荐目标分数"""
        targets = {}

        # 使用学习到的阈值
        if score_thresholds:
            for dimension, threshold in score_thresholds.items():
                targets[dimension] = max(threshold + 0.5, 4.0)  # 目标高于阈值
        else:
            # 默认目标
            targets = {
                "structure": 4.0,
                "content": 4.0,
                "innovation": 4.0,
                "rigor": 4.0,
                "clarity": 4.0,
                "professionalism": 4.0,
            }

        return targets

    def _identify_common_pitfalls(
        self,
        idea: Dict,
        paper_type: str,
    ) -> List[str]:
        """识别常见陷阱"""
        pitfalls = []

        # 获取常见问题
        common_issues = self.kb.get_common_issues()

        if common_issues:
            pitfalls.extend(list(common_issues.keys())[:5])

        # 基于论文类型的常见问题
        if paper_type == "neurips":
            pitfalls.extend([
                "理论分析不够深入",
                "缺少更广泛影响声明",
                "实验验证不充分",
            ])
        elif paper_type == "iclr":
            pitfalls.extend([
                "表示学习的理论分析不足",
                "收敛性分析缺失",
                "可视化表示不清楚",
            ])
        elif paper_type == "cvpr":
            pitfalls.extend([
                "架构图不够清晰",
                "定性结果不足",
                "效率分析缺失",
            ])

        return pitfalls[:5]

    def _calculate_confidence(
        self,
        similar_papers: List[Dict],
        success_factors: Dict,
    ) -> float:
        """计算推荐置信度"""
        confidence = 0.5

        # 基于相似论文数量
        if similar_papers:
            confidence += min(len(similar_papers) * 0.1, 0.3)

        # 基于成功因素完整性
        if success_factors.get("common_traits"):
            confidence += 0.1

        if success_factors.get("score_patterns"):
            confidence += 0.1

        return min(confidence, 1.0)

    def _save_recommendation(self, recommendation: Dict, idea: Dict):
        """保存推荐记录"""
        recommendation_dir = self.kb.knowledge_dir
        recommendation_dir.mkdir(parents=True, exist_ok=True)

        idea_name = str(
            idea.get("Name")
            or idea.get("Title")
            or idea.get("idea_name")
            or "untitled_idea"
        ).strip()
        idea_slug = re.sub(r"[^a-zA-Z0-9]+", "_", idea_name).strip("_") or "untitled_idea"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        payload = {
            "timestamp": datetime.now().isoformat(),
            "idea_name": idea_name,
            "idea": idea,
            "recommendation": recommendation,
        }

        latest_file = recommendation_dir / "latest_recommendation.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        snapshot_file = recommendation_dir / f"recommendation_{idea_slug}_{timestamp}.json"
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        history_file = recommendation_dir / "recommendation_history.jsonl"
        with open(history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def adapt_writing_prompt(
        self,
        base_prompt: str,
        recommendation: Dict,
        section: str = None,
    ) -> str:
        """
        根据推荐调整写作提示

        Args:
            base_prompt: 基础提示
            recommendation: 推荐策略
            section: 章节

        Returns:
            调整后的提示
        """
        adapted_prompt = base_prompt

        # 添加写作策略建议
        writing_strategy = recommendation.get("writing_strategy", {})

        if writing_strategy.get("emphasis_sections"):
            adapted_prompt += f"\n\n**重点章节**: {', '.join(writing_strategy['emphasis_sections'])}"

        if writing_strategy.get("writing_tips"):
            adapted_prompt += "\n\n**写作建议**:\n"
            for tip in writing_strategy["writing_tips"]:
                adapted_prompt += f"- {tip}\n"

        # 添加常见陷阱警告
        pitfalls = recommendation.get("common_pitfalls", [])
        if pitfalls:
            adapted_prompt += "\n\n**需要避免的常见问题**:\n"
            for pitfall in pitfalls:
                adapted_prompt += f"- {pitfall}\n"

        # 添加目标分数
        target_scores = recommendation.get("target_scores", {})
        if target_scores and section:
            if section in target_scores:
                adapted_prompt += f"\n\n**目标分数**: 该章节应达到 {target_scores[section]} 分"

        return adapted_prompt

    def learn_from_generation(
        self,
        idea: Dict,
        paper_data: Dict,
        outcome: str,
        reviews: List[Dict] = None,
        improvements: List[Dict] = None,
        final_scores: Dict = None,
    ):
        """
        从生成过程中学习

        Args:
            idea: 研究想法
            paper_data: 论文数据
            outcome: 结果
            reviews: 审查
            improvements: 改进
            final_scores: 最终分数
        """
        print("\n📚 自适应学习引擎正在学习...")

        # 存储经验
        self.kb.store_paper_experience(
            paper_data=paper_data,
            outcome=outcome,
            final_scores=final_scores,
            reviews=reviews,
            improvements=improvements,
        )

        # 分析新的模式
        success_factors = self.analyzer.analyze_success_factors()

        print(f"✅ 学习完成")
        print(f"   成功模式数: {len(self.kb.success_patterns)}")
        print(f"   失败模式数: {len(self.kb.failure_patterns)}")

        # 返回学习洞察
        return {
            "success_factors": success_factors,
            "common_issues": self.kb.get_common_issues(),
            "effective_strategies": self.kb.get_effective_strategies(),
        }

    def generate_adaptive_improvement_plan(
        self,
        current_review: Dict,
        recommendation: Dict,
    ) -> Dict:
        """
        生成自适应改进计划

        Args:
            current_review: 当前审查
            recommendation: 推荐策略

        Returns:
            改进计划
        """
        plan = {
            "priority_improvements": [],
            "suggested_strategies": [],
            "expected_improvement": 0.0,
            "rounds_needed": 1,
        }

        # 提取主要问题
        if "review" in current_review:
            main_issues = current_review["review"].get("main_issues", [])
            low_scores = []

            for dimension, score in current_review["review"].get("scores", {}).items():
                if isinstance(score, (int, float)) and score < 4.0:
                    low_scores.append((dimension, score))

            # 优先处理低分维度
            for dimension, score in sorted(low_scores, key=lambda x: x[1]):
                plan["priority_improvements"].append({
                    "dimension": dimension,
                    "current_score": score,
                    "target_score": recommendation.get("target_scores", {}).get(dimension, 4.0),
                })

            # 使用推荐的策略
            preferred_strategies = recommendation.get("improvement_strategy", {}).get("preferred_strategies", [])

            if preferred_strategies:
                plan["suggested_strategies"] = preferred_strategies[:3]

            # 估计需要的轮数
            improvement_strategy = recommendation.get("improvement_strategy", {})
            plan["rounds_needed"] = improvement_strategy.get("max_rounds", 2)

            # 估计预期改进
            if plan["priority_improvements"]:
                avg_gap = sum(
                    item["target_score"] - item["current_score"]
                    for item in plan["priority_improvements"]
                ) / len(plan["priority_improvements"])

                plan["expected_improvement"] = min(avg_gap * 0.5, 2.0)

        return plan


class AdaptiveWriter:
    """自适应写作器 - 根据学习到的模式写作"""

    def __init__(
        self,
        learning_engine: AdaptiveLearningEngine,
        model: str = "claude-3-5-sonnet",
    ):
        """
        初始化自适应写作器

        Args:
            learning_engine: 学习引擎
            model: 使用的模型
        """
        self.engine = learning_engine
        self.model = model
        self.client, self.client_model = create_client(model)

    def write_with_learning(
        self,
        section: str,
        idea: Dict,
        paper_type: str,
        context: Dict,
    ) -> str:
        """
        使用学习到的模式写作

        Args:
            section: 章节
            idea: 想法
            paper_type: 论文类型
            context: 上下文

        Returns:
            章节内容
        """
        # 获取推荐策略
        recommendation = self.engine.recommend_strategy(idea, paper_type, context)

        # 构建基础提示
        base_prompt = self._build_base_prompt(section, idea, context)

        # 调整提示
        adapted_prompt = self.engine.adapt_writing_prompt(
            base_prompt,
            recommendation,
            section,
        )

        # 添加相似论文示例
        similar_papers = recommendation.get("similar_papers", [])
        if similar_papers:
            adapted_prompt += "\n\n**参考成功论文**:\n"
            for paper in similar_papers[:2]:
                adapted_prompt += f"- {paper.get('idea_name', 'Unknown')}: "
                adapted_prompt += f"相似度 {paper.get('similarity', 0):.2f}\n"

        # 使用LLM生成
        response, _ = get_response_from_llm(
            prompt=adapted_prompt,
            client=self.client,
            model=self.client_model,
            system_message=f"""
你是资深的学术写作专家，专门撰写{paper_type}论文的{section}章节。

**自适应学习指导**:
根据历史成功论文的模式，你将应用最有效的写作策略。

**写作要求**:
1. 遵循顶级会议的写作标准
2. 应用从成功论文中学到的模式
3. 避免常见的错误
4. 确保逻辑清晰、论证严密
""",
            temperature=0.7,
        )

        return response

    def _build_base_prompt(self, section: str, idea: Dict, context: Dict) -> str:
        """构建基础提示"""
        prompt = f"""
请为以下研究想法撰写 **{section}** 章节。

**研究想法**:
标题: {idea.get('Title', '')}
摘要: {idea.get('Abstract', '')}
方法: {idea.get('Method', '')}

**上下文**:
{json.dumps(context, indent=2, ensure_ascii=False)[:1000]}
"""
        return prompt
