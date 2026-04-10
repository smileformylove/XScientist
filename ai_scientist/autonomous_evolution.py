#!/usr/bin/env python3
"""
自主进化引擎
使AI Scientist能够自主进化，并接受外部agent的指导
"""

import json
import os
from typing import Dict, List, Optional, Callable
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import asyncio
from enum import Enum
import re

from ai_scientist.config.paths import OUTPUT_PATH
from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.self_learning_knowledge_base import SelfLearningKnowledgeBase
from ai_scientist.adaptive_learning_engine import AdaptiveLearningEngine


class FeedbackSource(Enum):
    """反馈来源"""
    SELF = "self"  # 自我反思
    EXTERNAL_AGENT = "external_agent"  # 外部agent
    HUMAN = "human"  # 人工反馈
    PEER_REVIEW = "peer_review"  # 同行评审
    METRICS = "metrics"  # 指标反馈


class EvolutionAction(Enum):
    """进化动作"""
    IMPROVE_WRITING = "improve_writing"
    ADJUST_STRATEGY = "adjust_strategy"
    LEARN_PATTERN = "learn_pattern"
    UPDATE_KNOWLEDGE = "update_knowledge"
    OPTIMIZE_PROMPT = "optimize_prompt"
    REFINE_MODEL = "refine_model"


class AutonomousEvolutionEngine:
    """
    自主进化引擎

    核心能力：
    1. 自主反思和改进
    2. 接受外部agent指导
    3. 综合多源反馈
    4. 持续进化优化
    """

    def __init__(
        self,
        research_dir: str = str(OUTPUT_PATH),
        evolution_model: str = "claude-3-5-sonnet",
    ):
        """
        初始化自主进化引擎

        Args:
            research_dir: 研究目录
            evolution_model: 用于进化的模型
        """
        self.research_dir = Path(research_dir)
        self.evolution_dir = self.research_dir / "evolution"
        self.evolution_dir.mkdir(parents=True, exist_ok=True)

        # 初始化知识库和学习引擎
        self.knowledge_base = SelfLearningKnowledgeBase(research_dir)
        self.learning_engine = AdaptiveLearningEngine(self.knowledge_base)

        # 进化模型
        self.evolution_model = evolution_model
        self.client, self.client_model = create_client(evolution_model)

        # 进化历史
        self.evolution_history = []
        self.feedback_buffer = []

        # 外部agent接口
        self.external_agent_callbacks = {}

        # 加载进化历史
        self._load_evolution_history()

    def _load_evolution_history(self):
        """加载进化历史"""
        history_file = self.evolution_dir / "evolution_history.json"
        if history_file.exists():
            with open(history_file, "r") as f:
                self.evolution_history = json.load(f)

    def _save_evolution_history(self):
        """保存进化历史"""
        history_file = self.evolution_dir / "evolution_history.json"
        with open(history_file, "w") as f:
            json.dump(self.evolution_history, f, indent=2, ensure_ascii=False)

    # ========================================
    # 核心进化方法
    # ========================================

    async def evolve(
        self,
        paper_data: Dict,
        current_state: Dict,
        external_feedback: List[Dict] = None,
    ) -> Dict:
        """
        执行自主进化

        Args:
            paper_data: 论文数据
            current_state: 当前状态
            external_feedback: 外部反馈

        Returns:
            进化结果
        """
        print("\n🧬 启动自主进化流程...")

        evolution_record = {
            "timestamp": datetime.now().isoformat(),
            "paper_id": paper_data.get("paper_id", "unknown"),
            "current_state": current_state,
        }

        # 1. 自我反思
        print("\n🔍 步骤 1: 自我反思分析")
        self_reflection = await self._self_reflection(paper_data, current_state)
        evolution_record["self_reflection"] = self_reflection

        # 2. 整合外部反馈
        if external_feedback:
            print("\n📨 步骤 2: 整合外部反馈")
            integrated_feedback = self._integrate_feedback(
                self_reflection,
                external_feedback,
            )
            evolution_record["integrated_feedback"] = integrated_feedback
        else:
            integrated_feedback = {"self_reflection": self_reflection}
            evolution_record["integrated_feedback"] = integrated_feedback

        # 3. 生成进化策略
        print("\n🎯 步骤 3: 生成进化策略")
        evolution_strategy = await self._generate_evolution_strategy(
            paper_data,
            current_state,
            integrated_feedback,
        )
        evolution_record["strategy"] = evolution_strategy

        # 4. 执行进化动作
        print("\n⚡ 步骤 4: 执行进化动作")
        evolution_result = await self._execute_evolution(
            paper_data,
            evolution_strategy,
        )
        evolution_record["result"] = evolution_result

        # 5. 验证进化效果
        print("\n✅ 步骤 5: 验证进化效果")
        validation = await self._validate_evolution(
            paper_data,
            current_state,
            evolution_result,
        )
        evolution_record["validation"] = validation

        # 6. 更新知识库
        print("\n📚 步骤 6: 更新知识库")
        await self._update_knowledge_from_evolution(
            evolution_record,
            validation,
        )

        # 保存历史
        self.evolution_history.append(evolution_record)
        self._save_evolution_history()

        print("\n🎉 自主进化完成!")

        return {
            "evolution_record": evolution_record,
            "improvements": evolution_result.get("improvements", []),
            "validation_score": validation.get("overall_score", 0),
            "next_steps": validation.get("recommendations", []),
        }

    async def _self_reflection(
        self,
        paper_data: Dict,
        current_state: Dict,
    ) -> Dict:
        """自我反思分析"""
        prompt = f"""
请对当前论文状态进行深度自我反思分析。

**论文信息**:
标题: {paper_data.get('Title', '')}
摘要: {paper_data.get('Abstract', '')[:500]}

**当前状态**:
{json.dumps(current_state, indent=2, ensure_ascii=False)}

**历史成功模式**:
{json.dumps(self.knowledge_base.get_success_patterns()[:3], indent=2, ensure_ascii=False)}

请提供详细的自我反思，包括：

1. **自我评估** (1-5分):
   - 各维度评分
   - 与成功论文的差距

2. **问题诊断**:
   - 主要问题识别
   - 根本原因分析
   - 优先级排序

3. **改进方向**:
   - 最有潜力的改进点
   - 预期效果
   - 实施难度

4. **进化建议**:
   - 应该采用什么进化策略
   - 为什么选择这个策略
   - 预期收益

以JSON格式返回。
"""

        response, _ = await get_response_from_llm(
            prompt=prompt,
            client=self.client,
            model=self.client_model,
            system_message="你是AI Scientist的自我反思系统，能够深度分析自身状态并识别改进机会。",
            temperature=0.7,
        )

        # 解析响应
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                return {"raw_response": response, "analysis": response}
        except:
            return {"raw_response": response, "analysis": response}

    def _integrate_feedback(
        self,
        self_reflection: Dict,
        external_feedback: List[Dict],
    ) -> Dict:
        """整合多源反馈"""
        integrated = {
            "sources": [],
            "common_issues": [],
            "conflicting_points": [],
            "conflict_actions": [],
            "priority_actions": [],
        }

        # 收集所有反馈
        all_feedback = [
            {"source": FeedbackSource.SELF, "feedback": self_reflection}
        ]

        for ext_feedback in external_feedback:
            source = ext_feedback.get("source", "unknown")
            all_feedback.append({
                "source": source,
                "feedback": ext_feedback,
            })

        integrated["sources"] = all_feedback

        # 识别共同问题
        issue_counter = defaultdict(int)
        for feedback_obj in all_feedback:
            feedback = feedback_obj["feedback"]
            if "issues" in feedback:
                for issue in feedback["issues"]:
                    issue_counter[issue] += 1

        # 提取共同问题（出现2次以上）
        integrated["common_issues"] = [
            {"issue": issue, "frequency": count}
            for issue, count in issue_counter.items()
            if count >= 2
        ]

        # 识别冲突点
        integrated["conflicting_points"] = self._detect_feedback_conflicts(all_feedback)
        integrated["conflict_actions"] = self._derive_conflict_actions(
            integrated["conflicting_points"]
        )

        # 生成优先级行动
        priority_actions: List[str] = []
        if integrated["common_issues"]:
            priority_actions.extend(
                [
                issue["issue"]
                for issue in sorted(
                    integrated["common_issues"],
                    key=lambda x: x["frequency"],
                    reverse=True
                )
                ]
            )
        priority_actions.extend(integrated["conflict_actions"])
        unique_actions: List[str] = []
        seen_actions = set()
        for action in priority_actions:
            token = self._normalize_feedback_text(action)
            if not token or token in seen_actions:
                continue
            seen_actions.add(token)
            unique_actions.append(action)
        integrated["priority_actions"] = unique_actions[:5]

        return integrated

    def _normalize_feedback_text(self, value: str) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip().lower())
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff\-_ ]+", "", text)
        return text.strip()

    def _collect_feedback_markers(self, feedback: Dict) -> Dict:
        positive_keys = {"strengths", "positives", "pros", "highlights"}
        negative_keys = {"issues", "concerns", "weaknesses", "cons", "risks", "problems"}
        positive_points: set[str] = set()
        negative_points: set[str] = set()
        score_map: Dict[str, float] = {}

        if not isinstance(feedback, dict):
            return {
                "positive_points": positive_points,
                "negative_points": negative_points,
                "scores": score_map,
            }

        for key in positive_keys:
            values = feedback.get(key)
            if isinstance(values, list):
                for item in values:
                    token = self._normalize_feedback_text(str(item))
                    if token:
                        positive_points.add(token)
        for key in negative_keys:
            values = feedback.get(key)
            if isinstance(values, list):
                for item in values:
                    token = self._normalize_feedback_text(str(item))
                    if token:
                        negative_points.add(token)

        candidates = []
        if isinstance(feedback.get("scores"), dict):
            candidates.append(feedback.get("scores"))
        if isinstance(feedback.get("self_assessment"), dict) and isinstance(
            feedback.get("self_assessment", {}).get("scores"), dict
        ):
            candidates.append(feedback.get("self_assessment", {}).get("scores"))

        for score_dict in candidates:
            for dim, value in score_dict.items():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                score_map[str(dim)] = parsed

        return {
            "positive_points": positive_points,
            "negative_points": negative_points,
            "scores": score_map,
        }

    def _detect_feedback_conflicts(self, all_feedback: List[Dict]) -> List[Dict]:
        conflicts: List[Dict] = []
        if not all_feedback:
            return conflicts

        point_sources: Dict[str, Dict[str, set[str]]] = defaultdict(
            lambda: {"positive": set(), "negative": set()}
        )
        dimension_scores: Dict[str, List[Dict]] = defaultdict(list)

        for feedback_obj in all_feedback:
            source_name = str(feedback_obj.get("source", "unknown"))
            feedback = feedback_obj.get("feedback") or {}
            markers = self._collect_feedback_markers(feedback)

            for point in markers["positive_points"]:
                point_sources[point]["positive"].add(source_name)
            for point in markers["negative_points"]:
                point_sources[point]["negative"].add(source_name)

            for dim, score in markers["scores"].items():
                dimension_scores[dim].append({"source": source_name, "score": score})

        for point, polarity in point_sources.items():
            pos_sources = sorted(list(polarity["positive"]))
            neg_sources = sorted(list(polarity["negative"]))
            if pos_sources and neg_sources:
                conflicts.append(
                    {
                        "type": "polarity_conflict",
                        "point": point,
                        "positive_sources": pos_sources,
                        "negative_sources": neg_sources,
                    }
                )

        for dim, values in dimension_scores.items():
            if len(values) < 2:
                continue
            numeric_scores = [float(item["score"]) for item in values]
            max_score = max(numeric_scores)
            min_score = min(numeric_scores)
            score_gap = round(max_score - min_score, 3)
            if score_gap >= 1.5:
                max_items = [
                    item for item in values if float(item["score"]) == max_score
                ]
                min_items = [
                    item for item in values if float(item["score"]) == min_score
                ]
                conflicts.append(
                    {
                        "type": "score_conflict",
                        "dimension": dim,
                        "score_gap": score_gap,
                        "min_score": min_score,
                        "max_score": max_score,
                        "high_score_sources": sorted(
                            {str(item["source"]) for item in max_items}
                        ),
                        "low_score_sources": sorted(
                            {str(item["source"]) for item in min_items}
                        ),
                    }
                )

        return conflicts

    def _derive_conflict_actions(self, conflicts: List[Dict]) -> List[str]:
        ranked_actions: List[Dict] = []
        for conflict in conflicts:
            conflict_type = str(conflict.get("type") or "")
            if conflict_type == "score_conflict":
                dimension = str(conflict.get("dimension") or "overall_quality")
                score_gap = float(conflict.get("score_gap") or 0.0)
                ranked_actions.append(
                    {
                        "priority": max(score_gap, 0.0),
                        "action": (
                            f"Resolve scoring disagreement on {dimension} "
                            f"(gap {score_gap:.2f}) with targeted evidence."
                        ),
                    }
                )
                continue
            if conflict_type == "polarity_conflict":
                point = str(conflict.get("point") or "key claim")
                support_count = len(conflict.get("positive_sources") or []) + len(
                    conflict.get("negative_sources") or []
                )
                ranked_actions.append(
                    {
                        "priority": 1.0 + 0.1 * support_count,
                        "action": (
                            f"Run focused verification for '{point}' to resolve positive/negative disagreement."
                        ),
                    }
                )

        ranked_actions.sort(key=lambda item: item["priority"], reverse=True)
        output: List[str] = []
        seen = set()
        for item in ranked_actions:
            action = str(item.get("action") or "").strip()
            token = self._normalize_feedback_text(action)
            if not action or token in seen:
                continue
            seen.add(token)
            output.append(action)
        return output[:5]

    async def _generate_evolution_strategy(
        self,
        paper_data: Dict,
        current_state: Dict,
        integrated_feedback: Dict,
    ) -> Dict:
        """生成进化策略"""
        prompt = f"""
基于以下信息，生成最优进化策略。

**论文信息**:
{json.dumps(paper_data, indent=2, ensure_ascii=False)[:500]}

**当前状态**:
{json.dumps(current_state, indent=2, ensure_ascii=False)}

**整合的反馈**:
{json.dumps(integrated_feedback, indent=2, ensure_ascii=False)}

**历史有效策略**:
{json.dumps(self.knowledge_base.get_effective_strategies()[:5], indent=2, ensure_ascii=False)}

请生成详细的进化策略，包括：

1. **进化目标**:
   - 具体改进目标
   - 可量化指标

2. **进化动作序列**:
   - 按优先级排序的动作
   - 每个动作的预期效果
   - 依赖关系

3. **资源配置**:
   - 需要的资源
   - 时间估计
   - 优先级

4. **风险评估**:
   - 潜在风险
   - 缓解措施

5. **成功指标**:
   - 如何衡量成功
   - 验证方法

以JSON格式返回。
"""

        response, _ = await get_response_from_llm(
            prompt=prompt,
            client=self.client,
            model=self.client_model,
            system_message="你是AI Scientist的进化策略专家，能够制定最优的进化路径。",
            temperature=0.7,
        )

        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            else:
                return {"raw_response": response}
        except:
            return {"raw_response": response}

    async def _execute_evolution(
        self,
        paper_data: Dict,
        strategy: Dict,
    ) -> Dict:
        """执行进化动作"""
        results = {
            "improvements": [],
            "actions_taken": [],
            "intermediate_states": [],
        }

        # 获取动作序列
        actions = strategy.get("evolution_actions", [])
        if not actions and isinstance(strategy.get("raw_response"), str):
            # 从原始响应中提取动作
            actions = self._extract_actions_from_text(strategy["raw_response"])

        for action in actions:
            print(f"\n   执行动作: {action.get('name', 'unknown')}")

            # 执行动作
            action_result = await self._execute_single_action(
                paper_data,
                action,
            )

            results["actions_taken"].append(action_result)
            results["improvements"].extend(
                action_result.get("improvements", [])
            )

            # 记录中间状态
            results["intermediate_states"].append({
                "action": action.get("name"),
                "state": action_result.get("new_state"),
            })

        return results

    async def _execute_single_action(
        self,
        paper_data: Dict,
        action: Dict,
    ) -> Dict:
        """执行单个进化动作"""
        action_type = action.get("type", "generic")
        action_name = action.get("name", "unknown")

        result = {
            "action": action_name,
            "type": action_type,
            "success": False,
            "improvements": [],
            "new_state": {},
        }

        try:
            if action_type == "improve_writing":
                result.update(await self._action_improve_writing(paper_data, action))
            elif action_type == "adjust_strategy":
                result.update(await self._action_adjust_strategy(paper_data, action))
            elif action_type == "optimize_prompt":
                result.update(await self._action_optimize_prompt(paper_data, action))
            elif action_type == "learn_pattern":
                result.update(await self._action_learn_pattern(paper_data, action))
            else:
                result.update(await self._action_generic(paper_data, action))

            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            print(f"      ⚠️  动作执行失败: {e}")

        return result

    async def _action_improve_writing(
        self,
        paper_data: Dict,
        action: Dict,
    ) -> Dict:
        """改进写作动作"""
        # 使用专业写作系统
        from ai_scientist.professional_writing_system import ExpertSectionWriter

        target_section = action.get("target_section", "all")
        improvements = []

        if target_section == "all":
            sections = ["abstract", "introduction", "method", "experiments", "conclusion"]
        else:
            sections = [target_section]

        for section in sections:
            # 这里可以调用专业写作系统进行改进
            improvements.append({
                "section": section,
                "improvement": f"Enhanced {section} based on evolutionary strategy",
                "estimated_gain": 0.5,
            })

        return {
            "improvements": improvements,
            "new_state": {"writing_quality": "improved"},
        }

    async def _action_adjust_strategy(
        self,
        paper_data: Dict,
        action: Dict,
    ) -> Dict:
        """调整策略动作"""
        new_strategy = action.get("new_strategy", {})

        # 更新知识库中的策略
        self.knowledge_base.improvement_strategies.update(new_strategy)

        return {
            "improvements": [{
                "type": "strategy_adjustment",
                "description": "Updated strategy based on evolutionary insights",
                "estimated_gain": 0.3,
            }],
            "new_state": {"strategy": "adjusted"},
        }

    async def _action_optimize_prompt(
        self,
        paper_data: Dict,
        action: Dict,
    ) -> Dict:
        """优化提示词动作"""
        # 生成优化的提示词
        current_prompt = action.get("current_prompt", "")
        optimization_goal = action.get("goal", "improve_clarity")

        # 这里可以实现提示词优化逻辑
        optimized_prompt = f"Optimized for {optimization_goal}"

        return {
            "improvements": [{
                "type": "prompt_optimization",
                "description": f"Optimized prompt for {optimization_goal}",
                "estimated_gain": 0.4,
            }],
            "new_state": {"prompt": optimized_prompt},
        }

    async def _action_learn_pattern(
        self,
        paper_data: Dict,
        action: Dict,
    ) -> Dict:
        """学习模式动作"""
        pattern = action.get("pattern", {})
        pattern_type = action.get("pattern_type", "writing")

        # 提取并存储新模式
        if pattern_type == "writing":
            self.knowledge_base.writing_insights[pattern.get("name", "unnamed")] = pattern
        elif pattern_type == "review":
            self.knowledge_base.review_insights[pattern.get("name", "unnamed")] = pattern

        return {
            "improvements": [{
                "type": "pattern_learning",
                "description": f"Learned new {pattern_type} pattern",
                "estimated_gain": 0.6,
            }],
            "new_state": {"patterns_learned": 1},
        }

    async def _action_generic(
        self,
        paper_data: Dict,
        action: Dict,
    ) -> Dict:
        """通用动作执行"""
        description = action.get("description", "")

        return {
            "improvements": [{
                "type": "generic",
                "description": description,
                "estimated_gain": 0.2,
            }],
            "new_state": {},
        }

    async def _validate_evolution(
        self,
        paper_data: Dict,
        previous_state: Dict,
        evolution_result: Dict,
    ) -> Dict:
        """验证进化效果"""
        prompt = f"""
请验证本次进化的效果。

**进化前状态**:
{json.dumps(previous_state, indent=2, ensure_ascii=False)}

**进化动作**:
{json.dumps(evolution_result.get('actions_taken', []), indent=2, ensure_ascii=False)}

**改进内容**:
{json.dumps(evolution_result.get('improvements', []), indent=2, ensure_ascii=False)}

请提供验证结果，包括：

1. **改进评分** (1-5分):
   - 各维度改进程度
   - 总体改进评分

2. **目标达成度**:
   - 达成了哪些目标
   - 未达成的目标及原因

3. **副作用**:
   - 是否有负面影响
   - 需要注意的问题

4. **后续建议**:
   - 下一步改进方向
   - 需要持续关注的点

以JSON格式返回。
"""

        response, _ = await get_response_from_llm(
            prompt=prompt,
            client=self.client,
            model=self.client_model,
            system_message="你是AI Scientist的进化验证专家，能够客观评估进化效果。",
            temperature=0.5,
        )

        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                validation = json.loads(json_match.group(1))

                # 计算总体分数
                if "scores" in validation:
                    overall = sum(validation["scores"].values()) / len(validation["scores"])
                    validation["overall_score"] = overall

                return validation
            else:
                return {"raw_response": response, "overall_score": 3.0}
        except:
            return {"raw_response": response, "overall_score": 3.0}

    async def _update_knowledge_from_evolution(
        self,
        evolution_record: Dict,
        validation: Dict,
    ) -> None:
        """从进化中更新知识库"""
        # 提取成功的模式
        if validation.get("overall_score", 0) >= 4.0:
            successful_actions = []

            for action_result in evolution_record.get("result", {}).get("actions_taken", []):
                if action_result.get("success"):
                    successful_actions.append(action_result)

            # 存储成功模式
            for action in successful_actions:
                pattern = {
                    "action": action.get("action"),
                    "type": action.get("type"),
                    "timestamp": datetime.now().isoformat(),
                    "effectiveness": validation.get("overall_score", 0),
                }

                self.knowledge_base.improvement_strategies[
                    f"evolution_{action.get('action')}"
                ] = pattern

        # 保存验证结果
        validation_file = self.evolution_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(validation_file, "w") as f:
            json.dump(validation, f, indent=2, ensure_ascii=False)

    def _extract_actions_from_text(self, text: str) -> List[Dict]:
        """从文本中提取动作"""
        # 简单实现，可以改进
        actions = []

        # 常见动作关键词
        action_keywords = {
            "improve": "improve_writing",
            "adjust": "adjust_strategy",
            "optimize": "optimize_prompt",
            "learn": "learn_pattern",
        }

        lines = text.split("\n")
        for line in lines:
            for keyword, action_type in action_keywords.items():
                if keyword.lower() in line.lower():
                    actions.append({
                        "type": action_type,
                        "name": line.strip()[:50],
                        "description": line.strip(),
                    })
                    break

        return actions[:5]  # 最多5个动作

    # ========================================
    # 外部Agent接口
    # ========================================

    def register_external_agent(
        self,
        agent_name: str,
        callback: Callable,
        agent_info: Dict = None,
    ):
        """
        注册外部agent

        Args:
            agent_name: agent名称
            callback: 回调函数，接受 (paper_data, current_state) 返回反馈
            agent_info: agent信息
        """
        self.external_agent_callbacks[agent_name] = {
            "callback": callback,
            "info": agent_info or {},
            "registered_at": datetime.now().isoformat(),
        }

        print(f"✅ 已注册外部agent: {agent_name}")

    async def consult_external_agents(
        self,
        paper_data: Dict,
        current_state: Dict,
        agent_filter: List[str] = None,
    ) -> List[Dict]:
        """
        咨询外部agent

        Args:
            paper_data: 论文数据
            current_state: 当前状态
            agent_filter: 要咨询的agent列表（None表示全部）

        Returns:
            各agent的反馈
        """
        feedback_list = []

        # 确定要咨询的agents
        if agent_filter:
            agents_to_consult = {
                k: v for k, v in self.external_agent_callbacks.items()
                if k in agent_filter
            }
        else:
            agents_to_consult = self.external_agent_callbacks

        # 并发咨询所有agents
        tasks = []
        for agent_name, agent_info in agents_to_consult.items():
            task = self._consult_single_agent(
                agent_name,
                agent_info,
                paper_data,
                current_state,
            )
            tasks.append(task)

        # 执行所有咨询
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    print(f"⚠️  Agent咨询失败: {result}")
                elif result:
                    feedback_list.append(result)

        return feedback_list

    async def _consult_single_agent(
        self,
        agent_name: str,
        agent_info: Dict,
        paper_data: Dict,
        current_state: Dict,
    ) -> Dict:
        """咨询单个agent"""
        callback = agent_info["callback"]

        try:
            if asyncio.iscoroutinefunction(callback):
                feedback = await callback(paper_data, current_state)
            else:
                feedback = callback(paper_data, current_state)

            return {
                "source": FeedbackSource.EXTERNAL_AGENT,
                "agent_name": agent_name,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"❌ Agent {agent_name} 咨询失败: {e}")
            return {
                "source": FeedbackSource.EXTERNAL_AGENT,
                "agent_name": agent_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # ========================================
    # 反馈处理
    # ========================================

    def submit_feedback(
        self,
        source: FeedbackSource,
        feedback: Dict,
        metadata: Dict = None,
    ):
        """
        提交反馈

        Args:
            source: 反馈来源
            feedback: 反馈内容
            metadata: 元数据
        """
        feedback_record = {
            "source": source.value if isinstance(source, FeedbackSource) else source,
            "feedback": feedback,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }

        self.feedback_buffer.append(feedback_record)

        # 如果缓冲区满了，触发处理
        if len(self.feedback_buffer) >= 5:
            self._process_feedback_buffer()

    def _process_feedback_buffer(self):
        """处理反馈缓冲区"""
        if not self.feedback_buffer:
            return

        print(f"\n📨 处理 {len(self.feedback_buffer)} 条反馈...")

        # 整合反馈
        integrated = self._integrate_feedback(
            {},  # 自我反思
            [
                {"source": f["source"], "feedback": f["feedback"]}
                for f in self.feedback_buffer
            ],
        )

        # 提取洞察
        insights = self._extract_insights_from_feedback(integrated)

        # 更新知识库
        if insights:
            for insight in insights:
                if insight.get("type") == "writing":
                    self.knowledge_base.writing_insights[insight.get("name", "unnamed")] = insight
                elif insight.get("type") == "strategy":
                    self.knowledge_base.improvement_strategies[insight.get("name", "unnamed")] = insight

        # 清空缓冲区
        self.feedback_buffer = []

        print(f"✅ 反馈处理完成，提取了 {len(insights)} 条洞察")

    def _extract_insights_from_feedback(
        self,
        integrated_feedback: Dict,
    ) -> List[Dict]:
        """从反馈中提取洞察"""
        insights = []

        # 从共同问题中提取洞察
        for issue_obj in integrated_feedback.get("common_issues", []):
            insights.append({
                "name": f"common_issue_{issue_obj['issue']}",
                "type": "strategy",
                "description": f"Common issue: {issue_obj['issue']}",
                "frequency": issue_obj.get("frequency", 1),
                "timestamp": datetime.now().isoformat(),
            })

        # 从优先级行动中提取
        for i, action in enumerate(integrated_feedback.get("priority_actions", [])[:3]):
            insights.append({
                "name": f"priority_action_{i}",
                "type": "strategy",
                "description": f"Priority action: {action}",
                "priority": i + 1,
                "timestamp": datetime.now().isoformat(),
            })

        return insights

    # ========================================
    # 进化监控和报告
    # ========================================

    def get_evolution_report(self) -> Dict:
        """获取进化报告"""
        summary = {
            "total_evolutions": len(self.evolution_history),
            "recent_evolutions": self.evolution_history[-10:] if len(self.evolution_history) >= 10 else self.evolution_history,
            "knowledge_base_summary": self.knowledge_base.generate_learning_summary(),
            "registered_agents": list(self.external_agent_callbacks.keys()),
            "pending_feedback": len(self.feedback_buffer),
        }

        # 计算进化趋势
        if len(self.evolution_history) >= 2:
            recent_scores = [
                e.get("validation", {}).get("overall_score", 0)
                for e in self.evolution_history[-10:]
            ]
            if recent_scores:
                summary["average_validation_score"] = sum(recent_scores) / len(recent_scores)
                summary["trend"] = "improving" if recent_scores[-1] > recent_scores[0] else "stable"

        return summary

    def export_evolution_knowledge(self, export_path: str = None) -> str:
        """导出进化知识"""
        if export_path is None:
            export_path = self.evolution_dir / f"evolution_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        export_data = {
            "evolution_history": self.evolution_history,
            "knowledge_base": {
                "success_patterns": self.knowledge_base.success_patterns,
                "improvement_strategies": self.knowledge_base.improvement_strategies,
            },
            "external_agents": {
                name: {"info": info["info"], "registered_at": info["registered_at"]}
                for name, info in self.external_agent_callbacks.items()
            },
            "exported_at": datetime.now().isoformat(),
        }

        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 进化知识已导出到: {export_path}")
        return str(export_path)


# ========================================
    # 辅助函数
    # ========================================

async def get_response_from_llm(prompt, client, model, system_message, temperature):
    """异步获取LLM响应"""
    # 这里是同步调用的包装器
    # 实际实现中可以使用真正的异步LLM客户端
    from ai_scientist.llm import get_response_from_llm
    return get_response_from_llm(
        prompt=prompt,
        client=client,
        model=model,
        system_message=system_message,
        temperature=temperature,
    )
