#!/usr/bin/env python3
"""
外部Agent接口规范
定义其他Agent与AI Scientist交互的标准接口
"""

from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
from enum import Enum
import json


class AgentCapability(Enum):
    """Agent能力类型"""
    WRITING_CRITIQUE = "writing_critique"  # 写作批评
    TECHNICAL_REVIEW = "technical_review"  # 技术审查
    DOMAIN_EXPERTISE = "domain_expertise"  # 领域专长
    METHODOLOGY_ADVICE = "methodology_advice"  # 方法论建议
    EXPERIMENT_DESIGN = "experiment_design"  # 实验设计
    LITERATURE_ANALYSIS = "literature_analysis"  # 文献分析
    STATISTICAL_ANALYSIS = "statistical_analysis"  # 统计分析
    STYLE_GUIDANCE = "style_guidance"  # 风格指导


class FeedbackPriority(Enum):
    """反馈优先级"""
    CRITICAL = "critical"  # 必须处理
    HIGH = "high"  # 高优先级
    MEDIUM = "medium"  # 中等优先级
    LOW = "low"  # 低优先级
    INFORMATIONAL = "informational"  # 仅供参考


class StandardFeedback:
    """标准反馈格式"""

    @staticmethod
    def create(
        score: float,
        issues: List[str] = None,
        suggestions: List[str] = None,
        strengths: List[str] = None,
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        metadata: Dict = None,
    ) -> Dict:
        """
        创建标准反馈

        Args:
            score: 评分 (0-5)
            issues: 发现的问题列表
            suggestions: 改进建议列表
            strengths: 优点列表
            priority: 优先级
            metadata: 额外元数据

        Returns:
            标准格式的反馈字典
        """
        return {
            "score": score,
            "issues": issues or [],
            "suggestions": suggestions or [],
            "strengths": strengths or [],
            "priority": priority.value,
            "metadata": metadata or {},
            "format_version": "1.0",
        }

    @staticmethod
    def validate(feedback: Dict) -> bool:
        """验证反馈格式"""
        required_fields = ["score"]
        return all(field in feedback for field in required_fields)


class BaseAgent(ABC):
    """
    Agent基类

    所有与AI Scientist交互的外部Agent都应该继承这个类
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0",
        capabilities: List[AgentCapability] = None,
    ):
        """
        初始化Agent

        Args:
            name: Agent名称
            version: 版本号
            capabilities: 能力列表
        """
        self.name = name
        self.version = version
        self.capabilities = capabilities or []
        self.interaction_count = 0
        self.success_count = 0

    @abstractmethod
    async def analyze(
        self,
        paper_data: Dict,
        current_state: Dict,
        context: Dict = None,
    ) -> Dict:
        """
        分析论文并提供建议

        Args:
            paper_data: 论文数据
            current_state: 当前状态
            context: 额外上下文

        Returns:
            标准格式的反馈
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict:
        """
        获取Agent信息

        Returns:
            Agent信息字典
        """
        pass

    def _record_interaction(self, success: bool = True):
        """记录交互"""
        self.interaction_count += 1
        if success:
            self.success_count += 1

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.interaction_count == 0:
            return 0.0
        return self.success_count / self.interaction_count


class ExampleWritingCriticAgent(BaseAgent):
    """
    示例：写作批评Agent

    专门分析论文的写作质量并提供改进建议
    """

    def __init__(self):
        super().__init__(
            name="WritingCriticAgent",
            version="1.0",
            capabilities=[
                AgentCapability.WRITING_CRITIQUE,
                AgentCapability.STYLE_GUIDANCE,
            ],
        )

    async def analyze(
        self,
        paper_data: Dict,
        current_state: Dict,
        context: Dict = None,
    ) -> Dict:
        """分析写作质量"""
        # 这里可以实现具体的分析逻辑
        # 例如：检查语法、结构、清晰度等

        issues = []
        suggestions = []
        strengths = []

        # 分析标题
        title = paper_data.get("Title", "")
        if len(title) < 10:
            issues.append("标题过短，建议提供更多细节")
        elif len(title) > 150:
            issues.append("标题过长，建议精简")

        # 分析摘要
        abstract = paper_data.get("Abstract", "")
        if len(abstract) < 100:
            issues.append("摘要过短，应该包含更多细节")
        elif "we propose" in abstract.lower() and abstract.count(".") < 5:
            suggestions.append("摘要应该更清楚地说明贡献和方法")

        # 识别优点
        if "novel" in abstract.lower() or "new" in abstract.lower():
            strengths.append("强调了创新性")

        # 计算分数
        score = 3.0
        if len(issues) == 0:
            score += 1.0
        elif len(issues) > 3:
            score -= 1.0

        if len(strengths) > 0:
            score += 0.5

        score = max(1.0, min(5.0, score))

        self._record_interaction()

        return StandardFeedback.create(
            score=score,
            issues=issues,
            suggestions=suggestions,
            strengths=strengths,
            priority=FeedbackPriority.HIGH if len(issues) > 2 else FeedbackPriority.MEDIUM,
            metadata={
                "agent": self.name,
                "version": self.version,
                "analysis_type": "writing_quality",
            },
        )

    def get_info(self) -> Dict:
        """获取Agent信息"""
        return {
            "name": self.name,
            "version": self.version,
            "description": "分析论文写作质量并提供改进建议",
            "capabilities": [c.value for c in self.capabilities],
            "interaction_count": self.interaction_count,
            "success_rate": self.get_success_rate(),
        }


class ExampleTechnicalReviewerAgent(BaseAgent):
    """
    示例：技术审查Agent

    专门分析论文的技术内容和方法论
    """

    def __init__(self):
        super().__init__(
            name="TechnicalReviewerAgent",
            version="1.0",
            capabilities=[
                AgentCapability.TECHNICAL_REVIEW,
                AgentCapability.METHODOLOGY_ADVICE,
            ],
        )

    async def analyze(
        self,
        paper_data: Dict,
        current_state: Dict,
        context: Dict = None,
    ) -> Dict:
        """分析技术内容"""
        issues = []
        suggestions = []
        strengths = []

        # 分析方法描述
        method = paper_data.get("Method", "")
        if len(method) < 100:
            issues.append("方法描述过于简单，需要更多技术细节")
        else:
            strengths.append("方法描述较为详细")

        # 检查是否提到了实验
        if "experiment" not in method.lower():
            suggestions.append("建议在方法部分描述实验设置")

        # 检查是否提到了基线
        if "baseline" not in method.lower():
            suggestions.append("建议说明与基线方法的对比")

        # 分析假设
        hypothesis = paper_data.get("Hypothesis", "")
        if hypothesis:
            strengths.append("明确提出了研究假设")
        else:
            issues.append("缺少明确的研究假设")

        # 计算分数
        score = 3.0
        if len(strengths) > len(issues):
            score += 0.5
        if len(issues) > 3:
            score -= 0.5

        score = max(1.0, min(5.0, score))

        self._record_interaction()

        return StandardFeedback.create(
            score=score,
            issues=issues,
            suggestions=suggestions,
            strengths=strengths,
            priority=FeedbackPriority.HIGH if "缺少明确的研究假设" in issues else FeedbackPriority.MEDIUM,
            metadata={
                "agent": self.name,
                "version": self.version,
                "analysis_type": "technical_review",
            },
        )

    def get_info(self) -> Dict:
        """获取Agent信息"""
        return {
            "name": self.name,
            "version": self.version,
            "description": "分析论文技术内容和方法论",
            "capabilities": [c.value for c in self.capabilities],
            "interaction_count": self.interaction_count,
            "success_rate": self.get_success_rate(),
        }


class AgentOrchestrator:
    """
    Agent编排器

    管理多个外部Agent，协调它们与AI Scientist的交互
    """

    def __init__(self):
        """初始化编排器"""
        self.registered_agents = {}
        self.agent_groups = {}

    def register_agent(self, agent: BaseAgent, group: str = "default") -> bool:
        """
        注册Agent

        Args:
            agent: Agent实例
            group: 分组名称

        Returns:
            是否成功注册
        """
        if not isinstance(agent, BaseAgent):
            print(f"❌ Agent必须是BaseActor的子类")
            return False

        self.registered_agents[agent.name] = agent

        # 添加到分组
        if group not in self.agent_groups:
            self.agent_groups[group] = []
        self.agent_groups[group].append(agent.name)

        print(f"✅ 已注册Agent: {agent.name} (分组: {group})")
        return True

    def unregister_agent(self, agent_name: str) -> bool:
        """注销Agent"""
        if agent_name in self.registered_agents:
            del self.registered_agents[agent_name]

            # 从分组中移除
            for group_name, agents in self.agent_groups.items():
                if agent_name in agents:
                    agents.remove(agent_name)

            print(f"✅ 已注销Agent: {agent_name}")
            return True

        return False

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """获取Agent实例"""
        return self.registered_agents.get(agent_name)

    def get_agents_by_capability(
        self,
        capability: AgentCapability,
    ) -> List[BaseAgent]:
        """根据能力获取Agent"""
        return [
            agent
            for agent in self.registered_agents.values()
            if capability in agent.capabilities
        ]

    def get_agents_in_group(self, group: str) -> List[BaseAgent]:
        """获取分组中的Agent"""
        agent_names = self.agent_groups.get(group, [])
        return [
            self.registered_agents[name]
            for name in agent_names
            if name in self.registered_agents
        ]

    async def consult_agents(
        self,
        paper_data: Dict,
        current_state: Dict,
        agent_names: List[str] = None,
        group: str = None,
        capability: AgentCapability = None,
        context: Dict = None,
    ) -> List[Dict]:
        """
        咨询多个Agent

        Args:
            paper_data: 论文数据
            current_state: 当前状态
            agent_names: 指定Agent名称列表
            group: 指定分组
            capability: 指定能力
            context: 额外上下文

        Returns:
            所有Agent的反馈列表
        """
        # 确定要咨询的Agents
        agents_to_consult = []

        if agent_names:
            # 按名称指定
            for name in agent_names:
                agent = self.get_agent(name)
                if agent:
                    agents_to_consult.append(agent)

        elif group:
            # 按分组指定
            agents_to_consult = self.get_agents_in_group(group)

        elif capability:
            # 按能力指定
            agents_to_consult = self.get_agents_by_capability(capability)

        else:
            # 咨询所有注册的Agents
            agents_to_consult = list(self.registered_agents.values())

        # 并发咨询
        import asyncio

        tasks = []
        for agent in agents_to_consult:
            task = agent.analyze(paper_data, current_state, context)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 整理结果
        feedback_list = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"⚠️  Agent {agents_to_consult[i].name} 咨询失败: {result}")
            else:
                feedback_list.append({
                    "agent_name": agents_to_consult[i].name,
                    "feedback": result,
                    "agent_info": agents_to_consult[i].get_info(),
                })

        return feedback_list

    def get_agent_statistics(self) -> Dict:
        """获取Agent统计信息"""
        stats = {
            "total_agents": len(self.registered_agents),
            "agents": {},
            "groups": self.agent_groups,
        }

        for name, agent in self.registered_agents.items():
            stats["agents"][name] = agent.get_info()

        return stats


# ========================================
# 便捷函数
# ========================================

def create_agent_callback(agent: BaseAgent) -> Callable:
    """
    为Agent创建标准回调函数

    这个函数使得Agent可以与AutonomousEvolutionEngine集成

    Args:
        agent: Agent实例

    Returns:
        回调函数
    """
    async def callback(paper_data: Dict, current_state: Dict) -> Dict:
        """回调函数"""
        feedback = await agent.analyze(paper_data, current_state)
        return {
            "feedback": feedback,
            "agent_info": agent.get_info(),
        }

    return callback


def register_agent_with_evolution(
    evolution_engine,
    agent: BaseAgent,
    group: str = "default",
) -> bool:
    """
    将Agent注册到进化引擎

    Args:
        evolution_engine: AutonomousEvolutionEngine实例
        agent: Agent实例
        group: 分组名称

    Returns:
        是否成功注册
    """
    callback = create_agent_callback(agent)

    evolution_engine.register_external_agent(
        agent_name=agent.name,
        callback=callback,
        agent_info={
            "group": group,
            "capabilities": [c.value for c in agent.capabilities],
            "version": agent.version,
        },
    )

    return True
