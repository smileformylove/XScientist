# 自主进化系统使用指南

## 📋 概述

自主进化系统使 AI Scientist 能够：
1. **自主反思** - 自我分析并识别改进点
2. **接受指导** - 接受外部 Agent 的建议
3. **综合进化** - 整合多源反馈进行优化
4. **持续进化** - 随着使用不断进步

## 🧬 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    自主进化引擎                              │
│                  (AutonomousEvolutionEngine)                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  自我反思    │  │  外部指导    │  │  反馈整合    │    │
│  │  Self        │  │  External    │  │  Integration │    │
│  │  Reflection  │  │  Agents      │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         ↓                  ↓                  ↓             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              进化策略生成                              │ │
│  │         Evolution Strategy Generation                 │ │
│  └──────────────────────────────────────────────────────┘ │
│         ↓                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  改进写作    │  │  调整策略    │  │  优化提示    │    │
│  │  Improve     │  │  Adjust      │  │  Optimize    │    │
│  │  Writing     │  │  Strategy    │  │  Prompt      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         ↓                  ↓                  ↓             │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              进化验证                                  │ │
│  │           Evolution Validation                        │ │
│  └──────────────────────────────────────────────────────┘ │
│         ↓                                                     │
│  ┌──────────────────────────────────────────────────────┐ │
│  │              知识库更新                                │ │
│  │         Knowledge Base Update                         │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 主要功能

### 1. 自我反思 (Self-Reflection)

系统自动分析当前状态：
- 各维度评分（1-5分）
- 与成功论文的差距
- 主要问题识别
- 改进方向建议

### 2. 外部 Agent 指导

接受外部 Agent 的反馈和建议：
- 写作批评 Agent
- 技术审查 Agent
- 领域专家 Agent
- 方法论顾问 Agent
- 等等...

### 3. 反馈整合

智能整合多源反馈：
- 识别共同问题
- 检测冲突点
- 优先级行动排序
- 生成综合建议

### 4. 进化执行

执行进化动作：
- 改进写作质量
- 调整生成策略
- 优化提示词
- 学习新模式

### 5. 效果验证

验证进化效果：
- 改进评分
- 目标达成度
- 副作用检测
- 后续建议

## 🚀 使用方法

### 基础使用：自主进化生成论文

```python
from ai_scientist.continuous_paper_generator import ContinuousPaperGenerator

# 创建生成器（自动启用进化系统）
generator = ContinuousPaperGenerator(enable_learning=True)

# 使用自主进化生成
result = await generator.generate_paper_with_evolution(
    idea=idea,
    paper_type="neurips",
    evolution_rounds=3,  # 3轮进化
)

# 查看进化报告
print(result["evolution_report"])
```

### 注册外部 Agent

```python
from ai_scientist.agent_interface import BaseAgent, ExampleWritingCriticAgent

# 创建 Agent
writing_agent = ExampleWritingCriticAgent()

# 注册到生成器
generator.register_external_agent(writing_agent, group="writing")

# 现在生成时会自动咨询这个 Agent
result = await generator.generate_paper_with_evolution(
    idea=idea,
    paper_type="neurips",
    enable_external_agents=True,
)
```

### 创建自定义 Agent

```python
from ai_scientist.agent_interface import BaseAgent, AgentCapability, StandardFeedback

class MyCustomAgent(BaseAgent):
    """自定义 Agent"""

    def __init__(self):
        super().__init__(
            name="MyCustomAgent",
            version="1.0",
            capabilities=[AgentCapability.DOMAIN_EXPERTISE],
        )

    async def analyze(
        self,
        paper_data: Dict,
        current_state: Dict,
        context: Dict = None,
    ) -> Dict:
        """分析论文"""
        # 你的分析逻辑
        issues = []
        suggestions = []
        strengths = []

        # ... 分析代码 ...

        # 计算分数
        score = 4.0

        return StandardFeedback.create(
            score=score,
            issues=issues,
            suggestions=suggestions,
            strengths=strengths,
        )

    def get_info(self) -> Dict:
        """返回 Agent 信息"""
        return {
            "name": self.name,
            "version": self.version,
            "description": "我的自定义 Agent",
            "capabilities": [c.value for c in self.capabilities],
        }

# 注册自定义 Agent
agent = MyCustomAgent()
generator.register_external_agent(agent)
```

### 直接提交反馈

```python
# 提交人工反馈
generator.submit_feedback(
    source="human",
    feedback={
        "score": 3.5,
        "issues": ["实验部分不够详细"],
        "suggestions": ["增加更多实验细节"],
    },
    metadata={"reviewer": "Expert Name"},
)

# 提交指标反馈
generator.submit_feedback(
    source="metrics",
    feedback={
        "bleu_score": 0.75,
        "readability": 0.68,
        "issues": ["可读性偏低"],
    },
)
```

## 🔌 外部 Agent 接口规范

### Agent 基类

所有外部 Agent 都应该继承 `BaseAgent`：

```python
from ai_scientist.agent_interface import BaseAgent, AgentCapability

class MyAgent(BaseAgent):
    async def analyze(self, paper_data, current_state, context=None):
        """分析论文并返回反馈"""
        # 必须返回标准格式反馈
        return {
            "score": 4.0,
            "issues": [],
            "suggestions": [],
            "strengths": [],
        }

    def get_info(self):
        """返回 Agent 信息"""
        return {"name": "MyAgent", "version": "1.0"}
```

### 标准反馈格式

使用 `StandardFeedback.create()` 创建标准反馈：

```python
from ai_scientist.agent_interface import StandardFeedback, FeedbackPriority

feedback = StandardFeedback.create(
    score=4.2,  # 0-5分
    issues=["问题1", "问题2"],
    suggestions=["建议1", "建议2"],
    strengths=["优点1", "优点2"],
    priority=FeedbackPriority.HIGH,
    metadata={"custom_field": "value"},
)
```

### Agent 能力类型

可用能力：
- `WRITING_CRITIQUE` - 写作批评
- `TECHNICAL_REVIEW` - 技术审查
- `DOMAIN_EXPERTISE` - 领域专长
- `METHODOLOGY_ADVICE` - 方法论建议
- `EXPERIMENT_DESIGN` - 实验设计
- `LITERATURE_ANALYSIS` - 文献分析
- `STATISTICAL_ANALYSIS` - 统计分析
- `STYLE_GUIDANCE` - 风格指导

## 📊 进化监控

### 查看进化状态

```python
# 获取完整状态
status = generator.get_evolution_status()

print(f"进化系统启用: {status['enabled']}")
print(f"总进化次数: {status['evolution']['total_evolutions']}")
print(f"注册的 Agent: {status['agents']['total_agents']}")

# 查看最近进化
for evolution in status['evolution']['recent_evolutions']:
    print(f"验证分数: {evolution['validation']['overall_score']}")
```

### Agent 统计

```python
# 获取 Agent 统计
stats = generator.agent_orchestrator.get_agent_statistics()

for agent_name, info in stats['agents'].items():
    print(f"{agent_name}:")
    print(f"  交互次数: {info['interaction_count']}")
    print(f"  成功率: {info['success_rate']:.1%}")
```

### 导出进化知识

```python
# 导出进化历史和知识
export_path = generator.evolution_engine.export_evolution_knowledge()

print(f"进化知识已导出到: {export_path}")
```

## 🔄 完整工作流

### 1. 设置进化系统

```python
from ai_scientist.continuous_paper_generator import ContinuousPaperGenerator
from ai_scientist.agent_interface import (
    ExampleWritingCriticAgent,
    ExampleTechnicalReviewerAgent,
)

# 创建生成器
generator = ContinuousPaperGenerator(enable_learning=True)

# 注册多个 Agent
generator.register_external_agent(
    ExampleWritingCriticAgent(),
    group="quality"
)
generator.register_external_agent(
    ExampleTechnicalReviewerAgent(),
    group="technical"
)
```

### 2. 生成论文并进化

```python
# 使用自主进化生成
result = await generator.generate_paper_with_evolution(
    idea=my_idea,
    paper_type="neurips",
    enable_external_agents=True,
    evolution_rounds=3,
)

# 检查结果
if result["status"] == "success":
    print(f"✅ 论文生成成功!")
    print(f"位置: {result['latex_path']}")
    print(f"最终评分: {result['evaluation']['overall']['score']:.1f}/5")
```

### 3. 收集反馈并继续进化

```python
# 收集人工反馈
generator.submit_feedback(
    source="human",
    feedback={
        "score": 4.0,
        "issues": ["需要更多实验"],
        "suggestions": ["添加消融实验"],
    },
)

# 基于反馈继续进化
improved_result = await generator.generate_paper_with_evolution(
    idea=my_idea,
    paper_type="neurips",
    evolution_rounds=1,  # 再进化一轮
)
```

### 4. 监控进化进展

```python
# 查看进化状态
status = generator.get_evolution_status()

# 生成报告
report = status['evolution']
print(f"平均验证分数: {report.get('average_validation_score', 0):.1f}/5")
print(f"进化趋势: {report.get('trend', 'unknown')}")

# 查看 Agent 贡献
for agent_name, info in status['agents']['agents'].items():
    if info['interaction_count'] > 0:
        print(f"{agent_name}: {info['interaction_count']} 次交互")
```

## 💡 最佳实践

### 1. 渐进式 Agent 注册

先注册基础 Agent，再逐步添加专业 Agent：

```python
# 第一步：基础质量 Agent
generator.register_external_agent(ExampleWritingCriticAgent())

# 运行几次，观察效果
# ...

# 第二步：添加技术审查 Agent
generator.register_external_agent(ExampleTechnicalReviewerAgent())

# 继续观察
# ...

# 第三步：添加领域特定 Agent
generator.register_external_agent(DomainSpecificAgent())
```

### 2. 反馈优先级

使用合适的优先级：

```python
from ai_scientist.agent_interface import FeedbackPriority

# 关键问题
critical_feedback = StandardFeedback.create(
    score=2.0,
    issues=["致命错误"],
    priority=FeedbackPriority.CRITICAL,
)

# 一般建议
info_feedback = StandardFeedback.create(
    score=4.0,
    suggestions=["小改进"],
    priority=FeedbackPriority.INFORMATIONAL,
)
```

### 3. Agent 分组管理

使用分组来组织 Agent：

```python
# 写作质量组
generator.register_external_agent(WritingAgent(), group="writing")
generator.register_external_agent(GrammarAgent(), group="writing")

# 技术内容组
generator.register_external_agent(MethodologyAgent(), group="technical")
generator.register_external_agent(ExperimentAgent(), group="technical")

# 可以按组咨询
# feedback = await generator.agent_orchestrator.consult_agents(
#     paper_data, current_state, group="writing"
# )
```

### 4. 定期知识清理

定期清理旧知识保持相关性：

```python
# 清理6个月前的旧知识
generator.knowledge_base.prune_old_knowledge(days=180)
```

## 🎯 进化效果

### 预期改进

随着时间推移，系统会：

1. **提高自我认知**
   - 更准确的问题识别
   - 更好的自我评估

2. **增强学习能力**
   - 更快的模式识别
   - 更有效的策略选择

3. **优化 Agent 协作**
   - 识别最有效的 Agent
   - 优化咨询顺序

4. **持续质量提升**
   - 生成质量稳步提高
   - 减少迭代轮数

### 指标跟踪

关键指标：

- **验证分数趋势**: 应该持续上升
- **进化轮数**: 应该逐渐减少（效率提高）
- **Agent 成功率**: 高质量 Agent 的权重增加
- **知识库大小**: 持续增长

## 🐛 故障排除

### 问题1: Agent 咨询失败

**症状**: 外部 Agent 返回错误

**解决**:
- 检查 Agent 是否正确实现 `analyze()` 方法
- 验证返回的反馈格式是否符合标准
- 查看错误日志

### 问题2: 进化没有改进

**症状**: 验证分数没有提升

**解决**:
- 检查是否有足够的相似历史论文
- 确认反馈质量是否足够高
- 尝试增加进化轮数
- 注册更多相关 Agent

### 问题3: Agent 之间冲突

**症状**: 不同 Agent 给出矛盾的建议

**解决**:
- 系统会自动检测并标记冲突
- 可以调整 Agent 优先级
- 使用分组管理 Agent 咨询顺序

## 📚 相关文档

- [ADAPTIVE_LEARNING_README.md](ADAPTIVE_LEARNING_README.md) - 自适应学习系统
- [PROFESSIONAL_WRITING_README.md](PROFESSIONAL_WRITING_README.md) - 专业写作系统
- [AUTO_IMPROVEMENT_README.md](AUTO_IMPROVEMENT_README.md) - 自动改进系统

## 🎉 总结

自主进化系统提供了：

1. ✅ **自我反思能力**
   - 自动分析状态
   - 识别改进点

2. ✅ **外部 Agent 接口**
   - 标准化的 Agent 接口
   - 灵活的注册机制
   - 智能的反馈整合

3. ✅ **自主进化**
   - 多源反馈综合
   - 智能策略生成
   - 自动执行和验证

4. ✅ **持续改进**
   - 知识积累
   - 模式学习
   - 效果跟踪

现在 AI Scientist 不仅能生成论文，还能**自主思考、接受指导、持续进化**，真正成为了一个智能的研究助手！
