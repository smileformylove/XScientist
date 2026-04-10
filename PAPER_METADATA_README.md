# 论文元数据标记系统使用指南

## 📋 概述

论文元数据标记系统为每个论文文件夹添加标准化标记，使得：
1. **其他Agent能够快速了解论文状态**
2. **明确需要什么样的帮助**
3. **提供标准化的指导接口**
4. **追踪Agent的贡献和影响**

## 🎯 核心概念

### 论文状态 (PaperStatus)

```python
- IDEATION: 想法阶段
- GENERATING: 生成中
- DRAFT: 初稿完成
- UNDER_REVIEW: 审查中
- IMPROVING: 改进中
- COMPLETED: 完成
- PUBLISHED: 已发表
```

### 质量等级 (QualityLevel)

```python
- POOR: 差 (<3.0)
- FAIR: 一般 (3.0-3.5)
- GOOD: 良好 (3.5-4.0)
- EXCELLENT: 优秀 (4.0-4.5)
- OUTSTANDING: 卓越 (>4.5)
```

### 指导优先级 (GuidancePriority)

```python
- CRITICAL: 紧急（质量差或有严重问题）
- HIGH: 高优先级（有明显改进空间）
- MEDIUM: 中等优先级（正常改进需求）
- LOW: 低优先级（小幅优化）
- OPTIONAL: 可选（已经很好）
```

## 📁 标准化文件结构

每个论文文件夹包含以下标记文件：

```
paper_YYYYMMDD_HHMMSS_idea_name/
├── .paper_metadata.json          # 论文元数据
├── .status.json                  # 状态信息
├── .agent_comments.json          # Agent评论
├── README.md                     # 论文概览（供其他Agent阅读）
├── .agent_instructions.md        # Agent指导说明
└── .status_badge.txt            # 状态徽章（ASCII艺术）
```

## 🚀 使用方法

### 1. 创建论文时自动添加标记

```python
from ai_scientist.continuous_paper_generator import ContinuousPaperGenerator

generator = ContinuousPaperGenerator(enable_learning=True)

# 生成论文时会自动创建元数据标记
result = generator.generate_paper_with_professional_writing(
    idea=idea,
    paper_type="neurips",
)

# 或者在生成后手动创建
metadata = generator.create_paper_metadata_markers(
    paper_dir=result["paper_dir"],
    idea=idea,
    paper_type="neurips",
)
```

### 2. 外部Agent发现需要帮助的论文

```python
from ai_scientist.agent_guidance_coordinator import AgentGuidanceAPI

# 创建API
api = AgentGuidanceAPI()

# Agent发现论文
papers = api.discover_papers(
    agent_name="WritingCriticAgent",
    agent_capabilities=["writing_critique", "style_guidance"],
    max_papers=5,
)

for paper in papers:
    print(f"论文: {paper['paper_id']}")
    print(f"优先级: {paper['priority']}")
    print(f"评分: {paper['info']['status']['overall_score']}")
    print(f"可执行项: {len(paper['actionable_items'])}")
```

### 3. Agent提供指导

```python
# 提交指导
result = api.submit_guidance(
    agent_name="WritingCriticAgent",
    paper_id="paper_20240223_120000_my_idea_neurips",
    comment="Introduction需要更清晰地阐述贡献",
    score=3.5,
    issues=[
        "Introduction过长",
        "贡献声明不够突出",
    ],
    suggestions=[
        "精简Introduction",
        "在开头明确列出3个主要贡献",
    ],
    priority="high",
)

print(f"指导已提交: {result}")
```

### 4. 查看论文信息

```python
# 获取论文详细信息
info = api.get_paper_info("paper_20240223_120000_my_idea_neurips")

print(f"标题: {info['paper_info']['title']}")
print(f"状态: {info['status']['current_status']}")
print(f"质量: {info['status']['quality_level']}")
print(f"评分: {info['status']['overall_score']}")

# 查看Agent评论摘要
agent_summary = info['agent_summary']
print(f"评论数: {agent_summary['total_comments']}")
print(f"平均分: {agent_summary['average_score']}")

# 查看可执行项
actionable_items = api.get_actionable_items("paper_20240223_120000_my_idea_neurips")
for item in actionable_items:
    print(f"[{item['priority']}] {item['description']}")
```

## 🤖 Agent工作流

### 标准Agent工作流程

```python
from ai_scientist.paper_metadata import AgentGuidanceAPI

class MyAgent:
    def __init__(self):
        self.api = AgentGuidanceAPI()
        self.name = "MyAgent"
        self.capabilities = ["writing_critique", "technical_review"]

    def find_and_review_papers(self):
        """发现并审查论文"""

        # 1. 发现需要帮助的论文
        papers = self.api.discover_papers(
            agent_name=self.name,
            agent_capabilities=self.capabilities,
            max_papers=5,
        )

        print(f"发现 {len(papers)} 篇需要帮助的论文")

        # 2. 逐个处理
        for paper in papers:
            paper_id = paper["paper_id"]

            # 3. 获取详细信息
            info = self.api.get_paper_info(paper_id)

            # 4. 分析论文
            analysis = self.analyze_paper(info)

            # 5. 提供指导
            self.api.submit_guidance(
                agent_name=self.name,
                paper_id=paper_id,
                **analysis
            )

            print(f"✅ 已处理: {paper_id}")

    def analyze_paper(self, paper_info: Dict) -> Dict:
        """分析论文（实现自己的逻辑）"""
        # 读取LaTeX文件
        latex_files = paper_info['files'].get('latex', [])
        # ... 分析逻辑 ...

        return {
            "comment": "我的评论",
            "score": 4.0,
            "issues": ["问题1"],
            "suggestions": ["建议1"],
            "priority": "medium",
        }

# 使用
agent = MyAgent()
agent.find_and_review_papers()
```

### 自动化Agent指导系统

```python
import asyncio
from ai_scientist.paper_metadata import AgentGuidanceAPI

class AutomatedGuidanceSystem:
    """自动化指导系统"""

    def __init__(self):
        self.api = AgentGuidanceAPI()
        self.agents = []

    def register_agent(self, agent):
        """注册Agent"""
        self.agents.append(agent)

    async def run_guidance_cycle(self):
        """运行指导周期"""

        # 1. 每个Agent发现论文
        all_papers = {}
        for agent in self.agents:
            papers = self.api.discover_papers(
                agent_name=agent.name,
                agent_capabilities=agent.capabilities,
            )

            for paper in papers:
                if paper["paper_id"] not in all_papers:
                    all_papers[paper["paper_id"]] = {
                        "paper": paper,
                        "agents": []
                    }
                all_papers[paper["paper_id"]]["agents"].append(agent.name)

        # 2. 并行处理
        tasks = []
        for paper_id, data in all_papers.items():
            for agent in self.agents:
                if agent.name in data["agents"]:
                    task = agent.review_paper(data["paper"])
                    tasks.append(task)

        results = await asyncio.gather(*tasks)

        print(f"✅ 处理完成: {len(results)} 条指导")

# 使用
guidance_system = AutomatedGuidanceSystem()
guidance_system.register_agent(WritingCriticAgent())
guidance_system.register_agent(TechnicalReviewerAgent())

await guidance_system.run_guidance_cycle()
```

## 📊 元数据文件详解

### .paper_metadata.json

```json
{
  "paper_id": "paper_20240223_120000_my_idea_neurips",
  "created_at": "2024-02-23T12:00:00",
  "updated_at": "2024-02-23T12:30:00",
  "paper_type": "neurips",
  "idea_name": "my_idea",
  "title": "My Paper Title",
  "abstract": "Paper abstract...",
  "authors": ["AI Scientist"],
  "keywords": ["machine learning", "optimization"],
  "field": "Computer Vision",
  "task": "Image Classification"
}
```

### .status.json

```json
{
  "current_status": "draft",
  "review_status": "agent_reviewed",
  "quality_level": "good",
  "overall_score": 3.8,
  "last_updated": "2024-02-23T12:30:00",
  "history": [
    {
      "from": "generating",
      "to": "draft",
      "timestamp": "2024-02-23T12:00:00",
      "note": "初稿完成"
    }
  ]
}
```

### .agent_comments.json

```json
[
  {
    "agent_name": "WritingCriticAgent",
    "agent_type": "writing_critique",
    "comment": "Introduction需要改进",
    "score": 3.5,
    "issues": ["过长", "不够清晰"],
    "suggestions": ["精简内容", "突出贡献"],
    "priority": "high",
    "timestamp": "2024-02-23T12:15:00",
    "addressed": false
  }
]
```

## 🎯 Agent能力类型

推荐的Agent能力分类：

```python
# 写作相关
"writing_critique"      # 写作批评
"style_guidance"        # 风格指导
"grammar_check"         # 语法检查
"clarity_analysis"      # 清晰度分析

# 技术相关
"technical_review"      # 技术审查
"methodology_advice"    # 方法学建议
"experiment_design"     # 实验设计
"statistical_analysis"  # 统计分析

# 内容相关
"literature_review"    # 文献综述
"related_work"         # 相关工作
"background_knowledge" # 背景知识

# 领域相关
"domain_expertise"     # 领域专长
"novelty_assessment"   # 新颖性评估
"impact_evaluation"    # 影响力评估
```

## 📈 监控和报告

### 查看全局状态

```python
from ai_scientist.continuous_paper_generator import ContinuousPaperGenerator

generator = ContinuousPaperGenerator(enable_learning=True)

# 1. 查看学习状态
generator.show_learning_status()

# 2. 查看进化状态
evolution_status = generator.get_evolution_status()

# 3. 查看指导报告
guidance_report = generator.get_guidance_report()

print(f"总论文数: {guidance_report['paper_registry']['total_papers']}")
print(f"需要审查: {guidance_report['paper_registry']['papers_needing_review']}")
print(f"需要改进: {guidance_report['paper_registry']['papers_needing_improvement']}")
print(f"未处理评论: {guidance_report['unaddressed_comments']['total']}")
```

### Agent贡献统计

```python
# 获取所有Agent统计
agent_stats = guidance_report['agent_statistics']

for agent_name, stats in agent_stats['agents'].items():
    print(f"\n{agent_name}:")
    print(f"  审查论文: {len(stats['papers_reviewed'])}")
    print(f"  贡献次数: {stats['total_contributions']}")
    print(f"  能力: {', '.join(stats['capabilities'])}")
```

## 🔍 快速开始

### 为现有论文添加标记

```python
from ai_scientist.paper_metadata import create_paper_metadata, create_standardized_markers

# 为现有论文创建标记
paper_dir = "./research_output/paper_20240223_120000_idea"

# 1. 创建元数据
metadata = create_paper_metadata(
    paper_dir=paper_dir,
    idea={
        "Name": "my_idea",
        "Title": "My Paper",
        "Abstract": "Abstract...",
        "Field": "ML",
        "Task": "Classification",
    },
    paper_type="neurips",
)

# 2. 创建标记文件
create_standardized_markers(paper_dir)

# 3. 设置状态
metadata.set_status(PaperStatus.DRAFT, "初稿完成")
metadata.set_quality(3.8)

print("✅ 标记创建完成")
```

### 查看论文README

```bash
# 直接查看
cat /path/to/paper/README.md

# 或在Python中
from ai_scientist.paper_metadata import PaperMetadata

metadata = PaperMetadata("/path/to/paper")
print(metadata.get_status_summary())
```

## 💡 最佳实践

### 1. 及时更新状态

```python
# 在不同阶段更新状态
metadata.set_status(PaperStatus.GENERATING, "开始生成")
# ... 生成过程 ...
metadata.set_status(PaperStatus.DRAFT, "初稿完成")
# ... 审查过程 ...
metadata.set_status(PaperStatus.IMPROVING, "根据反馈改进")
# ... 完成 ...
metadata.set_status(PaperStatus.COMPLETED, "最终版本完成")
```

### 2. 详细记录Agent评论

```python
# 提供详细的评论
metadata.add_agent_comment(
    agent_name="WritingCriticAgent",
    agent_type="writing_critique",
    comment="Introduction部分虽然涵盖了相关工作，但缺乏对本工作贡献的突出说明。",
    score=3.5,
    issues=[
        "Introduction过长（2.5页）",
        "贡献声明不够清晰",
        "缺少具体的技术路线图",
    ],
    suggestions=[
        "精简Introduction到1.5页",
        "在Introduction末尾添加bullet points列出3个主要贡献",
        "添加Figure 1展示整体架构",
    ],
    priority="high",
)
```

### 3. 定期清理和更新

```python
from ai_scientist.paper_metadata import MetadataRegistry

registry = MetadataRegistry()

# 清理旧论文
registry.prune_old_knowledge(days=180)

# 更新注册表
papers = registry.get_papers_needing_improvement()
print(f"有 {len(papers)} 篇论文需要改进")
```

## 📚 相关文档

- [AUTONOMOUS_EVOLUTION_README.md](AUTONOMOUS_EVOLUTION_README.md) - 自主进化系统
- [ADAPTIVE_LEARNING_README.md](ADAPTIVE_LEARNING_README.md) - 自适应学习
- [agent_interface.py](ai_scientist/agent_interface.py) - Agent接口规范

## 🎉 总结

论文元数据标记系统提供了：

1. ✅ **标准化标记**
   - 统一的文件结构
   - 清晰的状态定义
   - 易于理解的README

2. ✅ **Agent发现机制**
   - 自动发现需要帮助的论文
   - 基于能力匹配
   - 优先级排序

3. ✅ **指导接口**
   - 简单的API
   - 标准化评论格式
   - 可追踪的贡献

4. ✅ **状态管理**
   - 完整的历史记录
   - 多维度评估
   - 进度跟踪

现在外部Agent可以轻松地：
- 🔍 **发现**需要帮助的论文
- 📖 **了解**论文的当前状态
- 💬 **提供**标准化指导
- 📊 **追踪**自己的贡献
