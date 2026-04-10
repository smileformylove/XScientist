# 自适应学习系统使用指南

## 📋 概述

自适应学习系统使 AI Scientist 能够从过去的论文生成经验中学习，不断优化写作策略、审查方法和改进技巧。随着使用时间的增长，系统会变得越来越智能。

## 🎯 核心功能

### 1. 自学习知识库 (SelfLearningKnowledgeBase)

中央知识存储库，存储和管理所有历史经验：

```python
from ai_scientist.self_learning_knowledge_base import SelfLearningKnowledgeBase

# 初始化知识库
kb = SelfLearningKnowledgeBase(research_dir="./research_output")

# 存储论文经验
kb.store_paper_experience(
    paper_data=paper_data,
    outcome="accepted",  # accepted, rejected, minor_revision, major_revision
    final_scores={"structure": 4.5, "content": 4.0, ...},
    reviews=[...],
    improvements=[...],
)

# 查找相似论文
similar = kb.find_similar_papers(
    current_idea=idea,
    paper_type="neurips",
    top_k=5,
)

# 获取有效策略
effective = kb.get_effective_strategies(
    issue_type="structure",
    min_success_rate=0.7,
)

# 生成学习摘要
summary = kb.generate_learning_summary()
```

### 2. 模式分析器 (PatternAnalyzer)

分析成功和失败模式，提取洞察：

```python
from ai_scientist.self_learning_knowledge_base import PatternAnalyzer

analyzer = PatternAnalyzer(knowledge_base)

# 分析成功因素
success_factors = analyzer.analyze_success_factors()

# 预测成功概率
prob = analyzer.predict_success_probability(paper_data)
```

### 3. 自适应学习引擎 (AdaptiveLearningEngine)

根据历史经验推荐和调整策略：

```python
from ai_scientist.adaptive_learning_engine import AdaptiveLearningEngine

engine = AdaptiveLearningEngine(knowledge_base)

# 推荐策略
recommendation = engine.recommend_strategy(
    idea=idea,
    paper_type="neurips",
    context={...},
)

# 推荐结果里现在还会包含 reviewer 修复闭环沉淀出的跨项目指引
print(recommendation["self_evolution_guidance"])

# 调整写作提示
adapted_prompt = engine.adapt_writing_prompt(
    base_prompt=base_prompt,
    recommendation=recommendation,
    section="introduction",
)

# 从生成中学习
insights = engine.learn_from_generation(
    idea=idea,
    paper_data=paper_data,
    outcome="accepted",
    reviews=[...],
    improvements=[...],
    final_scores={...},
)

# 生成自适应改进计划
plan = engine.generate_adaptive_improvement_plan(
    current_review=review,
    recommendation=recommendation,
)
```

### 4. 自适应写作器 (AdaptiveWriter)

应用学习到的模式进行写作：

```python
from ai_scientist.adaptive_learning_engine import AdaptiveWriter

writer = AdaptiveWriter(learning_engine, model="claude-3-5-sonnet")

# 使用学习到的模式写作
content = writer.write_with_learning(
    section="introduction",
    idea=idea,
    paper_type="neurips",
    context=context,
)
```

## 🚀 使用方法

### 方法1: 启用自适应学习的连续生成器

```python
from ai_scientist.continuous_paper_generator import ContinuousPaperGenerator

# 创建启用学习的生成器
generator = ContinuousPaperGenerator(
    research_dir="./research_output",
    enable_learning=True,  # 启用自适应学习
)

# 使用自适应学习生成论文
result = generator.generate_paper_with_adaptive_learning(
    idea=idea,
    paper_type="neurips",
    experiment_results=results,
    enable_evaluation=True,
    learn_from_result=True,  # 从结果中学习
)

# 查看学习状态
generator.show_learning_status()
```

### 方法2: 命令行使用

```bash
# 创建简单的测试脚本
python test_adaptive_learning.py
```

## 📊 知识库结构

```
./research_output/knowledge_base/
├── success_patterns.json       # 成功模式存储
├── failure_patterns.json       # 失败模式存储
├── improvement_strategies.json # 改进策略效果
├── review_insights.json        # 审查洞察
├── writing_insights.json       # 写作洞察
├── self_evolution_history.jsonl   # reviewer 修复闭环历史快照
└── self_evolution_playbook.json   # 跨项目自我进化 playbook
```

## 🔍 学习机制

### 1. 模式存储

每次论文生成后，系统会存储：

- **成功模式**: 接受/小修的论文
  - 论文信息（标题、摘要、方法）
  - 评分细节
  - 审查反馈
  - 改进记录

- **失败模式**: 拒稿/大修的论文
  - 同样的信息结构
  - 用于学习避免错误

### 2. 相似性匹配

使用 TF-IDF 向量化计算论文相似度：

```python
# 自动查找相似的成功论文
similar_papers = kb.find_similar_papers(
    current_idea=idea,
    paper_type="neurips",
    top_k=5,
)
```

返回结果包含：
- 相似论文的详细信息
- 相似度分数 (0-1)
- 该论文的结果和评分

### 3. 策略学习

系统追踪每种改进策略的效果：

```json
{
  "rewrite_content_structure": {
    "strategy": "rewrite_content_structure",
    "issue_type": "structure",
    "success_count": 15,
    "total_count": 20,
    "avg_improvement": 1.2,
    "success_rate": 0.75
  }
}
```

### 4. 分数阈值学习

系统学习各维度的成功阈值：

```python
thresholds = kb.get_score_thresholds()
# {
#   "structure": 3.8,
#   "content": 3.9,
#   "innovation": 4.0,
#   ...
# }
```

## 📈 自适应推荐

### 推荐策略包含：

1. **写作策略**
   - 强调章节（基于成功论文）
   - 写作技巧
   - 结构建议

2. **审查策略**
   - 策略名称（neurips/iclr/cvpr/journal）
   - 审查深度
   - 重点关注领域
   - 建议轮数

3. **改进策略**
   - 首选策略（高成功率）
   - 避免策略
   - 最大轮数
   - 改进阈值

4. **目标分数**
   - 各维度目标
   - 基于学习到的阈值

5. **常见陷阱**
   - 基于历史失败
   - 特定于论文类型

### 使用推荐

```python
# 获取推荐
recommendation = engine.recommend_strategy(idea, "neurips")

# 查看推荐内容
print(f"成功概率: {recommendation['success_probability']}")
print(f"推荐置信度: {recommendation['confidence']}")
print(f"相似论文: {len(recommendation['similar_papers'])}")

# 调整写作提示
adapted_prompt = engine.adapt_writing_prompt(
    base_prompt=original_prompt,
    recommendation=recommendation,
)
```

## 🎓 学习进展跟踪

### 查看学习状态

```python
generator.show_learning_status()
```

输出示例：
```
🧠 自适应学习系统状态
================================================================================

📊 知识库统计:
   总论文数: 50
   成功论文: 35
   失败论文: 15
   整体成功率: 70.0%

⚠️  常见问题 (Top 5):
   理论分析不够深入: 12 次
   实验验证不充分: 10 次
   相关工作缺失: 8 次
   写作不清晰: 7 次
   缺少更广泛影响: 6 次

✅ 有效策略 (Top 5):
   85.0% - rewrite_content_structure (20 次)
   80.0% - enhance_theoretical_analysis (15 次)
   75.0% - add_missing_experiments (12 次)
   70.0% - improve_clarity (10 次)
   68.0% - expand_related_work (8 次)

📈 分数阈值:
   structure: 3.8
   content: 3.9
   innovation: 4.0
   rigor: 3.7
   clarity: 4.0
   professionalism: 3.9

📁 知识库位置: ./research_output/knowledge_base
```

## 🔧 高级功能

### 1. 知识清理

定期清理旧知识以保持相关性：

```python
# 清理180天前的旧知识
kb.prune_old_knowledge(days=180)
```

### 2. 自适应改进计划

根据当前审查生成改进计划：

```python
plan = engine.generate_adaptive_improvement_plan(
    current_review=review,
    recommendation=recommendation,
)

# plan包含：
# - priority_improvements: 优先改进项
# - suggested_strategies: 建议策略
# - expected_improvement: 预期改进
# - rounds_needed: 需要轮数
```

### 3. 成功概率预测

```python
prob = analyzer.predict_success_probability(paper_data)
print(f"预测成功概率: {prob:.1%}")
```

## 💡 最佳实践

### 1. 持续使用

系统需要积累足够的经验才能发挥最大效果：

- **初期** (1-10篇): 开始收集模式
- **中期** (10-50篇): 模式识别增强
- **成熟期** (50+篇): 强大的预测和推荐能力

### 2. 定期查看学习状态

```python
# 定期检查
generator.show_learning_status()
```

### 3. 结合人工反馈

虽然系统能自动学习，但人工反馈可以加速学习：

```python
# 手动标注结果
kb.store_paper_experience(
    paper_data=...,
    outcome="accepted",  # 人工标注的实际结果
    ...
)
```

### 4. 领域适应性

系统会自动适应不同研究领域：

- 计算机视觉
- 自然语言处理
- 理论机器学习
- 等

## 🔄 集成工作流

### 完整的自适应学习流程

```python
# 1. 创建启用学习的生成器
generator = ContinuousPaperGenerator(enable_learning=True)

# 2. 加载想法
ideas = load_ideas("ideas.json")

# 3. 生成论文并学习
for idea in ideas:
    # 生成论文（系统会自动学习）
    result = generator.generate_paper_with_adaptive_learning(
        idea=idea,
        paper_type="neurips",
        learn_from_result=True,
    )

    # 系统已经自动存储经验

# 4. 查看学习进展
generator.show_learning_status()
```

### 与现有系统集成

```python
# 与自动改进系统集成
from ai_scientist.perform_auto_improvement import improve_paper_with_review

# 获取推荐
recommendation = engine.recommend_strategy(idea, "neurips")

# 生成自适应改进计划
improvement_plan = engine.generate_adaptive_improvement_plan(
    current_review=review,
    recommendation=recommendation,
)

# 应用改进
result = improve_paper_with_review(
    paper_dir=paper_dir,
    text_review=review,
    img_review=img_review,
    strategy=improvement_plan["suggested_strategies"],
)
```

## 📊 预期效果

随着时间推移，系统会：

1. **提高成功率**
   - 识别并避免常见错误
   - 应用成功模式

2. **减少迭代次数**
   - 更准确的初始生成
   - 更有效的改进策略

3. **提高论文质量**
   - 基于成功案例的写作
   - 避免常见陷阱

4. **领域专业化**
   - 学习领域特定模式
   - 适应不同会议风格

## 🐛 故障排除

### 问题1: 知识库为空

**症状**: 没有相似论文，推荐置信度低

**解决**:
- 生成更多论文以积累经验
- 或从已有论文导入经验

### 问题2: 学习效果不明显

**症状**: 成功率没有提升

**解决**:
- 确保启用了 `learn_from_result=True`
- 检查是否有足够的样本
- 考虑清理旧知识

### 问题3: 相似度匹配不准确

**症状**: 找到的论文不相关

**解决**:
- 这是正常现象，随着数据增多会改善
- 可以调整相似度阈值

## 📚 相关文档

- [PROFESSIONAL_WRITING_README.md](PROFESSIONAL_WRITING_README.md) - 专业写作系统
- [AUTO_IMPROVEMENT_README.md](AUTO_IMPROVEMENT_README.md) - 自动改进系统
- [DEEP_OPTIMIZATION_README.md](DEEP_OPTIMIZATION_README.md) - 深度优化总结

## 🎉 总结

自适应学习系统提供了：

1. ✅ **自学习知识库**
   - 存储成功/失败模式
   - 相似性匹配
   - 策略效果追踪

2. ✅ **智能推荐系统**
   - 写作策略推荐
   - 审查策略调整
   - 改进策略优化

3. ✅ **持续改进**
   - 从每次生成中学习
   - 预测成功概率
   - 识别常见陷阱

4. ✅ **无缝集成**
   - 与现有系统完全集成
   - 自动化学习流程
   - 透明的状态跟踪

现在 AI Scientist 不仅能生成论文，还能**持续学习和进化**，随着使用时间的增长变得越来越智能！
