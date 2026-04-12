# 专业学术写作系统使用指南

## 📋 概述

本系统提供了顶会级别的论文写作能力，通过结构化、专业化的写作流程，大幅提升论文质量。

## 🎯 主要特性

### 1. 多会议论文模板

系统为顶级会议和期刊提供了详细的论文模板：

- **NeurIPS**: 理论机器学习会议，8页
- **ICLR**: 表示学习会议，8页
- **CVPR**: 计算机视觉会议，8页
- **ICBINB**: ICLR Workshop，4页
- **Journal**: 期刊论文，12页

每个模板包含：
- 详细的章节结构
- 每个章节的写作要点
- 篇幅分配建议
- 写作技巧提示

### 2. 专业章节写作器

`ExpertSectionWriter` 提供章节级的专业写作能力：

```python
from ai_scientist.professional_writing_system import ExpertSectionWriter

# 创建写作器
writer = ExpertSectionWriter(template="neurips", model="claude-3-5-sonnet")

# 生成详细大纲
outline = writer.generate_detailed_outline(idea, experiment_results)

# 编写单个章节
section_content = writer.write_section(
    section_name="introduction",
    idea=idea,
    context=context,
)

# 改进章节
improved_content = writer.refine_section(
    section_name="introduction",
    current_content=section_content,
    feedback="改进建议",
    context=context,
)

# 编写完整论文
full_paper = writer.write_full_paper(
    idea=idea,
    experiment_results=experiment_results,
    iterative=True,  # 迭代式写作
)
```

### 3. 专业论文评估器

`ProfessionalPaperEvaluator` 提供多维度质量评估：

```python
from ai_scientist.professional_writing_system import ProfessionalPaperEvaluator

# 创建评估器
evaluator = ProfessionalPaperEvaluator(template="neurips", model="gpt-4o")

# 评估论文质量
evaluation = evaluator.evaluate_paper_quality(
    paper_content=full_paper,
    idea=idea,
)

# 查看评估结果
print(f"总分: {evaluation['overall']['score']}/5")
print(f"等级: {evaluation['overall']['level']}")
print(f"优点: {evaluation['overall']['strengths']}")
print(f"弱点: {evaluation['overall']['weaknesses']}")
```

## 🚀 使用方法

### 方法1: 使用连续论文生成器（推荐）

```bash
# 使用专业写作系统生成论文
python continuous_paper_generator.py \
  --topic my_topic.md \
  --paper-types neurips iclr \
  --improvement-rounds 3
```

### 方法2: 直接使用专业写作API

```python
from ai_scientist.continuous_paper_generator import ContinuousPaperGenerator

# 创建生成器
generator = ContinuousPaperGenerator(
    research_dir="./research_output",
)

# 生成想法
ideas_json = generator.generate_ideas(
    topic_file="my_topic.md",
    num_ideas=3,
)

# 加载想法
with open(ideas_json, "r") as f:
    ideas = json.load(f)

# 使用专业写作系统
for idea in ideas:
    result = generator.generate_paper_with_professional_writing(
        idea=idea,
        paper_type="neurips",
        experiment_results=None,
        model="claude-3-5-sonnet",
        enable_evaluation=True,
    )

    if result["status"] == "success":
        print(f"✅ 论文生成成功: {result['latex_path']}")
        print(f"质量评分: {result['evaluation']['overall']['score']:.1f}/5")
```

### 方法3: 单独使用专业写作组件

```python
from ai_scientist.professional_writing_system import (
    ExpertSectionWriter,
    ProfessionalPaperEvaluator,
)

# 写作
writer = ExpertSectionWriter(template="neurips")
full_paper = writer.write_full_paper(idea, experiment_results)

# 评估
evaluator = ProfessionalPaperEvaluator(template="neurips")
evaluation = evaluator.evaluate_paper_quality(full_paper, idea)

# 根据评估改进
for section in evaluation['weaknesses']:
    improved = writer.refine_section(
        section_name=section,
        current_content=...,
        feedback=evaluation['recommendations'],
        context=...,
    )
```

## 📊 论文模板详解

### NeurIPS 模板

**特点**: 理论性强，重视方法创新

**章节结构**:
1. Abstract (150-250 words)
2. Introduction (1-1.5 pages)
3. Related Work (1-1.5 pages)
4. Method (2-3 pages)
5. Experiments (2-3 pages)
6. Results (1-1.5 pages)
7. Discussion (0.5-1 page)
8. Conclusion (0.25-0.5 page) - **必须包含Broader Impact**

**写作要点**:
- 强调理论贡献
- 清晰的数学表述
- 严格的实验验证
- 包含更广泛影响声明

### ICLR 模板

**特点**: 表示学习，重视学习机制

**章节结构**:
1. Abstract (150-250 words)
2. Introduction (1-1.5 pages)
3. Background (0.5-1 page)
4. Method (2.5-3.5 pages) - **重点**
5. Experiments (2-3 pages)
6. Related Work (1 page)
7. Conclusion (0.5 page)

**写作要点**:
- 重视表示学习的理论分析
- 强调学习机制的创新
- 包含收敛性或复杂度分析
- 分析学习到的表示

### CVPR 模板

**特点**: 视觉任务，重视实际效果

**章节结构**:
1. Abstract (150-250 words)
2. Introduction (1 page)
3. Related Work (1-1.5 pages)
4. Method (2.5-3.5 pages)
5. Experiments (2-3 pages) - **包含定性可视化**
6. Conclusion (0.25 page)

**写作要点**:
- 包含架构图
- 详细的层规范
- 定性和定量结果
- 计算效率分析
- 多数据集验证

### Journal 模板

**特点**: 全面深入，可复现性强

**章节结构**:
1. Abstract (200-300 words)
2. Introduction (2-3 pages)
3. Related Work (2-3 pages)
4. Background (1-2 pages)
5. Method (3-4 pages)
6. Experiments (3-4 pages)
7. Discussion (1-1.5 pages)
8. Conclusion (0.5 page)

**写作要点**:
- 全面的文献综述
- 自包含的背景介绍
- 严谨的理论分析
- 彻底的实验验证
- 可复现性细节

## 📈 学术写作标准

### 形式语言规范

**使用**:
- 精确的技术术语
- 完整的句子
- 主动语态以增加清晰度
- 第三人称或第一人称复数 ("we")
- 现在时描述通用真理
- 过去时描述方法和结果

**避免**:
- 口语表达
- 没有证据的主观语言
- 过度陈述 ("always", "never", "prove")
- 第一人称单数 ("I")
- 反问句
- 情感化语言

### 段落结构

1. **主题句**: 引入主要观点
2. **支撑句**: 提供证据和论据
3. **分析/解释**: 解释证据如何支持观点
4. **过渡/总结句**: 连接到下一段

### 逻辑流畅性

- 清晰的问题-解决方案结构
- 连贯的论证进展
- 明确的选择理由

### 技术写作规范

**数学表达**:
- 在使用前定义所有符号
- 对重要方程进行编号
- 解释方程的含义
- 检查量纲一致性

**图表**:
- 信息丰富的标题
- 清晰的标签和图例
- 高分辨率
- 在正文中引用

**算法**:
- 清晰的伪代码
- 一致的符号
- 在文中解释

## 🎯 写作质量评估

### 评估维度

1. **结构** (Structure)
   - 符合会议/期刊的章节结构
   - 各章节比例合理
   - 逻辑流程清晰
   - 过渡自然

2. **内容** (Content)
   - 内容全面覆盖主题
   - 技术细节充分
   - 论证有力
   - 分析深入

3. **创新性** (Innovation)
   - 创新点明确
   - 与现有工作的区别清晰
   - 贡献具有重要性
   - 方法新颖

4. **严谨性** (Rigor)
   - 实验设计合理
   - 基线选择恰当
   - 统计分析正确
   - 可复现性强

5. **清晰度** (Clarity)
   - 语言简洁明了
   - 术语定义清楚
   - 数学表达规范
   - 图表清晰易懂

6. **专业性** (Professionalism)
   - 格式规范
   - 引用正确完整
   - 无语法错误
   - 学术语言得体

### 质量等级

- **4.5-5.0**: 优秀 (Excellent) - 顶级会议水平
- **4.0-4.5**: 良好 (Very Good) - 可能被接收
- **3.5-4.0**: 中等 (Good) - 需要 minor revision
- **3.0-3.5**: 一般 (Fair) - 需要 major revision
- **<3.0**: 较差 (Poor) - 需要重大改进

## 🔄 与现有系统集成

### 与自动改进系统集成

```python
# 先用专业写作系统生成初稿
result = generator.generate_paper_with_professional_writing(...)

# 然后进行自动改进
improved_result = improve_paper_with_review(
    paper_dir=result["paper_dir"],
    text_review=...,
    img_review=...,
    model="claude-3-5-sonnet",
)
```

### 与创新评估系统集成

```python
from ai_scientist.innovation_enhancer import InnovationEvaluator

# 先评估创新性
innovation_eval = InnovationEvaluator()
novelty = innovation_eval.evaluate_idea_novelty(idea)

# 根据创新性选择合适的模板
template = recommend_template(idea)

# 使用推荐的模板写作
writer = ExpertSectionWriter(template=template)
```

## 📝 最佳实践

### 1. 选择合适的模板

根据研究内容选择最合适的会议/期刊：

```python
from ai_scientist.professional_writing_system import recommend_template

# 自动推荐
template = recommend_template(idea)
print(f"推荐模板: {template}")
```

### 2. 迭代式改进

```python
# 第一轮：生成初稿
paper = writer.write_full_paper(idea, experiment_results)

# 评估质量
evaluation = evaluator.evaluate_paper_quality(paper, idea)

# 根据反馈改进
for section_name, feedback in evaluation["recommendations"].items():
    improved_section = writer.refine_section(
        section_name=section_name,
        current_content=...,
        feedback=feedback,
        context=...,
    )
```

### 3. 章节并行写作

```python
# 对于较长的论文，可以并行写作多个章节
from concurrent.futures import ThreadPoolExecutor

sections = ["introduction", "method", "experiments"]

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(
            writer.write_section,
            section_name=section,
            idea=idea,
            context=context,
        ): section
        for section in sections
    }

    results = {futures[future]: future.result() for future in as_completed(futures)}
```

## 🐛 故障排除

### 常见问题

1. **模板不匹配**
   - 症状：生成的论文结构不符合预期
   - 解决：检查`paper_type`参数是否正确

2. **质量评估过低**
   - 症状：评估分数低于3.0
   - 解决：
     - 检查实验结果是否充分
     - 使用`refine_section`方法改进薄弱章节
     - 增加更多基线对比

3. **LaTeX编译错误**
   - 症状：生成的LaTeX无法编译
   - 解决：
     - 检查LaTeX语法
     - 确保所有包都正确引用
     - 验证图片路径

## 📚 相关文档

- [DEEP_OPTIMIZATION_README.md](DEEP_OPTIMIZATION_README.md) - 深度优化总结
- [AUTO_IMPROVEMENT_README.md](AUTO_IMPROVEMENT_README.md) - 自动改进系统
- [RESEARCH_GENERATOR_README.md](RESEARCH_GENERATOR_README.md) - 研究生成器

## 🎉 总结

专业学术写作系统提供了：

1. ✅ **多会议模板支持**
   - NeurIPS, ICLR, CVPR, Journal等
   - 详细的章节结构和写作指南

2. ✅ **章节级专业写作**
   - 迭代式写作流程
   - 基于上下文的连贯性
   - 支持反馈改进

3. ✅ **多维度质量评估**
   - 6个维度全面评估
   - 详细的优缺点分析
   - 具体的改进建议

4. ✅ **学术写作标准**
   - 形式语言规范
   - 段落结构指导
   - 技术写作要求

5. ✅ **无缝集成**
   - 与自动改进系统集成
   - 与创新评估系统集成
   - 支持连续论文生成

现在你可以生成**顶会级别**的高质量学术论文了！
