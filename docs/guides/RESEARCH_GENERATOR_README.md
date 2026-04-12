# AI Scientist 连续论文生成系统

## 📖 概述

这是一个优化的 AI Scientist 系统，支持**连续产生论文**，不仅限于 workshop 格式。所有生成的文件都统一存放在 `./research_output` 目录下。

## ✨ 主要特性

### 1. **多类型论文支持**
- **ICBINB (Workshop)**: 4页 workshop 论文
- **Normal**: 8页标准会议论文
- **Journal**: 12页期刊论文
- **Extended Abstract**: 2页扩展摘要

### 2. **统一输出目录**
所有文件统一输出到 `./research_output`，包含以下结构：

```
./research_output/
├── cache/              # 缓存文件
├── ideas/              # 生成的想法
├── experiments/        # 实验结果
├── papers/             # 每篇论文独立文件夹
│   └── paper_YYYYMMDD_HHMMSS_idea_name_type/  # 单篇论文完整文件夹
│       ├── idea.json           # 想法
│       ├── idea.md             # 想法描述
│       ├── experiment/         # 实验代码和结果
│       ├── latex/              # LaTeX源文件
│       ├── paper.pdf           # 最终PDF
│       ├── reviews/            # 审查结果
│       │   └── round_1/        # 第1轮审查
│       │       ├── review_text.json
│       │       └── review_img.json
│       └── logs/               # 日志
└── batches/            # 批次管理
    └── batch_YYYYMMDD_HHMMSS/  # 批次记录
        ├── ideas/              # 本批次想法
        ├── progress.json       # 进度跟踪
        └── final_report.json   # 最终报告
```

### 3. **连续生成能力**
- 支持一次性生成多个想法
- 为每个想法生成多种类型的论文
- 支持并行处理提升效率
- 自动进度跟踪和断点续传

### 4. **研究管理工具**
提供命令行工具管理研究目录：
- 列出所有批次和论文
- 搜索论文内容
- 清理旧文件
- 查看统计信息

## 🚀 快速开始

### 1. 环境设置

```bash
# 设置智谱API密钥
export ZHIPU_API_KEY="your_api_key_here"

# 进入项目目录
cd <repo_root>
```

### 2. 基础用法

#### 生成想法并创建workshop论文

```bash
python continuous_paper_generator.py \
  --topic my_research_topic.md \
  --num-ideas 5 \
  --paper-types icbinb
```

#### 生成所有类型的论文

```bash
python continuous_paper_generator.py \
  --topic my_research_topic.md \
  --num-ideas 3 \
  --all-types
```

#### 从已有想法生成

```bash
python continuous_paper_generator.py \
  --ideas existing_ideas.json \
  --paper-types normal journal
```

### 3. 并行处理

```bash
# 使用2个worker并行处理
python continuous_paper_generator.py \
  --topic my_research_topic.md \
  --all-types \
  --num-workers 2
```

### 4. 选择特定想法

```bash
# 只处理第0、1、2个想法
python continuous_paper_generator.py \
  --ideas my_ideas.json \
  --all-types \
  --idea-indices 0,1,2
```

## 📋 研究管理工具使用

### 列出所有批次

```bash
python research_manager.py list-batches
```

### 查看批次详情

```bash
python research_manager.py batch-summary 20240101_120000
```

### 列出所有论文

```bash
# 所有论文
python research_manager.py list-papers

# 特定类型
python research_manager.py list-papers --type normal

# 显示详细信息
python research_manager.py list-papers --detailed
```

### 查看论文详情

```bash
# 查看特定论文文件夹的详细信息
python research_manager.py paper-details paper_20240101_120000_my_idea_icbinb
```

### 搜索论文

```bash
python research_manager.py search-papers "transformer"
```

### 查看统计信息

```bash
python research_manager.py stats
```

### 清理旧文件

```bash
# 预览30天前的文件
python research_manager.py cleanup --days 30 --dry-run

# 实际删除
python research_manager.py cleanup --days 30
```

## ⚙️ 配置选项

### 论文类型配置

论文类型在 `ai_scientist/config/paths.py` 中定义：

```python
PAPER_TYPES = {
    "icbinb": {
        "name": "ICLR Workshop (ICBINB)",
        "page_limit": 4,
        "template": "blank_icbinb_latex",
        "description": "4页 workshop 论文"
    },
    "normal": {
        "name": "Standard Conference Paper",
        "page_limit": 8,
        "template": "blank_icml_latex",
        "description": "8页标准会议论文"
    },
    "journal": {
        "name": "Journal Paper",
        "page_limit": 12,
        "template": "blank_icml_latex",
        "description": "12页期刊论文"
    },
    "extended": {
        "name": "Extended Abstract",
        "page_limit": 2,
        "template": "blank_icbinb_latex",
        "description": "2页扩展摘要"
    }
}
```

### 自定义输出目录

通过环境变量自定义输出目录：

```bash
export RESEARCH_OUTPUT_DIR="/path/to/your/research"
python continuous_paper_generator.py --topic topic.md
```

## 🔧 高级用法

### 1. 创建研究批次

```bash
python continuous_paper_generator.py \
  --topic deep_learning.md \
  --num-ideas 10 \
  --all-types \
  --batch-name "deep_learning_batch_001" \
  --num-workers 2
```

### 2. 多轮改进

```bash
python continuous_paper_generator.py \
  --ideas ideas.json \
  --paper-types normal \
  --improvement-rounds 3
```

### 3. 自定义模型选择

```bash
python continuous_paper_generator.py \
  --topic topic.md \
  --model-ideation glm-4-flash \
  --model-writeup glm-4-plus \
  --model-citation glm-4-air \
  --model-review glm-4-plus
```

### 4. 调整引用检索

```bash
python continuous_paper_generator.py \
  --topic topic.md \
  --num-cite-rounds 20
```

## 📊 进度跟踪

每个批次都有进度跟踪文件 `progress.json`：

```json
{
  "batch_name": "20240101_120000",
  "started_at": "2024-01-01T12:00:00",
  "papers_generated": [],
  "papers_completed": [],
  "papers_failed": [],
  "current_stage": "generating_ideas"
}
```

完成后的最终报告 `final_report.json`：

```json
{
  "batch_name": "20240101_120000",
  "statistics": {
    "total_papers": 12,
    "completed": 10,
    "failed": 2,
    "by_type": {
      "icbinb": {"completed": 3, "failed": 0},
      "normal": {"completed": 3, "failed": 1},
      "journal": {"completed": 2, "failed": 1},
      "extended": {"completed": 2, "failed": 0}
    }
  }
}
```

## 🛠️ 故障排除

### 问题：PDF生成失败

```bash
# 检查LaTeX安装
which pdflatex

# 增加重试次数
python continuous_paper_generator.py \
  --topic topic.md \
  --writeup-retries 5
```

### 问题：内存不足

```bash
# 减少并行worker数量
python continuous_paper_generator.py \
  --topic topic.md \
  --num-workers 1
```

### 问题：API限流

```bash
# 使用更小的模型
python continuous_paper_generator.py \
  --topic topic.md \
  --model-writeup glm-4-air \
  --model-citation glm-4-flash
```

## 📝 示例工作流

### 工作流1：快速生成workshop论文

```bash
# 1. 创建主题文件
cat > my_topic.md << EOF
I want to explore novel attention mechanisms for transformer models.
Focus on efficiency improvements while maintaining performance.
EOF

# 2. 生成论文
python continuous_paper_generator.py \
  --topic my_topic.md \
  --num-ideas 3 \
  --paper-types icbinb

# 3. 查看结果
python research_manager.py list-papers --type icbinb
```

### 工作流2：完整研究流程

```bash
# 1. 生成想法
python continuous_paper_generator.py \
  --topic my_research_area.md \
  --num-ideas 5 \
  --batch-name "research_001"

# 2. 为所有想法生成各种类型论文
python continuous_paper_generator.py \
  --ideas batches/batch_research_001/ideas/generated_ideas.json \
  --all-types \
  --improvement-rounds 2

# 3. 查看批次摘要
python research_manager.py batch-summary research_001

# 4. 搜索相关论文
python research_manager.py search-papers "attention"
```

## 📂 文件组织

### 输入文件
- `topic.md`: 研究主题描述
- `ideas.json`: 预生成的想法JSON

### 输出文件
- `papers/`: 最终生成的PDF论文
- `batches/batch_*/`: 批次目录，包含所有中间文件
- `batches/batch_*/progress.json`: 实时进度
- `batches/batch_*/final_report.json`: 最终报告

## 🔗 与原系统的兼容性

本系统兼容原有的 ai_scientist 功能：

```bash
# 使用原有的启动脚本
python launch_scientist_zhipu.py \
  --load_ideas ideas.json \
  --writeup-type normal

# 新系统会自动使用统一的输出目录
```

## 📚 相关文件

- `continuous_paper_generator.py`: 连续论文生成器
- `research_manager.py`: 研究管理工具
- `ai_scientist/config/paths.py`: 路径配置
- `launch_scientist_zhipu.py`: 原有启动脚本（已更新）
- `run_project.py`: 项目管理脚本（已更新）

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可

与 ai_scientist 项目相同
