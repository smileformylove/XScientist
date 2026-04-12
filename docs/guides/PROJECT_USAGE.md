# AI Scientist 项目管理器使用指南

## 🚀 简介

`run_project.py` 是一个增强版的项目管理脚本，支持：

- ✅ **并行处理多个论文** - 同时生成多篇论文
- ✅ **自动反思改进** - 根据审查意见自动修改论文
- ✅ **独立项目目录** - 所有输出在指定文件夹中
- ✅ **进度跟踪** - 实时保存执行进度

## 快速开始

### 0. 先登录（必需）

```bash
python3 auth_cli.py login --user <你的用户名>
python3 auth_cli.py status
```

### 1. 最简单的方式 - 生成1篇论文

```bash
python3 run_project.py my_research \
  --topic ai_scientist/ideas/i_cant_believe_its_not_better.md
```

### 2. 🌟 并行生成3篇论文 + 自动改进

```bash
python3 run_project.py my_research \
  --topic ai_scientist/ideas/i_cant_believe_its_not_better.md \
  --num-ideas 3 \
  --parallel \
  --num-workers 2 \
  --improvement-rounds 2
```

这会：
1. 生成3个研究想法
2. 并行运行3个实验（同时处理2个）
3. 每篇论文自动反思改进2轮
4. 输出3篇最终论文

### 3. 处理已有的想法

```bash
python3 run_project.py my_research \
  --ideas existing_ideas.json \
  --parallel \
  --improvement-rounds 3
```

### 4. 只处理特定想法

```bash
python3 run_project.py my_research \
  --ideas ideas.json \
  --idea-indices 0,2,4 \
  --parallel \
  --improvement-rounds 1
```

## 项目目录结构

```
my_research/
├── 01_ideas/                     # 生成的想法
│   └── generated_ideas.json
│
├── 02_experiments/               # 实验运行记录
│   ├── 20250125_143022_idea_1/
│   │   ├── idea.json
│   │   ├── logs/                # 实验日志
│   │   ├── plots/               # 图表
│   │   ├── reviews_round_1/     # 第1轮审查
│   │   ├── reviews_round_2/     # 第2轮审查
│   │   └── *.pdf                # 最终论文
│   ├── 20250125_143145_idea_2/
│   └── ...
│
├── 03_papers/                    # 最终论文汇总
│   ├── idea_1_final.pdf
│   ├── idea_2_final.pdf
│   └── idea_3_final.pdf
│
└── 04_logs/                      # 运行日志
    └── progress.json             # 实时进度
```

## 完整参数说明

### 项目设置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `project_dir` | 项目目录路径 | 必需 |

### 想法生成

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--topic` | 主题描述 Markdown 文件 | - |
| `--ideas` | 已有想法 JSON 文件 | - |
| `--model-ideation` | 想法生成模型 | `glm-4-flash` |
| `--num-ideas` | 生成想法数量 | `3` |
| `--num-reflections` | 每个想法反思轮数 | `5` |
| `--skip-ideation` | 跳过想法生成 | `False` |

### 并行处理 🆕

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--parallel` | 启用并行处理 | `False` |
| `--num-workers` | 并行worker数量 | `2` |
| `--idea-indices` | 要处理的想法索引（逗号分隔） | 全部 |

### 自动改进 🆕

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--improvement-rounds` | 每篇论文的反思改进轮数 | `1` |

改进流程：
1. LLM 审查论文（文本 + 图表）
2. 根据审查意见自动修改LaTeX
3. 重新编译PDF
4. 重复 N 轮

### 模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-writeup` | 论文写作主模型 | `glm-4-plus` |
| `--model-writeup-small` | 论文写作辅助模型 | `glm-4-air` |
| `--model-citation` | 文献检索模型 | `glm-4-air` |
| `--model-review` | 论文审查模型 | `glm-4-plus` |
| `--model-agg-plots` | 图表聚合模型 | `glm-4-flash` |

### 其他参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-cite-rounds` | 文献检索轮数 | `15` |
| `--writeup-retries` | 论文写作重试次数 | `3` |
| `--writeup-type` | 论文类型 | `icbinb` |
| `--skip-experiment` | 跳过实验 | `False` |
| `--override-strict-fallbacks` | 临时放宽严格兜底（idea ranking 或质量链路 fallback 时继续运行） | `False` |

## 使用场景

### 场景1: 快速生成单篇论文

```bash
python3 run_project.py quick_paper \
  --topic my_topic.md \
  --improvement-rounds 1
```

### 场景2: 批量生成论文（并行）

```bash
python3 run_project.py batch_papers \
  --topic my_topic.md \
  --num-ideas 5 \
  --parallel \
  --num-workers 3 \
  --improvement-rounds 2
```

预期时间：
- 单篇论文：~2-3小时
- 5篇论文（3个worker）：~3-5小时

### 场景3: 高质量论文（多轮改进）

```bash
python3 run_project.py high_quality \
  --topic my_topic.md \
  --num-ideas 1 \
  --improvement-rounds 5 \
  --num-cite-rounds 25
```

### 场景4: 从中断处继续

如果进程中断，可以查看 `04_logs/progress.json`，然后：

```bash
# 只处理剩余的想法
python3 run_project.py my_research \
  --skip-ideation \
  --idea-indices 2,3,4 \
  --parallel
```

## 工作流程详解

### 单篇论文流程

```
想法 #0
├── 步骤1: 运行实验 (BFTS树搜索)
│   └── 输出: 实验结果 + 图表
│
├── 步骤2: 生成论文
│   ├── 聚合图表
│   ├── 收集文献
│   └── LLM写作 LaTeX → PDF
│
├── 步骤3: 反思改进 (N轮)
│   ├── 第1轮:
│   │   ├── LLM审查论文
│   │   ├── LLM修改LaTeX
│   │   └── 重新编译PDF
│   ├── 第2轮:
│   │   └── ...
│   └── 第N轮
│
└── 步骤4: 最终审查
    ├── LLM文本审查
    ├── VLM图表审查
    └── 保存最终PDF
```

### 并行处理流程

```
想法生成 (串行)
    ↓
    ┌─────────────────────────┐
    │   Idea 0    Idea 1    Idea 2  │
    │      ↓         ↓         ↓      │
    │   实验 #0   实验 #1   实验 #2   │  ← 并行执行
    │      ↓         ↓         ↓      │
    │   论文 #0   论文 #1   论文 #2   │
    │      ↓         ↓         ↓      │
    │  改进 #0   改进 #1   改进 #2   │
    └─────────────────────────┘
                ↓
        最终PDF汇总
```

## 进度监控

查看实时进度：

```bash
# 查看进度文件
cat my_research/04_logs/progress.json

# 输出示例:
{
  "completed": 2,
  "total": 3,
  "results": [
    {"idea_idx": 0, "status": "success", "pdf_path": "..."},
    {"idea_idx": 1, "status": "success", "pdf_path": "..."}
  ]
}
```

## 注意事项

### 1. API 限流
并行处理会增加 API 调用频率，建议：
- 使用 `--num-workers` 控制并发数
- 监控 API 配额使用情况

### 2. 磁盘空间
每个想法约需 500MB-1GB，3个想法建议预留 3GB 空间

### 3. 运行时间
- 单个想法：2-3小时
- 3个想法（串行）：6-9小时
- 3个想法（2个worker并行）：3-5小时

### 4. 改进轮数
- `--improvement-rounds 1`: 快速，质量可接受
- `--improvement-rounds 2-3`: 推荐，质量较好
- `--improvement-rounds 5+`: 高质量，耗时较长

### 5. 严格兜底策略
- 当选择 `program_driven / writing_studio / review_board` workflow，或开启 `--submission-mode`/`--high-quality-mode`/`--target-venue journal|nature` 时，系统会默认启用严格兜底：若 idea ranking 或质量管线触发 fallback，`run_project.py` 会直接终止。其中质量 fallback 会落成 `quality_fallback_blocked` 这类明确失败阶段；idea ranking fallback 会在 ranking 阶段直接早停并输出严格兜底错误。
- 只有在调试阶段需要观测 fallback 行为时，才建议加上 `--override-strict-fallbacks` 放宽限制（fallback 仍会被记录并在看板中暴露）。

## 故障排除

### Q: 进程中断了怎么办？
A: 查看 `04_logs/progress.json`，使用 `--skip-ideation --idea-indices X,Y` 继续处理剩余的想法

### Q: 某个想法失败怎么办？
A: 其他想法会继续执行，失败的可以在日志中查看错误信息

### Q: 如何只重新生成某篇论文？
A: 删除对应的实验目录，然后使用 `--skip-ideation --idea-indices X` 重新运行

## 成本估算（智谱AI）

| 任务 | 模型 | 成本 (CNY) |
|------|------|-----------|
| 生成1个想法 | glm-4-flash | ¥0.5 |
| 运行实验 | glm-4-flash | ¥10-15 |
| 生成论文 | glm-4-plus | ¥3-5 |
| 文献检索 | glm-4-air | ¥2-3 |
| 审查改进 (1轮) | glm-4-plus | ¥2-3 |
| **单篇论文总计** | - | **¥20-30** |
| **3篇论文 (并行)** | - | **¥60-90** |

## 对比原脚本

| 特性 | `launch_scientist_zhipu.py` | `run_project.py` |
|------|----------------------------|------------------|
| 输出隔离 | ❌ 混在 experiments/ | ✅ 独立项目目录 |
| 多想法管理 | ❌ 困难 | ✅ 原生支持 |
| 并行处理 | ❌ 不支持 | ✅ 多进程并行 |
| 自动改进 | ❌ 不支持 | ✅ 多轮反思 |
| 进度跟踪 | ❌ 不支持 | ✅ 实时保存 |
| 断点续传 | ❌ 困难 | ✅ 易于恢复 |
