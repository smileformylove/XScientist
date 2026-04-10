# AI Scientist 输出目录优化说明

## 概述

本项目已进行优化，将所有运行时生成的文件夹统一管理到 `_outputs` 目录下，使项目根目录更加整洁。

## 目录结构

```
ai_scientist/
├── _outputs/               # 统一的输出目录（新增）
│   ├── cache/             # 缓存文件
│   ├── ideas/             # 生成的idea文件（统一存放）
│   ├── experiments/       # 实验运行结果
│   └── projects/          # 完整的项目文件夹
├── ai_scientist/          # 核心代码
│   └── config/           # 路径配置模块
├── launch_scientist_zhipu.py
├── launch_scientist_bfts.py
├── run_project.py
└── ...
```

### 各目录用途

| 目录 | 用途 | 示例 |
|------|------|------|
| `_outputs/cache/` | 缓存文件（如数据集、模型等） | - |
| `_outputs/ideas/` | **所有生成的idea JSON文件** | `my_idea.json`, `data_compression.json` |
| `_outputs/experiments/` | **运行实验的结果** | `20260127_205142_NeuroEntropy_attempt_0/` |
| `_outputs/projects/` | **完整的项目**（包含idea、实验、论文） | `my_project/` |

## 核心改进

### 1. Idea文件统一存放

**之前**: idea文件散落在各处
- `ai_scientist/ideas/my_idea.json`
- `experiments/.../idea.json`

**现在**: 所有idea统一存放在 `_outputs/ideas/`
- `_outputs/ideas/my_idea.json`
- `_outputs/ideas/data_compression.json`

### 2. 实验结果统一存放

**之前**: `experiments/` 在项目根目录

**现在**: `_outputs/experiments/` 清晰分离

### 3. 完整项目统一存放

**之前**: 项目文件夹可能散落在各处

**现在**: 所有项目在 `_outputs/projects/` 下

## 修改内容

### 新增的文件

1. **[ai_scientist/config/paths.py](ai_scientist/config/paths.py)** - 路径配置模块
2. **[ai_scientist/config/__init__.py](ai_scientist/config/__init__.py)** - 配置模块初始化
3. **[test_output_config.py](test_output_config.py)** - 配置测试脚本

### 修改的文件

1. **[launch_scientist_zhipu.py](launch_scientist_zhipu.py)** - 使用新的路径配置
2. **[launch_scientist_bfts.py](launch_scientist_bfts.py)** - 使用新的路径配置
3. **[run_project.py](run_project.py)** - 使用新的路径配置
4. **[.gitignore](.gitignore)** - 添加 `_outputs/` 到忽略列表

### 新增的API

```python
from ai_scientist.config import paths

# 获取实验目录（运行实验的结果）
exp_dir = paths.get_experiment_dir("MyIdea", attempt_id=0)
# -> _outputs/experiments/20260127_205142_MyIdea_attempt_0/

# 获取idea文件路径（统一存放idea JSON）
idea_path = paths.get_idea_path("my_idea")
# -> _outputs/ideas/my_idea.json

# 获取项目目录（完整的项目）
proj_dir = paths.get_project_dir("my_project")
# -> _outputs/projects/my_project/

# 确保所有输出目录存在
paths.ensure_output_dirs()
```

## 使用方法

### 方法 1: 使用默认配置

直接运行脚本，输出将自动保存到 `_outputs` 目录：

```bash
# 智谱 API 版本
python launch_scientist_zhipu.py --load_ideas "ai_scientist/ideas/my_research_topic.json"
# 结果: _outputs/experiments/20260127_205142_MyIdea_attempt_0/

# OpenAI 版本
python launch_scientist_bfts.py --load_ideas "ai_scientist/ideas/my_research_topic.json"
# 结果: _outputs/experiments/20260127_205142_MyIdea_attempt_0/

# 项目管理器
python run_project.py my_project --topic topic.md
# 结果: _outputs/projects/my_project/
```

### 方法 2: 自定义输出目录

设置环境变量 `RESEARCH_OUTPUT_DIR`：

```bash
# 使用相对路径
export RESEARCH_OUTPUT_DIR="my_outputs"
python launch_scientist_zhipu.py --load_ideas "ai_scientist/ideas/my_research_topic.json"

# 使用绝对路径
export RESEARCH_OUTPUT_DIR="/path/to/output"
python launch_scientist_zhipu.py --load_ideas "ai_scientist/ideas/my_research_topic.json"
```

兼容性说明：旧变量 `AI_SCIENTIST_OUTPUT_DIR` 仍可用，但建议统一迁移到 `RESEARCH_OUTPUT_DIR`。

### 方法 3: 在代码中使用

```python
from ai_scientist.config import paths

# 获取实验目录
exp_dir = paths.get_experiment_dir("my_idea", attempt_id=0)
print(exp_dir)  # -> _outputs/experiments/20260127_205142_my_idea_attempt_0/

# 获取idea文件路径
idea_path = paths.get_idea_path("my_idea")
print(idea_path)  # -> _outputs/ideas/my_idea.json

# 获取项目目录
proj_dir = paths.get_project_dir("my_project")
print(proj_dir)  # -> _outputs/projects/my_project/

# 确保输出目录存在
paths.ensure_output_dirs()
```

## 迁移现有数据

如果你有旧的实验结果和idea文件，可以迁移到新位置：

```bash
# 迁移实验结果
mv experiments/* _outputs/experiments/

# 迁移idea文件（从 ai_scientist/ideas/ 或其他位置）
mv ai_scientist/ideas/*.json _outputs/ideas/

# 迁移项目文件夹
mv projects/* _outputs/projects/
```

## 验证配置

运行测试脚本验证配置是否正确：

```bash
python test_output_config.py
```

## 优势

1. **更清晰的项目结构**: 源代码和生成文件完全分离
2. **idea文件统一管理**: 所有idea JSON文件都在 `_outputs/ideas/` 下，易于查找和管理
3. **便于版本控制**: 所有生成内容在一个目录，易于忽略
4. **便于清理**: 可以直接删除 `_outputs` 目录清理所有生成文件
5. **灵活配置**: 支持通过环境变量自定义输出位置
6. **便于备份**: 可以只备份源代码，或只备份生成结果

## 兼容性

- **向后兼容**: 绝对路径的项目目录仍然有效
- **不影响现有代码**: 只是修改了输出位置，核心逻辑不变
- **可配置**: 支持通过环境变量自定义输出目录

## 目录结构对比

### 优化前
```
ai_scientist/
├── experiments/              # 散落的实验结果
│   └── 2024-01-27_20-30-00_MyIdea/
├── projects/                 # 散落的项目
│   └── my_project/
├── ai_scientist/
│   └── ideas/               # idea文件在这里
│       └── my_idea.json
├── cache/                   # 缓存文件
└── ...
```

### 优化后
```
ai_scientist/
├── _outputs/                # 所有生成内容统一在这里
│   ├── cache/              # 缓存文件
│   ├── ideas/              # 所有idea文件
│   │   └── my_idea.json
│   ├── experiments/        # 所有实验结果
│   │   └── 20260127_205142_MyIdea_attempt_0/
│   └── projects/           # 所有项目
│       └── my_project/
├── ai_scientist/           # 纯粹的源代码
└── ...
```
