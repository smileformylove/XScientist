# XScientist

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Smoke](https://github.com/smileformylove/ai_scientist/actions/workflows/smoke.yml/badge.svg)](https://github.com/smileformylove/ai_scientist/actions/workflows/smoke.yml)

English README: [README.en.md](README.en.md)

> 面向"可持续自我迭代"的 AI 科研系统：从想法生成、实验执行、论文写作，到自评审闭环、策略调度与长期运行（daemon）。

本仓库的目标不是一次性"生成一篇论文"，而是把自动科研系统做成**可长期运行、可观测、可回放、可交接**的研究流水线：每次运行都产出结构化工件（计划、证据、评审、修复任务、质量门禁与报告），便于持续改进与协作。

重要提示（建议先读）：

- 成本：运行会调用大模型/检索服务，可能产生 API 费用与较长运行时间。
- 可靠性：生成内容可能存在错误或幻觉；请务必自行复核关键结论、数据与引用。
- 输出目录：默认**不会**把运行产物写入仓库目录（避免污染开源仓库）。

---

## 目录 (Contents)

- [XScientist](#xscientist)
  - [目录 (Contents)](#目录-contents)
  - [项目概览](#项目概览)
  - [核心能力](#核心能力)
  - [快速开始](#快速开始)
    - [0) 依赖说明](#0-依赖说明)
    - [1) 安装（推荐 conda）](#1-安装推荐-conda)
    - [2) 配置 API Key（按需）](#2-配置-api-key按需)
    - [3) 登录（必需）](#3-登录必需)
    - [4) 预检（推荐）](#4-预检推荐)
  - [配置](#配置)
    - [输出目录（默认不写入仓库）](#输出目录默认不写入仓库)
    - [严格兜底策略（调试提示）](#严格兜底策略调试提示)
  - [使用方法](#使用方法)
    - [A) 从 Topic 跑一个项目（最常用）](#a-从-topic-跑一个项目最常用)
    - [B) 连续运行/批量生成（适合跑一段时间）](#b-连续运行批量生成适合跑一段时间)
    - [C) Daemon 长期自治运行（推荐用于"持续迭代"）](#c-daemon-长期自治运行推荐用于持续迭代)
  - [输出与可观测性](#输出与可观测性)
  - [示例论文](#示例论文)
  - [文档索引](#文档索引)
  - [开发与测试](#开发与测试)
  - [路线图](#路线图)
  - [贡献与社区](#贡献与社区)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
  - [Citation and References](#citation-and-references)

---

## 项目概览

你可以把它理解为一个"研究操作系统（research operating system）"：

- 输入：Topic / Sources / 研究约束（预算、停止条件、质量门禁）
- 过程：ideation -> 实验 -> 写作 -> 自评审 -> 修复/重写 -> 打包交付
- 输出：可复用的研究资产（报告、论文草稿、评审与修复队列、运行索引、交接简报）

核心循环（简化）：

```mermaid
flowchart LR
  A["Topic / Sources"] --> B["Ideation & Ranking"]
  B --> C["Experiments"]
  C --> D["Writeup & Quality Gates"]
  D --> E["Self-Review & Repair"]
  E --> F["Artifacts + Index + Dossier"]
  F --> G["Daemon Strategy Feedback"]
  G --> B
```

---

## 核心能力

- **自评审闭环**：多轮 self-review 生成结构化 issue 与修复计划，并将修复覆盖率/回归检查纳入门禁。
- **实验 TODO 可度量闭环**：把"还差什么实验/证据"显式落成 TODO，并持续跟踪 closure 进度。
- **长期自治运行（Daemon）**：支持持续运行、失败保护、来源调度、趋势报告、交接简报与策略反馈。
- **增强反馈系统**：多源反馈收集、实时健康监控、趋势分析、自动行动生成。
- **可观测与可回放**：关键阶段工件结构化落盘（JSON/MD），便于对比、复盘与二次加工。
- **工程化安全**：登录守卫、预检/仓库校验、配置 schema、默认输出目录隔离。

相关入口脚本：

- `run_project.py`：单项目端到端（适合本地调试/复现实验）
- `continuous_paper_generator.py`：批量/连续运行入口
- `continuous_research_daemon.py`：长期自治调度入口
- `research_manager.py`：索引与看板（筛选、导出、打包）

---

## 快速开始

### 0) 依赖说明

- Python: 3.10+（推荐 3.11）
- 系统依赖（建议安装）：
  - LaTeX 工具链（用于编译论文 PDF，例如 TeX Live / MacTeX）
  - `poppler`（用于 PDF 处理/抽取）
  - `chktex`（可选，LaTeX 静态检查）

> GPU/CUDA 并非必需；如需 GPU，请按 PyTorch 官方指引安装匹配版本。

### 1) 安装（推荐 conda）

```bash
conda create -n xscientist python=3.11 -y
conda activate xscientist

pip install -r requirements.txt
```

更稳定的"CI 风格"安装（可选）：

```bash
pip install -r requirements.txt -c constraints-ci.txt
```

### 2) 配置 API Key（按需）

按你使用的提供商设置环境变量（不需要全部设置）：

```bash
export OPENAI_API_KEY="..."
export ZHIPU_API_KEY="..."
export GEMINI_API_KEY="..."
export S2_API_KEY="..."
```

### 3) 登录（必需）

```bash
python3 auth_cli.py login --user <your_name>
python3 auth_cli.py status
```

登录守卫说明：`docs/LOGIN_GUARDRAIL.md`

### 4) 预检（推荐）

```bash
python3 preflight_check.py --strict
python3 validate_repo.py
make smoke
```

---

## 配置

### 输出目录（默认不写入仓库）

为避免运行产物污染仓库，默认输出到**仓库平级目录**：

- 默认输出根目录：`../ai_scientist_outputs`（相对当前仓库）
- 优先级：`RESEARCH_OUTPUT_DIR` > `AI_SCIENTIST_OUTPUT_DIR` > 默认平级目录
- 若平级目录不可写：回退到系统数据目录（如 `~/.local/share/ai_scientist/research`）

推荐显式指定输出路径：

```bash
export RESEARCH_OUTPUT_DIR="/path/to/my_ai_scientist_outputs"
```

### 严格兜底策略（调试提示）

多数脚本都支持更严格的质量门禁。若你在调试阶段需要放宽兜底策略，可查阅参数 `--override-strict-fallbacks`（仅建议本地调试使用）。

---

## 使用方法

### A) 从 Topic 跑一个项目（最常用）

```bash
python3 run_project.py my_project \
  --output-root "$RESEARCH_OUTPUT_DIR" \
  --topic examples/example_topic.md
```

更多用法：`docs/guides/PROJECT_USAGE.md`

### B) 连续运行/批量生成（适合跑一段时间）

```bash
python3 continuous_paper_generator.py \
  --research-dir "$RESEARCH_OUTPUT_DIR" \
  --topic examples/example_topic.md \
  --paper-types icbinb
```

### C) Daemon 长期自治运行（推荐用于"持续迭代"）

```bash
python3 continuous_research_daemon.py \
  --source-config configs/sources/stable_source_priority.example.json \
  --duration-hours 24 \
  --enable-rewrite-followup \
  --auto-source-quality-feedback \
  --auto-quality-strategy-feedback \
  --auto-quality-governor \
  --auto-evidence-strategy-feedback \
  --auto-export-submission-dossier \
  --auto-failure-guard \
  --serve-dashboard \
  -- --submission-mode --num-ideas 3
```

常用运维命令：

```bash
bash run_stable_daemon.sh status
bash run_stable_daemon.sh brief
bash run_stable_daemon.sh handoff
bash run_stable_daemon.sh report-trends
bash run_stable_daemon.sh source-plan
```

### D) 反馈系统监控（新增）

```bash
# 查看系统健康状态
python3 feedback_cli.py --feedback-dir ./feedback status

# 查看推荐行动
python3 feedback_cli.py --feedback-dir ./feedback actions

# 分析趋势
python3 feedback_cli.py --feedback-dir ./feedback trends \
  --metrics quality_score success_rate error_rate

# 导出报告
python3 feedback_cli.py --feedback-dir ./feedback report
```

更多用法：`docs/guides/FEEDBACK_QUICKSTART.md`

---

## 输出与可观测性

本项目会在输出根目录下生成结构化工件，便于索引与复盘（目录名可能随版本演进）：

- `projects/`：每个研究项目的完整目录
- `experiments/`：实验运行结果与日志
- `ideas/`：生成/整理后的 idea 工件
- `reports/`：趋势报告/交接报告等（daemon 场景）
- `knowledge_base/`：跨项目沉淀（例如 self-evolution history/playbook）

常用看板/索引命令（更多见 `research_manager.py --help`）：

```bash
python3 research_manager.py rebuild-index
python3 research_manager.py submission-board --top 5 --require-gate
python3 research_manager.py rewrite-board --top 10
python3 research_manager.py repair-board --top 20 --priority-tier p0
python3 research_manager.py evolution-board --top 20
python3 research_manager.py process-board --status blocked --top 30
```

---

## 示例论文

示例论文与相关提交材料统一放在 `example/` 目录，便于检查论文排版、补充材料组织方式和最终交付格式。

当前已整理的示例文件：

- `example/icml2026_arxiv_paper.pdf`：ICML 2026 arXiv 论文 PDF。

如需继续补充 NeurIPS 2026 test1 的论文与补充材料，请将对应 PDF 放入 `example/`，建议命名为：

- `example/nips2026_test1_paper.pdf`
- `example/nips2026_test1_supplementary.pdf`

---

## 文档索引

- `docs/guides/PROJECT_USAGE.md`：`run_project.py` 项目流用法与参数说明
- `docs/guides/FEEDBACK_QUICKSTART.md`：反馈系统快速入门指南（新增）
- `docs/CONFIG_REFERENCE.md`：更细的配置/参数参考
- `docs/SOURCE_ORCHESTRATION.md`：source queue 编排与运行姿态建议
- `docs/LONG_RUNNING_GUIDE.md`：长时运行操作指南（新增）
- `docs/LOGIN_GUARDRAIL.md`：登录守卫与会话管理
- `docs/guides/OUTPUT_DIRECTORIES.md`：输出目录策略说明（如与代码不一致，请以 `ai_scientist/config/paths.py` 为准）
- `ARCHITECTURE.md`：系统架构文档（新增）
- `OPTIMIZATION_SUMMARY.md`：优化总结（新增）
- `OPTIMIZATION_UPDATE_PHASE2.md`：第二阶段更新（新增）

---

## 开发与测试

- 单元测试：`make test`
- 语法/导入/校验 smoke：`make smoke`
- 更严格的本地 doctor：`make doctor`（需要有效登录会话）
- 代码格式化：`make format`

---

## 路线图

XScientist 的路线图聚焦于把自动科研流程从“单次生成”推进到“长期可运行、可复现、可评审、可交付”的科研基础设施。欢迎 issue / PR 协作（见 `CONTRIBUTING.md`）。

**近期目标：稳定交付与可复现**

- [ ] 完善 `example/` 示例论文与补充材料，形成可直接对照的 submission-ready 样例。
- [ ] 增强 preflight / smoke 检查，覆盖 API Key、LaTeX、输出目录、登录状态和依赖版本。
- [ ] 建立论文交付清单，自动检查 PDF、图表、表格、引用、实验日志和复现实验配置。
- [ ] 把 TODO closure 信号并入质量门禁，明确每篇论文的未闭环实验与证据缺口。

**中期目标：自评审、自修复与质量提升**

- [ ] 强化 evidence 指标与实验结果（figure/table/metric）的双向绑定。
- [ ] 增加 submission-ready 套件的一致性检查与回归测试（dossier checks）。
- [ ] 将 self-evolution / playbook 更直接接入自动 rewrite、repair 和 follow-up 执行器。
- [ ] 引入多评审视角聚合，区分 novelty、soundness、clarity、reproducibility 和 ethics 风险。

**长期目标：持续自治科研系统**

- [ ] 让 daemon 根据历史成功率、成本、质量分数和失败模式自动调整研究策略。
- [ ] 建立跨项目知识库，沉淀高质量 idea、失败案例、可复用实验模板和写作经验。
- [ ] 提供更完整的英文文档、API 文档和插件接口，便于外部团队协作与二次开发。
- [ ] 支持更标准的 benchmark / leaderboard，用于评估自动科研系统的长期表现。

---

## 系统架构

详细架构文档请参阅：[ARCHITECTURE.md](ARCHITECTURE.md)

核心组件：
- **Ideation Engine**: 想法生成与排序
- **Experiments Engine**: 实验执行与证据收集
- **Writeup Engine**: 论文写作与编译
- **Self-Review Engine**: 自评审与修复
- **Autonomous Evolution Engine**: 自主进化与策略优化
- **Adaptive Learning Engine**: 自适应学习与推荐
- **Enhanced Feedback System**: 增强反馈与监控

## 贡献与社区

- 贡献指南：`CONTRIBUTING.md`
- 行为准则：`CODE_OF_CONDUCT.md`
- 安全策略：`SECURITY.md`
- 架构文档：`ARCHITECTURE.md`

---

## License

Apache-2.0，详见 `LICENSE`。

---

## Acknowledgements

感谢以下开源工作提供的经验与启发：

- [Sakana AI: AI Scientist](https://github.com/SakanaAI/AI-Scientist)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing)
- [AIDE](https://github.com/WecoAI/aideml)
- [DeepReviewer-v2](https://github.com/ResearAI/DeepReviewer-v2)

---

## Citation and References

如果你在研究中使用了 XScientist，建议引用本项目和具体生成论文；用于论文或报告时，请注明使用的 commit hash、实验配置、模型版本和输出目录，以便复现。

### XScientist

XScientist（软件/代码仓库）：

```bibtex
@software{xscientist,
  title        = {XScientist: A Long-Running Autonomous Scientific Research System},
  author       = {{XScientist}},
  year         = {2026},
  url          = {https://github.com/smileformylove/ai_scientist}
}
```

XScientist Board（使用本系统写作/打磨的论文或报告）：

```bibtex
@misc{xscientist_board,
  title        = {XScientist Board: Artifact-Routed Submission Hardening for Autonomous Research Systems},
  author       = {{XScientist}},
  year         = {2026},
  url          = {https://github.com/smileformylove/ai_scientist}
}
```

### Citation Notes

- 引用 XScientist 生成的论文时，请同时引用本仓库和具体生成论文。
- 引用自动生成结果时，请明确标注人工复核、修改和筛选过程。
