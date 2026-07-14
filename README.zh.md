# XScientist

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

English README: [README.md](README.md)

> 面向"可持续自我迭代"的 AI 科研系统：从想法生成、实验执行、论文写作，到自评审闭环、策略调度与长期运行（daemon）。
> 更进一步——我们想做的不只是「更好的全自动科研」，而是**可 git 的科研协议**，沿着自动化的科技树，从数学与物理这类可验证的根节点向外扩展。

本仓库的目标不是一次性"生成一篇论文"，而是把自动科研系统做成**可长期运行、可观测、可回放、可交接**的研究流水线：每次运行都产出结构化工件（计划、证据、评审、修复任务、质量门禁与报告），便于持续改进与协作。这些工件对齐一份独立的协议规范（`ai_scientist/protocol/`，ARA v1），因此可以被别的实现读、写、diff、fork。

重要提示（建议先读）：

- 成本：运行会调用大模型/检索服务，可能产生 API 费用与较长运行时间。
- 可靠性：生成内容可能存在错误或幻觉；请务必自行复核关键结论、数据与引用。
- 输出目录：默认**不会**把运行产物写入仓库目录（避免污染开源仓库）。

如果你对该仓库来时路感兴趣，欢迎阅读对应的知乎随笔：[XScientist踩坑实录](https://zhuanlan.zhihu.com/p/2027818800238666075) 。如果对你有帮助，感谢star和小心心～
---

## 目录 (Contents)

- [XScientist](#xscientist)
  - [目录 (Contents)](#目录-contents)
  - [愿景：可 git 的科研协议](#愿景可-git-的科研协议)
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
    - [D) 反馈系统监控](#d-反馈系统监控)
  - [输出与可观测性](#输出与可观测性)
    - [科研完整性取证（Integrity Forensics）](#科研完整性取证integrity-forensics)
    - [ARA 工件（面向下游智能体）](#ara-工件面向下游智能体)
      - [科技探索树视图](#科技探索树视图)
  - [示例论文](#示例论文)
  - [文档索引](#文档索引)
  - [开发与测试](#开发与测试)
  - [路线图](#路线图)
  - [系统架构](#系统架构)
  - [贡献与社区](#贡献与社区)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
  - [Citation and References](#citation-and-references)

---

## 愿景：可 git 的科研协议

XScientist 想做的不只是「一个更好的全自动科研系统」，更是一层**可 git 的科研协议**——让研究这件事像代码一样可 diff、可 fork、可 review、可 rebase：

- **协议先于系统**：`ai_scientist/protocol/`（ARA v1）用 6 份 JSON Schema + 一个 `content_hash` 归一化算法，把一次研究运行沉淀成机读工件。第三方 producer / consumer 无需依赖 XScientist 本身也能实现同一协议——就像 git 之于代码。
- **每次运行都是一个 commit**：ARA 把 exploration graph、每个节点的 `code / term_out / metrics / plots`、失败分支、修复轨迹、Pareto 池和环境指纹一起归档；每一份 manuscript 断言用 `\claimref{node_id}` 反向锚回其证据节点。
- **fork-continue 而非冷启动**：任一节点都可被 `run_ara_fork.py fork` 出一份「本身也是合规 ARA」的种子，下一次运行直接接力（provenance 自动落进 child ARA），跨系统 / 跨团队都可以延续。
- **自动化的科技树，数学与物理是根节点**：我们相信科研的可自动化程度沿着一棵树展开——**数学和物理是根节点**，越靠近根的问题「协议 / 证据 / 复核」信号越强、更适合被机器扩展；越靠近叶子（工程、经验、社会科学）越依赖人类判断。XScientist 的重心先落在根附近：把「可验证、可复现、可 fork」的部分自动化，把「值得人类花时间的」部分显式暴露给评审者。

一句话：**把科研做成协议，把系统做成协议的一个实现**。<br/>
配套细节见 [`ai_scientist/protocol/SPEC.md`](ai_scientist/protocol/SPEC.md) 与下文的 [ARA 工件](#ara-工件面向下游智能体) / [从 ARA 接力](#从-ara-接力fork-continue) 章节。

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
- **Pareto 前沿候选池**：自动追踪不同 rewrite round 的稿件质量向量，保留 Pareto 前沿候选稿，为后续改写提供互补强项参考与种子稿。
- **修复反思与验证**：在生成修复计划前进行 LLM 反思（reflection），执行修复后自动验证（verifier），并记录修复尝试历史（repair attempts）用于趋势分析与策略反馈。
- **实验 TODO 可度量闭环**：把"还差什么实验/证据"显式落成 TODO，并持续跟踪 closure 进度。
- **长期自治运行（Daemon）**：支持持续运行、失败保护、来源调度、趋势报告、交接简报与策略反馈。
- **增强反馈系统**：多源反馈收集、实时健康监控、趋势分析、自动行动生成。
- **可观测与可回放**：关键阶段工件结构化落盘（JSON/MD），便于对比、复盘与二次加工。
- **工程化安全**：登录守卫、预检/仓库校验、配置 schema、默认输出目录隔离。
- **ARA（Agent-Native Research Artifact）导出**：每次运行结束会在 `<project_dir>/ara/` 下额外落一份「面向下游智能体」的机读工件——完整的 exploration graph、每个节点的 `code.py`/`term_out.log`/`metrics.json`/`plots.json`、Pareto 池、修复历史、环境指纹，以及从 LaTeX 中扫描出的 `\claimref{node_id}` 声明到节点的映射。配套的 `run_ara_fork.py` CLI 可以 inspect / re-exec / fork 任意节点，让另一个 AI Scientist 无需解码 PDF 就能续跑或验证前作；`exploration_graph.html` 则把每篇小论文的探索过程展示成可浏览的科技探索树。

相关入口脚本：

- `run_project.py`：单项目端到端（适合本地调试/复现实验）
- `continuous_paper_generator.py`：批量/连续运行入口
- `continuous_research_daemon.py`：长期自治调度入口
- `research_manager.py`：索引与看板（筛选、导出、打包）
- `run_ara_fork.py`：从 ARA 工件里 inspect / re-exec / fork 单个节点

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

- 默认输出根目录：仓库平级的 `<仓库名>_outputs`；本仓库默认是 `../XScientist_outputs`
- 优先级：`RESEARCH_OUTPUT_DIR` > `AI_SCIENTIST_OUTPUT_DIR` > 默认平级目录
- 若平级目录不可写：回退到系统数据目录（如 `~/.local/share/ai_scientist/research`）

推荐显式指定输出路径：

```bash
export RESEARCH_OUTPUT_DIR="/path/to/my_xscientist_outputs"
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

投稿级/高质量运行会默认启用 deterministic integrity forensics。你也可以显式控制：

```bash
# 强制启用最终稿完整性取证
python3 run_project.py my_project \
  --output-root "$RESEARCH_OUTPUT_DIR" \
  --topic examples/example_topic.md \
  --integrity-forensics

# 在高质量调试中临时关闭
python3 continuous_paper_generator.py \
  --research-dir "$RESEARCH_OUTPUT_DIR" \
  --topic examples/example_topic.md \
  --paper-types icbinb \
  --high-quality-mode \
  --no-integrity-forensics
```

常用运维命令：

```bash
bash run_stable_daemon.sh status
bash run_stable_daemon.sh brief
bash run_stable_daemon.sh handoff
bash run_stable_daemon.sh report-trends
bash run_stable_daemon.sh source-plan
```

### D) 反馈系统监控

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
- `papers/`：批量生成的单篇论文目录
- `batches/`：连续生成器批次记录与进度
- `cache/`：HuggingFace / Torch / wandb 等运行缓存
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

### 科研完整性取证（Integrity Forensics）

XScientist 会在最终稿阶段运行一组 deterministic integrity forensics 检查，用于在进入 submission gate 前发现更像“硬阻断”的稿件风险，例如证据/断言一致性、结构化报告中的异常信号等。它不是人工审稿或事实核查的替代品，而是一层可复现、可落盘、可被后续 agent 消费的机器检查。

默认行为：

- `--submission-mode` 或 `--high-quality-mode` 下默认启用。
- 其他普通运行默认关闭，但可用 `--integrity-forensics` 强制启用。
- 调试或成本敏感场景可用 `--no-integrity-forensics` 显式关闭。
- 入口覆盖 `run_project.py`、`continuous_paper_generator.py`、`launch_scientist_bfts.py` 与 `launch_scientist_zhipu.py`。

每篇稿件的取证产物写入对应运行目录下的 `integrity_forensics/`，通常包括 JSON 报告和 Markdown 摘要。批量/项目 summary 会记录 `integrity_forensics_status`、`integrity_forensics_verdict`、finding 数量和报告路径，并在 shortlist 中展示。`HARD_FLAGS` 会阻断 submission-ready 判定；`SOFT_FLAGS` 会被报告出来，但不会单独阻断。

### ARA 工件（面向下游智能体）

除了给人看的 PDF，每次成功跑完 `run_project.py` 之后，还会在 `<project_dir>/ara/<timestamp>_<idea>/` 下写入一份「机读」的研究工件（Agent-Native Research Artifact，简称 ARA），设计目标是让另一个 AI Scientist 可以直接 fork / re-execute，而不必去逆向 PDF。

典型目录结构：

```
<project_dir>/ara/<timestamp>_<idea>/
├── manifest.json              # 顶层入口，指向下面的所有文件
├── exploration_graph.json     # 树搜索 DAG：每个节点与 parent/child 边（含失败分支）
├── exploration_graph.html     # 可在浏览器打开的科技探索树视图
├── exploration_graph.summary.json # DAG 校验摘要（root / leaf / 拓扑序）
├── nodes/<node_id>/
│   ├── code.py                # 该节点原样执行的代码
│   ├── term_out.log           # 完整的 stdout/stderr
│   ├── metrics.json           # metric + analysis + is_buggy
│   ├── plots.json             # 图表路径与 VLM 分析
│   ├── env.json               # Python 版本 / 预期 cwd
│   └── run.sh                 # 一键复跑脚本
├── claims/                    # 从 .tex 里扫描出的 `\claimref{node_id}` 映射
├── repair_history.jsonl       # 修复反思 / 验证器 / 尝试历史
├── pareto_pool.json           # 非支配的候选稿池
├── env/
│   ├── bfts_config.yaml
│   └── model_fingerprint.json
└── README.md                  # 面向 agent 的入口说明
```

#### 科技探索树视图

每一份 ARA 都把一次小论文生成过程记录成一个有向无环图（DAG）：根节点通常是初始方案或 baseline，子节点是后续实验、消融、修复、失败分支或候选稿改写。用户可以直接打开 `exploration_graph.html` 查看这棵科技探索树，也可以用 `run_ara_fork.py graph --json` 读取同一份结构化图。

示意图如下：

```mermaid
flowchart TD
  root["root: 研究问题 / baseline"]
  exp1["exp-1: 第一版实验"]
  fail1["fail-1: 失败分支 / bug"]
  repair1["repair-1: 修复与重跑"]
  ablate1["ablate-1: 消融实验"]
  candidate1["paper-a: 候选小论文"]
  candidate2["paper-b: Pareto 候选"]
  claim1["claimref: 论文结论锚点"]
  fork1["fork: 下一轮接力种子"]

  root --> exp1
  exp1 --> fail1
  fail1 --> repair1
  exp1 --> ablate1
  repair1 --> candidate1
  ablate1 --> candidate2
  candidate1 --> claim1
  candidate2 --> claim1
  candidate2 --> fork1
```

这棵树和 git-like 记录、CLI log、节点 diff 使用的是同一份 provenance：`exploration_graph.json` 是源数据，`exploration_graph.html`、`exploration_graph.summary.json`、`run_ara_fork.py log`、`run_ara_fork.py diff --only-node` 和 `run_ara_fork.py fork` 都是它的不同视图。如果把 ARA 目录提交进 git，git 记录的是这份图数据的文件级快照；XScientist 的 log/diff/fork 则给出节点级历史。因此某篇小论文的结论如果来自 `candidate2`，就能继续追到它的父实验、失败修复、消融对照，以及可以从哪个节点 fork 出下一轮研究。

配套 CLI：`run_ara_fork.py` 提供 `inspect` / `exec` / `fork` / `freeze` / `validate` / `verify` / `graph` 等子命令：

```bash
# 打印某个节点的 metric / analysis / 代码大小
python3 run_ara_fork.py inspect \
  --ara <project_dir>/ara/<timestamp>_<idea> \
  --node-id <node_id>

# 重跑一个节点并把 fresh vs recorded metric 写进 verify/*.json
python3 run_ara_fork.py exec \
  --ara <project_dir>/ara/<timestamp>_<idea> \
  --node-id <node_id>

# 把节点拷贝出一份「本身也是合规 ARA」的 fork 目录（可再次 fork / validate）
python3 run_ara_fork.py fork \
  --ara <project_dir>/ara/<timestamp>_<idea> \
  --node-id <node_id> \
  --dest /path/to/fork_seed

# 快照当前解释器的 pip freeze
python3 run_ara_fork.py freeze --ara <project_dir>/ara/<timestamp>_<idea>

# 对照 ai_scientist/protocol/SPEC.md 做 conformance 校验
python3 run_ara_fork.py validate --ara <project_dir>/ara/<timestamp>_<idea>

# 检查 DAG invariant，并按需重建可视化
python3 run_ara_fork.py graph \
  --ara <project_dir>/ara/<timestamp>_<idea> \
  --write-html

# 批量重跑若干节点并写一份 verify/reexec_batch_*.json（对齐 CI 门禁）
python3 run_ara_fork.py verify \
  --ara <project_dir>/ara/<timestamp>_<idea> \
  --limit 3
```

`exploration_graph.json` 是每篇小论文对应的科技探索 DAG：节点是一次具体实验、修复或失败分支，边是 parent -> child 的演化关系。`validate` 会校验它是有向无环图；`graph --json` 给出 root、leaf、拓扑序和问题列表；`exploration_graph.html` 则是给人看的可视化入口。这样 `run_ara_fork.py log --node <id>` 的祖先链、`diff --only-node <id>` 的节点差异和浏览器里的探索树使用同一份图数据。

写作阶段的 prompt 会引导模型在关键定量结论后附加 `\claimref{<node_id>}`。该宏在 PDF 里不可见，但会被 `ai_scientist/utils/claim_registry.py` 扫描，把每一条 claim 落到 `ara/.../claims/<claim_id>.json`——完成「论文 assertion ↔ 探索节点」的双向锚定。`ai_scientist/utils/claim_coverage.py` 会把这些标记聚合成 `coverage_score` 与 severity（`ok` / `sparse` / `unresolved` / `insufficient` / `none`），存入 `ara/.../claims/coverage.json`，供质量门禁 / 排行 / dossier 打分使用。

可选：批量 re-execution 验证。设置环境变量后，`run_project.py` 结束时会挑选 top-metric 节点重跑并生成 verify 报告：

```bash
export AI_SCIENTIST_ARA_REEXEC=1
```

默认关闭，避免非预期地重复调用外部 API/GPU。

### 从 ARA 接力（Fork-Continue）

任何一个 XScientist 实例产出的 ARA 都可以作为下一次运行的种子——tree search 的首个 draft 直接使用指定节点的 code，跳过 LLM 冷启动，`provenance` 会自动写进 child ARA 的 `manifest.json`：

```bash
# 用 fork 目录作为种子（推荐）
python3 run_project.py \
  --project-dir <B_project> \
  --seed-from-ara /path/to/fork_seed \
  --topic ...   # 其他常规参数

# 或直接从上一次 ARA 的某个节点起接力（等价于 fork + seed 一步到位）
python3 run_project.py \
  --project-dir <B_project> \
  --seed-from-ara <A_project>/ara/<timestamp>_<idea> \
  --seed-node-id <node_id>
```

底层通过环境变量 `AI_SCIENTIST_ARA_SEED_PATH` 传递种子清单——同一机制会自动跨越 subprocess 边界（并行 worker 也能生效）。协议细节见 [`ai_scientist/protocol/SPEC.md`](ai_scientist/protocol/SPEC.md) §7。

### 协议规范

`ai_scientist/protocol/` 是独立可移植的协议包（`ara.v1`），包含 6 份 JSON Schema、`content_hash` 归一化算法与最小 conformance validator。第三方 producer/consumer 无需依赖 XScientist 也能实现同一协议——用途包括：让另一个 agent 消费我们的 ARA、跨系统的 provenance 追踪、CI 中把 `--strict` 校验作为门禁。规范正文见 [`ai_scientist/protocol/SPEC.md`](ai_scientist/protocol/SPEC.md)。

### A/B 加速证据实验

想验证「从 ARA 接力真的省事」而不是自嗨，可以跑 `ai_scientist/experiments/ara_ab/`：

```bash
# CI 安全：不调用真实 LLM，只验证 seed 短路机制
python -m ai_scientist.experiments.ara_ab.harness stub \
    --seed-manifest <project>/.ara_seed/ara_seed.json \
    --out-dir /tmp/ab_out

# 真跑：两次 run_project.py（baseline vs seeded），需 API key
python -m ai_scientist.experiments.ara_ab.harness real \
    --project-dir-baseline /tmp/ab_baseline \
    --project-dir-seeded   /tmp/ab_seeded \
    --seed-from-ara /path/to/fork \
    --out-dir /tmp/ab_out \
    -- --topic mytopic.md   # 后接的参数直传 run_project.py
```

产物 `ab_report.json`（schema `ara.ab_report.v1`）会同时给出两侧的 wall-clock、LLM 调用数、node 数、content_hash 重叠度，以及最终 verdict（`seed_saved_llm_calls` / `seed_wall_clock_faster` / `seed_did_not_short_circuit` / `seed_inconclusive`）。

---

## 示例论文

示例论文与相关提交材料统一放在 `example/` 目录，便于检查论文排版、补充材料组织方式和最终交付格式。

当前已整理的示例文件：

- [example/XScientist_Board.pdf](example/XScientist_Board.pdf)：XScientist Board 论文/报告 PDF。
- [example/icml_submitted_gravitation_paper.pdf](example/icml_submitted_gravitation_paper.pdf)：ICML 投稿中的 gravitation 论文稿件 PDF。

---

## 文档索引

- `docs/guides/PROJECT_USAGE.md`：`run_project.py` 项目流用法与参数说明
- `docs/guides/FEEDBACK_QUICKSTART.md`：反馈系统快速入门指南
- `docs/CONFIG_REFERENCE.md`：更细的配置/参数参考
- `docs/SOURCE_ORCHESTRATION.md`：source queue 编排与运行姿态建议
- `docs/LONG_RUNNING_GUIDE.md`：长时运行操作指南
- `docs/LOGIN_GUARDRAIL.md`：登录守卫与会话管理
- `docs/guides/OUTPUT_DIRECTORIES.md`：输出目录策略说明（如与代码不一致，请以 `ai_scientist/config/paths.py` 为准）
- `ARCHITECTURE.md`：系统架构文档
- `OPTIMIZATION_SUMMARY.md`：优化总结

---

## 开发与测试

- 单元测试：`make test`
- 语法/导入/校验 smoke：`make smoke`
- 更严格的本地 doctor：`make doctor`（需要有效登录会话）
- 代码格式化：`make format`

---

## 路线图

XScientist 的目标是把自动科研从「单次生成一篇论文」推进为「长期可运行、可复现、可评审、可交付」的科研基础设施。欢迎 issue / PR 协作（见 `CONTRIBUTING.md`）。

- **近期**：交付可复现的 submission-ready 样例；补齐 preflight/交付清单；把 TODO closure 并入质量门禁。
- **中期**：evidence 与 figure/table/metric 双向绑定；dossier 一致性/回归检查；多评审视角聚合。
- **长期**：daemon 依据历史指标自适应策略；跨项目知识库；标准 benchmark / leaderboard；更完整的英文文档与插件接口。

---

## 系统架构

详细架构文档请参阅：[ARCHITECTURE.md](ARCHITECTURE.md)

核心组件：
- **Ideation Engine**: 想法生成与排序
- **Experiments Engine**: 实验执行与证据收集
- **Writeup Engine**: 论文写作与编译
- **Self-Review Engine**: 自评审与修复
- **Repair Reflection & Verifier**: 修复反思与验证
- **Pareto Pool**: 前沿候选稿管理
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
  url          = {https://github.com/smileformylove/XScientist}
}
```

XScientist Board（使用本系统写作/打磨的论文或报告）：

```bibtex
@misc{xscientist_board,
  title        = {XScientist Board: Artifact-Routed Submission Hardening for Autonomous Research Systems},
  author       = {{XScientist}},
  year         = {2026},
  url          = {https://github.com/smileformylove/XScientist/blob/main/example/XScientist_Board.pdf}
}
```

ICML 投稿中的 gravitation 示例论文：

```bibtex
@misc{xscientist_icml_submitted_gravitation,
  title        = {A Gravitational Field Theory for Deep Networks},
  author       = {{XScientist}},
  year         = {2026},
  url          = {https://github.com/smileformylove/XScientist/blob/main/example/icml_submitted_gravitation_paper.pdf}
}
```

### Citation Notes

- 引用 XScientist 生成的论文时，请同时引用本仓库和具体生成论文。
- 引用自动生成结果时，请明确标注人工复核、修改和筛选过程。
