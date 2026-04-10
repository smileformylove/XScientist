# AI Scientist Autonomous Research Lab

> 面向“可持续自我迭代”的 AI 科研系统：从想法生成、实验执行、论文写作，到自评审闭环、策略调度和提交级运营。

本仓库在公开研究系统的基础上，重点建设了**长期运行能力**与**自评审驱动改进闭环**，目标不是一次性“生成一篇论文”，而是持续产出可验证、可迭代、可交接的高价值研究资产。

---

## 目录

1. [项目定位](#项目定位)
2. [本仓库亮点（差异化能力）](#本仓库亮点差异化能力)
3. [项目全景梳理](#项目全景梳理)
4. [环境安装与快速启动](#环境安装与快速启动)
5. [核心工作流](#核心工作流)
6. [关键产物与观测点](#关键产物与观测点)
7. [参考项目对齐与执行模式](#参考项目对齐与执行模式)
8. [公开 TODO（Roadmap）](#公开-todoroadmap)
9. [Thanks / 参考资料](#thanks--参考资料)
10. [Citation](#citation)

---

## 项目定位

本项目聚焦于“**自动科研系统的可运营化**”，核心目标是：

- 从“单次生成”升级到“**多轮自评审 + 实验 TODO 闭环 + 策略自适应**”；
- 从“离线跑批”升级到“**Daemon 连续运行 + 控制台运维 + 交接报告**”；
- 从“写出草稿”升级到“**提交优先级排序 + 阻塞项治理 + dossier 打包**”。

一句话：这是一个面向真实研究迭代的自动化科研引擎，而不仅是论文生成脚本集合。

---

## 本仓库亮点（差异化能力）

### 1) 自评审闭环（Deep Review Loop）

- 结构化问题账本与跨轮跟踪：`ai_scientist/utils/self_review_optimizer.py`
- 回合级 gate（critical/persistent/coverage 等）与最终回归检查：`run_project.py`
- fallback 改进链路同构：`ai_scientist/perform_auto_improvement.py`

### 2) 实验 TODO 闭环度量（可机器判定）

- 自动生成可执行 TODO（含 `completion_rule`）：`continuous_paper_generator.py`
- 回合/最终闭环快照（closure_rate、p0_closure_rate）：`ai_scientist/utils/experiment_todo_progress.py`
- 指标进入 run index 与管理面板：`ai_scientist/utils/run_index.py`

### 3) 研究运营层（Manager + Daemon）

- 提交板、重写板、短名单与 dossier 导出：`research_manager.py`
- 连续运行、源调度、失败保护、自动策略反馈：`continuous_research_daemon.py`
- 运维入口与控制命令：`run_stable_daemon.sh`
- fallback debt 看板：`python research_manager.py fallback-board`
- source lineage 看板：`python research_manager.py source-board`
- source mix 建议：`python research_manager.py source-mix --desired-policy program_driven`
- 下一批 source 组合建议：`python research_manager.py source-next-batch --desired-policy program_driven`
- self-evolution 看板：`python research_manager.py evolution-board`
- process alignment 看板：`python research_manager.py process-board --status blocked`

### 4) 策略自适应（不是固定参数）

- 质量策略反馈（venue/type/idea budget 动态调整）
- 证据策略反馈（claim support / coverage / unsupported claims）
- governor 自适应治理（含 closure_repair 模式）
- workflow execution policy（`program_driven / agentic_tree / review_board / writing_studio / multi_agent_board`）
- policy-aware daemon feedback（按 `budget_status / execution_policy / workflow_mode` 自动切换下一轮运行姿态）

### 5) 工程化安全与可用性

- 登录守卫：`auth_cli.py`
- 仓库校验与预检：`validate_repo.py`, `preflight_check.py`
- 配置样例与 schema：`*.example.json`, `*.schema.json`
- workflow-aware source planning：`source_queue.example.toml`, `source_queue.schema.json`, `docs/SOURCE_ORCHESTRATION.md`

---

## 项目全景梳理

### 顶层结构

- `ai_scientist/`：核心执行逻辑（ideation / experiment / writeup / review / quality）
- `ai_scientist/utils/`：共享工具（index、guardrail、optimizer、workflow、auth session）
- `continuous_paper_generator.py`：批量连续生成入口
- `run_project.py`：单项目端到端执行入口
- `continuous_research_daemon.py`：长期自治调度入口
- `research_manager.py`：索引重建、看板筛选、导出工具
- `docs/`：对齐说明、配置参考、运维检查单

### 生命周期（简化视图）

```mermaid
flowchart LR
  A["Topic / Ideas"] --> B["Ideation & Ranking"]
  B --> C["Experiment Tree Search"]
  C --> D["Writeup & Quality Workflow"]
  D --> E["Self-Review Iterations"]
  E --> F["Experiment TODO & Progress"]
  F --> G["Run Index + Manager Boards"]
  G --> H["Daemon Governor & Strategy Feedback"]
  H --> B
```

---

## 参考项目对齐与执行模式

本仓库不是简单照搬单一路线，而是把多个开源项目的长处显式做成可切换的执行模式和调度策略：

- `AI-Scientist`：对应 `classic_pipeline`
  - 强调稳定端到端模板流、较高成功率和完整产物链。
- `AI-Scientist-v2`：对应 `agentic_tree`
  - 强调探索分支、开放搜索和更强的 breakthrough 压力。
- `autoresearch`：对应 `program_driven`
  - 强调 research program、预算纪律、stop condition 和 acceptance rules。
- `awesome-ai-research-writing`：对应 `writing_studio`
  - 强调 evidence-to-writing、caption / analysis / polish 技能层。
- `DeepReviewer-v2`：对应 `review_board`
  - 强调 multi-role review、tool-grounded hardening 和 reviewer-facing repair。
- `全链路组合拳`：对应 `multi_agent_board`
  - 把 research program、agentic search、证据写作、review board、hostile critic 串成一条 submission-grade 多 agent 论文冲刺线。

### 1) Workflow Runtime

- `--workflow-mode classic_pipeline`
- `--workflow-mode agentic_tree`
- `--workflow-mode program_driven`
- `--workflow-mode writing_studio`
- `--workflow-mode review_board`
- `--workflow-mode multi_agent_board`

这些模式不只是 prompt 或默认值变化，现在已经影响：

- review 角色编排
- `research_plan.json` 中的 `execution_policy / task_kind / acceptance_checks`
- `research_program.md` 中的 operating policy
- `experiment_registry.jsonl` 中的 `policy_name / budget_status`
- `stage_standards.json` 中的 `stage_results / score / required_failures / top_risks`
- `process_alignment.json` 中逐过程的 point-to-point 开源对齐审计
- daemon 的下一轮调度反馈与 source planning

### 1.5) Fallback Debt Is Explicit

为了减少“静默兜底”带来的研究债务，ranking 与高质量写作阶段的 fallback 现在会进入 pipeline contracts：

- `pipeline_manifest.json` 会记录 `fallback_events` 和 `fallback_summary`
- `python research_manager.py pipeline-status` 会显示每个项目的 fallback 数量
- `python research_manager.py fallback-board --stage idea_ranking`
  可以直接排查 heuristic ranking、auto-improvement rewrite 等兜底来源
- `python research_manager.py submission-board`
  现在默认不接受 `strict fallback` 稿件进入投稿榜单，除非显式放宽阈值

这意味着系统不再只是“能继续跑完”，而是会把哪里在靠 fallback 维持运行明确暴露出来，方便后续持续压降。

进一步地，这些 fallback 信号现在也会进入：

- `source-board / source-mix / source-next-batch`
  source 画像会考虑 fallback debt，优先放大更“干净”的研究来源
- `readiness-benchmark`
  readiness score 会对 strict fallback debt 施加惩罚，避免“分数看起来不错但过程靠兜底撑住”的假阳性
- `submission-board / shortlist`
  现在会同时结合 `stage_standards + self_evolution` 过滤“看起来质量高，但修复纪律和自检闭环没过关”的稿件
- `continuous_research_daemon.py`
  daemon 会把 strict fallback debt 视作真实调度压力，而不只是日志信息

与此同时，高质量链路本身也更严格了：

- `program_driven`、`writing_studio`、`review_board`、`multi_agent_board`，以及 `journal / nature` 目标下，
  `auto-improvement fallback` 默认不再被当作可接受的 submission-grade 修复手段
- `research_program.md` 会写出当前 workflow 的 `quality fallback policy`
- 缓存的 `high_quality_result.json` 如果和当前 fallback discipline 不一致，会自动失效并重跑

与此同时，`run_project.py` 与 `continuous_paper_generator.py` 在 `program_driven / writing_studio / review_board / multi_agent_board`、以及所有 submission / high_quality / journal & nature 目标下会**默认启用严格兜底策略**：一旦 idea ranking 或质量链路触发 fallback，流程会立即终止。quality fallback 会落成 `quality_fallback_blocked` 这类明确失败阶段；idea ranking fallback 则会在 ranking 阶段直接早停并输出严格兜底错误。确需在调试阶段放宽时，可显式传入 `--override-strict-fallbacks`，但所有 fallback 仍会被记录进 `pipeline_manifest.json` 并在各看板中暴露。

### 2) Execution Policy

当前每个 workflow 都会生成显式执行策略，覆盖：

- `execution_style`
- `evidence_pressure`
- `budget.max_steps`
- `budget.max_wallclock_minutes`
- `budget.max_retry_per_task`
- `acceptance_rules`
- `quality_fallback_policy`
- `registry_expectations`

这让系统更接近真正的 research operating system，而不是“生成一次就结束”的脚本集合。

### 2.5) Explicit Stage Standards

现在每个 contracts 驱动项目都会有结构化的阶段评估标准：

- `stage_standards.json`
  覆盖 `ideation / planning / experiment / figure / manuscript / review`
- 每个阶段都会记录
  `status / score / criteria / required_failures / signals`
- `python research_manager.py stage-standards`
  可以直接查看每个阶段的通过情况、得分和关键阻塞项
- `python research_manager.py pipeline-status`
  现在也会汇总 `stage_overall_score` 和 `blocked_standard_stages`
- `python research_manager.py shortlist` / `submission-board`
  默认会过滤 `blocked stage standards` 的稿件，避免“结果看起来不错但流程自检没过”的假阳性
- `python research_manager.py shortlist --min-self-evolution-score 85`
  可以进一步要求 reviewer 驱动的自我进化分数达到阈值；默认也会拦掉 `blocked self_evolution` 和存在 required failure 的稿件
- `python research_manager.py readiness-benchmark`
  现在也会把 `stage_overall_score / blocked_stage_count / top_stage_standard_risks / self_evolution_score / self_evolution_required_failures` 纳入评分和导出报告
- `review_state.json`
  现在会显式记录 `active / resolved / persistent issues`、`repair_metrics`、`resolution_rate`、`verification_coverage`
  并把 reviewer 问题绑定到具体 `claim / figure / section`，避免“知道有问题但不知道该修哪”的宽松修复
- `review_state.json` 里的 `repair_queue`
  会把 active reviewer issues 转成结构化 repair tasks，记录 `priority_tier / primary_target / repair_actions / verification_checks / blocking_reasons`
- `repair_plan.json`
  会把 `repair_queue` 提升成 agentic repair plan，按 `figure_repair / claim_repair / evidence_followup / section_rewrite / method_repair / triage` 分 lane，并记录 `execution_steps / success_criteria / verification_checks / ready_rate`
- `self_evolution.json`
  会把 `review_state + repair_plan + stage signals` 提升成跨轮自我进化工件，记录 `status / score / lessons / next_cycle_defaults`
- `knowledge_base/self_evolution_playbook.json`
  会把多个项目最近一次的 `self_evolution` 聚合成跨项目 playbook，让下一轮 adaptive learning 能直接复用 reviewer 修复经验
- `python research_manager.py rewrite-board`
  会优先暴露 reviewer debt 高、resolution 低、persistent issues 多、target binding 弱、repair-ready coverage 低的稿件，推动 issue-by-issue 自我修复
- `python research_manager.py repair-board`
  会直接列出最值得先修的 reviewer repair tasks，明确该修哪条 claim、哪张 figure、哪一节，以及修完要怎么验证
- `python research_manager.py evolution-board`
  会直接显示每个项目当前的 self-evolution 状态、主导 repair lane、top lessons，以及下一轮默认该往哪些 stage 收紧

### 2.6) Explicit Process Alignment Against the Five Reference Repos

为了避免“有很多新工件，但每个过程到底对齐到了哪个开源研究系统并不清楚”，现在每个 contracts 驱动项目还会生成：

- `process_alignment.json`
  覆盖 `ideation / program / exploration / experiment / figure / writing / review / evolution / packaging`
- 每个过程都会记录：
  `status / score / criteria / required_failures / signals / risks / references`
- `python research_manager.py process-board`
  可以直接按过程、状态查看哪些 run 在哪个环节没有对齐好
- `python research_manager.py shortlist --min-process-alignment-score 80`
  可以强制只保留 process alignment 足够完整的稿件
- `python research_manager.py submission-board --max-blocked-processes 0`
  默认也会过滤 blocked process 的 run
- `python research_manager.py readiness-benchmark`
  现在会对 `process_alignment` 的 blocked/missing 过程施加显式惩罚

当前 point-to-point 映射是显式固定下来的：

- `ideation` -> `AI-Scientist` + `autoresearch`
- `program` -> `autoresearch`
- `exploration` -> `AI-Scientist-v2` + `autoresearch`
- `experiment` -> `AI-Scientist` + `AI-Scientist-v2` + `autoresearch`
- `figure` -> `AI-Scientist` + `awesome-ai-research-writing`
- `writing` -> `awesome-ai-research-writing`
- `review` -> `DeepReviewer-v2`
- `evolution` -> `DeepReviewer-v2` + `AI-Scientist-v2`
- `packaging` -> `AI-Scientist` + `awesome-ai-research-writing` + `DeepReviewer-v2`

这样系统不再只是“有一条流水线”，而是会明确回答：这一轮 research runtime 的每一个过程，究竟和哪类成熟开源范式对齐到了什么程度。

### 3) Policy-Aware Daemon

`continuous_research_daemon.py` 现在会基于 contracts 和 registry 聚合这些信号：

- `dominant_execution_policy`
- `execution_policy_counts`
- `budget_status_counts`
- `budget_exhausted_experiment_count`
- `stage_blocked_project_count`
- `stage_missing_project_count`
- `avg_stage_overall_score`
- `blocked_self_evolution_project_count`
- `self_evolution_required_failure_count`
- `avg_self_evolution_score`
- `process_alignment_blocked_project_count`
- `process_alignment_missing_project_count`
- `avg_process_alignment_score`

并自动进入不同 repair / rebuild 模式：

- `program_budget_repair`
  - 预算耗尽偏多时切向 `program_driven`，收缩探索并强化研究程序纪律。
- `agentic_exploration_rebuild`
  - 探索失败多但 figure/packaging 还不是瓶颈时切向 `agentic_tree`，扩大搜索和 followup。
- `review_board_hardening`
  - blocked artifacts 偏多时切向 `review_board`，加强 guardrail 与 reviewer-facing 修复。
- `self_evolution_rebuild`
  - self-evolution 自检被卡住时，强制切到 `review_board` 风格收敛，抬高 rewrite / verification / repair 纪律，先修清楚再继续放大探索。
- `process_alignment_repair`
  - 某些关键过程长期 blocked/missing 时，强制切向 `program_driven` 的收敛姿态，优先补齐 research program、exploration graph、figure provenance 和 packaging discipline。

现在 daemon 也会把 `blocked stage standards` 当成真实调度压力，而不是只把它们显示在看板里。
review feedback 也开始进入调度：低 `review_resolution_rate`、高 `persistent reviewer issues`、弱 `issue target binding` 和低 `repair-ready coverage` 都会被视作真实修复压力，而不是普通日志噪音。
新的 `repair_plan.json` 也让这条链更像 agentic runtime：系统现在不仅知道“哪里有 reviewer debt”，还知道“下一轮该走哪条 repair lane、每个 lane 的 success criteria 是什么”。
现在又进一步加上了 `self_evolution.json` 和 `knowledge_base/self_evolution_playbook.json`：系统不只会为当前 run 生成 repair lane，还会把这些 reviewer 问题沉淀成跨项目 playbook，并在下一轮 adaptive strategy recommendation 里回用这些经验。
现在再叠加 `process_alignment.json`，daemon 不只知道“质量不够”，还知道“到底是 ideation/program/exploration/figure/writing/review 哪条 process 没有对齐到位”。

同时 source runtime board 也会展示 `Preferred Policy` 和 `Workflow Alignment`，让 source 选择和当前修复目标保持一致。

### 4) Workflow-Aware Source Planning

这轮进一步把 source queue 从“按优先级轮询”升级成“可表达研究范式的 orchestration 输入”。每个 source 现在都可以显式声明：

- `workflow_mode / workflow_modes`
- `source_archetype`
- `batch_profile`
- `alignment_tags`
- `planning_notes`

daemon 会据此产出新的 `latest_source_batch_plan.md`，并把 source 映射到不同开源风格：

- `template_first` -> `classic_pipeline` -> `AI-Scientist`
- `frontier_exploration` -> `agentic_tree` -> `AI-Scientist-v2`
- `program_guarded` -> `program_driven` -> `autoresearch`
- `writing_polish` -> `writing_studio` -> `awesome-ai-research-writing`
- `review_hardening` -> `review_board` -> `DeepReviewer-v2`

这让系统不再只是“统一参数套不同 topic”，而是能按 source 的研究画像决定下一批次应该走 discovery、submission、evidence packing 还是 review hardening。

现在这套 metadata 也会继续进入 batch 产物和索引层：

- `progress.json` / `final_report.json` 会记录 `source_provenance`
- `source_provenance.json` 会进入 run index
- `python research_manager.py source-board` 可以按 source archetype / batch profile / workflow 直接看历史表现
- `python research_manager.py source-mix` 可以直接看当前 source 组合是否偏科、下一批该往哪类 research posture 倾斜
- `python research_manager.py source-next-batch` 会把这种倾斜进一步压成主力 lane / 多样化 lane / hardening lane 的下一批组合建议
- `auto_apply_source_plan` 现在也会消费这层 mix advisory，在存在合适 source 时自动做 `policy-align` 或 `mix-rebalance`
- daemon 现在还会额外写出 `latest_source_next_batch.{json,md}`，把推荐 cadence 和 lane 组合归档下来

### 5) 提交与 README 同步原则

为了让仓库更像成熟开源项目，而不是“代码在跑但思路不可见”的实验仓库，后续每轮功能提交都会尽量同步更新：

- `README.md`：高层能力、用法、参考项目映射
- `docs/`：更细的运行说明、运维说明、对齐说明
  - `docs/SOURCE_ORCHESTRATION.md`：source archetype、batch profile 和参考项目映射
- 对应测试：保证新增能力可回归验证
- GitHub Smoke Checks：优先安装 `requirements-smoke.txt`，并上传 `.ci-output/` 日志工件，减少依赖安装波动并提升失败可诊断性

---

## 环境安装与快速启动

### 1) 安装依赖

```bash
conda create -n ai_scientist python=3.11 -y
conda activate ai_scientist

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -y anaconda::poppler conda-forge::chktex
pip install -r requirements.txt
# Reproducible CI-style install (recommended for stable environments)
pip install -r requirements.txt -c constraints-ci.txt
```

### 2) 配置 API Key（按需）

```bash
export ZHIPU_API_KEY="YOUR_ZHIPU_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
export S2_API_KEY="YOUR_S2_KEY"
```

### 3) 登录（必需）

```bash
python3 auth_cli.py login --user <your_name>
python3 auth_cli.py status
```

登录守卫说明：`docs/LOGIN_GUARDRAIL.md`

### 4) 预检

```bash
python3 preflight_check.py --strict
python3 validate_repo.py
```

推荐在提交前跑一轮本地质量检查：

```bash
make smoke
```

如果你的系统默认 `python3` 仍然是 3.9，可显式指定解释器：

```bash
make smoke PYTHON=python3.11
```

如果你已经登录，或者想显式指定登录会话文件，可以用更严格的本地 doctor 流程：

```bash
make doctor
# 或
make doctor AUTH_FILE=/path/to/session.json
```

如果你在 CI、远程容器或临时 smoke 环境里使用单独的登录会话文件，可以显式指定：

```bash
python3 preflight_check.py --strict --auth-file /path/to/session.json
```

### 5) 输出路径策略（默认不写入仓库）

为避免运行产物污染开源仓库，本项目默认把输出写到**仓库平级目录**：

- 默认输出根目录：`../ai_scientist_outputs`（相对于当前仓库）
- 优先级：`RESEARCH_OUTPUT_DIR` > `AI_SCIENTIST_OUTPUT_DIR` > 默认仓库平级目录
- 若仓库平级目录不可写，会自动回退到系统数据目录（如 `~/.local/share/ai_scientist/research`）

推荐显式指定输出路径：

```bash
# 方式1：环境变量（全局生效）
export RESEARCH_OUTPUT_DIR="/path/to/my_ai_scientist_outputs"
```

```bash
# 方式2：连续论文生成器（按次指定）
python3 continuous_paper_generator.py \
  --research-dir "/path/to/my_ai_scientist_outputs" \
  --topic example_topic.md \
  --paper-types icbinb
```

```bash
# 方式3：项目流程（相对 project_dir 会落在 --output-root 下）
python3 run_project.py my_project \
  --output-root "/path/to/my_ai_scientist_outputs" \
  --topic example_topic.md
```

```bash
# 方式4：直接给 run_project.py 绝对项目路径
python3 run_project.py /path/to/my_ai_scientist_outputs/projects/my_project \
  --topic example_topic.md
```

---

## 核心工作流

### A. 从 Topic 到单次项目输出

1) 先做 ideation：

```bash
python3 ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/my_topic.md \
  --model glm-4-flash \
  --max-num-generations 20 \
  --num-reflections 5
```

2) 再做生成与质量链路（提交导向示例）：

```bash
python3 launch_scientist_zhipu.py \
  --load_ideas ai_scientist/ideas/my_topic.json \
  --submission-mode \
  --target-venue nature \
  --quality-preset publishable \
  --review-reflections 2 \
  --review-ensemble 3
```

### B. 连续自治运行（推荐长期迭代）

```bash
python3 continuous_research_daemon.py \
  --source-config stable_source_priority.example.json \
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

---

## 关键产物与观测点

### 单项目目录下

- `self_review_iteration_summary.json`
- `self_review_final_progress.json`
- `research_program.md`
- `stage_standards.json`
- `process_alignment.json`
- `self_evolution.json`
- `repair_plan.json`
- `experiment_todo.json` / `experiment_todo.md`
- `experiment_todo_progress.json` / `experiment_todo_progress.md`
- `quality/high_quality_result.json`

### 跨项目知识库

- `knowledge_base/self_evolution_history.jsonl`
- `knowledge_base/self_evolution_playbook.json`

### Daemon 目录下

- `daemon_status.json`
- `cycle_history.jsonl`
- `latest_cycle_summary.{json,md}`
- `latest_operator_brief.{json,md}`
- `latest_handoff_report.{json,md}`
- `reports/index.{json,md}`
- `reports/trends.{json,md}`
- `latest_live_dashboard.{json,html}`

### 管理索引

```bash
python3 research_manager.py rebuild-index
python3 research_manager.py submission-board --top 5 --require-gate
python3 research_manager.py submission-board --top 5 --require-gate --min-self-evolution-score 85
python3 research_manager.py submission-board --top 5 --require-gate --min-process-alignment-score 80
python3 research_manager.py rewrite-board --top 10 --min-rewrite-gain 0.5
python3 research_manager.py repair-board --top 20 --priority-tier p0
python3 research_manager.py evolution-board --top 20
python3 research_manager.py process-board --status blocked --top 30
python3 research_manager.py shortlist --top 5 --require-gate --require-ready --min-self-evolution-score 85 --min-process-alignment-score 80
```

---

## 公开 TODO（Roadmap）

> 这些是明确公开的后续迭代项，欢迎 issue / PR 协作。

- [ ] 把 TODO closure 信号进一步并入 `guardrail_mode` 相位切换（不仅 governor）
- [ ] 增加跨周期 closure trend 的回归检测与自动阈值建议
- [ ] 增强 evidence 指标与具体实验结果（figure/table/metric）的直接绑定
- [ ] 增加“submission-ready 套件”质量回归测试（dossier consistency checks）
- [ ] 把 `evolution-board` 和 `self_evolution_playbook` 进一步接到自动 rewrite / experiment follow-up 执行器，形成更主动的 reviewer-driven closure loop
- [ ] 把 `repair_plan.json` 和 `process_alignment.json` 进一步接到自动执行器，形成更直接的 process-level self-repair runtime
- [ ] 建立标准化 benchmark topic 集，用于比较不同模型/策略组合
- [ ] 增加更细粒度成本观测（按 stage/round/source 计费）
- [ ] 提供更完整的英文文档与 API 风格文档（便于外部协作）

---

## Thanks / 参考资料

我们感谢以下工作提供的公开经验与启发（按相关性排序）：

- [Sakana AI: AI Scientist](https://github.com/SakanaAI/AI-Scientist)
- [Sakana AI: AI Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2)
- [awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing)
- [AIDE](https://github.com/WecoAI/aideml)
- [DeepReviewer-v2](https://github.com/ResearAI/DeepReviewer-v2)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

同时感谢开源社区在代理系统、科研自动化、论文写作工具链方面的长期贡献。

---

## Citation

如果你在研究中使用了本仓库的工程方案，请同时引用上游基础工作与本仓库（建议在方法或系统实现部分注明“fork + substantial extensions”）。

上游论文（AI Scientist-v2）：

```bibtex
@article{aiscientist_v2,
  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},
  author={Yamada, Yutaro and Lange, Robert Tjarko and Lu, Cong and Hu, Shengran and Lu, Chris and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2504.08066},
  year={2025}
}
```

---

如你希望，我可以下一步把 `PROJECT_USAGE.md` 和 `docs/CONFIG_REFERENCE.md` 也统一成同一套“运营手册风格”，并补一版英文 README（README.en.md）。

---

## License

本项目采用 Apache-2.0 License，详见 `LICENSE`。

## Contributing

欢迎 Issue / PR。贡献指南见 `CONTRIBUTING.md`，行为准则见 `CODE_OF_CONDUCT.md`。
