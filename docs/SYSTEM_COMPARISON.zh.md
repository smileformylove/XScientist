# 科研系统全方位对比（来源审计，不是排行榜）

本页根据附带的 Expo Talk 先整理方案，再以截至 2026-08-23 可获得的论文、
官方项目页和代码仓为主来源。演讲稿是“发现线索”的资料，不是统一实验协议；
只在演讲稿出现的细节统一标成 `talk_reported`，不会当作本项目已经复现的结果。

可以离线生成同一份机器可审查报告：

```bash
xscientist benchmark systems --json > system-comparison.json
xscientist benchmark systems --workspace ./first-study --show-process
```

该命令不调用 Provider、不联网、不启动外部 rollout，也不聚合跨系统分数。传入
`--workspace` 后，会把与 AutoResearchEval-inspired pilot 相同的有界、脱敏过程视图
带入报告：commit、分支拓扑、中间 artifact 数量和公平性状态都能审查，但不导出
prompt、completion 或自由 payload。
机器字段还会明确标注 rollout/cost 只属于“本次审计”（`this_audit_only`）；历史轨迹的
真实成本是 `unobserved`，绝不会把审计的 $0 写成历史成本。

## 证据标签怎么读

| 标签 | 含义 | 明确不代表 |
| --- | --- | --- |
| `reported_primary` | 论文或官方项目描述/评测了该能力 | XScientist 已复现，或该能力普遍成立 |
| `reported_talk` | 附带演讲稿报告了该细节 | 同条件、同行评审或独立审计结果 |
| `local_observed` | XScientist 本地产物暴露了该信号 | 模型质量、科研新颖性或胜率 |
| `local_structural_only` | 本地 pilot 只检查契约/证据形状 | 自主 rollout |
| `scoped_component` | 系统只覆盖科研栈中的一个层 | 端到端科研能力 |
| `not_measured_here` | 本仓库没有运行该外部系统 | 外部系统失败、较弱或不具备该能力 |
| `not_in_scope` | 来源定义了更窄的组件边界 | 系统永远不能做这件事 |

矩阵故意没有“排名”列。论文写作、科研绘图、MLE 搜索、证据完整性审计和端到端
发现 benchmark 解决的是不同问题，不能压成一个总分。

JSON 的每一行还带有 `human_evidence`：本次审计中 `score` 始终为 `null`、
`same_condition` 始终为 `false`。`human_reference_proxy` 表示历史/参考结果，
`human_SOTA_reference` 表示已发表的人类设计 SOTA/参考结果，
`human_judgment_calibration` 表示人类只参与评审器校准，`human_agent_process` 表示
人机干预或流程增益研究；没有任务性能臂时保留 `not_reported`。这样不会把人类论文、
评审分数或目标解悄悄冒充人类基线。

演讲稿中只出现的架构也不会被悄悄删掉：Deep Researcher Agent 和 Autonomous
Science Team（AST）以 `reported_talk` 行保留；CAST 示例论文、Paper Assistant
Tool 集成、SingleAgent/人类 benchmark 行以及 AutoResearchEval 本身则列在 JSON
的 `talk_inventory` 邻接参考中，不伪装成独立竞品。只在背景页出现的名称和
ScientistTwo 等未来概念会另列 `context_only_mentions`，保留页码但不生成 benchmark 行。

人类对照是另一层证据。请看[来源审计的人类基线清单](HUMAN_BASELINES.md)：它把真正
招募参与者运行的任务（如 RE-Bench、PaperBench、DiscoveryWorld）与已发表 human-SOTA
参考、专家验证、人机协作过程研究分开。清单中的数字不会注入本矩阵，也不会写入
`workspace.score`。

MLE-STAR、DS-STAR 是为了覆盖执行层而补充的相邻主来源方案。FAR
（Find–Attempt–Recommend）则是为了覆盖“文献 → 开放问题 → 尝试/分配”入口而补充的
相邻主来源发现方案。它们没有被声称出现在这份 107 页附件中，因此 JSON 中的
`talk_slides` 为空；结果仍只按各自论文、仓库和版本化评测协议解释，XScientist 没有在
本地重跑这些方案。

## 对比对象和边界

| 系统 | 层级 / 输入输出边界 | 来源中最突出的机制 | 可核对的评测锚点（仅为来源报告，本仓库未重跑） |
| --- | --- | --- | --- |
| **XScientist** | Git-like 证据与科研工作流底座；pilot 从任务契约或工作区开始 | typed objects、追加式 checkpoint、分支审计、`trace → replay → verify`、公平性门禁 | first-run 与 AutoResearchEval-inspired 结构 conformance；**0 次 rollout、没有科研分数** |
| **Deep Researcher Agent（演讲稿参考）** | 广义 deep-research / 科研助手参考 | 演讲稿提到 deep research 与 test-time diffusion | 只有演讲稿来源；没有匹配 task slice 或本地重跑 |
| **Autonomous Science Team（AST）framework** | 演讲稿中的多角色分解 | Generator → Implementor → Paper Writer → Paper Reviewer | 只有架构页；未识别独立分数或可运行协议 |
| **Science One / ScientistOne** | 端到端：问题调查 → 发现 → 论文/claim verification | Chain-of-Evidence 在生成时建立，另有四项完整性审计 | ADRS 五任务、75 篇论文 CoE audit、MLE-Bench/Parameter Golf；论文报告 0/337 幻觉引用和 12/12 score verification，但 XScientist 未复现（ADRS 的“human”是已发表参考，不是新招募的人类臂） |
| **Sakana AI Scientist v2** | 端到端 idea → 实验 → 分析 → manuscript | progressive best-first experiment tree search 与 VLM review loop | 三篇 ICLR workshop 自主稿件（运行外围仍有人工主题/idea 设置与选择）；不是 XScientist 的匹配运行 |
| **AutoResearchClaw** | 端到端 23 阶段流水线 | 多 Agent debate、自修复 `Pivot/Refine`、只读结果、定点 HITL、跨 run lessons | ARC-Bench 实验/端到端模式与消融；比较前必须锁论文/仓库 commit |
| **DeepScientist** | 长时目标驱动发现 | 分层 hypothesize → verify → analyze 与 Findings Memory | 大规模、GPU 密集型 progressive discovery；其预算不能和 provider-free pilot 横比 |
| **AI-Researcher** | 端到端 survey → 算法实现 → 可投稿论文 | survey、coding、writing 专门 Agent 与 code-validate-refine | Scientist-Bench guided/open-ended；是 Agent benchmark，不是人类同条件运行 |
| **FAR（Find–Attempt–Recommend）** | 文献 → 开放问题池 → 候选尝试 → judge/grade 发现 | 从研究方向提取未解决问题，对每个看似良定义候选尝试，再把通过的结果送去分级/专家审阅 | 组合数学 pilot 报告 4,717 个猜想已尝试、1,050 个 `NEW`、598 个 judge `PASS`、77 个 grade 可发表；**仅为主来源报告，本仓库未重跑，也没有人类任务性能臂** |
| **MARS** | 面向 MLE 的执行/搜索组件 | budget-aware MCTS、Design–Decompose–Implement、反思记忆 | MLE-Bench 与跨分支 lesson transfer；不是完整文献到论文系统 |
| **AdaEvolve** | 自适应演化优化组件 | 按累计改进信号调整探索强度和资源分配 | ADRS / open-ended optimization；属于搜索组件，不是完整科研流水线 |
| **EvoX（Meta-Evolution）** | 元进化搜索策略组件 | 同时演化候选解和选择/变异策略 | ADRS / 广泛优化任务；必须锁定任务和 evaluator 版本 |
| **MLE-STAR** | MLE 搜索与定向代码细化组件 | web search、消融选高影响代码块、内外层 refinement | MLE-Bench Lite；没有论文写作/claim grounding 契约 |
| **DS-STAR** | 异构数据科学规划/执行组件 | 数据文件分析、verifier、顺序式计划修正 | DABStep、KramaBench、DA-Code；不是文献到论文发现 |
| **ScholarPeer** | 只做同行评审组件 | live literature context、historian/baseline scout、多维质询 | ScholarEval（DeepReview-Bench + AgentReview，另有 DeepReview-13K 子集）；不执行被评论文的实验 |
| **PaperOrchestra** | 只做写作：原始材料 → LaTeX | 文献综合、章节规划、视觉生成、论文组装 | PaperWritingBench（200 篇论文反向构造材料；数据发布需单独锁版本核验）；不能验证实验本身 |
| **PaperBanana / PaperVizAgent** | 只做图：上下文/参考 → diagram 或 plot | Retriever/Planner/Stylist/Renderer/Critic 迭代 | PaperBananaBench（292 张方法图）；图质量不是发现质量；论文的 `Human=50` 是评测器参考刻度，不是人类绘图实测分数 |

主要来源：[ScientistOne](https://arxiv.org/abs/2605.26340)、[AI
Scientist-v2](https://arxiv.org/abs/2504.08066)、
[AutoResearchClaw](https://arxiv.org/abs/2605.20025)、
[DeepScientist](https://arxiv.org/abs/2509.26603)、
[AI-Researcher](https://arxiv.org/abs/2505.18705)、
[FAR](https://arxiv.org/abs/2608.16977)（[官方仓库](https://github.com/zeyu-zheng/FAR)）、
[MARS](https://arxiv.org/abs/2602.02660)、
[AdaEvolve](https://arxiv.org/abs/2602.20133)、
[EvoX](https://arxiv.org/abs/2602.23413)、
[MLE-STAR](https://arxiv.org/abs/2506.15692)、
[DS-STAR](https://arxiv.org/abs/2509.21825)、
[ScholarPeer](https://arxiv.org/abs/2601.22638)、
[PaperOrchestra](https://arxiv.org/abs/2604.05018) 和
[PaperBanana](https://arxiv.org/abs/2601.23265)。官方/作者仓库也会保留在 JSON 报告中；
这里的“官方”仅指作者发布的研究代码，并不等于产品支持。Google Research 仓库自身的
“not an officially supported Google product”免责声明仍然有效。

对 FAR，漏斗数量和分配分析只作为来源报告的锚点保留。论文中人工抽查的子集是作者
按兴趣选择的，不能读成 100% 准确率；judge/专家审阅也不是人类完成任务的对照臂。
该来源是 arXiv 预印本，不是独立审计的跨系统 benchmark。XScientist 没有在这里运行
FAR 仓库或其组合数学语料，因此 JSON 明确保持 `reported_primary` /
`not_measured_here`。

## 按能力维度看差异

| 维度 | XScientist 当前实况 | 演讲稿中的端到端系统 | 演讲稿中的专门系统与相邻参考 |
| --- | --- | --- | --- |
| 问题定义 | 本地检查 question/plan/falsifier；pilot 不自动发现 benchmark | 通常由人给定问题；ScientistOne/ARC 描述 investigator/scoping | FAR 从人指定研究方向形成开放问题池；MARS/MLE-STAR/DS-STAR 接受任务 tuple 或数据文件；评审/写作/绘图从后段开始 |
| 文献与来源 | 有 provenance/契约表面，但 pilot 不做 provider 检索 | ScientistOne、ARC 明确检索并 grounding；AI-Researcher 有 collector/filter | ScholarPeer 把实时上下文检索作为核心；PaperOrchestra 消费输入材料 |
| 探索与分支 | 可审计 branch/checkpoint 元数据；不声称有模型生成搜索轨迹 | AI Scientist v2 是实验树；ScientistOne 是 explore/exploit；DeepScientist 是持久分层搜索 | FAR 枚举并尝试文献导出的候选；MARS 用资源约束 MCTS；AdaEvolve/EvoX 自适应演化搜索；其他组件不负责完整发现搜索 |
| 执行 | typed attempts、receipt、失败和 closure 可审计；无 rollout 分数 | 五个端到端系统都报告执行，但预算/环境不同 | MARS/MLE-STAR/DS-STAR 聚焦资源/执行；其余组件位于下游 |
| 结论与 claim | closure、review debt、provenance、gate 是本地可见信号；不推断质量 | ScientistOne 明确 claim evidence chain；ARC 报告 verified result；其余需 artifact audit | ScholarPeer 可质询 claim；PaperOrchestra/PaperBanana 不能证明实验真实 |
| 写作/图 | 当前 pilot 不测生成质量 | 端到端系统都含写作，但协议不同 | PaperOrchestra/PaperBanana 应单独评测 |
| 反馈/进化 | 追加历史、repair/gate、自进化契约可审查 | ARC、DeepScientist 明确持久 lessons；AI Scientist v2 主要是 tree search | FAR 报告资源分配分析但没有持久自进化臂；MARS 明确研究跨分支 lesson transfer |
| 过程/公平 | 当前最强本地信号：branch membership、时间线、来源总量、公平性校验、脱敏 | 外部论文记录预算/分支的方式各异；不能自动称为 Git-like | 组件 benchmark 需各自的匹配 harness |
| 复现/导出 | `fsck`、bundle、CAS/ARA pointer、`trace → replay → verify` | 外部系统公开的 logs/code 不一，必须独立重跑 | 写作/绘图输出仍需独立 artifact/evaluator 检查 |

## 真正公平的比较需要什么

若要把 XScientist 与某一个端到端系统做实测，必须把以下内容写入 manifest，并在
每条轨迹中记录 digest：

1. 相同 task、starting artifact、数据快照和网络/工具政策；
2. 相同 backbone/model 版本、prompt/适配 commit、硬件、wall-clock/turn/GPU
   预算、成本上限和 seed 数；
3. 相同 evaluator、优化方向、容差、重试规则和停止规则；
4. 相同输出契约（代码、日志、claim、论文和证据包）；
5. 独立 canonical rerun，以及对 citation、数字、method-code alignment、
   specification violation 的 artifact-aware audit；
6. 过程指标：分支数与分叉基点、失败/完成尝试、修复、人类介入、时间/成本、
   证据完整度和评审不确定性。

`workspace.process.fairness` 在 task slice、budget、evaluator、base 都有证据前不会
把 `eligible` 设为真。看到两条分支不等于有公平实验臂；当无法证明每条分支都有结果
时，报告会明确 `artifact_scope: current_checkout_only`。

## 对 XScientist 的优化启发

这是一张集成路线图，不是“一个系统吞并其他系统”的结论：

- 借鉴 ScientistOne 的 claim-level evidence-chain 词汇，把 typed research object
  接到论文 claim；
- 把 AutoResearchClaw 的失败→修复→进化记录接到 XScientist 的追加历史和公平性 gate；
- 吸收 DeepScientist/MARS 的 Findings Memory 与资源约束规划，但把 lesson provenance
  和 credit assignment 固化成 typed objects；
- 借鉴 FAR 的“研究方向 → 候选池 → 尝试 → judge → grade”漏斗，完整保留 `NONE`、
  `KNOWN`、失败/无效等结果，不只统计成功发现；
- 将 FAR 风格的预期产出/预期重要性分配作为可选策略，并记录校准元数据；不能把其
  组合数学比率外推到其他领域；
- 将 AdaEvolve/EvoX 式搜索策略自适应做成可选 optimizer adapter，同时保存候选解/策略
  的 lineage 和 evaluator digest；
- 将 MLE-STAR/DS-STAR 作为执行层 adapter，任务分数与 claim/论文质量严格分开；
- 把 ScholarPeer 接成 adversarial review adapter，保存 search receipt，而不是复制评审 prose；
- 把 PaperOrchestra/PaperBanana 放在下游，输入只能来自已验证 artifact，不能反过来成为实验真值；
- 只有锁定版本、evaluator、数据和许可边界后，才为 ADRS、ARC-Bench、Scientist-Bench、
  MLE-Bench、PaperWritingBench、PaperBananaBench 编写匹配 adapter。

## 明确不声称

当前仓库**不声称** XScientist：

- 超过人类、ScientistOne、AI Scientist、AutoResearchClaw、DeepScientist、
  AI-Researcher、FAR、MARS、ScholarPeer、PaperOrchestra 或 PaperBanana；
- 复现了演讲稿或外部论文中的任何数字；
- 仅因为本地 process audit 完成，就完成了 provider-backed autonomous trajectory；
- 可以把评审/绘图/写作分数转换成科研发现分数。

当前诚实结论是过程/证据结论：项目能在不导出 prompt 和隐藏自由推理的前提下，展示
中间决策、分支、closure level 和公平性阻塞项。下一步应是注册一个匹配 rollout，
而不是再堆一张没有协议边界的“大表”。
