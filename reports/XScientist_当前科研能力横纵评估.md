# XScientist 当前科研能力横纵评估

> 研究时间：2026-08-08 | 评估基线：XScientist main@0ef61fe | 类型：本地代码、协议、用户旅程与两轮优化验证

## 一、结论先行

XScientist 已经跨过了“论文生成 Demo”阶段，形成了一套可运行的计算科研工作台：它能从选题、文献检索、想法生成、实验代码搜索与执行，一直走到写作、审稿、修复、归档和本地科研版本控制。它也能把失败、超时、反证和未完成验证保留下来，而不是只展示最好看的成功结果。本报告先保留审计基线的缺口判断，再在第十三、十四节记录针对这些缺口完成的两轮实现；下表同时给出优化前后评分。

但它还不是一个能够独立产出可信科学结论的通用“自主科学家”。更准确的定位是：

> 面向计算型、尤其是机器学习研究的受监督自主科研系统，加上一套已进入本地 Beta、仍需外部信任基础设施继续硬化的科研原生版本控制与治理协议。

五个维度的当前成熟度如下。评分不是代码量评分，而是“一个真实用户能否在不绕过系统的情况下稳定获得对应保证”的评分。

| 维度 | 审计基线 | 优化后 | 当前判断 |
|---|---:|---:|---|
| 自科研能力 | 3.4 / 5 | 4.0 / 5 | 研究计划已进入 BFTS 执行合约，高层命令自动采集复现 provenance；最终科学验证仍需独立主体 |
| 自进化能力 | 2.6 / 5 | 4.0 / 5 | 已有真实候选、成对 benchmark、canary、签名发布和精确回滚；云平台与物理隔离 evaluator 尚待适配 |
| 科研透明度 | 3.2 / 5 | 4.0 / 5 | 对象、运行、失败、部署与回滚均可寻址和验证；还缺外部不可变时间与公开透明日志 |
| 科研协议可行性 | 2.8 / 5 | 4.1 / 5 | 33 份 schema、跨语言 canonical JSON 和五类互操作导出已落地；仍需第二个独立完整实现验证生态兼容性 |
| 用户“科研 Git”体验 | 3.2 / 5 | 4.1 / 5 | 已补选择器、恢复/回退、分支维护、结构化错误、完整旅程和标准导出；TUI、bisect 与远程协作仍可继续打磨 |

一句话概括：优化后的系统已从“功能很强的 Alpha 原型”进入“可用于受监督真实项目的本地 Beta”——执行闭环、透明留痕和科研 Git 日常操作已经成立，但独立第三方信任、外部时间锚与具体生产平台权限仍不能由本地代码自行证明。

## 二、这套系统是怎样长出来的

从 Git 历史看，XScientist 的路线不是一开始就设计出完整科研协议，而是沿着真实痛点逐层加固。

### 第一阶段：自动化论文流水线

2026 年 4 月至 5 月，系统的重心是自动生成和改进论文：想法生成、代码实验、写作、审稿与修复。这个阶段解决的是“能不能完成一次端到端计算研究”。

### 第二阶段：ARA 把一次运行变成研究对象

2026 年 7 月开始，Agent-Native Research Artifact（ARA）进入主线。系统不再只保留 PDF，而是保存探索 DAG、节点代码、终端输出、指标、图、失败分支、LLM 调用、环境和 claim 锚点。`xscientist ara inspect/exec/fork/validate/verify/graph/context` 让一次研究运行可以被检查、重放和继续。

这一步非常关键。它把“论文是结果”改成“研究过程和证据图才是结果，论文只是一个视图”。

### 第三阶段：科研完整性与受控自进化

2026 年 8 月初加入了确证研究门禁、独立评估、科学宪法、认识论图和自进化治理。设计开始明确区分：

- 生成者的自评分和独立验证；
- 探索性结论和确证性结论；
- 当前运行的反思和生产系统的永久改动；
- 候选变更、影子测试、canary、人工批准和生产晋升。

### 第四阶段：Research VCS 成为用户层

2026 年 8 月 6 日至 8 日，系统连续加入本地科研 Git、CAS、大文件指针、离线 bundle、语义对象、语义分支与合并、用户友好的生命周期命令，以及最新的 payload-free closure audit。

今天的架构因此包含四个互补层：

1. 自动科研流水线负责做研究；
2. ARA 负责保存一次运行的详细研究过程；
3. Research VCS 负责保存跨运行的科学对象与状态变迁；
4. Git 和本地 CAS 负责可靠持久化。

这个历史解释了当前的优势，也解释了当前的主要问题：系统的不同层是在很短时间内叠加出来的，功能增长快于协议收敛，ARA 与 Research VCS 之间仍存在语义重复和桥接不完整。

## 三、自科研能力：能完成研究，但不能把“完成”自动等同于“可信”

### 3.1 已经真实可运行的闭环

主项目入口并非空壳。`ai_scientist/apps/project.py` 实际串起：

- 文献检索与想法生成；
- idea card、研究计划、预注册和阶段合约；
- BFTS 多阶段实验搜索；
- Python 代码生成、执行、keep/discard/crash 记录和多 seed 重评；
- 图表聚合、论文写作、引用收集；
- 多轮 reviewer、hostile critic、issue ledger 和 owner-aware repair；
- 质量门、投稿门、完整性检查；
- ARA 导出和 Research VCS checkpoint。

实验执行也有真实的运行安全与恢复能力。默认 BFTS 配置要求 Docker 隔离、禁网、只读根文件系统、能力降权和 CPU/内存/PID 约束；journal 与 checkpoint 在派生账本之前落盘；预算耗尽、中断和异常都会保存可恢复状态；恢复时会校验 checkpoint 与 ledger 一致性；实验锁会阻止并发覆盖。

因此，把它称为“自主科研流水线”是合理的。把它称为“通用自主科学家”则过早。

### 3.2 最值得肯定的科学边界

自动项目即使内部质量分很高，也不会把结果直接写成已验证科学结论。项目投影到 Research VCS 时：

- evidence 标记为 `awaiting_independent_verification`；
- gate 写入 `hold`；
- manuscript 保持 `draft`；
- `claim_promotion_allowed` 为 false。

对应实现位于 `ai_scientist/apps/project.py:618-733`。这说明系统清楚地区分了“生成了一篇看起来不错的论文”和“获得了可以提升的科学结论”。这是整个项目最重要的设计自觉之一。

### 3.3 自科研尚未闭合的地方

第一，研究计划目前更多是可审计合约，而不是实验执行器的硬约束。

研究规划会生成 rival hypotheses、零效应、机制替代、测量伪影、边界条件和鉴别实验。但 BFTS 的实际输入主要仍由原始 idea 转成的 `idea.md` 提供，`ai_scientist/treesearch/` 没有完整消费 `research_plan` 和 Socratic challenge 的强制路径。代理可能得到更高指标，却没有完成计划要求的负对照、机制消融和边界探测。

第二，严格科研模式不是默认路径。

`--high-quality-mode`、`--require-quality-gate`、严格写作守护需要用户显式开启；integrity forensics 主要在投稿或高质量模式下默认启用。默认模式适合探索和产出草稿，不应被解释成完成了强科学验证。

第三，ARA 归档偏后且通常是 best-effort。

完整 ARA finalization 发生在论文与审稿流程后段。早期失败有 BFTS 日志和 checkpoint，但不一定有同等级 ARA。ARA export、claim scan 或 re-exec 失败不会默认阻断主任务，这对保存昂贵研究产物有利，却不适合需要 fail-closed 的严格透明场景。

第四，领域范围仍窄。

当前执行器围绕 AI 生成的 Python 代码、指标、图和论文组织，适合机器学习和其他纯计算研究。湿实验、仪器采样、实验室审批、实体设备和真实世界现场研究没有通用执行适配层。项目自己在 README 中也承认，越远离数学、物理和可计算验证，人类判断越不可替代。

### 3.4 自科研结论

XScientist 已经能作为“受监督的计算科研工作台”使用。它最适合：

- 有明确数据集、指标和基线的计算实验；
- 可以在隔离环境中运行代码的研究；
- 需要批量探索、保留失败、生成论文草稿和审稿修复队列的团队；
- 愿意由人类或独立服务完成最终科学验证的用户。

它不适合被承诺为：无需专家监督、跨学科通用、自动获得可信新知识的系统。

## 四、自进化能力：治理设计先进，真实自修改闭环还没接上

### 4.1 已实现的三层模型

XScientist 把自进化拆为三层，这是正确方向。

- L0 episodic：当前运行中的反思与修复，不产生生产级永久变更；
- L1 playbook：跨项目聚合重复问题，形成建议性默认策略；
- L2 system：对 prompt、工具、搜索、资源分配或恢复策略提出候选变更，并走独立门禁。

L0 的旧 `AutonomousEvolutionEngine` 能收集反馈、反思并生成策略，但部分“执行动作”仍是占位性质，例如返回固定的 enhanced 文本或估计收益。因此它更像真实的反思原型，不是系统修改器。

L1 已经比较扎实：reviewer 指标、repair lane 和阶段 blocker 会被转换成结构化 lesson，写入 append-only history 和跨项目 playbook；自评分学习进入 quarantine，不会直接成为可信全局知识。

L2 的控制面也有大量真实实现：

- 固定效用 epoch；
- 每 epoch 与每组件预算；
- explore/exploit 配额；
- 单变更因果约束；
- 重复机制与 futility stopping；
- 宪法保护的不可变组件；
- 消融、hidden/prospective benchmark、独立 evaluator stack；
- canary、OOD、长尾、common-mode、回滚与人工批准；
- `evolve/*` 分支和稳定线合并门禁。

单看治理设计，自进化安全能力可以达到 4.0 / 5。

### 4.2 为什么综合只有 2.6 / 5

因为当前缺的不是规则，而是数据面执行器。

`evolution_program.json` 生成的 intent 状态是 `awaiting_candidate_artifact`。仓库里没有一条完整自动路径把 intent 转成真实 prompt/tool/code diff，放入隔离工作树，生成基线和候选 artifact，再自动运行消融与盲测。

`xscientist evolution-gate` 读取用户提供的 candidate、benchmark、ablation 和 canary JSON，并验证字段、哈希和阈值。它不负责产生这些证据。`ResearchEvolution.promote()` 和 `rollback()` 也主要是记录被提升或回滚的语义对象，不会真正部署候选文件或恢复生产文件。

哈希能够证明“这份记录后来没有变化”，但不能证明“这份记录描述的实验真实发生过”。当前 attestation 没有签名、密钥身份、远程证明或受控 benchmark custody。一个拥有本地写权限的调用者可以构造形式合格的 JSON。

所以当前自进化应描述为：

> 可信自进化协议和控制面原型，而不是已经可以独立、安全地改写、评估、部署并回滚自身的系统。

### 4.3 下一步最关键的三件事

1. 实现 intent 到真实候选 diff/CAS artifact 的 candidate builder；
2. 实现受控 benchmark runner、签名 attestation 和独立 evaluator service；
3. 实现真实 canary/deploy/rollback adapter，让 receipt 来自执行动作，而不是字段比较。

## 五、科研透明度：内部可观测性高，第三方可信度中等

### 5.1 透明度做得好的地方

ARA 可以保存探索 DAG、失败分支、代码、终端输出、指标、图、LLM 调用、环境、claim 锚点、修复轨迹和验证报告。Research VCS 又保存跨运行的 typed objects、relations、checkpoint、语义分支、merge 冲突和 reproduction receipt。

用户可以从多个视图理解同一研究：

- `research log/show/diff/blame/tree` 看跨阶段历史；
- `research audit` 看 claim 到 evidence、attempt、plan、gate、reproduction 的闭环；
- `ara graph/log/diff/claims/context` 看单次运行的探索树和上下文；
- `exploration_graph.html` 看人类可读 DAG；
- API 提供项目 Research VCS status 和 payload-free audit。

大数据、模型和日志可以进入本地 CAS，Git 中只保留不可变指针。payload-free audit 只返回 ID、哈希和缺口，兼顾隐私、规模和可审计性。

### 5.2 透明不等于可信

当前的透明度更接近“记录丰富、方便复盘”，还不是“第三方不可伪造账本”。

最严重的问题是 manifest history 验证。`verify_manifest_lock()` 只检查当前 manifest hash 是否等于 lock，或者是否等于 history 最后一行的 `new_hash`；它没有完整验证第一行、相邻 hash 链、revision 连续性、历史快照和签名。一个本地写入者可以同时修改 manifest 与最后一条 history，让结果被报告为 `revised`。因此这里应该称为本地一致性检查，而不是对抗性不可篡改证明。

LLM tracing 也可能被关闭，异常会被吞掉；普通 JSONL 账本缺少锁、fsync 和 hash chain；`validate_ara` 不覆盖所有 LLM、event、lock、history、context 和 CAS 关系。Git hash 能保证当前内容自洽，但本地 Git 历史仍可被强制重写。没有签名或外部不可变锚点时，系统不能证明作者身份和真实时间顺序。

### 5.3 透明度的合理分级

建议对外公开五个等级，避免一个 `ok` 承载过多含义：

1. syntax：JSON Schema 结构合法；
2. structure：DAG、计数和跨文件引用合法；
3. integrity：lock/history/CAS/hash 完整；
4. replay：代码、数据、环境与命令闭包可重放；
5. verify：独立 gate 和独立 reproduction 可信通过。

当前 XScientist 在 syntax 与本地 integrity 上较强，在 replay 上部分可用，在对抗性 verify 上仍弱。

## 六、科研协议是否可行

### 6.1 作为本地单实现协议：可行

25 个 JSON Schema、Draft 2020-12、SHA-256、普通 Git、tar 和本地 CAS 都是务实选择。对象和 checkpoint 有内容哈希；读取时会重算 Research Object 身份；`fsck` 能校验 checkpoint 父图、对象指针、CAS 字节和 ARA manifest hash；bundle verifier 对路径逃逸、重复成员、未声明成员、大小与 hash 不一致做了较强防护；restore 先在临时目录验证再原子发布。

对于单个可信用户、同一 Python 实现、本地离线使用，这套协议是可行的。

### 6.2 作为跨工具科研标准：暂不可行

当前有六个阻断点。

第一，参考验证器的跨 schema `$ref` 没有本地 registry。合法 manifest 一旦使用 `references.pipeline_artifacts`，相对引用可能触发 `Unresolvable` 异常，而不是返回 ValidationReport。

第二，规范要求 kind-specific payload 在入库前验证，但 `build_research_object()` 没调用已有的 `validate_research_payload()`。实测 `research record hypothesis --data '{}'` 可以成功创建空语义对象，直到后续 audit 才可能暴露问题。

第三，canonical hash 使用 Python JSON 行为和 `default=str`，没有采用跨语言 canonical JSON，也没有显式禁止 NaN/Infinity。独立语言实现难以保证产生同一 hash。

第四，`rso-*`、`rcp-*` 等公开短 ID 只使用 hash 前 16 个十六进制字符。64 bit 适合本地显示，不适合作为长期跨组织的权威科研标识。

第五，ARA、Research VCS、Research Git、Epistemic Graph 和 pipeline contracts 同时存在，协议版本与映射关系还没有完整兼容矩阵。一个完整 ARA 可以没有 Research VCS claim；一个 Research VCS verify 也不一定证明对应 ARA 的 graph/claim/verify 真实完整。

第六，规范列出 bisect、promote、revert、clone、gc 等操作，实际 Research VCS CLI 并未完整实现；规范默认稳定 ref 是 `stable`，实际初始化分支是 `main`。这类漂移会让独立实现无法只依赖规范。

### 6.3 verify 目前能证明什么

当前 closure verify 主要证明“对象形状、状态和关系满足约定”，不能证明“独立验证真实发生”。

- verifier 独立性主要靠调用者提供一个非空 ID；
- `review --decision pass` 可以由用户直接产生 passing gate；
- `evidence --verified` 是用户布尔开关；
- reproduction receipt 进入 lifecycle 时只做 schema 校验，没有统一重算 receipt ID/content hash；
- generic `record` 允许直接创建 state=`verified` 的对象。

更严重的是，正常不可变演进会被旧草稿拖累。closure audit 遍历当前 ref 中所有 claim；任何历史 draft 都会阻塞整库 verify。现有测试明确验证了：新的 verified claim 可以自身 complete，但旧 draft 仍让整个 ref `complete=false`。正确语义应该审计未被 supersedes 的有效 frontier，而不是让所有历史草稿一票否决。

因此，现阶段对外应把 `verify complete` 解释为“本地协议关系闭合”，不能解释成“第三方可信科学证明”。

## 七、用户是否真的拥有一套方便的“科研 Git”

### 7.1 已经成立的部分

用户不必理解 Git 内部对象就能使用：

```text
hypothesis -> preregister -> experiment -> evidence -> review -> claim
```

这些高层命令默认创建 typed object 和精确 checkpoint。失败、超时和取消是正式历史；确证实验强制绑定锁定预注册；verified claim 必须有 passing gate。

用户还可以：

- `stage/add/unstage/commit` 精确选择研究变化；
- `branch/switch/tag` 管理独立研究线；
- `diff/blame/tree` 查看科学语义变化和来源；
- 语义 merge 阻止相反证据、锁定预注册、指标定义和未门禁 agent candidate 被静默覆盖；
- 用 CAS 保存大对象；
- 用 bundle 离线备份、验证和恢复；
- materialize 指定 checkpoint 并生成 reproduction receipt；
- 自愿添加普通 Git remote，但系统永不自动 push。

尤其是安全默认：deny-first 隐私策略、文件白名单、秘密扫描、干净工作区才能 switch/merge、bundle 的严格路径与 hash 校验，都比很多实验跟踪工具更适合个人科研档案。

### 7.2 临时仓库实测

本次评估实际创建了一个临时科研仓库，并使用高层命令完成：

```text
init -> hypothesis -> preregistration -> plan -> confirmatory experiment
     -> evidence -> independent review/gate -> verified claim
```

结果：

- `research fsck`：通过；
- `research audit --level trace`：通过；
- `research audit --level replay`：阻塞；
- `research audit --level verify`：阻塞。

replay 的阻塞项是：缺 code identity、dependency lock identity、environment identity 和 evidence hash anchor。虽然命令传入了 `--seed 42`，审计仍报告 seed policy unspecified，因为高层命令把 seed 写在 payload，closure 却只在 provenance 中找 seeds。

这说明日常记录链已经好用，但日常命令还没有自动采集 closure 要求的完整 provenance。用户想从 trace 走到 replay/verify，往往需要退回 Python API、generic record、CAS 和 reproduction 子命令。

### 7.3 当前最影响使用的摩擦

第一，ID 搬运过重。完整链路要反复复制 hypothesis、plan、preregistration、attempt、evidence 和 gate 的 `rso-*` ID。系统缺少 `@latest:hypothesis`、人类别名和交互选择。

第二，Git 用户自然期待的操作不全。没有工作区 restore、语义 revert、branch 删除/重命名、bisect 和 clone。当前 restore 只是把离线 bundle 恢复到新目录。

第三，`--json` 的错误仍是普通文本，没有稳定错误类别，与规范的 structured result 要求不一致。

第四，语义 diff 存在确定性实现错误。`_manifest_delta()` 只有 `before_map`，真正的 after/return 逻辑被放到 `_typed_object_delta()` 的 return 之后，导致 `semantic.ara_manifests` 恒为 null。全量测试仍然通过，说明这里缺少关键字段覆盖。

第五，CLI 和 README 的日常示例没有始终要求 experiment 绑定 plan，但 trace audit 会强制要求 attempt 到 plan 的关系。用户照文档操作也可能在审计阶段才发现链路不完整。

### 7.4 科研 Git 结论

科研 Git 的核心产品判断已经成立：研究者需要版本化的不是文件，而是假设、预注册、实验、证据、反证、声明和验证关系。

当前实现适合技术用户、小规模真实项目和协议验证；还不适合作为普通研究者唯一、无心理负担的日常科研档案。它更像功能完整度很高的 Alpha，而不是像 Git 一样经过多年打磨的基础设施。

## 八、横向对照：XScientist 的独特位置

现有工具大致分成三类。

第一类是自主科研生成系统。[The AI Scientist](https://arxiv.org/abs/2408.06292) 展示了从想法、代码实验、图表到论文和模拟审稿的自动闭环；[Google AI co-scientist](https://arxiv.org/abs/2502.18864) 更强调多代理提出和评估新假设。XScientist 与它们重叠在自动科研，但额外把失败、证据关系、科研分支和自进化治理提升为版本控制对象。

第二类是实验与数据版本工具。[DVC](https://dvc.org/doc/command-reference/) 用 Git-like 模型管理数据、模型、pipeline 和实验；[MLflow Tracking](https://mlflow.org/docs/latest/tracking) 擅长记录 run、参数、指标和 artifacts，并提供服务器与 UI。XScientist 的不同点是，它试图版本化“科学意义”：falsifier、locked preregistration、support/refute、independent gate 和 contested merge，而不只是 run 和文件。

第三类是科研互操作标准。[RO-Crate](https://www.researchobject.org/ro-crate/specification/1.3/introduction.html) 用 JSON-LD 聚合研究数据及其上下文；[Workflow Run RO-Crate](https://www.researchobject.org/workflow-run-crate/profiles/workflow_run_crate/) 描述工作流运行 provenance；[W3C PROV](https://www.w3.org/TR/prov-overview/) 提供通用 provenance 数据模型；[CWL](https://www.commonwl.org/specification/) 提供可移植、跨平台的计算工作流描述。

XScientist 当前最有价值的生态位不是取代这些工具，而是成为它们之上的“科学状态机”：

- DVC/MLflow 管数据、模型和运行；
- CWL 管可执行工作流；
- RO-Crate/PROV 管交换和 provenance；
- XScientist 管假设、反证、预注册、claim、gate、科研分支和自进化候选。

如果 XScientist 尝试独自重新定义所有存储、工作流、provenance、身份和交换标准，会背上过重的协议负担。更可行的路线是提供 RO-Crate/PROV/CWL/DVC/MLflow adapter，把自己的优势集中在科学语义与状态转换上。

## 九、最优先的硬化路线

### P0：先修“结论可能错误”的问题

1. 修复 `_manifest_delta()`，增加 ARA manifest diff 的回归测试；
2. 在 Research Object 持久化前强制调用 kind-specific payload validation；
3. closure 改为审计有效 frontier，并明确 supersedes/promotion 语义；
4. 修复 schema `$ref` 离线 registry；
5. 完整验证 manifest lock/history hash 链和 revision 连续性；
6. gate 必须绑定精确 claim/evidence/attempt closure hash，reproduction receipt 必须重算身份；
7. 禁止公共 raw record 直接产生 verified/promoted 状态。

### P1：打通“可运行但不闭合”的路径

1. 将 research plan 与 required discriminating tests 注入 BFTS 硬执行合约；
2. 提供 `research-strict` 默认 profile：隔离、质量 gate、forensics、ARA strict、VCS strict；
3. checkpoint 时验证 ARA，并从 ARA graph/claims/verify 确定性派生 Research VCS 对象；
4. 高层 experiment/evidence 自动采集环境、代码、数据、依赖和 measurement hash；
5. 实现真实 candidate builder、benchmark runner、canary/deploy/rollback adapter；
6. 给 event、LLM trace 和 manifest history 增加并发安全、hash chain、签名或外部锚定。

### P2：让普通研究者真的愿意每天用

1. 增加 `@latest:<kind>`、人类别名、交互选择和摘要列表；
2. 实现 semantic revert、working-tree restore、branch delete/rename、bisect；
3. 为 status/log/tree/diff 提供 Rich/TUI 和浏览器视图；
4. `--json` 错误统一输出稳定 schema 与 error category；
5. 统一 init/setup/research init 的产品叙事，统一 bundle/export 术语；
6. 增加完整 golden journey 测试和一个非 Python conformance consumer。

## 十、三个未来剧本

### 最可能剧本

XScientist 先成为一套很强的个人本地计算科研工作台。用户把它用于探索、失败保留、论文草稿、研究分支和项目交接，独立验证仍由人或外部服务完成。只要 P0 被修复，这条路线有现实产品价值。

### 最危险剧本

产品文案过早把 `verify complete`、hash 和自进化 gate 描述成可信科学证明。用户看到全绿状态后忽略了自报 verifier、可构造 receipt、可改写本地 history 和弱 ARA bridge，系统反而制造了比普通日志更强的虚假确定感。

### 最乐观剧本

XScientist 收敛为一套小而清晰的“research commit protocol”，用完整 hash、跨语言 canonical JSON、签名身份、RO-Crate/PROV/CWL adapter 和非 Python conformance suite 建立生态。届时它不必取代 Git、DVC 或 MLflow，而是补上它们一直缺少的科学语义层：什么是假设，什么是反证，什么可以合并，什么仍有争议，什么经过独立复现。

## 十一、最终判断

现在的 XScientist 值得继续沿“科研 Git”方向投入，因为这个方向比单纯提高论文生成质量更有长期价值。它已经证明了三个核心命题：

1. 科研过程可以被结构化为 typed objects 和不可变关系；
2. 科研分支与合并需要理解 support/refute、预注册和 gate，而不是只做文本合并；
3. 自进化必须被当作科研对象来验证，而不是给 agent 直接改生产系统的权限。

接下来不应继续优先堆更多命令和 schema，而应完成一次“语义收敛”：修复 P0、统一 ARA 与 Research VCS、把强完整性门禁接入日常 CLI、建立可信身份与跨语言 conformance。完成这些之后，XScientist 才能从“功能丰富的科研基础设施原型”走向“别人敢把真实科研历史长期交给它保存的协议”。

## 十二、证据与方法

本报告基于以下证据：

- 优化基线 `main@0ef61fe`、本轮实现后的代码、协议、文档与 Git 历史；
- 最终全量测试：1125 passed、3 skipped、47 subtests passed；
- 自进化、Research VCS、closure、bundle、onboarding 等定向测试；
- 临时科研仓库的真实 CLI 用户旅程；
- DVC、MLflow、The AI Scientist、AI co-scientist、RO-Crate、W3C PROV 与 CWL 的官方文档或原始论文。

本报告采用横纵分析法：纵向追踪 XScientist 从论文流水线到 ARA、自进化治理和 Research VCS 的演进；横向对照自主科研、实验跟踪和科研互操作标准；最终用代码和实测结果交叉判断当前能力。

## 十三、本轮优化实施更新

评估完成后，仓库已按 P0 → P1 → P2 顺序实施第一轮全面硬化：修复 ARA manifest semantic diff；在 Research Object 构造与读取时强制语义 payload validation；为 manifest schema 建立离线 `$ref` registry；完整验证 manifest history 的 revision、base/new hash 和历史快照链；closure 改为审计有效 claim frontier，并对不可变 draft → verified 晋级做 supersession 处理。

`verify` 语义也已收紧：通过门禁必须来自独立 verified review 并由 deterministic gate 绑定当前 claim/evidence；verified reproduction 必须内嵌可重算 ID/content hash 的成功 computational-rerun receipt，且 verifier 不能与 lineage producer 相同；公共 raw record 不再允许直接创建 verified/promoted 状态。

日常科研 Git 已增加 `@latest:<kind>` 与唯一 ID 前缀、branch 删除/重命名、显式 path restore、语义 revert、结构化 JSON 错误和对象摘要。高层 experiment 自动采集代码提交、环境、依赖锁、数据 hash 与 seed，evidence 自动生成 measurement hash；完整 CLI golden journey 已覆盖 hypothesis → preregistration → experiment → evidence → review/gate → verified claim → execute reproduction → verify closure。

自动科研侧，BFTS 的真实输入现在包含 binding research contract，把 research plan 的 tasks、判别实验、acceptance rules、产物和 execution policy 从旁路文档提升为执行约束。自进化 API 则明确返回 `semantic_receipt_only` 与 `production_mutated=false`，防止把语义晋级记录误解为生产部署。

仍需后续生态建设的项目包括：硬件或远程密钥托管、外部时间锚、真正隔离的 benchmark custody/evaluator 服务，以及 Kubernetes、托管智能体平台等生产适配器。这些不再被当前实现描述成已经闭合的能力。

## 十四、第二轮优化实施更新

针对上一版报告仍列出的执行与生态短板，仓库现在增加了一条可真实运行、默认不触碰生产的自进化路径。`xscientist evolution candidate` 会从基线与候选目录生成不可变 CAS 文件树，拒绝符号链接、路径穿越、保护区修改与未声明 diff；`benchmark` 对同一任务成对运行基线和候选的 argv 命令，记录输出哈希、hard gates 与运行回执；`canary` 把候选部署到显式受控目录，运行真实项目命令，并在异常路径同样强制恢复精确基线。

生产执行不再只依赖形式合格的 JSON。新 `xscientist.canonical-json.v1` 规范化配置由 Python 和 Node.js conformance consumer 共用测试向量；attestation 支持核心 HMAC-SHA256 与可选 Ed25519。生产 apply 要同时通过原有 promotion 语义校验，以及候选生成者、独立 benchmark、canary executor 和独立人类批准的签名验证。production rollback 也需要单独的签名授权。默认命令只产生 plan，只有显式 `--apply` 才会在限定 deployment root 内做可恢复目录交换。

科研 Git 的生态交换也已补齐：`xscientist research export` 对一个已提交 ref 生成 hash-bound export manifest、RO-Crate、W3C PROV-JSON、CWL、DVC 和 MLflow 文件；默认只含 ID、关系、状态与 hash，只有 `--include-payloads` 才输出科研 payload。由此，上一轮列出的 candidate builder、benchmark runner、本地 canary/deploy/rollback、签名身份、五类 adapter 和非 Python conformance consumer 已从路线图转为有自动测试覆盖的实现。仍然不能由本地代码替代的是物理 benchmark 保管、远程独立评估、外部不可变时间、硬件密钥和具体生产平台授权。
