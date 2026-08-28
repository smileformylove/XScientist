# 信念上下文投影

XScientist 从
[Belief Context Graph（BCG）](https://github.com/bigai-nlco/belief-context-graph)
借鉴了一个重要思想：Agent 不能只做检索，还需要在行动前明确看到支持、质疑、
来源、时效和不确定性。

但实现边界刻意不同。XScientist 不创建第二套可变 memory database，也不把概率
信念图注入模型 prompt；它只从一个不可变的 Research VCS 来源闭包，派生出有界、
确定性、仅含元数据的投影。Research VCS 始终是单一事实源。

## 借鉴了什么，没有继承什么

| 关注点 | BCG 的启发 | XScientist 的契约 |
| --- | --- | --- |
| Agent 上下文 | 显式呈现信念、证据、冲突和时间 | 从某个 Research VCS ref 的对象派生只读决策视图 |
| 状态 | 维护以信念为中心的图上下文 | 输出确定性的序数证据状态 |
| 溯源 | 把信念链接到原始上下文 | 每条信号绑定 Research Object ID、内容 hash、关系和来源族 hash |
| 不确定性 | 用置信度判断是否应阻止行动 | 使用具名 blocker 和保守决策姿态；绝不输出校准概率 |
| 生命周期 | 跟踪信念随时间的变化 | 在声明的逻辑时间检查 `valid_until`、失效、撤稿和取代 |
| Agent 集成 | 把信念快照提供给模型上下文 | 将 hash 绑定的投影放入既有 Research VCS context snapshot |

XScientist **不会**复现 BCG 的图构建后端、置信度公式、HTTP 服务、参考 Agent、
prompt 注入方式、benchmark harness 或 benchmark 结果。BCG 公布的准确率和 token
成本数字不能迁移到 XScientist。任何关于 XScientist 的性能主张仍需另行声明
harness、模型、数据集、资源边界和本地结果 artifact。

## Research VCS 是单一事实源

`build_belief_context_projection(...)` 接收精确上下文闭包中已经存在的 Research
Object，并产生派生视图。它不会写入 belief row、更新 confidence 字段或维护隐藏
状态。在相同逻辑时间和限制下，对同一个规范化闭包重新构建会得到相同的
`projection_hash`，与输入顺序无关。

投影会绑定：

- 规范化的目标和来源对象 ID；
- 通过 `source_closure_hash` 绑定每个来源对象的内容 hash；
- 逻辑 `as_of` 边界及其选择方式；
- 支持、质疑、血缘、失效和取代关系；
- 图限制、截断、循环、冲突、blocker 和 warning；
- 通过 `projection_hash` 绑定完整投影。

投影可以辅助决策，但不能让科研主张晋级。顶层和逐目标的
`scientific_promotion_allowed` 始终为 `false`；`quality_claim_allowed` 和
`causal_claim_allowed` 也始终为 `false`。现有 Research VCS 闭包、独立评估和
promotion gate 保留最终权限。因此，序数状态只是后续 gate 的上下文，绝不能
单独作为充分条件或唯一晋级门。

## 序数语义，不是校准置信度

固定的语义标识为：

```text
deterministic_ordinal_evidence_state_not_calibrated_probability
```

这些状态是用于决策的有序标签，不是概率、贝叶斯后验或科学真实性度量：

| 状态 | 观察条件 | 默认姿态 |
| --- | --- | --- |
| `unassessed` | 未观察到有效支持或质疑信号 | `collect_discriminating_evidence` |
| `supported` | 存在有效支持，但观察到的不同来源族少于两个 | 通常为 `seek_independent_review` |
| `corroborated` | 有效支持解析到至少两个不同来源族 | 同时观察到独立权限时为 `review_with_scientific_gate` |
| `contested` | 有效支持与有效质疑同时存在 | `investigate_conflict` |
| `challenged` | 存在有效质疑，或目标处于终止性的负面状态 | `collect_discriminating_evidence` |
| `stale` | 存在支持但已不再有效，或目标已经过期 | `refresh_evidence` |
| `superseded` | 目标已被取代或失效 | 不基于旧目标采取行动 |

`corroborated` 不等于“已验证”。它只是在确定性规则下记录来源族数量。独立权限
会另行检查，并且只有 `human` 或 `independent_evaluator` actor 才算被观察到。
即使目标处于 corroborated 状态，也仍须通过科学 gate。

## 同源去重

来自同一篇论文的多段 passage 不是多个独立来源。投影会沿 `depends_on`、
`derived_from`、`quotes`、`observes` 和 `tested_by` 等血缘关系追溯来源根节点，
然后统计唯一来源族 hash。

支持关系本身仍受对象类型与关系类型约束。显式的 `supports`、
`qualified_supports`、`replicates` 和 `reproduces` 会成为支持信号；此外，当
`claim` 通过 `depends_on` 指向 `evidence`、`passage_evidence`、`inference` 或
`evidence_synthesis` 时，投影会派生一个 `depends_on_evidence` 支持绑定，以免
常规 claim/evidence 结构丢失。指向其他任意对象的普通 `depends_on`，或任意血缘
关系，都**不会**被静默重分类为支持。

对于 `source_snapshot`，会按顺序选择第一个已声明的规范身份：DOI、PMID、arXiv
ID、URL 或来源内容 hash。若无法到达 source snapshot，则可用声明的 producer
actor 作为来源族的后备身份。投影不会复制原始标识，只输出确定性 hash。

这样可以防止重复 passage、summary 或派生对象抬高
`independent_support_source_count`。但它不能证明法律、机构或实验层面的真正独立。
缺失来源根节点会通过 `independence_observed=false` 暴露，血缘截断则成为行动
blocker。

## 时效、失效与冲突

若决策需要在明确边界上重放，请提供带时区的 `--as-of`。未提供时，投影使用
所观察来源闭包中最新的有效 `created_at` 时间；若两者都不可用，逻辑时间标记为
`unavailable`，不会用当前系统时间代替。

显式历史边界是真正的时间截断：在 `as_of` 之后创建的证据会标记为
`not_yet_observed`，不能支持或质疑目标；在该边界之后才创建的撤稿、失效和取代
对象也不会影响这次历史投影，避免后来事件改写过去的决策上下文。若目标本身晚于
该边界创建，投影会变为不完整，而不会假装当时已经存在。

信号可以声明 `valid_until`。过期信号，以及因撤稿、撤回、失效或取代关系而失效
的信号不能继续提供有效支持；格式错误的时效声明会标记为 `invalid`，同样不能
作为有效证据。失效还会沿有界血缘传播。过期目标变为 `stale`，失效或被取代的
目标变为 `superseded`。

支持与质疑会同时保留，不会通过求平均而消失。两者并存时，目标变为
`contested`，获得未解决且确定性的 `conflict_id`，决策姿态指向
`investigate_conflict`。在底层 Research VCS 记录发生变化前，冲突会一直可见。

这些是 fail-closed 的决策规则：`stale`、`superseded`、`challenged`、
`contested` 或 `unassessed` 目标都会收到行动 blocker，而所有状态的科研晋级
权限始终关闭。

## 循环与有界计算

公开投影受硬限制约束：

- 最多 1,024 个节点；
- 最多观察 8,192 条关系；
- 解析来源族时最多追溯 8 层血缘。

调用方可以选择更小的正数 `max_nodes` 和 `max_relations`，但不能超过硬上限。
节点或关系截断、端点落在观察图之外、重复或无效的对象身份，以及血缘循环，都会
让投影变为不完整。此时 `complete=false`、`decision_context_usable=false`，包含
它的 Research VCS context 也会被阻止。

CLI 的 `--budget` 控制包含该投影的 context snapshot 中供人阅读的语义 working
set；它绝不会裁剪构成硬来源闭包的不可变 ID 和 hash。若必需的决策语义无法放入
预算，context 会 fail-closed，而不是静默删除不方便的证据。

## 审计边界

`audit_belief_context_projection(...)` 在不返回来源陈述或原始证据 payload 的前提
下检查公开契约。它检查 policy 与 semantics 标识、规范化目标、完整性一致性、
不可提升的权限标志和 projection hash。输出仅包含有界且稳定的 issue code，并有
自己的 `audit_hash`。

审计结果为 `verification_allowed=true`，只表示投影 artifact 通过了这些完整性
检查；它不表示某个 belief 为真，不表示来源独立性已经被外部证明，也不表示科学
主张可以晋级。审计结果会再次把三个 claim 权限标志固定为 `false`。

## CLI

为一个或多个 Research Object selector 构建投影：

```bash
xscientist research belief @latest:hypothesis \
  --repo ./first-study \
  --ref HEAD \
  --as-of 2026-08-28T00:00:00+00:00 \
  --budget 4000 \
  --json > belief.json
```

`--ref` 默认是 `WORKTREE`；需要精确历史重放时应使用已提交的 ref。只有投影完整
时，命令才成功退出。不使用 `--json` 时，它会打印 projection hash、逻辑时间、
完整性、冲突数，以及每个目标的状态和下一步姿态。

审计原始投影、含 `belief_context` 的 Research Context，或含该 context 的 JSON
对象：

```bash
xscientist research belief-audit belief.json --json
```

只有 artifact 通过投影契约时，审计命令才以状态码 `0` 退出。它刻意不输出来源
原文。

## 范围与限制

该功能是受 BCG 启发的科研决策上下文投影，不是 BCG 的安装或 fork。当前不会：

- 学习 belief 更新或运行图构建模型；
- 估计校准概率或因果效应；
- 证明两个来源族真正相互独立；
- 自动解决科学冲突；
- 替代 Research VCS 证据闭包、独立评审或 promotion；
- 复现或继承任何 BCG benchmark 数字。

目标更窄，也更可审计：让 Agent 即将依赖的证据状态显式、确定、有界，并且无法
静默授予自己科学权威。
