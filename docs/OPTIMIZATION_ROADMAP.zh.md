# 基于 benchmark 的优化完成状态

这是一份当前状态和验收边界说明，不是排行榜。本地 pilot 是离线 conformance
审计：没有运行 AutoResearchEval 的 100 任务/800 轨迹 rollout，也没有把人类或
竞品分数注入报告。因此下面写的是“缺少什么证据/能力”，不是宣称某方案全面更强。

## 当前实测真正发现了什么

内置 balanced fixture 是完整性测试：

| 观察 | 证据 | 含义 |
| --- | --- | --- |
| 任务契约 | 已记录运行中开放式 20/20、目标锚定 20/20 | 说明 framing 稳定，不代表解题质量 |
| 生命周期覆盖 | 5/6 阶段，83.3% 结构覆盖 | fixture 故意缺少检索产物，不能当质量分数 |
| 闭环 | trace/replay 通过，verify 阻塞 | 仍能看见 held-out 冲突和独立复现缺口 |
| 反馈 | `contained`，2 个未解决问题，0 个带问题发布 | 门禁拦截了发布，但 containment 不是修复 |
| 过程 | 3 commits、1 branch、16 个 typed artifacts | 中间决策可审查，但没有证明探索多样性 |
| 首次运行易用性 | `benchmark_first_run` 只记录当前机器、当前运行的耗时；不把过期单点数字当固定证据 | 仅是零 Provider 本地耗时，不能与模型或网络耗时比较 |
| 公平性 | 分支 fixture 仍为 `NOT VERIFIED` | 没有证明同任务切片、基点、预算、evaluator |
| 外部比较 | 0 次模型 rollout、没有匹配人类臂 | 目前不存在诚实的跨系统或人机科研分数 |

JSON 现在把这些转成有序的 `diagnostics` 清单：P0 阻断质量声明，P1 是证据/生命
周期债务，P2 是探索或易用性改进。

## 能力差距地图

| 参照方案族 | 它展示的能力 | XScientist 当前状态 |
| --- | --- | --- |
| AutoResearchEval / ARFT | 长时 rollout、artifact-aware judge、六阶段失败分类 | 本仓库只提供离线 conformance 报告；官方 rollout、artifact-aware judge 和标注轨迹包均未在仓库内实现。 |
| ScientistOne / Chain-of-Evidence | 生成物的 claim provenance 与完整性审计 | 已有 typed objects 和显式导出；claim 级证据包不是默认质量分数，项目也没有这样宣称。 |
| MARS / AdaEvolve / EvoX | 预算约束搜索、分支、适应性探索 | Git-like 分支可见，但当前 fixture 只有一条分支，公平性元数据未验证。 |
| ScholarPeer | 检索增强的质询和 reviewer 校准 | review 对象可以记录；本地 pilot 未评估检索质量和 citation entailment。 |
| PaperOrchestra / PaperBanana | 专门的写作/绘图输出和对应 benchmark | XScientist 覆盖较广生命周期，但不报告组件级质量分数。 |
| 人类研究 | 一些邻近任务有匹配人类臂 | 本地人类臂明确为 `not_reported`；外部数字只作上下文证据，不会替代本地结果。 |

附件演讲稿只用于发现范围；具体证据以[系统矩阵](SYSTEM_COMPARISON.zh.md)里的原始论文和
仓库为准。演讲 PDF 的文件名和 SHA-256 已保存在机器可读 source manifest。

## 当前完成状态与明确阻断

仓库中已经完成、且无需冒充外部 rollout 即可验证的优化包括：schema 校验的
conformance 报告、原子脱敏输出、固定枚举 diagnostics、有界 Git-like 过程审计、
带固定未验证原因的 fail-closed 公平性元数据、ARA/evidence 有界只读索引、探索图计数、
可复现输入指纹、离线报告验证、保守的 feedback 归因标签、惰性 SDK 导出，以及中英文
文档同步。这些是仓库能力的完成状态，不是 benchmark 质量声明。

以下阻断会继续原样显示在报告中。它们是发布条件，不是带日期的待办：

| 阻断项 | 当前证据 | 只有在什么条件下才能改变状态 |
| --- | --- | --- |
| 没有匹配的模型 rollout | `rollouts_evaluated: 0`；`official_comparable: false` | 同时具备固定 evaluator、任务切片、环境、预算、seed 策略和可记录 rollout。 |
| 没有匹配的人类臂 | `human_baseline.status: not_reported`；分数为 `null` | 真实人类运行遵循同一任务/工具/预算/verifier 契约并报告不确定性。 |
| 分支公平性未验证 | 当前 fixture 只有一条分支；任务切片、基点、预算、evaluator 相同性没有证据 | 每条比较分支都有机器可读契约，且所有必要字段验证通过。 |
| 检索与独立验证缺口 | 本地 fixture 为 5/6 结构覆盖；`verify` 被阻断 | 必需检索产物和独立 held-out 验证 receipt 同时存在。 |
| 反馈债务只是 containment，不是修复 | 仍有 2 个问题处于 `contained`，带问题发布数为 0 | 问题完成修复、回归测试，并记录新的证据。 |
| ARFT adapter 缺失 | 报告明确给出 `AUDIT.ARFT_ADAPTER_MISSING` | 实现 typed adapter 和 evaluator，并保留 `unassessed`/error 状态。 |

在这些条件得到证据前，报告必须保持 `quality_claim_allowed: false`，不能把结构覆盖率
转成科研质量分数，也不暗示任何时间表或未经验证的完成日期。

## 操作命令

```bash
# 离线结构审计并保存脱敏报告
xscientist benchmark autoresearch \
  --tasks ./open-ended_tasks.jsonl --workspace ./first-study \
  --limit 20 --kind open-ended --show-process --json \
  --output ./benchmark-evidence/autoresearch-report.json

# 来源审计能力矩阵（不运行外部 rollout）
xscientist benchmark systems --json \
  --output ./benchmark-evidence/system-matrix.json

# 离线校验已保存报告的 schema 与不可比较边界
xscientist benchmark verify --report ./benchmark-evidence/autoresearch-report.json --json
```

两条命令都不会生成科研排行榜；它们固定当前完成状态和明确阻断，使任何单独进行的
受控重跑都能被审计，而不会事后改变声明边界。
