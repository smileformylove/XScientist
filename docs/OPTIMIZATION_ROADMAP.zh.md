# 基于 benchmark 的优化路线图

这是一份决策和验收路线图，不是排行榜。本地 pilot 是离线 conformance
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
| 首次运行易用性 | 最近一次本地 `benchmark_first_run(max_seconds=30)` 为 15.99 秒 | 仅是零 Provider 本地耗时，不能与模型或网络耗时比较 |
| 公平性 | 分支 fixture 仍为 `NOT VERIFIED` | 没有证明同任务切片、基点、预算、evaluator |
| 外部比较 | 0 次模型 rollout、没有匹配人类臂 | 目前不存在诚实的跨系统或人机科研分数 |

JSON 现在把这些转成有序的 `diagnostics` 清单：P0 阻断质量声明，P1 是证据/生命
周期债务，P2 是探索或易用性改进。

## 能力差距地图

| 参照方案族 | 它展示的能力 | 暴露出的 XScientist 缺口 | 优化目标 |
| --- | --- | --- | --- |
| AutoResearchEval / ARFT | 长时 rollout、artifact-aware judge、六阶段失败分类 | 本仓库没有官方 rollout harness、judge 和标注轨迹包 | 固定版本 evaluator adapter 与轨迹 bundle 契约 |
| ScientistOne / Chain-of-Evidence | 生成物的 claim provenance 与完整性审计 | typed objects 已有，但 claim 级可分享证据包仍需显式导出 | claim → evidence → verifier 清单与独立 replay receipt |
| MARS / AdaEvolve / EvoX | 预算约束搜索、分支、适应性探索 | Git-like 分支可见，但各分支结果、预算、evaluator 尚不能公平比较 | 分支实验 manifest 与资源感知调度 |
| ScholarPeer | 检索增强的质询和 reviewer 校准 | review 对象可记录，但检索质量/citation entailment 缺专用 evaluator | citation/claim entailment 与 adversarial review 集 |
| PaperOrchestra / PaperBanana | 专门的写作/绘图输出和对应 benchmark | XScientist 生命周期广，但没有组件级质量分数 | manuscript、figure、citation 插件 evaluator |
| 人类研究 | 一些邻近任务有匹配人类臂 | 本 pilot 没有统一人类协议；外部数字不可互换 | 先固定 agent evaluator，再预注册小规模匹配人类臂 |

附件演讲稿只用于发现范围；具体证据以[系统矩阵](SYSTEM_COMPARISON.zh.md)里的原始论文和
仓库为准。演讲 PDF 的文件名和 SHA-256 已保存在机器可读 source manifest。

## 30 / 90 / 180 天计划

### 未来 30 天：让审计边界一眼可见

- 在注册 evaluator、重复 seed 策略之前保持 `quality_claim_allowed: false`。
- 用 `--output` 保存脱敏报告；ARA/VCS 完整包仍单独、显式、受控导出。
- 关闭当前 P0：为每条分支记录 task slice、fork base、budget、evaluator；否则保持
  `eligible=false`。
- 只有在显式 adapter 下做 typed-object → ARFT evidence-channel 映射，并保留输入错误
  和 `unassessed`，不能把缺失当成功。

验收：报告通过 schema、没有任务/gold 文本，且每个“通过”都有固定验证条件。

### 未来 90 天：运行公平的本地 benchmark

- 固定一份 task manifest、evaluator revision、环境/container、预算和至少 3 个 seed。
- 用相同切片运行 XScientist 与至少一个可复现 baseline，发布 hash、run receipt、失败
  分类、时间/成本区间。
- 为每个分支保存 manifest、合并/拒绝决策，让探索可比较但不泄露隐藏思维链。
- 增加 retrieval provenance/entailment、execution receipt、不确定性/负结果、claim
  trace、独立复现等组件 evaluator。

验收：只有全部 fairness checks 为 true 时才允许 `official_comparable=true`，否则始终
明确 `unverified`。

### 未来 180 天：测科研质量，而不只是可观测性

- 对固定公开子集复现官方 evaluator，使用 artifact-aware judge 和盲法仲裁。
- 仅在完全相同任务/工具/预算下预注册人类臂，报告人数、不确定性、流失和原始过程 receipt。
- 测反馈自进化：修复成功率、回归率、恢复时间、分支复用，以及 hold 门禁是否真的阻止发布。
- 双语发布正负结果和 unavailable evidence 清单；绝不把不同 benchmark 的人类基线平均成一个数。

验收：发布结果必须附 manifest、evaluator、环境、seed、证据包和明确不确定性说明。

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
```

两条命令都不会生成科研排行榜；它们把边界、缺失证据、过程分支和下一步行动固定下来，
让后续受控测试可以被审计，而不是事后解释。
