# 研究策略 Rollout 审计

Faraday 最值得迁移的不是参数量比较，而是系统结构：外层研究策略选择下一步实验，内层编码工具负责执行，任务级评估器检查结果是否真的支持科学结论。XScientist 现在把这条边界记录为 metadata-only 的 `research_rollout` 对象。

## 记录内容

- 带 hash 的任务与 split（`train`、`test`、`holdout` 或 `external`）；
- 五维 rubric：结果一致性、主张支持、实现忠实度、资源效率、科学完整性；
- 工具调用的 provider/model 指纹、输入/输出 hash、预算、决策类型和结果；绝不保存 prompt、stdout、凭证或原始响应；
- turn 元数据与仅供观察的事后正向 reward-delta credit；
- 零个或多个评估样本、各维分数、均值和分歧；
- 明确写入 `quality_claim_allowed=false` 与 `causal_claim_allowed=false`。

评估摘要只是测量记录，不是 ground truth。科研主张晋级仍需独立 gate；缺失的 reward trace 会保持缺失，builder 不会静默插值 credit。

## CLI

准备包含 `task_id`、完整 `task_hash`、`time_budget_seconds`，以及可选
`tool_delegations`、`turns`、`evaluations` 的 `episode.json`，执行：

```bash
xscientist research rollout episode.json --repo ./first-study --json
```

命令离线且幂等，只保存脱敏的内容寻址 Research VCS 对象，并生成标准 experiment checkpoint。

## 工具替换与对比边界

`assess_tool_swap_compatibility(reference, candidate)` 会检查任务 hash、rubric hash、split 和时间预算。边界合格只是进行受控比较的前提，不代表任何模型或工具更好。

当前实现不复现 Faraday 的 RL 训练、三评审协议、编码工具提供商、benchmark 任务集或论文分数。这是受 [Faraday 论文](https://arxiv.org/abs/2608.13331) 启发的 XScientist 审计/训练数据契约，不是本地复现实验。
