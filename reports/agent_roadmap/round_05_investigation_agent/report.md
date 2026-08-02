# Round 05 可评测的多步调查 Agent — 数据与前提审计

## 结论

结论：**数据协议已建立；Agent 准确率与性能优化证据不足**。

本轮读取了 Round 01–04 报告。只有 Round 01 的 XScientist CLI 冷启动修改通过并保留；
Round 02–04 均因目标视频检索实现、runtime 和真实标签不在 XScientist 而未进入候选修改。
因此不存在通过 scene → time → region/track → instance/trajectory 全部前序门禁的 Agent 基线。

本轮开始前仓库 HEAD 已变为 `0bf98866c4f313753766450dbec98f26c43724bf`，但该提交没有
Round 02–04 验收报告，未被用作通过前序门禁的视频调查基线。

## 组件与数据核验

在当前 XScientist 仓库中未找到：

- `QuerySpec`
- `CapabilityRegistry`
- `EvidenceLedger`
- `TrajectoryEngine`
- scene/frame/region/track/entity 类型化视频工具
- 授权真实或真实脱敏的调查任务集
- 一次 QuerySpec / 一次检索的视频基线 runtime

由于这些组件不存在，本轮没有绕过或另建同名替代实现，也没有让 Agent 直接比较原始向量。

## 本轮交付

- `investigation_task.schema.json`：定义授权、用户目标、媒体范围、人工期望证据、不相关证据、
  冲突、no-match、禁止结论、成功条件和双人复核来源。
- `investigation_trace.schema.json`：定义假设分支、类型化工具调用、质量门、模型指纹、
  query-bound 缓存、资源成本、停止原因、可回放线索和完整重放哈希。
- `collection_protocol.md`：定义授权核验、人工双标、裁决、分区、防泄漏、数据冻结、一次查询
  基线、候选 Agent 轨迹和发布门禁。

Schema 明确禁止把大模型生成的答案作为标签：`llm_generated_labels` 必须为 `false`。

## 实测指标状态

| 指标 | 一次查询基线 | 候选 Agent | 原因 |
|---|---:|---:|---|
| investigation task completion rate | N/A | N/A | 无授权冻结任务集与视频 runtime |
| 关键证据 Recall | N/A | N/A | 无人工期望证据 |
| 关键证据 Precision | N/A | N/A | 无人工不相关证据 |
| 工具路由正确率 | N/A | N/A | 无类型化视频工具与人工可接受路由 |
| 错误对象类型选择率 | N/A | N/A | 无对象真值 |
| 无依据陈述率 | N/A | N/A | 无 Agent 输出与 Evidence Ledger |
| 错误身份合并率 | N/A | N/A | 无 identity/trajectory 真值 |
| 冲突证据发现率 | N/A | N/A | 无人工冲突标签 |
| no-match 正确停止率 | N/A | N/A | 无真实 no-match 任务 |
| 平均 / p95 工具调用数 | N/A | N/A | 无 Agent runtime |
| 平均 / p95 端到端耗时 | N/A | N/A | 无相同任务上的基线和候选运行 |
| GPU 秒 | N/A | N/A | 无视频模型 runtime |
| CPU 时间 | N/A | N/A | 无 Agent runtime |
| 读取视频时长 | N/A | N/A | 无源视频执行链路 |
| 轨迹可重放率 | N/A | N/A | 尚无执行轨迹；Schema 要求通过时必须为 100% |

没有报告单个工具变快，也没有声称 Agent 准确率、任务完成率或 p95 获得改善。

## 代码与验收状态

- 未实现多步 Agent，未修改现有运行时代码。
- 未生成模型答案或合成任务标签。
- 未建立一次查询 baseline 或候选结果。
- 未运行精度、延迟或资源成本 benchmark。
- 验收结论：任务数据格式和采集流程完成；Agent 优化验收未开始。

## 下一次可执行门禁

1. 由数据管理员按协议登记授权并锁定媒体 manifest。
2. 由两名人工标注者和独立裁决员采集至少一个 development 与一个 holdout 版本。
3. 对 task manifest、媒体、代码、配置、模型和索引记录 SHA-256。
4. 在正确的 `video_search` 仓库确认 QuerySpec、能力注册、质量门、证据账本和轨迹引擎版本。
5. 先在完全相同任务和媒体范围运行一次查询基线，再冻结候选 Agent。
6. 候选轨迹逐条通过 trace Schema，任何无法回到原始视频的结论判为不受支持。

在上述输入形成前，所有 Agent 业务准确率与性能指标继续保持 N/A。
