# Round 06 调查 Agent → 案件候选 → 人工复核发布审计

## 最终发布判断

```json
{
  "release_ready": false,
  "status": "blocked"
}
```

结论：**本轮未成功，禁止发布为视频调查或案件关联系统。**

当前没有授权真实案件数据、案件隔离的锁定 test、可运行案件关联链路、三种同集基线或
真实人工复核实验。按数据前提，本轮没有实施案件匹配优化，也没有使用 LLM 生成伪案件、
伪 case_id 或伪标签。

机器可读判定见 `release_decision.json`。

## 前五轮证据使用规则

- Round 01：只有 XScientist CLI / 导入冷启动优化通过并保留。
- Round 02：temporal localization 证据不足，未形成候选。
- Round 03：region / track 证据不足，未形成候选。
- Round 04：instance / trajectory 证据不足，未形成候选。
- Round 05：建立了任务与可重放轨迹 Schema 及采集协议；没有 Agent runtime 或准确率结果。

本报告没有把 Round 02–05 的 N/A、Schema 或代理结果转换成系统精度结论。

## 案件数据与领域对象审计

当前 XScientist 仓库中不存在满足要求的授权案件数据集：没有固定 `case_id`、时间、地点、
事件类型、关联证据、no-match 标签，也没有按案件隔离的 train/dev/test manifest。

要求的案件域对象状态：

| 对象 | 状态 | 说明 |
|---|---|---|
| Investigation | Missing | 无案件调查领域模型 |
| Incident | Missing | 无警情领域模型 |
| Lead | Missing | 代码中的普通单词匹配属于科研写作语义，不是案件 Lead |
| Hypothesis | Missing | 现有科研假设对象不是案件调查假设 |
| EntityCandidate | Missing | 无案件实体候选模型 |
| Observation | Missing | 无视频观测案件对象 |
| Evidence | Missing | 现有科研证据概念不是案件 Evidence 对象 |
| Conflict | Missing | 现有测试/文本词汇不是案件冲突证据对象 |
| HumanDecision | Missing | 无人工复核决定模型或审计记录 |

因此 Agent 不能形成符合限定输出的“可能相关案件候选 + 支持/冲突证据 + 未决问题 + 人工
核查动作”。没有输出“已确定同案”或“已确认同一人”等结论。

## 三种基线

| 方法 | 状态 | 原因 |
|---|---|---|
| 用户查询直接检索案件 | Unavailable | 无 case index、case metadata 和锁定 test |
| 视频一次检索后检索案件 | Unavailable | 前序视频链路未通过且无 case linkage runtime |
| Round 05 调查 Agent 检索案件 | Unavailable | Round 05 只有数据协议，无 Agent runtime |

三种方法没有在任何任务上运行，因此不存在可比较差异。

## 精度指标

| 指标 | 直接案件检索 | 视频一次检索 → 案件 | 调查 Agent → 案件 |
|---|---:|---:|---:|
| Case Recall@1/3/5 | N/A | N/A | N/A |
| Case Precision@1/3/5 | N/A | N/A | N/A |
| MRR / NDCG | N/A | N/A | N/A |
| 错误案件关联率 | N/A | N/A | N/A |
| no-match 拒识 | N/A | N/A | N/A |
| 无依据案件陈述率 | N/A | N/A | N/A |
| 关键证据覆盖率 | N/A | N/A | N/A |
| 冲突证据发现率 | N/A | N/A | N/A |
| 人物/车辆错误串联率 | N/A | N/A | N/A |

没有真实人工复核实验，因此平均复核时间和操作数为 N/A，没有估算值。

## 速度与规模指标

| 指标 | 结果 | 原因 |
|---|---:|---|
| 单案件端到端 p50/p95/p99 | N/A | 无可运行链路和固定任务 |
| 每阶段耗时 | N/A | 无案件关联阶段实现 |
| 工具调用数 | N/A | 无 Agent runtime |
| GPU 秒 | N/A | 无视频/案件模型执行 |
| CPU 时间 | N/A | 无执行 workload |
| 峰值内存 | N/A | 无执行 workload |
| 索引规模 | N/A | 无 case index |
| 并发任务 QPS | N/A | 无可运行服务 |
| 100 路/多天 | N/A | 没有真实运行或明确 performance 压测 |

没有线性外推 100 路或多天规模，也没有用单工具耗时替代端到端案件任务耗时。

## 全系统 operational coverage

| 链路 | Labeled | Proxy | Performance | 最终状态 |
|---|---:|---:|---:|---|
| scene_event | No | No | No | Missing |
| temporal/spatial localization | No | No | No | Missing |
| person | No | No | No | Missing |
| plate | No | No | No | Missing |
| vehicle | No | No | No | Missing |
| object_instance | No | No | No | Missing |
| trajectory | No | No | No | Missing |
| agent investigation | No | No | No | Missing；只有 Schema/采集协议 |
| case linkage | No | No | No | Missing |
| ingest/realtime | No | No | No | Missing |

Operational coverage：**0/10，0%**。Round 01 的 CLI 冷启动不属于上述十条视频案件业务链路，
不能增加业务覆盖率。

## 消融实验

| 消融 | 状态 | 原因 |
|---|---|---|
| 无 Agent | Not run | 无锁定 test 与案件 runtime |
| 无局部证据 | Not run | 无局部证据链路 |
| 无时空拓扑 | Not run | 无轨迹/拓扑链路 |
| 无冲突搜索 | Not run | 无调查 Agent runtime |
| 无案件元数据 | Not run | 无案件数据 |
| 完整系统 | Not run | 全链路不具备运行条件 |

没有以不同 test、预算或代理数据运行任何消融。

## 已知退化与失败实验

已知数值退化：Round 01 候选的 `project_import` 中位数回归 1.9241%，`batch_import`
中位数回归 0.1767%；两项当时仍在既有回归门限内。除此之外没有可比较业务候选，不能将
Missing 写成“无退化”。

失败或被阻塞项：

- Round 01 `make smoke` 最后的 `validate --full-import-smoke` 被过期登录会话阻塞。
- Round 02–04 因错误仓库范围、缺 runtime 和真实标签而未进入实验。
- Round 05 仅完成评测数据与轨迹协议，Agent baseline/candidate 未运行。
- Round 06 三种基线、六种消融、完整系统与人工复核实验均未运行。

## 客户承诺边界

当前可以承诺：

- 已定义可校验的授权调查任务格式、人工双标/冻结流程和 100% 可重放轨迹格式。
- 可以在严格限定为 XScientist CLI 冷启动的范围内引用 Round 01 实测结果。

当前禁止承诺：

- scene/event 检索、时间/空间定位、人物/车辆/车牌/物体匹配的业务准确率；
- 跨摄像头轨迹或同一身份确认；
- 调查 Agent 任务完成率、案件关联准确率或人工复核提效；
- 100 路、多天规模性能；
- 视频调查或案件关联生产发布就绪。

## 下一轮唯一瓶颈

**缺少经授权、按案件隔离、人工标注并以哈希锁定的端到端案件关联 test manifest。**

该 manifest 必须绑定正确的 `video_search` 代码版本、媒体范围、case metadata、期望/冲突
证据、no-match 和禁止结论。它形成之前，继续调整 Agent、案件分数或阈值无法产生可验收证据。
