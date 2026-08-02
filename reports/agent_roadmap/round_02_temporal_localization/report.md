# Round 02 temporal localization — 前提审计

## 结论

结论：**证据不足，未进入候选修改阶段**。

本轮已读取第 1 轮 `comparison.md`。该文件把
`be2ca0d2a4bfc6a8d24fb20fadde2adee533804a` 冻结为第 2 轮唯一代码基线，
仓库为 XScientist。

在该基线中，下列要求检查的路径全部不存在：

- `app/core/localization.py`
- `app/search/service.py`
- `scripts/eval_labeled_retrieval.py`
- `scripts/bench_search_cpu.py`

XScientist 中也不存在 `ObservationCatalog`、temporal localization 实现、视频索引、
固定检索 query 集或包含真实 `start_ms` / `end_ms` 的标注集。因此不能在不更换仓库、
基线和数据的前提下执行本轮实验。

同一工作区的另一个仓库
`<private-machine>/Documents/startup/业务/江源雪亮工程/video_search` 包含上述全部文件，
但它不包含第 1 轮 `comparison.md`，其当前 HEAD 为
`36f567670bbd0675866170c3371ad44b3a1c9033`，且存在未提交的用户修改。把该仓库或其
当前 HEAD 自动替换成第 1 轮冻结基线，会违反“以第 1 轮最终保留版本作为唯一基线”
以及“不得覆盖用户已有修改”的约束。

## 指标状态

| 指标 | Baseline | Candidate | 原因 |
|---|---:|---:|---|
| segment Recall@K | N/A | N/A | XScientist 无视频 segment 标注与检索链路 |
| temporal IoU | N/A | N/A | 无真实起止时间标签 |
| 边界 MAE | N/A | N/A | 无真实起止时间标签 |
| 中心点误差 | N/A | N/A | 无真实起止时间标签 |
| NDCG | N/A | N/A | 无冻结检索 query / relevance 集 |
| no-match 拒识 | N/A | N/A | 无冻结 no-match query 集 |
| `requires_source_refinement` 覆盖率 | N/A | N/A | 基线没有该字段或实现 |
| 时间聚合 p50/p95/p99 | N/A | N/A | 基线没有该阶段 |
| 补帧 p50/p95/p99 | N/A | N/A | 基线没有该阶段 |
| 源视频抽帧 p50/p95/p99 | N/A | N/A | 基线没有该阶段 |
| 端到端 p50/p95/p99 | N/A | N/A | 无视频端到端服务与固定 query 集 |
| cold/warm QPS | N/A | N/A | 无视频端到端服务与固定 query 集 |
| 场景 Hit@1/Hit@10/MRR | N/A | N/A | 第 1 轮 XScientist comparison 未测该指标 |
| 误合并率 | N/A | N/A | 无真实独立事件边界标签 |

没有用示例标签、代理类别标签、合成时间段或 CPU 微基准替代真实时间段验收数据，
也没有声称任何定位精度或速度提高。

## 代码与测试状态

- 本轮没有修改 temporal localization 或检索代码。
- 没有创建 candidate 指标。
- 没有重新选择数据或 query。
- 第 1 轮保留提交未被改写。

## 继续执行所需的唯一输入

需要明确将第 2 轮目标切换到 `video_search`，并为该仓库提供或确认：

1. 第 1 轮最终保留提交；
2. 第 1 轮 `comparison.md`；
3. 第 1 轮使用的固定 development / holdout 时间段标注文件及 query 集；
4. 第 1 轮相同的远端运行环境和启动命令。

获得这些前提后，才能在同一数据、query、环境和代码基线上形成 coarse-to-fine
baseline，并依据验收条件决定是否保留候选修改。
