# Round 03 frame → region → track — 前提审计

## 结论

结论：**证据不足，未进入候选修改阶段**。

第 3 轮要求以前两轮“通过验收并保留”的版本为基线。实际报告状态为：

- Round 01：改善并保留，代码提交
  `be2ca0d2a4bfc6a8d24fb20fadde2adee533804a`。
- Round 02：证据不足，未修改 temporal localization，未形成通过验收的候选版本。

因此不存在满足本轮前提的“两轮均通过”基线。

## 路径核验

冻结的 XScientist 基线中，下列要求检查的路径全部不存在：

- `app/core/assets.py`
- `app/core/localization.py`
- `app/regions/`
- `app/person/tracking.py`
- `app/plate/tracking.py`
- `scripts/benchmark_region_grounding_contract.py`
- `scripts/benchmark_region_grounding_runtime.py`
- `scripts/bench_spatial_catalog.py`
- `scripts/eval_labeled_retrieval.py`
- `scripts/eval_object_instances.py`

XScientist 也不存在 `ObservationCatalog`、grounding、detector、tracklet 或源视频读取链路。
这些实现位于另一个 `video_search` 仓库，而该仓库没有可供本轮读取的 Round 01 / 02
验收报告及冻结数据指纹，不能自动替换当前基线。

## 数据门禁

当前 XScientist 基线没有同时包含真实 `bbox`、`frame_idx` 和 `target_type` 的固定
development / holdout 标注集，也没有 track identity 真值。因此没有使用示例数据、
合成框、缩略图或 detector 输出代替人工真值。

## 指标状态

| 指标 | Baseline | Candidate | 原因 |
|---|---:|---:|---|
| spatial coverage | N/A | N/A | 无空间检索链路与真实 bbox 标注 |
| region Recall@0.5 | N/A | N/A | 无真实 bbox / frame_idx / target_type 标注 |
| region Precision@0.5 | N/A | N/A | 无真实 bbox / frame_idx / target_type 标注 |
| mean IoU | N/A | N/A | 无真实 bbox 标注 |
| 错框率 | N/A | N/A | 无冻结空间 query 与真值 |
| 空结果正确率 | N/A | N/A | 无冻结 no-match 空间 query 集 |
| 多目标串框率 | N/A | N/A | 无多目标 identity / bbox 真值 |
| track 建立成功率 | N/A | N/A | 无 tracking 实现与 track 真值 |
| ID switch | N/A | N/A | 无 identity trajectory 真值 |
| 原图不可用降级率 | N/A | N/A | 无源图读取链路 |
| grounding cold/warm p50/p95/p99 | N/A | N/A | 无 grounding runtime |
| 打开候选端到端延迟 | N/A | N/A | 无候选回放链路 |
| grounding 调用次数 | N/A | N/A | 无 grounding runtime |
| 缓存命中率 | N/A | N/A | 无 region cache |
| GPU 时间 | N/A | N/A | 无 grounding/detector 模型链路 |
| Round 01 场景指标 | N/A | N/A | Round 01 是 XScientist CLI 冷启动评测 |
| Round 02 时间定位指标 | N/A | N/A | Round 02 未形成可验收基线 |

## 禁止项状态

本轮没有生成或绘制任何 bbox，没有把 `contextual_candidate` 当成
`grounded_target`，没有跨查询复用缓存，也没有对缩略图放大后声称恢复真实细节。

## 代码、测试与验收状态

- 没有修改 region、grounding、detector 或 tracking 代码。
- 没有生成 candidate 性能或精度结果。
- 没有重新选择数据、query 或仓库。
- Recall、Precision、IoU、延迟和 GPU 成本验收均为 N/A。
- 最终判定：未满足开始实验的前提，不能声称空间精度或成本优化成功。

## 继续执行所需输入

若本轮目标确实是工作区中的 `video_search`，需要先为该仓库补齐并确认：

1. Round 01 和 Round 02 的最终保留提交与报告；
2. 两轮共同使用且不可更换的 query 清单和数据文件 SHA-256；
3. 包含真实 `bbox`、`frame_idx`、`target_type` 的 development / holdout 标签；
4. 若验收 tracking，包含 track identity 和逐帧轨迹的真值；
5. 相同的远端服务、模型、GPU、缓存冷/热策略和运行命令。

这些前提满足后，才能建立 frame → grounded region → tracklet 的可比较 baseline，
并按验收门禁决定是否保留候选修改。
