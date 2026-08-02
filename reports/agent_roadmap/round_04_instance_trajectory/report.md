# Round 04 region/track → instance candidate → trajectory — 前提审计

## 结论

结论：**证据不足，未进入候选修改阶段**。

本轮已读取 Round 01、Round 02 和 Round 03 报告。前序状态如下：

- Round 01：CLI / 模块导入优化通过并保留，代码提交
  `be2ca0d2a4bfc6a8d24fb20fadde2adee533804a`。
- Round 02：证据不足，未形成通过时间定位门禁的候选。
- Round 03：证据不足，未形成通过 region / track 门禁的候选。

本轮要求“只使用通过前序门禁的版本”。由于 region / track 前序门禁未通过，不能从
不存在的 grounded region / track 基线继续构造 instance candidate 或 trajectory。

## 四条链路清点

| 链路 | Runtime | 锁定 labeled test | 正式优化资格 | 证据 |
|---|---|---|---|---|
| person | Missing | Missing | 不具备 | XScientist 无 `app/person/` 和 `scripts/eval_person_reid.py` |
| vehicle | Missing | Missing | 不具备 | 无 vehicle ReID runtime 和 `scripts/eval_vehicle_reid.py` |
| plate | Missing | Missing | 不具备 | XScientist 无 `app/plate/` 和 `scripts/eval_plate_ocr.py` |
| object | Missing | Missing | 不具备 | XScientist 无 `app/object_search/` 和 `scripts/eval_object_instances.py` |

同时缺少：

- `app/entity/`
- `app/core/scoring.py`
- `app/core/trajectory.py`
- 对应实例质量门、ANN、局部精排、OCR 和 trajectory benchmark

因此四条链路中没有任何一条同时具备“可运行模型 + 锁定 labeled test”，无法选择满足
本轮限制的主链路。没有盲目同时优化四类对象。

## 标签与 hard-negative 门禁

当前通过门禁的 XScientist 基线不包含以下数据：

- person / vehicle / object identity、gallery、query 和 no-match 真值；
- plate 整牌文本、逐字符真值和易混字符分桶；
- track identity、跨摄像头对应关系和逐帧轨迹；
- 像素、模糊、曝光、遮挡、天气、昼夜属性；
- 同衣不同人、同车型不同车、易混车牌等锁定 hard-negative。

没有从 test 反向构造训练数据，也没有用 detector、OCR 自输出或 ANN 自召回结果充当真值。

## 指标状态

### Person / vehicle / object

| 指标 | Baseline | Candidate | 状态 |
|---|---:|---:|---|
| Rank-1 | N/A | N/A | runtime 与 identity 标签均缺失 |
| Rank-5 | N/A | N/A | runtime 与 identity 标签均缺失 |
| mAP | N/A | N/A | runtime 与 identity 标签均缺失 |
| no-match 拒识 | N/A | N/A | no-match 标签缺失 |
| hard-negative 误接受率 | N/A | N/A | 锁定 hard-negative 缺失 |

### Plate

| 指标 | Baseline | Candidate | 状态 |
|---|---:|---:|---|
| 整牌准确率 | N/A | N/A | OCR runtime 与整牌真值缺失 |
| 字符准确率 | N/A | N/A | 字符级真值缺失 |
| Top-K | N/A | N/A | OCR 候选与标签缺失 |
| 易混字符 | N/A | N/A | 易混字符分桶缺失 |
| no-match | N/A | N/A | no-match 标签缺失 |

### Trajectory / quality / latency

| 指标 | Baseline | Candidate | 状态 |
|---|---:|---:|---|
| IDF1 | N/A | N/A | trajectory runtime 和 identity 真值缺失 |
| ID switch | N/A | N/A | 逐帧 identity 真值缺失 |
| 错误跨摄像头连接率 | N/A | N/A | 跨摄像头 identity 真值缺失 |
| 不可达候选排除率 | N/A | N/A | 摄像头拓扑与真值缺失 |
| 质量门覆盖率 | N/A | N/A | 实例质量门缺失 |
| 质量门拒识正确率 | N/A | N/A | 低质量 / no-match 标签缺失 |
| ANN Recall | N/A | N/A | 实例 ANN 索引与锁定真值缺失 |
| 查询编码 p95 / QPS | N/A | N/A | runtime 缺失 |
| ANN p95 / QPS | N/A | N/A | runtime 缺失 |
| 局部精排 p95 / QPS | N/A | N/A | runtime 缺失 |
| OCR p95 / QPS | N/A | N/A | runtime 缺失 |
| 轨迹组织 p95 / QPS | N/A | N/A | runtime 缺失 |
| 端到端 p95 / QPS | N/A | N/A | runtime 与固定 query 缺失 |

像素、模糊、曝光、遮挡、天气和昼夜分桶全部为 N/A，因为对应真实属性标签不存在。
ANN Recall 没有被用来替代业务准确率。

## 前三轮保护指标

- Round 01 CLI 指标未受本轮影响，因为本轮没有代码修改。
- Round 02 时间定位指标为 N/A，未形成通过门禁的基线。
- Round 03 空间 / track 指标为 N/A，未形成通过门禁的基线。

## 代码、测试与验收状态

- 没有修改 instance、OCR、ReID、scoring 或 trajectory 代码。
- 没有训练、调参或从锁定 test 构造 hard-negative。
- 没有生成 candidate 指标。
- 主链路的 Rank-1 / mAP / 整牌准确率和端到端性能验收均无法执行。
- 未实测的 person、vehicle、plate、object 四条链路继续标为 Missing / Unavailable。

## 继续执行所需输入

若目标仓库是工作区中的 `video_search`，需要先完成以下事项：

1. 确认该仓库 Round 01–03 的保留提交、报告和数据 SHA-256；
2. 在 person、vehicle、plate、object 中指定或确认至少一条具备可运行模型的主链路；
3. 提供该链路锁定的 development / holdout identity 或 OCR 标签和固定 query；
4. 单独锁定 hard-negative，不得由 holdout 结果反向生成；
5. 若验收 trajectory，提供 identity trajectory、摄像头拓扑及时间同步真值；
6. 固定远端模型、GPU、索引、缓存策略和完整端到端启动命令。

满足这些前提后，才能只选择一条合格主链路完成正式 before / after，并把其余链路继续
标记为 Missing / Unavailable。
