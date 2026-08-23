# 研究机会漏斗（受 FAR 启发）

XScientist 现在提供一个领域无关、可审计的“研究方向 → 候选问题池 → 尝试 →
独立判定 → 重要性分级 → 资源分配”接口，对应 [FAR](https://arxiv.org/abs/2608.16977)
论文的 Find → Attempt → Recommend 思路。FAR [作者仓库](https://github.com/zeyu-zheng/FAR)
仍是外部参考；本项目没有重跑它的组合数学 pilot，也没有把论文数字写成本地结果。

接口会复用既有 Research VCS kind，并用显式 `protocol_kind` 区分新协议，因此不会
改变旧仓库的 semantic-profile digest：

1. `save_research_direction` 记录研究方向（`research_goal`）。
2. `save_opportunity_pool` 记录完整候选集（`question`）。未知来源、未尝试候选和
   负结果都必须保留，不能只留下成功项。候选池是有界、由调用者提供的抽取结果；
   XScientist 不声称实现 FAR 的全语料 Label/Extract/Check importer。
3. `save_opportunity_attempt` 的结果只能是 `known`、`new`、`fix`、`none`。接口记录
   外部 runner 的结果与证据，但不会自动调用 FAR solver 或导入其 workdir。
4. `save_opportunity_judgment` 要求来源谱系不重叠的 evaluator，记录 `pass`、`fail`
   或 `known`；这只是声明的 actor-disjointness，不是身份或科学正确性证明。
5. `save_opportunity_grade` 记录 `known`、`minor`、`substantial`，对象状态保持
   `completed`，不会自动晋升 claim 或生成“发表分数”。
6. `save_opportunity_allocation` 采用 fail-closed：只有候选集完整且每个候选的
   `source_status=open` 时才会锁定分配。纯排序函数仍可查看临时/不完整行，但不会
   隐藏它们。

判定与分级有明确阶段门禁：正常情况下只有 `outcome=new` 才能进入 judgment，只有
`verdict=pass` 或 `known` 才能进入 grade。回溯性例外必须显式使用
`--allow-stage-override` 和非空 `--override-reason`，理由会进入不可变哈希。

`difficulty`、`importance`、`expected_success_probability`、
`expected_artifact_probability`、`expected_importance` 都采用连续 `[0,1]` 标度。
默认 `probability_semantics=conditional_artifact_given_success` 将前者解释为
“尝试被接受”的概率、后者解释为“在被接受后可发表”的条件概率；后者缺失时只使用
明确记录的中性假设 `1.0`。若指定 `joint_artifact_probability`，
`expected_artifact_probability` 已是联合概率，不会再次乘成功概率；缺失时该候选不会
被选中。
默认 `calibration_status=declared_inputs_not_calibrated`，表示没有把事后结果泄漏到
事前排序，也没有把 FAR 的领域特定 AUC/产出率外推到别的学科。

候选可用 `source_object_ids` 绑定本地 `search_receipt`、`source_snapshot`、
`passage_evidence` 或 `question`；只有外部 URL/标签时会明确标为未完成 lineage。原始
响应、凭据、超长文本和过深嵌套引用会在落盘前拒绝。attempt、judgment、grade 也可用
规范的 `evidence_object_ids` 绑定证据；这些绑定会落成 `derived_from` 关系，而不是
无法核验的自由文本引用。`source_status=open` 只表示在当时记录的检索证据中没有找到
可信解答，不是全球新颖性证明，也不是“人类尚未解决”的断言。
当前协议每个 review object 记录一个 judgment；重复 judgment 会被标记为漏斗不完整，
并没有实现 FAR 的“三个 judge 全部通过”门槛。因此本地的 `pass` 数量不能与 FAR
论文 pilot 的报告数字直接比较。
为保留审计轨迹，attempt 接口也允许对非 `open` 候选记录回溯结果；但这类候选不能
进入锁定的 allocation 计划，因此不等同于 FAR 的 open-only attempt 阶段。

CLI 入口：

```text
xscientist research opportunity direction DIRECTION STATEMENT OBJECTIVE
xscientist research opportunity pool DIRECTION_ID candidates.json
xscientist research opportunity attempt POOL_ID CANDIDATE_ID none "没有结果"
xscientist research opportunity inspect POOL_ID --json
```

这是一套过程与分配契约，不是 benchmark 分数；不声称自主数学发现、达到人类水平、
论文可发表，或超过 FAR/其他系统。候选只有在明确的下游人工决定和既有 XScientist
门禁之后，才能转成 hypothesis 或 experiment design。
