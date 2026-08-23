# XScientist 项目劣势与科研可信度审计（2026-08）

> 审计对象：`main`（含本轮闭环、provider 和反馈优化；历史基线仍保留在 Git 中）
> 审计方式：源码审查、临时工作区 CLI 旅程、真实 OpenAI-compatible provider 最小请求、协议/隐私/定向测试。
> 结论性质：这是工程和科研治理审计，不是对任何模型、论文或科学结论的独立复核。

## 结论先行

XScientist 的优势已经不在“能否生成一篇论文”，而在于它把问题、假设、实验尝试、证据、反证、审稿、门禁、复现和失败分支组织成可寻址的本地研究历史。它适合作为受监督的计算科研工作台和科研版本控制层。

它的主要劣势也很明确：功能层叠快于协议收敛；默认路径仍把探索、可追溯、可复现和独立验证混在一个产品叙事里；真实端点、容器、数据和第三方评估的证据远少于本地单元测试。因而目前最危险的不是“功能不够”，而是用户看到绿色状态后高估了科学可信度。

本次优先修复了四个会直接伤害复现和审查的工程问题：

- 多工作区连续运行时，之前加载的 provider endpoint/credential 可能污染当前工作区；现在对 XScientist 注入的环境值做所有权跟踪和切换清理。
- `provider test` 之前对 Anthropic/Bedrock/Vertex 直接抛出 OpenAI-compatible 错误；现在返回结构化的 `live_probe_not_supported`，不会伪装成成功。
- 网关返回带路由前缀的模型别名时，之前只有布尔 mismatch；现在输出 `exact/alias/mismatch/unavailable`，仍对严格验证 fail-closed。
- LLM 调用之前只记录 provider/model；现在记录无密钥的 provider provenance（client model、endpoint 指纹、配置指纹和环境变量名）。`research blame` 的 `@latest:<kind>` 也改为按指定历史 commit 解析。

本轮继续收紧三个容易被误读的表面：

- `provider check` 默认仍是零请求；只有显式 `--live` 才做一次最小请求，并将
  `configuration_only`、`local_service`、`live_request` 分开报告。
- 每份 closure audit 一次性包含 `trace`、`replay`、`verify` 三层的完成状态、claim
  覆盖数、blocker/warning 计数和代码列表；workspace `status.review` 复用同一摘要。
- feedback 可绑定 `intervention_id → outcome_id → evaluator_id`。默认状态是
  observational，报告明确显示未归因/部分绑定/独立成对，并且任何反馈都不能绕过
  Research VCS evolution gate。

这些修复提高了“记录是否与实际调用一致”的能力，但没有把本地记录变成不可伪造的第三方证明。

## 证据与可重复测试

### 已执行

| 检查 | 结果 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| `python -m pytest --collect-only -q` | 可收集 1398 项 | 测试入口可发现 | 不代表真实 provider 或 Docker 可用 |
| `python -m pytest -q`（本轮） | 1398 passed，5 warnings，53 subtests（534.13s） | 当前源码全量回归通过 | 测试主要是本地 mock/fixture |
| provider/closure/feedback 定向回归 | 155 passed，13 subtests | 本轮改动覆盖的 live 探针、闭环摘要、反馈归因和跨进程持久化成立 | 不代表跨平台和外部服务稳定 |
| 临时工作区 `setup → provider add → provider test` | 配置成功；真实最小请求成功 | 自定义 OpenAI-compatible 路由可被调用，返回了模型身份和 token 汇总 | 不代表完整科研运行成功 |
| `provider check --json` | 明确区分 configuration-only 与 live verification | 不再把“有 key”当成“API 已验证” | hosted provider 仍需 opt-in live test |
| `provider check --live --json` | 一次显式最小请求 | 验证传输和模型身份，不保存响应正文 | 不代表科学质量或完整运行成功 |
| closure audit / workspace status | 一次输出三层闭环摘要 | 审查者可直接看到每层 blocker/warning 数 | 本地账本仍不是第三方证明 |
| feedback attribution | 干预、结果、评估者可寻址 | 自进化线索不再只有无因果的趋势分数 | 成对观察仍需独立门禁才可晋级 |
| AutoResearchEval-inspired pilot | 官方任务清单的本地契约检查；内置 demo 的六阶段证据覆盖与 `trace/replay/verify` | 过程证据是否可审查、问题是否被门禁拦截 | 不产生官方模型分数；没有 rollout、gold 或独立 judge |
| Git-like process audit | 2 分支 fixture 测得 3 个有界 commit，并保留每个 commit 的分支归属；同时暴露 typed artifact、失败/恢复和公平性检查 | 同一输入下的中间决策证据与分支保留情况 | 不导出隐藏思维链；未验证同预算/evaluator/base 时不声称分支对比公平 |
| `privacy_audit.py . --json` | 0 findings | 当前仓库没有扫描到明显凭据/隐私泄露 | 不能证明未来输入不会泄露 |
| 分发 inventory + isolated wheel smoke | wheel/sdist 通过；隔离导入显示 39 个 schema、demo closure `blocked` | 新 benchmark/process schema 确实随包发布 | 不代表外部依赖、Docker 或 provider 在每台机器都可用 |

完整自动科研旅程在本机仍受到环境前置条件限制（例如可选 `sklearn`、认证会话和 Docker executable）。这不是测试失败应被掩盖的细节，而是首次运行体验的核心风险：用户很容易把“provider ready”误读成“研究可以运行”。

### 自定义 provider 的真实行为

测试只通过环境变量提供凭据，未将密钥写入仓库、报告或命令历史。endpoint 的 `/models` 返回可用模型列表；随后用一个极小请求验证了 OpenAI-compatible chat 路由、返回模型 ID、延迟和 aggregate token usage。探针刻意不保存 prompt/completion 内容。

仍需注意：最小请求只证明一次网络调用和模型身份回报，不证明模型回答的科学质量、数据分析正确性、实验统计有效性或论文结论可信。

## 六个维度的劣势

### 1. 易用性：离线审查容易，完整科研运行仍是专家路径（P1）

优点是 `explore`、Research VCS 和离线 DAG 不需要 provider；缺点是从离线对象到真实 Autopilot 需要同时理解 provider/model 前缀、credential env、可选 Python extras、Docker、数据边界、预算、ARA/CAS 和质量门。

具体摩擦：

- provider、ARA、CAS、context receipt、closure 等术语在同一层出现，首次错误往往暴露内部实现而不是下一步动作；
- 模型路由历史兼容规则复杂，裸模型 ID 与 `openai_compat/<model>` 的区别不直观；
- `setup` 能创建工作区，但可选依赖缺失、认证 session 缺失和 Docker 缺失可能在后续阶段才一起出现；
- `--json` 已覆盖很多命令，但错误类别、remediation 和 human output 还没有完全统一；
- 高层生命周期虽支持 selector，用户仍经常需要在命令间搬运不可变 ID；
- provider 的价格未知、live probe、配置存在性和模型可用性仍是不同检查，用户需要知道它们的语义差异。

改进方向：把首屏拆成“离线 DAG / 真实 provider / 复现已有研究”三条路径；provider add 后始终显示 resolved provider、client model 和匿名 endpoint 指纹；doctor 为每个阻塞项给一个可复制命令；把内部术语放到 `--verbose`/JSON 详情层。

### 2. 可审查性：记录丰富，但信任边界容易被误读（P0）

ARA、Research VCS、Git 和 CAS 能保存大量过程证据，这是项目最强的部分；但本地 hash 只能证明“当前字节与记录一致”，不能证明“事件真的发生过”。本地 Git 可被重写，manifest/history、普通 JSONL trace 和用户提供的 attestation 仍缺外部签名、可信时间锚和独立 custody。

审查者还要面对多个相近协议层：ARA graph、Research VCS objects、epistemic graph、pipeline contract 和 adapter receipt 之间并非每条路径都自动闭合。`traceable`、`replayable`、`verified` 若被压成一个 `ok`，会制造虚假的确定感。

当前最合理的分级是：

1. `syntax`：Schema 合法；
2. `structure`：关系和计数闭合；
3. `integrity`：hash/CAS/lock 本地一致；
4. `replay`：代码、数据、依赖、环境、seed 可重放；
5. `verify`：独立主体和独立 reproduction 通过。

产品文案和 CLI 应始终显示等级及 blocker，而不是只显示绿色/红色。

### 3. 科研质量：生成质量与科学验证仍是两回事（P0）

系统能生成假设、实验代码、图表、论文和 reviewer 修复，但默认探索模式不等于严格确证模式。研究计划中的 rival hypothesis、零效应、机制替代、边界条件和鉴别实验，仍可能没有成为 BFTS 的硬执行约束；高质量模式、quality gate、独立 review 和 reproduction 需要明确启用/满足。

科研质量的硬短板：

- 领域适用性主要由用户定义，系统无法替代实验设计、测量误差判断和因果识别；
- 合成数据开关虽显式，但用户仍可能把 synthetic result 当作真实外部结论；
- 模型自评、review pass 和 evidence verified 标记不能自动成为独立证据；
- 统计功效、多重比较、泄漏、负结果选择和数据漂移需要领域级检查器；
- 失败被保存是优点，但失败为何发生、是否具有代表性仍需要人类解释。

因此应把系统定位为“生成和治理科研过程”，而不是“自动保证科研真值”。

### 4. 科研探索：搜索广度高，探索偏差与计划消费仍不透明（P1）

BFTS/ARA 保存父子树、keep/discard/crash、指标和失败分支，适合回看搜索轨迹；但探索策略可能被单一代理偏好、预算、上下文截断和模型路由决定。当前用户很难在一屏回答：

- 哪些候选从未被尝试，为什么没有尝试；
- 是模型判断、预算耗尽、工具错误还是数据不可用导致分支结束；
- 搜索是否覆盖了预注册的负对照和机制消融；
- 当前最佳结果相对所有失败/未完成候选是否存在选择偏差；
- provider/model 变化是否造成策略变化而非科学变化。

探索 DAG 需要增加“未探索但计划要求”的节点、停止原因 taxonomy、预算/上下文影响、模型 provenance 和候选覆盖率。否则 DAG 透明的是轨迹，不一定透明的是搜索空间。

### 5. 科研线索透明：对象可寻址，语义链仍有断点（P1）

Research VCS 的 typed object、relation、checkpoint、diff、blame 和 offline bundle 让线索比普通实验目录清晰得多。本次修复后，历史 commit 上的 `@latest:<kind>` 也能稳定解析。

仍有几类线索断点：

- 早期失败可能只有 BFTS journal，没有同等级 ARA/Research VCS 锚点；
- 研究计划、BFTS 实际任务、ARA 节点和最终 claim 之间仍需桥接检查；
- evidence/claim 的 hash 能证明内容未改，但不等于数据源、运行器和评估者真实存在；
- provider provenance 现在能记录匿名 endpoint/config 指纹，但旧产物没有该字段，跨版本迁移需要解释；
- `blame` 找到的是文件首次加入的 Git 变更，不能自动回答“哪次实验真正产生了这个科学结论”；
- 历史 draft、supersedes、contest 和 active frontier 的语义仍比文件 Git 复杂。

本轮加入的离线 AutoResearchEval-inspired pilot 在内置 balanced demo 上测得
`5/6` 个阶段达到最低 typed-evidence coverage（83.3%）；缺失的是检索产物。该 demo
的 `trace` 与 `replay` 通过、`verify` 因独立 reproduction 和 passing gate 阻塞，审查中
发现的两个问题被标为 `contained` 而不是“已修复”。若没有明确 hold/reject 门禁，
pilot 会标为 `open`，不会推断问题已经遏制。这正是过程 benchmark 能补足普通最终
分数的地方，但它仍不是 100-task/800-trajectory 的官方性能结果。
同一报告还保留有界的 commit/checkpoint 轨迹、branch 拓扑、typed artifact 计数、失败尝试
和结构化决策事件。可分享输出不包含 commit/branch 自由文本、prompt、completion 或
gold；它只在 manifest/task slice、fork base、预算和 evaluator 都可核验时才允许“公平分支
对比”，否则显式标记 `unverified`。

建议在 DAG 节点上同时显示 `selector → resolved id → checkpoint → source revision → provider fingerprint → evidence hash → gate/reproduction`，缺任何一段都显示可行动 blocker。

### 6. 反馈与自进化：反馈能积累，因果闭环尚未完成（P0）

反馈系统、review issue ledger、repair lane、playbook 和 evolution gate 已形成治理骨架；但“收到反馈”不等于“知道哪项策略导致改进”。目前仍缺：

- 反馈与具体 prompt/tool/model/data/预算变更的单变量因果绑定；
- 独立 holdout/prospective benchmark，避免在训练反馈上自证；
- 从 evolution intent 自动生成候选 diff、隔离评估、签名 attestation 的完整数据面；
- 真正的 deploy/canary/rollback adapter（很多 API 仍是 semantic receipt，而非生产文件变更）；
- 对跨项目 lesson 的去重、时效、反事实和灾难性回归评估；
- 反馈缺失、延迟、选择性提交和用户不采纳的显式状态。

所以当前更准确的说法是“受控自进化协议/控制面原型”，不是“系统已经会安全地自我改写并证明变好了”。

## 风险优先级与验收指标

| 优先级 | 风险 | 用户/科研后果 | 建议验收 |
|---|---|---|---|
| P0 | 本地账本被误读为独立证明 | 错把内部 green 当科学真值 | 所有 UI/报告显示 closure level、独立性和外部锚状态 |
| P0 | provider/模型/endpoint 不一致 | 请求发错账户或模型，结果不可复现 | 每次 LLM call 有 provenance；跨工作区切换回归；旧产物迁移标注 unknown |
| P0 | 反馈自进化无独立因果评估 | 策略越改越偏却看似提升 | 每个候选绑定 baseline、holdout、独立 evaluator、可验证 receipt |
| P1 | 研究计划未变成硬执行约束 | 漏掉负对照、消融和边界实验 | plan task coverage 100%，停止原因可分类 |
| P1 | 完整运行前置条件集中爆发 | 新用户无法定位阻塞 | doctor 每项一个 remediation；clean-host journey CI |
| P1 | 多协议桥接断点 | 线索只能人工拼接 | ARA↔Research VCS mapping contract + conformance fixtures |
| P2 | 领域适配窄 | 湿实验/仪器/现场结论不可直接迁移 | adapter 生态和至少两个独立消费者 |

推荐每次 release 公开这些指标，而不是只公开论文/模型分数：首次离线 DAG 时间、首次真实运行成功率、人工 ID 次数、provider live verification 覆盖率、trace→replay→verify 转化率、失败分支保留率、负对照覆盖率、平均成本、独立复现成功率和自进化回滚率。

## 未完成能力（无时间承诺）

以下是仍可验证的能力缺口，按风险优先级排列；它们不是 30/90/180 天交付计划。

### P0：证据与独立性

1. 把 closure level、独立主体、外部时间/签名状态加入所有人类和 JSON 输出。
2. 为 LLM call、experiment attempt、evidence、gate 建立同一份 provenance contract，并为旧对象提供显式 `unknown` 而非补猜。
3. 增加一个真实、公开、带失败和负结果的端到端样例，记录成本、环境、数据 hash 和 reproduction receipt。
4. 将 provider live probe、价格未知、客户端缺失、Docker 缺失汇总为统一 doctor remediation。

### P1：执行与协议

1. 让研究计划任务直接驱动 BFTS queue，并在 DAG 中显示未覆盖的判别实验。
2. 为探索节点记录停止原因、覆盖率、预算/上下文影响和候选未尝试原因。
3. 完成 ARA↔Research VCS 的确定性桥接和跨版本兼容矩阵。
4. 加入独立 evaluator service、签名 attestation、prospective benchmark 和真实 canary adapter。

### P2：生态与领域适配

1. 以 conformance kit 培育第二个独立协议实现和多个外部 adapter。
2. 引入可选外部透明日志/时间锚，但保持本地离线可用和隐私默认。
3. 扩展到湿实验、仪器和现场研究时，坚持 provenance、权限和人工责任边界，不把计算型经验外推成通用科学能力。

## 使用建议

- 把 `explore`/Research VCS 当作低成本研究线索和审查入口。
- 把 `provider check` 理解为配置检查；要确认远端才运行明确收费提示的 `provider test`。
- 也可以使用 `provider check --live`，但必须把它视为一次明确授权、可能计费的最小
  请求；默认 `provider check` 永远不做网络调用。
- 把 `trace`、`replay`、`verify` 当作不同门槛；任何 `unknown` 或 `awaiting_independent_verification` 都不能写成已证实。
- 查看 audit/status 中的 `closure_levels`，不要只看顶层 `ok` 或 `promotion_ready`。
- 反馈自进化前先记录 intervention/outcome/evaluator 链；没有成对结果时只能当作观察，
  不能宣称策略导致改进。
- 生产或投稿前固定 provider/model/endpoint 指纹、数据 hash、代码 revision、依赖锁、seed 和独立 evaluator 身份。
- 不要把合成数据、模型自评、用户自报 gate 或本地 hash 当作第三方科学证明。
