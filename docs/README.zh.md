<p align="center">
  <img src="https://raw.githubusercontent.com/smileformylove/XScientist/main/docs/xscientist-evidence-mark.png" width="220" alt="XScientist 证据路径标志">
</p>

<h1 align="center">XScientist</h1>

<p align="center"><strong>把一个想法变成 Git 式科研历史：可检查、可复现、可回滚。</strong></p>

<p align="center">
  只带来一个想法也可以——不要求先懂模型或 API Key。XScientist 会帮助检验它，
  但不会隐藏不确定性、失败尝试或相反证据。
</p>

<p align="center">
  <a href="https://pypi.org/project/xscientist/"><img src="https://img.shields.io/pypi/v/xscientist.svg" alt="PyPI 版本"></a>
  <a href="https://pypi.org/project/xscientist/"><img src="https://img.shields.io/pypi/pyversions/xscientist.svg" alt="Python 版本"></a>
  <a href="https://github.com/smileformylove/XScientist/actions/workflows/smoke.yml"><img src="https://github.com/smileformylove/XScientist/actions/workflows/smoke.yml/badge.svg?branch=main" alt="Smoke 检查"></a>
  <a href="https://github.com/smileformylove/XScientist/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache-2.0 许可证"></a>
  <a href="https://arxiv.org/abs/2607.12301"><img src="https://img.shields.io/badge/arXiv-2607.12301-b31b1b.svg" alt="arXiv 论文"></a>
</p>

<p align="center">
  <a href="#从自己的想法开始不需要-api-key">快速开始</a> ·
  <a href="#运行自主研究">自主研究</a> ·
  <a href="#检查审计与复现">审计复现</a> ·
  <a href="#安装方式">安装</a> ·
  <a href="../README.md">English</a>
</p>

XScientist 既是本地优先的自主科研系统，也是一套开放科学协议。它可以比较竞争
解释、选择更有信息量的实验、在隔离边界内执行、主动批判结果，并把整个过程保存
为带类型、机器可读的科研对象。一次运行完成，并不等于科学结论已经成立；只有证据
和评审门禁真正通过后，系统才会把它标记为已验证。

> [!IMPORTANT]
> XScientist 目前是 Alpha 科研软件，不是科学事实机器。自主运行可能调用付费模型；
> 生成代码必须经过配置好的隔离执行器；机器生成的结论只有补齐证据和独立评审
> 门禁后，才能成为“已验证”。

本文描述 `main` 上的 `0.1.3` 候选版。PyPI 当前正式版是 `0.1.2`；如需使用下方
新流程，请从 `main` 安装。

## 先选最短路径

| 你的起点 | 先运行 | Provider 或成本 | 立即得到什么 |
| --- | --- | --- | --- |
| 有想法，但没有模型或 API Key | `xscientist explore ./my-study --lang zh` | 无 | 本地保存、带版本、可证伪的科研起点 |
| 想先看看系统是否实用 | `xscientist demo ./first-study --autopilot --lang zh --open` | 无；`$0.00` | 完整但故意保留争议的证据历史 |
| 已有本地 Ollama 模型 | `xscientist provider list` | 本地算力；无需托管 Key | 可用模型和下一条配置命令 |
| 已有托管模型 Key | `xscientist start ./my-study` | 可能产生模型费用 | 在同一历史中启动受控自主科研 |

如果不确定，从 `explore` 开始。它只记录你真正知道的内容，未知项会诚实保持为空。

## 从自己的想法开始：不需要 API Key

只需要 Python 3.10+ 和 Git。安装完成后，不需要 API Key、模型、Docker 或网络调用。

```bash
python -m pip install \
  "xscientist @ git+https://github.com/smileformylove/XScientist.git@main"
xscientist explore ./my-study --lang zh
```

引导过程不要求理解 Provider 或科研协议，只会问四个普通问题：

- 你想研究什么想法？
- 如果想法成立，你预计能观察到什么变化？
- 什么结果会让你改变看法？
- 先做哪一个公平比较或检验？

只回答第一个问题也可以，之后再次运行同一命令继续。XScientist 会把真实状态明确
保存为“想法已保存”“可证伪”或“已有计划”，绝不会为了显得完整而替用户编造答案。
这条路径不使用 Provider，不调用模型，不执行生成代码，也不会伪造证据或结论。

脚本或自动化环境可以显式传入同样的信息：

```bash
xscientist explore ./my-study \
  --idea "每天散步是否改善睡眠质量？" \
  --expect "每天散步会改善预先选定的睡眠评分。" \
  --disprove "评分没有改善或变差。" \
  --test "比较散步阶段和日常活动阶段。" \
  --lang zh --non-interactive
```

无需阅读内部日志，也能理解工作区：

- `question.md` 保存人可以直接阅读的科研问题；
- `research.yaml` 保存本地策略与工作区身份；
- `.xscientist/objects/` 和 `checkpoints/` 保存有类型的决策与历史；
- 本地 Git 仓库没有远端，也不会自行推送。

日常使用 `status` 和 `history` 查看即可；新用户无需直接修改内部对象仓库。

如果想先看一条完整但存在争议的证据历史，可运行内置 `$0.00` 样例：

```bash
xscientist demo ./first-study --autopilot --lang zh --open
xscientist status ./first-study --lang zh
```

演示会诚实停在“运行完成，仍需补充证据”，因为留出结果挑战了过度宽泛的结论。
保留冲突是正确的科学行为，不是程序失败。

日常只看 `status` 即可。需要分支、流水线、Token 或后台任务细节时再加
`--verbose`；自动化程序使用 `--json`。

## 运行自主研究

离线引导能够整理用户亲自给出的判断，但不能诚实地凭空产生领域知识、数据或科研
发现。需要 AI 辅助探索时，再为已经保存的问题添加模型；同一个工作区可以安全升级，
不会替换原有科研历史。

先查看当前可用模型。这个命令不要求已有工作区，会优先发现正在运行的本地 Ollama，
再显示托管服务：

```bash
xscientist provider list
```

### 本地模型

[安装 Ollama](https://ollama.com/download)，下载一个本地模型，并确认本地服务正在
运行。桌面应用会启动服务；无界面环境可使用 `ollama serve`。这条路径不需要托管
API Key。当前[官方 CLI 文档](https://docs.ollama.com/cli)使用 `ollama pull` 下载、
`ollama ls` 查看本地模型：

```bash
ollama pull gemma3
ollama ls

python -m pip install \
  "xscientist[research,openai-compatible] @ git+https://github.com/smileformylove/XScientist.git@main"
xscientist provider list
xscientist start ./my-study
```

交互流程只询问缺失的信息：问题、Provider/模型、证据来源、本地科研身份和可选
预算。如果只发现一个可用 Provider，会自动选择，不要求用户重复确认。对于
`explore` 创建的工作区，会直接复用已保存的问题，并保留已有科研文件。本地模型
可以避免托管 API 费用，但仍会占用本机算力；生成实验代码仍需经过 Docker 隔离。

### 托管模型

安装科研运行时，再只安装一个需要的 Provider 客户端：

```bash
python -m pip install \
  "xscientist[research,openai] @ git+https://github.com/smileformylove/XScientist.git@main"
export OPENAI_API_KEY="..."
xscientist start ./hosted-study
```

客户端 extra 包括 `openai`、`anthropic`、`zhipu`、`bedrock`、`vertex` 和
`openai-compatible`。最后一个也覆盖 Ollama、DeepSeek、Gemini、OpenRouter 和
自定义兼容端点。

脚本和 CI 应显式写出所有重要选择：

```bash
xscientist start ./ood-study \
  --question "为什么检索增强反思在分布外失效？" \
  --provider openai \
  --model openai/gpt-4.1 \
  --user YOUR_NAME \
  --autopilot discovery \
  --data-dir ./data \
  --max-cost-usd 10 \
  --non-interactive
```

只有明确做探索性研究时，才用 `--allow-synthetic-data` 替代 `--data-dir`。输入
数据会先做内容哈希，再以只读方式挂载。启用成本上限后，未知模型价格会安全失败。

### 隔离与就绪检查

生成的实验代码不会静默运行在宿主 Python 进程里。模型实验需要 Docker 和版本
匹配的执行器：

```bash
xscientist executor prepare --workspace ./ood-study
xscientist provider check --workspace ./ood-study --max-cost-usd 10
xscientist doctor --workspace ./ood-study --deep
```

检查会区分客户端、凭据、本地模型、Docker CLI、Docker daemon 和执行器镜像等
问题，并按顺序给出修复命令；检查本身不会发起付费模型请求。

### 长任务

```bash
xscientist start ./ood-study \
  --question "机制会在哪里失效？" \
  --allow-synthetic-data --max-cost-usd 10 --detach

xscientist runs list --workspace ./ood-study
xscientist runs watch RUN_ID --workspace ./ood-study
xscientist runs logs RUN_ID --workspace ./ood-study --tail 100
xscientist runs cancel RUN_ID --workspace ./ood-study
xscientist runs resume RUN_ID --workspace ./ood-study
```

`status` 会优先显示运行中或失败的后台任务。失败态返回非零退出码，但 `--json`
仍会输出完整结构，便于自动诊断和恢复。

## 简单入口不会削弱自主科研

默认入口虽然精简，内部科研循环仍会根据 profile：

1. 提出竞争假设和零假设，而不是只维护第一个想法；
2. 提前锁定预测，并按预期信息价值选择实验；
3. 执行有边界的实验，保留失败和负结果；
4. 主动扫描异常、矛盾、证据质量和迁移边界；
5. 运行独立评审、有限修复，并在硬门禁处停止；
6. 打包论文、证据 DAG、来源信息和精确续研上下文。

自主不等于绕过科研权限。XScientist 不会静默编造用户没有回答的内容，不会把合成
数据冒充真实数据，不会在宿主机直接执行生成代码，不会晋级未经评审的结论，也不会
自行发表研究或推送工作区远端。

| Profile | 适用场景 | 侧重点 |
| --- | --- | --- |
| `balanced` | 第一次完整研究 | 有界搜索和标准评审 |
| `discovery` | 发现机制与边界 | 竞争假设、反驳压力、分支多样性 |
| `publication` | 论文候选 | 多角色独立评审和更严格门禁 |

深入策略能力仍在 `xscientist research` 下，但第一次使用无需学习几十个命令。详见
[深度科研协议](DEEP_RESEARCH_PROTOCOL.md)和[方法发现协议](METHOD_DISCOVERY_PROTOCOL.md)。

## 检查、审计与复现

科研 Git 保存的是有明确科学含义的对象，不要求用户从日志目录猜测上下文。Git 是
当前的本地存储适配器；无需 GitHub 账号或远端，XScientist 也不会自行推送研究。

```bash
xscientist history list ./first-study
xscientist audit ./first-study --level trace
xscientist audit ./first-study --level replay
xscientist audit ./first-study --level verify
```

审计会严格区分三个问题：

- `trace`：结论是否都能追溯到证据和决策？
- `replay`：代码、数据、环境、随机种子和命令是否足以重跑？
- `verify`：是否经过了要求的独立验证门禁？

三个层级只能逐级增强：已记录的结论可能可追踪但不可重放，也可能可重放但尚未经过
独立验证。审计显示阻塞，通常意味着存在明确的科研缺口，不一定是软件运行失败。

手工修改到达一个有意义的状态时，可以先保存检查点，再尝试风险更高的替代方案。
回滚默认只预览；显式使用 `--apply` 后也只会追加一条反向检查点，不删除、不改写
原始结果。

```bash
xscientist history save ./first-study -m "记录修正后的测量规则"
xscientist history rollback ./first-study --commit HEAD
# 先检查目标、影响、阻塞项以及自动生成的执行命令。
xscientist history rollback ./first-study --commit HEAD --apply
```

尚未保存的科研改动和仓库的第一个检查点都会阻止回滚。预览会检查这些本地前置
条件；撤销较早的检查点仍可能与后续工作冲突，此时
`--apply` 会停止，不会丢弃当前历史。复现、打包、对象检查、决策上下文、深度差异
与分支仍保留在完整协议入口中：

```bash
xscientist research reproduce HEAD --repo ./first-study --execute --record \
  --reproduces @latest:claim --verifier human:REPRODUCER

xscientist research bundle --repo ./first-study --dest ./study-backup
xscientist research export --repo ./first-study --dest ./exchange
```

生成的 DAG 是可随时重建的视图，不是科学源数据；重建它不会污染 checkpoint，也
不会阻止打包。真正的科研改动、已跟踪编辑或暂存内容仍必须先审查。

如果要挑战一个结论而不抹掉原历史：

```bash
xscientist research branch challenge/boundary --repo ./first-study --switch
xscientist research plan @latest:hypothesis --repo ./first-study \
  "寻找反例" --test "一个可复现失败将反驳当前机制"
xscientist research switch main --repo ./first-study
xscientist research merge challenge/boundary --repo ./first-study --preview
```

## 小入口，透明分层

```mermaid
flowchart TB
  U["explore · start · status"] --> O["自主科研循环"]
  O --> E["隔离实验与模型来源"]
  O --> R["有类型的科研 Git 历史"]
  E --> D["证据 DAG 与 ARA 产物"]
  R --> D
  D --> A["审计 · 历史 · 复现"]
```

日常入口只保留 `explore`、`start`、`status`、`audit` 和 `history`。环境修复放在
`doctor`，后台运行放在 `runs`，完整科研协议放在 `research`。文档开头的路径表就是
新用户需要的全部决策树。

公开编排层位于 `xscientist/`，实验工作流位于 `ai_scientist/`，版本化协议位于
`ai_scientist/protocol/`。更多细节见[架构文档](ARCHITECTURE.md)。

## 安装方式

| 渠道 | 命令 |
| --- | --- |
| PyPI 正式版 0.1.2 | `python -m pip install "xscientist==0.1.2"` |
| 0.1.3 候选版 | `python -m pip install "xscientist @ git+https://github.com/smileformylove/XScientist.git@main"` |
| 贡献者 | `python -m pip install -e ".[research,openai,dev]" -c requirements/constraints-ci.txt` |

需要实验严格复现时，请固定 commit，不要跟随变化中的 `main`。

| Extra | 用途 |
| --- | --- |
| `research` | 端到端自主科研运行时 |
| Provider extra | 一个模型客户端或兼容路由 |
| `plot`、`pdf`、`pdf-layout`、`ml` | 按需安装的专业实验能力 |
| `service` | FastAPI/Uvicorn 服务 |
| `trust` | 可选签名能力 |
| `full` | 向后兼容的全量环境 |

核心协议和 CLI 支持 Python 3.10–3.13。自主实验还取决于具体 Provider、Docker 和
研究所需的实验栈。

## 产物与边界

自主项目会分别保存配置、想法、实验、论文、运行日志与 ARA 移交产物。完整布局见
[输出目录](guides/OUTPUT_DIRECTORIES.md)。

| 边界 | 默认行为 |
| --- | --- |
| 生成代码 | 隔离执行；严格模式不满足条件即停止 |
| 实验网络 | 严格隔离时关闭 |
| 密钥 | 私有环境文件、Git 忽略、诊断脱敏 |
| 远端发布 | 从不自动进行 |
| 结论 | 补齐证据和独立门禁前保持草稿 |
| 负结果 | 作为一等科研历史保留 |
| 自我演化 | 影子候选 → 密封评估 → Canary → 签名晋级 |

敏感领域应把 XScientist 视为科研基础设施，而不是领域专家、伦理审查或合规验证的
替代品。

## SDK 与文档

```python
from xscientist import ProjectRequest, XScientist

client = XScientist(output_root="./research-output")
result = client.run_project(
    ProjectRequest(
        project="retrieval-study",
        question="检索增强反思会在什么条件下失效？",
        autopilot="discovery",
        allow_synthetic_data=True,
        max_cost_usd=10,
    )
)
print(result.returncode)
```

| 需求 | 文档 |
| --- | --- |
| 第一个项目与恢复 | [入门](GETTING_STARTED.md) · [长任务指南](LONG_RUNNING_GUIDE.md) |
| 科研历史与协议 | [本地科研 Git](LOCAL_RESEARCH_GIT.md) · [协议 v2](RESEARCH_PROTOCOL_V2.md) |
| 科研诚信与策略 | [科研诚信](RESEARCH_INTEGRITY.md) · [科学宪法](SCIENCE_CONSTITUTION.md) |
| SDK、HTTP API 与适配器 | [SDK/API](guides/SDK_AND_API.md) · [DAG/适配器](RESEARCH_DAG_AND_ADAPTERS.md) |
| 配置与运维 | [配置](CONFIG_REFERENCE.md) · [运维清单](OPERATIONS_CHECKLIST.md) |

日常命令看 `xscientist --help`；完整科研协议命令看
`xscientist research --help`。

## 项目状态

项目处于积极开发的 Alpha 阶段。贡献需包含测试，保持协议和 schema 兼容，并且
不能削弱来源追踪、隔离、成本或科学门禁。参见
[CONTRIBUTING.md](https://github.com/smileformylove/XScientist/blob/main/.github/CONTRIBUTING.md)
和[CHANGELOG.md](https://github.com/smileformylove/XScientist/blob/main/CHANGELOG.md)。

论文：[XScientist: Towards an AI-Driven Scientific Research Ecosystem](https://arxiv.org/abs/2607.12301)。

许可证：Apache-2.0，见
[LICENSE](https://github.com/smileformylove/XScientist/blob/main/LICENSE)。
