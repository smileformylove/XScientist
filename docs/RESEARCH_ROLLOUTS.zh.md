# 研究策略 Rollout 审计

Faraday 最值得迁移的不是参数量比较，而是系统结构：外层研究策略选择下一步实验，内层编码工具负责执行，任务级评估器检查结果是否真的支持科学结论。XScientist 现在把这条边界记录为 metadata-only 的 `research_rollout` 对象。

## 记录内容

- 带 hash 的任务与 split（`train`、`test`、`holdout` 或 `external`）；
- 可选的 harness/resource/evaluator 边界（`comparison_boundary`），用于在任务 hash 之外检查工具替换是否公平；
- 五维 rubric：结果一致性、主张支持、实现忠实度、资源效率、科学完整性；
- 工具调用的 provider/model 指纹、输入/输出 hash、预算、决策类型和结果；绝不保存 prompt、stdout、凭证或原始响应；
- 确定性的 `strategy_budget_summary`：决策归属、连续预算核算、失败/恢复观察和下一步提示；
- turn 元数据与仅供观察的事后正向 reward-delta credit；
- 零个或多个评估样本、各维分数、均值和分歧；
- 明确写入 `quality_claim_allowed=false` 与 `causal_claim_allowed=false`。

评估摘要只是测量记录，不是 ground truth。科研主张晋级仍需独立 gate；缺失的 reward trace 会保持缺失，builder 不会静默插值 credit。

## 这篇 Faraday 论文对本项目的真正借鉴

[Faraday 论文](https://arxiv.org/abs/2608.13331)训练的是外层研究策略
（Qwen3.6-27B），它把编码工作交给更强的工具，并在作者自己的 Replica
图表复现任务上做后训练。XScientist 是协议与本地审计 SDK：不包含
Faraday 权重，不运行 Replica harness，不调用 Codex，也不报告论文分数。
可迁移的是“边界契约”，而不是模型对比：

| Faraday 要素 | XScientist 的对应边界 |
| --- | --- |
| 策略选择下一步研究动作 | `tool_delegations` 与 `strategy_budget_summary` 记录角色、决策、顺序、预算和恢复，不保存 prompt/stdout |
| 编码工具执行实验 | 成功的 `coding_executor` 必须暴露 output hash；审计器要求该 artifact 被评估证据引用 |
| 任务级五维 judge | 锁定 rubric 与逐样本评估；分数仍是 observational、仅供 judge 参考 |
| 多样本与 turn credit | 有界评估样本与事后 turn credit 可追溯；不会推断因果归因或自动执行 RL 更新 |
| 人类校准与独立检查 | 评估器签名一个 receipt，绑定自身 principal、全部已观察 producer 身份和被检查 artifact hash；`rollout-audit` 使用本地 trust store 验签 |

因此，缺失的策略步骤、断裂的预算、未被评估的执行 artifact 或缺失的
独立评估器都会成为机器可见 blocker，而不是被默认为成功。这是受
Faraday 启发的审计/训练数据契约，不是 Replica importer、solver、三评审
协议的复现，也不是本地科研质量结果。

## CLI

准备包含 `task_id`、完整 `task_hash`、`time_budget_seconds`，以及可选
`tool_delegations`、`turns`、`evaluations` 的 `episode.json`，执行：

```bash
xscientist research rollout episode.json \
  --repo ./first-study --json > rollout.json
```

命令离线且幂等，只保存脱敏的内容寻址 Research VCS 对象，并生成标准 experiment
checkpoint。其 JSON wrapper 会把规范 payload 放在 `rollout` 字段中，因此捕获的
文件可直接传给 `rollout-audit`，无需手工提取或改写 payload。

若要审计已保存的原始 payload（或上面命令输出的 JSON wrapper），请把本地
证据索引中已知的 hash 显式传入：

```bash
xscientist research rollout-audit rollout.json \
  --evidence-hash sha256:... \
  --trust-store trust-store.json --json
```

审计器对 `completed` episode fail-closed：检查 schema 和全部内容 hash、任务/
rubric 绑定、预算边界、策略摘要、成功执行 artifact 与评估证据的绑定，以及
actor-disjoint evaluator receipt。输出只有有界 blocker/warning，并始终保持
`quality_claim_allowed=false` 与 `causal_claim_allowed=false`。不提供 evidence
resolver 时只能检查 hash 语法，不能验证 artifact 是否真实存在，因此完成态
不会获得 verification-eligible。没有可信评估器 attestation 时，调用方声明的
`identity_verified=true` 也只是一条观察元数据，同样不能放行完成态 rollout。

对于 `completed` rollout，预算与恢复是审计门，而不是 dashboard 提示。每次调用
都必须有 before/after 预算边界，第一次调用必须从声明预算起算，相邻调用必须组成
连续链，且结果仍在声明边界内。若失败或超时调用标记
`follow_up_required=true`，后续必须出现结果为成功的 `repair`/`delegate`，或显式
终止 `stop`。失败的修复尝试不算恢复；若之后仍无成功响应或 stop，完成态 rollout
会继续被阻止。`stop` 之后只要还有任何工具调用，就不是终止动作，同样阻断验证。

## 签名评估器 receipt

使用公开 builder 创建精确的规范绑定，通过现有 attestation 协议签名，再把
envelope 包装成 receipt：

```python
from ai_scientist.protocol.attestation import sign_attestation
from xscientist import (
    INDEPENDENCE_ATTESTATION_PURPOSE,
    build_independence_attestation_payload,
    build_independence_receipt,
)

binding = build_independence_attestation_payload(
    evaluator_id="judge-independent",
    evaluator_identity="human:reviewer-42",
    target_hashes=[executor_output_hash],
    producer_actor_ids=all_rollout_producer_ids,
)
attestation = sign_attestation(
    binding,
    purpose=INDEPENDENCE_ATTESTATION_PURPOSE,
    identity="human:reviewer-42",
    key_id="reviewer-42-ed25519",
    algorithm="ed25519",
    key=private_key_pem,
)
receipt = build_independence_receipt(
    evaluator_id="judge-independent",
    evaluator_identity="human:reviewer-42",
    target_hashes=[executor_output_hash],
    producer_actor_ids=all_rollout_producer_ids,
    attestation=attestation,
)
```

本地 trust store 以 `key_id` 为键，提供预期 identity、algorithm、public key 和
可选撤销状态。共享或长期记录应优先使用 Ed25519。HMAC 只适合本地流程，因为其
trust store 含 secret，不能提交或打印。receipt builder 只检查结构和内容绑定；
真正的信任与可选 freshness 由 `rollout-audit` 检查，需要时使用
`--max-attestation-age-seconds`。

## 工具替换与对比边界

`assess_tool_swap_compatibility(reference, candidate)` 会检查任务 hash、rubric hash、split 和时间预算。若两份报告都提供 `comparison_boundary`，还会检查 harness、资源指纹、评估协议、起始 artifact、网络策略和 seed 策略。边界合格只是进行受控比较的前提，不代表任何模型或工具更好。

若工具替换资格必须同时包含两份 rollout 自身的审计结果，请启用 strict mode：

```python
from xscientist import assess_tool_swap_compatibility

comparison = assess_tool_swap_compatibility(
    reference,
    candidate,
    strict=True,
    audit_evidence_hashes=all_reference_and_candidate_evidence_hashes,
    audit_trust_store=local_trust_store,
    max_attestation_age_seconds=3600,  # 可选
)
```

`audit_evidence_hashes` 必须是两份 rollout 证据 hash 的并集 resolver，
`audit_trust_store` 必须能验证两份评估器 receipt。任一输入缺失、任一 rollout 尚未
verification-ready，或 rollout/工具签名实际未变化时，strict mode 都会
fail-closed。即使通过，也不会授予质量或因果主张权限。

当前实现不复现 Faraday 的 RL 训练、三评审协议、编码工具提供商、benchmark 任务集或论文分数。这是受 [Faraday 论文](https://arxiv.org/abs/2608.13331) 启发的 XScientist 审计/训练数据契约，不是本地复现实验。
