"""Beginner-friendly research setup and next-step guidance."""

from __future__ import annotations

import locale
import shlex
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.privacy import redact_sensitive_payload

from .research_git import ResearchGitError, init_repository
from .research_vcs import ResearchRepository

GUIDE_SCHEMA = "xscientist.research-guide.v1"
WORKSPACE_ACTION_SCHEMA = "xscientist.workspace-action.v1"
WORKSPACE_ACTION_CONTEXT_SCHEMA = "xscientist.workspace-action-context.v1"
WORKSPACE_PLACEHOLDER = "{workspace}"
_COMMAND_PLACEHOLDER_TRANSLATIONS = {
    "YOUR HYPOTHESIS": "你的假设",
    "WHAT RESULT WOULD DISPROVE IT": "什么结果会推翻它",
    "RIVAL HYPOTHESIS": "竞争假设",
    "WHAT RESULT WOULD DISPROVE THE RIVAL": "什么结果会推翻竞争假设",
    "RESEARCH QUESTION": "研究问题",
    "WHAT TO TEST": "要检验什么",
    "WHAT RESULT SEPARATES THE EXPLANATIONS": "什么结果能区分这些解释",
    "NAME=VALUE": "名称=数值",
    "DATASET": "数据集",
    "METRIC": "指标",
    "BASELINE": "基线",
    "SPLIT_FILE": "划分文件",
    "YOUR_NAME": "你的姓名",
    "WHAT YOU RAN": "运行了什么",
    "WHAT THE RESULT SHOWS": "结果说明了什么",
    "BOUNDED CONCLUSION": "有边界的结论",
    "WHY THIS EVIDENCE JUSTIFIES THE CONCLUSION": "该证据为何支持这一结论",
    "REVIEW SUMMARY": "复核摘要",
    "REVIEWER": "复核人",
    "BOUNDED CLAIM": "有边界的主张",
    "TESTED CONDITIONS": "已检验条件",
    "TEST THE CONTESTED BOUNDARY": "检验存在争议的边界",
    "WHAT RESULT WOULD RESOLVE THE CONFLICT": "什么结果能解决冲突",
    "RESOLUTION EXPERIMENT": "争议解决实验",
    "RESOLUTION METRIC": "解决指标",
    "RESOLUTION RESULT": "争议解决结果",
    "RESOLUTION CONCLUSION": "争议解决结论",
    "WHY THE NEW EVIDENCE RESOLVES THE CONFLICT": "新证据为何能解决冲突",
    "RESOLUTION REVIEW": "争议解决复核",
    "NARROWED CLAIM": "缩小范围后的主张",
    "RESOLVED CONDITIONS": "已解决的适用条件",
    "REPRODUCER": "复现人",
}


def _required(value: str, *, label: str) -> str:
    normalized = " ".join(str(value or "").split())
    if not normalized:
        raise ResearchGitError(f"{label} cannot be empty")
    return normalized


def _optional(value: str | None) -> str:
    return " ".join(str(value or "").split())


def _human_actor(value: str) -> str:
    raw_actor = _required(value, label="actor")
    if ":" not in raw_actor:
        return "human:" + raw_actor
    if raw_actor.startswith("human:") and raw_actor != "human:":
        return raw_actor
    raise ResearchGitError("guided research actor must be a human identity")


def _language(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized == "auto":
        detected = (locale.getlocale()[0] or "en").lower()
        return "zh" if detected.startswith("zh") else "en"
    if normalized not in {"en", "zh"}:
        raise ResearchGitError("guide language must be en, zh, or auto")
    return normalized


def _text(language: str, english: str, chinese: str) -> str:
    return chinese if language == "zh" else english


def _step(
    language: str,
    *,
    code: str,
    title_en: str,
    title_zh: str,
    why_en: str,
    why_zh: str,
    command: str,
) -> dict[str, str]:
    return {
        "code": code,
        "title": _text(language, title_en, title_zh),
        "why": _text(language, why_en, why_zh),
        "command": _localize_command(language, command),
    }


def _localize_command(language: str, command: str) -> str:
    if language != "zh":
        return command
    rendered = command
    for source, target in _COMMAND_PLACEHOLDER_TRANSLATIONS.items():
        rendered = rendered.replace(source, target)
    return rendered


def _command_input_placeholders(command: str) -> list[str]:
    """Return unresolved human-input markers without mistaking HEAD for one."""

    found = {
        placeholder
        for pair in _COMMAND_PLACEHOLDER_TRANSLATIONS.items()
        for placeholder in pair
        if placeholder in command
    }
    return sorted(found)


def _command_for_repo(command: str, repo: str | Path) -> str:
    """Keep copy/paste guidance bound to the repository the user inspected."""

    if command == "xscientist explore .":
        return "xscientist explore " + shlex.quote(str(repo))
    if command == "xscientist explore . --lang zh":
        return "xscientist explore " + shlex.quote(str(repo)) + " --lang zh"
    prefix = "xscientist research "
    if not command.startswith(prefix) or " --repo " in f" {command} ":
        return command
    remainder = command[len(prefix) :]
    subcommand, separator, arguments = remainder.partition(" ")
    if subcommand in {"program", "literature", "discovery"} and separator:
        nested, nested_separator, nested_arguments = arguments.partition(" ")
        # Template generators are intentionally repository-neutral and do not
        # accept ``--repo``. Keep their copy/paste commands valid.
        if nested == "template" and subcommand in {"program", "discovery"}:
            return command
        contextual = f"{prefix}{subcommand} {nested} --repo {shlex.quote(str(repo))}"
        return f"{contextual} {nested_arguments}" if nested_separator else contextual
    contextual = f"{prefix}{subcommand} --repo {shlex.quote(str(repo))}"
    return f"{contextual} {arguments}" if separator else contextual


def workspace_action_contract(command: str | None) -> dict[str, Any] | None:
    """Return a host-path-free execution contract for one suggested command.

    Human output can bind a command directly to the path supplied by the user.
    Portable JSON cannot disclose that absolute host path, so it must expose an
    explicit placeholder and say whether the selected workspace is passed as an
    argument or used as the process working directory. A bare ``.`` therefore
    never masquerades as a command that is safe from an arbitrary caller cwd.
    """

    normalized = str(command or "").strip()
    if not normalized:
        return None
    try:
        argv = shlex.split(normalized)
    except ValueError:
        return {
            "schema_version": WORKSPACE_ACTION_SCHEMA,
            "command_template": None,
            "argv_template": [],
            "executable_after_binding": False,
            "workspace_binding": {
                "mode": "unavailable",
                "source": None,
                "placeholder": None,
                "required": False,
                "host_path_disclosed": False,
            },
            "cwd_binding": {"mode": "caller", "template": None},
            "input_binding": {
                "mode": "unavailable",
                "required": True,
                "placeholders": [],
            },
        }

    binding_mode = "none"
    cwd_mode = "caller"

    for flag in ("--repo", "--workspace"):
        if flag in argv:
            index = argv.index(flag)
            if index + 1 < len(argv):
                argv[index + 1] = WORKSPACE_PLACEHOLDER
                binding_mode = "argument"
                break

    if binding_mode == "none" and len(argv) >= 3:
        if argv[:2] in (
            ["xscientist", "explore"],
            ["xscientist", "start"],
            ["xscientist", "status"],
            ["xscientist", "audit"],
        ):
            argv[2] = WORKSPACE_PLACEHOLDER
            binding_mode = "argument"
        elif argv[:2] == ["xscientist", "history"] and len(argv) >= 4:
            argv[3] = WORKSPACE_PLACEHOLDER
            binding_mode = "argument"
        elif argv[:3] == ["xscientist", "research", "init"] and len(argv) >= 4:
            argv[3] = WORKSPACE_PLACEHOLDER
            binding_mode = "argument"

    if binding_mode == "none" and argv[:2] == ["xscientist", "research"]:
        repository_neutral_template = (
            len(argv) >= 4
            and argv[2] in {"program", "discovery"}
            and argv[3] == "template"
        )
        if repository_neutral_template:
            binding_mode = "cwd"
            cwd_mode = "workspace_root"
        else:
            argv.extend(["--repo", WORKSPACE_PLACEHOLDER])
            binding_mode = "argument"

    if binding_mode == "none" and argv[:2] == ["xscientist", "demo"]:
        # ``./xscientist-demo`` is intentionally relative, but portable status
        # JSON must bind that relative destination to the inspected workspace
        # instead of whichever directory an automation caller happens to use.
        binding_mode = "cwd"
        cwd_mode = "workspace_root"

    if binding_mode == "none" and argv[:2] in (
        ["xscientist", "doctor"],
        ["xscientist", "executor"],
    ):
        argv.extend(["--workspace", WORKSPACE_PLACEHOLDER])
        binding_mode = "argument"
    if (
        binding_mode == "none"
        and len(argv) >= 3
        and argv[:2] == ["xscientist", "provider"]
        and argv[2] in {"add", "check", "test", "activate", "remove"}
    ):
        argv.extend(["--workspace", WORKSPACE_PLACEHOLDER])
        binding_mode = "argument"
    if binding_mode == "none" and argv[:2] == ["xscientist", "preflight"]:
        binding_mode = "cwd"
        cwd_mode = "workspace_root"

    quoted_placeholder = shlex.quote(WORKSPACE_PLACEHOLDER)
    command_template = shlex.join(argv).replace(
        quoted_placeholder, WORKSPACE_PLACEHOLDER
    )
    workspace_required = binding_mode in {"argument", "cwd"}
    input_placeholders = _command_input_placeholders(command_template)
    return {
        "schema_version": WORKSPACE_ACTION_SCHEMA,
        "command_template": command_template,
        "argv_template": argv,
        "executable_after_binding": not input_placeholders,
        "workspace_binding": {
            "mode": binding_mode,
            "source": "invocation_workspace" if workspace_required else None,
            "placeholder": WORKSPACE_PLACEHOLDER if workspace_required else None,
            "required": workspace_required,
            "host_path_disclosed": False,
        },
        "cwd_binding": {
            "mode": cwd_mode,
            "template": (
                WORKSPACE_PLACEHOLDER if cwd_mode == "workspace_root" else None
            ),
        },
        "input_binding": {
            "mode": "template" if input_placeholders else "none",
            "required": bool(input_placeholders),
            "placeholders": input_placeholders,
        },
    }


def workspace_action_context() -> dict[str, Any]:
    """Describe how portable action templates bind the selected workspace."""

    return {
        "schema_version": WORKSPACE_ACTION_CONTEXT_SCHEMA,
        "workspace_placeholder": WORKSPACE_PLACEHOLDER,
        "workspace_source": "workspace argument supplied to this invocation",
        "binding_required_before_execution": True,
        "template_inputs_may_be_required": True,
        "host_path_disclosed": False,
    }


def public_workspace_action(step: dict[str, Any]) -> dict[str, Any]:
    """Make one next-step row safe and unambiguous for portable JSON."""

    safe = deepcopy(step)
    contract = workspace_action_contract(safe.get("command"))
    safe["action"] = contract
    if contract is not None:
        safe["command"] = contract.get("command_template")
    return safe


def public_research_guide_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a portable guide whose actions require explicit workspace binding."""

    safe = deepcopy(payload)
    safe["repository"] = "."
    safe["workspace_context"] = workspace_action_context()
    safe["next_steps"] = [
        public_workspace_action(step)
        for step in safe.get("next_steps") or []
        if isinstance(step, dict)
    ]
    primary = safe.get("primary_action")
    if isinstance(primary, dict):
        safe["primary_action"] = public_workspace_action(primary)
    return redact_sensitive_payload(safe)


_ACTION_OWNERS = {
    "record_hypothesis": "researcher",
    "record_rival_hypothesis": "researcher",
    "lock_hypothesis_portfolio": "researcher",
    "choose_first_test": "researcher",
    "choose_study_mode": "researcher",
    "preregister_confirmatory": "researcher",
    "lock_method_discovery": "researcher",
    "run_experiment": "experimenter",
    "bind_evidence": "experimenter",
    "record_inference": "researcher",
    "independent_review": "independent_reviewer",
    "state_claim": "researcher",
    "resolve_contested_claim": "researcher",
    "run_resolution_experiment": "experimenter",
    "bind_resolution_evidence": "experimenter",
    "infer_resolution": "researcher",
    "review_resolution": "independent_reviewer",
    "replace_contested_claim": "researcher",
    "reproduce": "independent_reproducer",
    "inspect_dag": "researcher",
    "strengthen_research_program": "researcher",
}

_ACTION_INPUTS = {
    "record_hypothesis": ["expected_observation", "disconfirming_result"],
    "record_rival_hypothesis": [
        "rival_hypothesis",
        "rival_disconfirming_result",
    ],
    "lock_hypothesis_portfolio": [
        "research_question",
        "primary_hypothesis",
        "rival_hypothesis",
    ],
    "choose_first_test": ["dataset_or_observation", "fair_comparison"],
    "choose_study_mode": ["research_question", "candidate_explanations"],
    "preregister_confirmatory": [
        "dataset",
        "metric",
        "baseline",
        "locked_split_hash",
        "estimand",
    ],
    "lock_method_discovery": ["intervention", "rival_explanation", "transfer_test"],
    "run_experiment": ["command_or_run_description", "seed", "result_artifact"],
    "bind_evidence": ["experiment_attempt", "result_artifact", "direction"],
    "record_inference": ["evidence", "warrant", "bounded_conclusion"],
    "independent_review": ["reviewer_identity", "review_decision", "review_scope"],
    "state_claim": ["reviewed_inference", "tested_conditions", "claim_scope"],
    "reproduce": ["checkpoint_or_ref", "independent_executor", "receipt"],
    "strengthen_research_program": [
        "rival_hypothesis",
        "mechanism",
        "transfer_boundary",
    ],
}


def _primary_action(
    language: str,
    next_steps: list[dict[str, str]],
    completed_stages: int,
) -> dict[str, Any]:
    """Expose one deterministic action while retaining alternatives for experts."""

    if not next_steps:
        return {
            "code": "none",
            "title": _text(language, "No pending action", "没有待处理动作"),
            "why": _text(
                language,
                "The guide has no unresolved step.",
                "当前指南没有未解决步骤。",
            ),
            "command": None,
            "priority": "P2",
            "blocking_level": "advisory",
            "owner": "researcher",
            "required_inputs": [],
        }
    step = next_steps[0]
    code = str(step.get("code") or "unknown")
    blocking = completed_stages < 8 and code not in {
        "strengthen_research_program",
        "inspect_dag",
    }
    return {
        "code": code,
        "title": step.get("title"),
        "why": step.get("why"),
        "command": step.get("command"),
        "priority": "P0" if blocking else "P1",
        "blocking_level": "blocking" if blocking else "advisory",
        "owner": _ACTION_OWNERS.get(code, "researcher"),
        "required_inputs": list(_ACTION_INPUTS.get(code, [])),
    }


def _created_at(item: dict[str, Any]) -> str:
    return str(item.get("created_at") or "")


def _latest(items: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    matches = [item for item in items if item.get("kind") == kind]
    return max(matches, key=_created_at) if matches else None


def inspect_idea_research(path: str | Path) -> dict[str, Any]:
    """Return the plain-language framing state of an existing repository."""

    repository = ResearchRepository(path)
    objects = repository.objects()
    question = _latest(objects, "question")
    if question is None:
        raise ResearchGitError(
            "research repository has no recorded question; use a new directory or "
            "record a question first"
        )
    hypothesis = _latest(objects, "hypothesis")
    plan = _latest(objects, "research_plan")
    question_payload = question.get("payload") or {}
    hypothesis_payload = (hypothesis or {}).get("payload") or {}
    plan_payload = (plan or {}).get("payload") or {}
    return {
        "repository": repository.path.as_posix(),
        "idea": str(question_payload.get("question") or "").strip(),
        "question_id": question["object_id"],
        "expectation": str(hypothesis_payload.get("statement") or "").strip(),
        "disconfirming_result": str(hypothesis_payload.get("falsifier") or "").strip(),
        "hypothesis_id": (hypothesis or {}).get("object_id"),
        "first_test": str(
            (plan_payload.get("discriminating_tests") or [""])[0]
        ).strip(),
        "success_rule": str(plan_payload.get("success_rule") or "").strip(),
        "plan_id": (plan or {}).get("object_id"),
    }


def _latest_after(
    items: list[dict[str, Any]], kind: str, created_at: str
) -> dict[str, Any] | None:
    matches = [
        item
        for item in items
        if item.get("kind") == kind and _created_at(item) > created_at
    ]
    return max(matches, key=_created_at) if matches else None


def build_research_guide(
    repo: str | Path,
    *,
    language: str = "auto",
    command_repo: str | Path | None = None,
) -> dict[str, Any]:
    """Explain current progress and return safe copy/paste next actions."""

    selected_language = _language(language)
    repository = ResearchRepository(repo)
    objects = repository.objects()
    counts = Counter(str(item["kind"]) for item in objects)
    user_idea_entry = any(
        item.get("kind") == "question"
        and (item.get("payload") or {}).get("source") == "user_idea"
        for item in objects
    )
    relations = [
        (str(item["object_id"]), relation)
        for item in objects
        for relation in item.get("relations") or []
    ]
    latest_claim = _latest(objects, "claim")
    latest_hypothesis = _latest(objects, "hypothesis")
    latest_plan = _latest(objects, "research_plan")
    latest_attempt = _latest(objects, "experiment_attempt")
    latest_evidence = _latest(objects, "evidence") or _latest(
        objects, "passage_evidence"
    )
    latest_inference = _latest(objects, "inference")
    contested_claim = (
        latest_claim
        if latest_claim is not None
        and (
            latest_claim.get("state") in {"contested", "rejected"}
            or any(
                relation.get("type") in {"refutes", "contradicts"}
                and relation.get("target") == latest_claim.get("object_id")
                for _source, relation in relations
            )
        )
        else None
    )
    next_steps: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not counts["hypothesis"]:
        next_steps.append(
            _step(
                selected_language,
                code="record_hypothesis",
                title_en=(
                    "Make your idea testable"
                    if user_idea_entry
                    else "Write one falsifiable hypothesis"
                ),
                title_zh=(
                    "把想法变成可以检验的问题"
                    if user_idea_entry
                    else "写下一个可证伪假设"
                ),
                why_en="A scientific question becomes testable only after you state what would prove it wrong.",
                why_zh="只有同时说明什么结果会推翻它，研究问题才真正可检验。",
                command=(
                    (
                        "xscientist explore ."
                        + (" --lang zh" if selected_language == "zh" else "")
                    )
                    if user_idea_entry
                    else (
                        'xscientist research hypothesis "YOUR HYPOTHESIS" '
                        '--falsifier "WHAT RESULT WOULD DISPROVE IT"'
                    )
                ),
            )
        )
    elif not counts["research_plan"]:
        hypotheses = sorted(
            (item for item in objects if item.get("kind") == "hypothesis"),
            key=lambda item: (_created_at(item), str(item.get("object_id") or "")),
        )
        primary_hypothesis_id = str(hypotheses[0]["object_id"])
        if len(hypotheses) == 1:
            next_steps.append(
                _step(
                    selected_language,
                    code="record_rival_hypothesis",
                    title_en="Record a falsifiable rival hypothesis",
                    title_zh="记录一个可证伪的竞争假设",
                    why_en=(
                        "A rival explanation makes the first study discriminate "
                        "between plausible accounts instead of only seeking support."
                    ),
                    why_zh=(
                        "竞争解释能让第一个研究区分多个可能解释，"
                        "而不是只寻找支持性结果。"
                    ),
                    command=(
                        'xscientist research hypothesis "RIVAL HYPOTHESIS" '
                        '--falsifier "WHAT RESULT WOULD DISPROVE THE RIVAL"'
                    ),
                )
            )
        elif not counts["hypothesis_portfolio"]:
            alternative_flags = " ".join(
                "--alternative " + shlex.quote(str(item["object_id"]))
                for item in hypotheses[1:]
            )
            next_steps.append(
                _step(
                    selected_language,
                    code="lock_hypothesis_portfolio",
                    title_en="Lock the competing hypotheses before choosing a test",
                    title_zh="在选择检验前锁定竞争假设组合",
                    why_en=(
                        "A locked portfolio preserves the alternatives and their "
                        "priors before results can influence the comparison."
                    ),
                    why_zh=(
                        "在结果影响比较之前锁定竞争假设及其先验，"
                        "可以避免事后改写解释。"
                    ),
                    command=(
                        "xscientist research program portfolio "
                        f"{shlex.quote(primary_hypothesis_id)} {alternative_flags} "
                        '--question "RESEARCH QUESTION"'
                    ),
                )
            )
        if user_idea_entry:
            next_steps.append(
                _step(
                    selected_language,
                    code="choose_first_test",
                    title_en="Choose the first fair comparison",
                    title_zh="选择第一个公平比较",
                    why_en=(
                        "A small discriminating test is more useful than collecting "
                        "only examples that support the idea."
                    ),
                    why_zh=(
                        "先做一个能够区分不同解释的小检验，比只收集支持想法的例子更有用。"
                    ),
                    command=(
                        "xscientist explore ."
                        + (" --lang zh" if selected_language == "zh" else "")
                    ),
                )
            )
        next_steps.append(
            _step(
                selected_language,
                code="choose_study_mode",
                title_en="Option A — plan exploratory work",
                title_zh="选项 A — 规划探索性研究",
                why_en="Use this while comparing explanations or discovering which test is most informative.",
                why_zh="当你仍在比较不同解释，或寻找最有信息量的检验时，使用这条路径。",
                command=(
                    "xscientist research plan "
                    f'{shlex.quote(primary_hypothesis_id)} "WHAT TO TEST" '
                    '--test "WHAT RESULT SEPARATES THE EXPLANATIONS"'
                ),
            )
        )
        next_steps.append(
            _step(
                selected_language,
                code="preregister_confirmatory",
                title_en="Option B — lock a confirmatory study",
                title_zh="选项 B — 锁定确证性研究",
                why_en="Use this when the metric, baseline, data split, and success rule are fixed before the test.",
                why_zh="当指标、基线、数据切分和成功标准都能在实验前确定时，使用这条路径。",
                command=(
                    "xscientist research preregister "
                    f"{shlex.quote(primary_hypothesis_id)} "
                    "--dataset DATASET --metric METRIC --baseline BASELINE "
                    "--split-file SPLIT_FILE --registered-by human:YOUR_NAME"
                ),
            )
        )
        next_steps.append(
            _step(
                selected_language,
                code="lock_method_discovery",
                title_en="Option C — test a transferable method",
                title_zh="选项 C — 检验可迁移的新方法",
                why_en=(
                    "Use a generated contract to prevent a local score gain or "
                    "larger resource budget from being mislabeled as discovery."
                ),
                why_zh=(
                    "用生成的契约隔离变量、锁定资源和盲测条件，避免把局部分数提升"
                    "或扩大算力误称为方法发现。"
                ),
                command=(
                    "xscientist research discovery template --output discovery.json"
                ),
            )
        )
    elif not counts["experiment_attempt"]:
        if latest_plan is None:  # pragma: no cover - guarded by the branch above
            raise ResearchGitError("research guide cannot resolve the active plan")
        next_steps.append(
            _step(
                selected_language,
                code="run_experiment",
                title_en="Record the first experiment",
                title_zh="记录第一次实验",
                why_en="Successes, failures, and timeouts are all evidence about the research path.",
                why_zh="成功、失败和超时都属于研究路径中的有效信息。",
                command=(
                    'xscientist research experiment "WHAT YOU RAN" --status completed '
                    f"--plan {shlex.quote(str(latest_plan['object_id']))} "
                    "--metric NAME=VALUE --seed 1"
                ),
            )
        )
    elif not (counts["evidence"] or counts["passage_evidence"]):
        if latest_attempt is None:  # pragma: no cover - guarded by the branch above
            raise ResearchGitError("research guide cannot resolve the active attempt")
        next_steps.append(
            _step(
                selected_language,
                code="bind_evidence",
                title_en="Record the result before interpreting its direction",
                title_zh="先记录结果，再判断它的方向",
                why_en=(
                    "Bind the result to the exact attempt first. Add --supports or "
                    "--refutes only after comparing it with the locked prediction."
                ),
                why_zh=(
                    "先把结果绑定到确切实验；与锁定预测比较后，再明确添加 "
                    "--supports 或 --refutes。"
                ),
                command=(
                    'xscientist research evidence "WHAT THE RESULT SHOWS" '
                    f"--attempt {shlex.quote(str(latest_attempt['object_id']))}"
                ),
            )
        )
    elif not counts["inference"]:
        if latest_evidence is None:  # pragma: no cover - guarded by the branch above
            raise ResearchGitError("research guide cannot resolve the active evidence")
        next_steps.append(
            _step(
                selected_language,
                code="record_inference",
                title_en="State why the evidence supports the conclusion",
                title_zh="明确证据为何能够支持结论",
                why_en=(
                    "Separating evidence from the warrant exposes assumptions and makes "
                    "the scientific argument reviewable."
                ),
                why_zh="把证据和推理依据分开，才能暴露隐藏假设并让科学论证可审查。",
                command=(
                    'xscientist research infer "BOUNDED CONCLUSION" '
                    f"--premise {shlex.quote(str(latest_evidence['object_id']))} "
                    '--warrant "WHY THIS EVIDENCE JUSTIFIES THE CONCLUSION"'
                ),
            )
        )
    elif not counts["review"]:
        if latest_inference is None:  # pragma: no cover - guarded by the branch above
            raise ResearchGitError("research guide cannot resolve the active inference")
        next_steps.append(
            _step(
                selected_language,
                code="independent_review",
                title_en="Ask an independent person or service to review",
                title_zh="请独立的人或服务进行复核",
                why_en="The producer must not be the only judge of whether its evidence passed.",
                why_zh="证据生产者不能同时成为判断证据是否通过的唯一裁判。",
                command=(
                    'xscientist research review "REVIEW SUMMARY" '
                    f"--evaluates {shlex.quote(str(latest_inference['object_id']))} "
                    "--verifier human:REVIEWER --decision hold"
                ),
            )
        )
    elif not counts["claim"]:
        if latest_inference is None:  # pragma: no cover - guarded by the branch above
            raise ResearchGitError(
                "research guide cannot resolve the reviewed inference"
            )
        next_steps.append(
            _step(
                selected_language,
                code="state_claim",
                title_en="State only the claim supported by the evidence",
                title_zh="只陈述证据能够支持的结论",
                why_en="Keep the scope no broader than the tested data, metric, and conditions.",
                why_zh="结论范围不能超过实际检验的数据、指标和条件。",
                command=(
                    'xscientist research claim "BOUNDED CLAIM" '
                    f"--evidence {shlex.quote(str(latest_inference['object_id']))} "
                    '--scope "TESTED CONDITIONS"'
                ),
            )
        )
    elif contested_claim is not None:
        resolution_plan = _latest_after(
            objects, "research_plan", _created_at(contested_claim)
        )
        resolution_attempt = (
            _latest_after(objects, "experiment_attempt", _created_at(resolution_plan))
            if resolution_plan is not None
            else None
        )
        resolution_evidence = (
            _latest_after(objects, "evidence", _created_at(resolution_attempt))
            if resolution_attempt is not None
            else None
        )
        resolution_inference = (
            _latest_after(objects, "inference", _created_at(resolution_evidence))
            if resolution_evidence is not None
            else None
        )
        resolution_review = (
            _latest_after(objects, "review", _created_at(resolution_inference))
            if resolution_inference is not None
            else None
        )
        if resolution_plan is None:
            if latest_hypothesis is None:  # pragma: no cover - a claim needs lineage
                raise ResearchGitError("research guide cannot resolve a hypothesis")
            step = _step(
                selected_language,
                code="resolve_contested_claim",
                title_en=(
                    "Resolve or narrow the contested claim: plan a boundary test"
                ),
                title_zh="解决争议或缩小结论范围：规划一次边界检验",
                why_en=(
                    "A held or refuted claim needs a discriminating condition "
                    "before it can be narrowed or replaced."
                ),
                why_zh="被暂缓或反驳的结论，需要先增加有区分度的条件检验。",
                command=(
                    "xscientist research plan "
                    f"{shlex.quote(str(latest_hypothesis['object_id']))} "
                    '"TEST THE CONTESTED BOUNDARY" '
                    '--test "WHAT RESULT WOULD RESOLVE THE CONFLICT"'
                ),
            )
        elif resolution_attempt is None:
            step = _step(
                selected_language,
                code="run_resolution_experiment",
                title_en="Run the planned boundary experiment",
                title_zh="运行已规划的边界实验",
                why_en="The new plan advances only after its result is recorded.",
                why_zh="只有记录新计划对应的实验结果，争议解决流程才会继续。",
                command=(
                    'xscientist research experiment "RESOLUTION EXPERIMENT" '
                    "--status completed "
                    f"--plan {shlex.quote(str(resolution_plan['object_id']))} "
                    "--metric RESOLUTION_METRIC=0 --seed 1"
                ),
            )
        elif resolution_evidence is None:
            step = _step(
                selected_language,
                code="bind_resolution_evidence",
                title_en="Bind the boundary result as evidence",
                title_zh="把边界实验结果绑定为证据",
                why_en="Record what the new attempt supports or refutes.",
                why_zh="明确新实验支持或反驳了什么。",
                command=(
                    'xscientist research evidence "RESOLUTION RESULT" '
                    f"--attempt {shlex.quote(str(resolution_attempt['object_id']))}"
                ),
            )
        elif resolution_inference is None:
            step = _step(
                selected_language,
                code="infer_resolution",
                title_en="State the bounded resolution",
                title_zh="写下有边界的争议解决结论",
                why_en="Explain why the new evidence changes the claim boundary.",
                why_zh="说明新证据为何改变了原结论的适用边界。",
                command=(
                    'xscientist research infer "RESOLUTION CONCLUSION" '
                    f"--premise {shlex.quote(str(resolution_evidence['object_id']))} "
                    '--warrant "WHY THE NEW EVIDENCE RESOLVES THE CONFLICT"'
                ),
            )
        elif resolution_review is None:
            step = _step(
                selected_language,
                code="review_resolution",
                title_en="Independently review the resolution",
                title_zh="独立复核争议解决结果",
                why_en="The revised boundary still needs an independent gate.",
                why_zh="修改后的结论边界仍需要独立门禁。",
                command=(
                    'xscientist research review "RESOLUTION REVIEW" '
                    f"--evaluates {shlex.quote(str(resolution_inference['object_id']))} "
                    "--verifier human:REVIEWER "
                    "--decision hold"
                ),
            )
        else:
            step = _step(
                selected_language,
                code="replace_contested_claim",
                title_en="Record the narrowed replacement claim",
                title_zh="记录缩小范围后的替代主张",
                why_en="Preserve the old contested claim and add the evidence-bounded replacement.",
                why_zh="保留原争议主张，同时新增受证据约束的替代主张。",
                command=(
                    'xscientist research claim "NARROWED CLAIM" '
                    f"--evidence {shlex.quote(str(resolution_inference['object_id']))} "
                    '--scope "RESOLVED CONDITIONS"'
                ),
            )
        next_steps.append(step)
    elif not counts["reproduction"]:
        if latest_claim is None:  # pragma: no cover - guarded by the branch above
            raise ResearchGitError("research guide cannot resolve the active claim")
        next_steps.append(
            _step(
                selected_language,
                code="reproduce",
                title_en="Re-run and preserve a reproduction receipt",
                title_zh="重新运行并保存复现回执",
                why_en="A saved claim is traceable; a successful independent rerun makes it verifiable.",
                why_zh="已保存的结论只是可追踪；独立重跑成功后才具备更强的可验证性。",
                command=(
                    "xscientist research reproduce HEAD --execute --record "
                    f"--reproduces {shlex.quote(str(latest_claim['object_id']))} "
                    "--verifier human:REPRODUCER"
                ),
            )
        )
    else:
        next_steps.append(
            _step(
                selected_language,
                code="inspect_dag",
                title_en="Inspect the complete evidence DAG",
                title_zh="检查完整证据 DAG",
                why_en="The graph shows which conclusions are merely recorded, replayable, verified, or contested.",
                why_zh="图中会区分哪些结论只是被记录、可重放、已验证或仍存在争议。",
                command="xscientist research dag --output research-dag",
            )
        )

    challenge_relations = [
        relation
        for _source, relation in relations
        if relation.get("type") in {"refutes", "contradicts"}
    ]
    if counts["evidence"] and not challenge_relations:
        warnings.append(
            {
                "code": "no_recorded_challenge_evidence",
                "message": _text(
                    selected_language,
                    "No refuting or contradictory evidence is recorded yet; actively test an alternative explanation.",
                    "尚未记录反驳或矛盾证据；请主动检验替代解释，而不只寻找支持结果。",
                ),
            }
        )

    from .research_strategy import review_research_program

    program_review = review_research_program(repository.path, record=False)
    program_report = program_review.get("report") or {}
    if counts["experiment_attempt"] and program_report.get("gaps"):
        next_steps.append(
            _step(
                selected_language,
                code="strengthen_research_program",
                title_en="Review the open scientific-strategy gaps",
                title_zh="检查尚未解决的科研策略缺口",
                why_en=(
                    f"The deterministic review found {len(program_report['gaps'])} "
                    "open gap(s), such as rival explanations, mechanisms, evidence "
                    "quality, or transfer boundaries."
                ),
                why_zh=(
                    f"确定性审查发现 {len(program_report['gaps'])} 个尚未解决的缺口，"
                    "可能涉及竞争解释、机制、证据质量或迁移边界。"
                ),
                command="xscientist research program review",
            ),
        )

    for step in next_steps:
        step["command"] = _command_for_repo(
            step["command"], command_repo if command_repo is not None else repo
        )

    completed_stages = sum(
        bool(counts[kind])
        for kind in (
            "hypothesis",
            "research_plan",
            "experiment_attempt",
            "evidence",
            "inference",
            "review",
            "claim",
            "reproduction",
        )
    )
    primary_action = _primary_action(selected_language, next_steps, completed_stages)
    return {
        "schema_version": GUIDE_SCHEMA,
        "language": selected_language,
        "repository": repository.path.as_posix(),
        "branch": repository.status().get("branch"),
        "progress": {
            "completed_stages": completed_stages,
            "total_stages": 8,
            "percent": round(completed_stages / 8 * 100),
        },
        "primary_action": primary_action,
        "counts": dict(sorted(counts.items())),
        "next_steps": next_steps,
        "warnings": warnings,
        "program_review": {
            "gap_count": len(program_report.get("gaps") or []),
            "gaps": program_report.get("gaps") or [],
            "recommendations": program_report.get("recommended_actions") or [],
        },
    }


def public_exploration_payload(
    payload: dict[str, Any], *, workspace: str | Path | None = None
) -> dict[str, Any]:
    """Return a portable exploration response safe for JSON/telemetry output."""

    safe = deepcopy(payload)
    raw_workspace = workspace or safe.get("repository") or "."
    root = Path(raw_workspace).expanduser().resolve()

    def relative_value(value: Any) -> str:
        try:
            candidate = Path(str(value)).expanduser().resolve()
            return candidate.relative_to(root).as_posix() or "."
        except (OSError, TypeError, ValueError):
            return "[REDACTED_PATH]"

    safe["repository"] = "."
    continue_action = workspace_action_contract("xscientist explore .")
    status_action = workspace_action_contract("xscientist status .")
    if continue_action is None or status_action is None:  # pragma: no cover
        raise ResearchGitError("portable workspace action contract is unavailable")
    safe["continue_action"] = continue_action
    safe["status_action"] = status_action
    safe["continue_command"] = continue_action["command_template"]
    safe["status_command"] = status_action["command_template"]
    safe["workspace_context"] = workspace_action_context()
    checkpoint = safe.get("checkpoint")
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_path"):
        checkpoint["checkpoint_path"] = relative_value(checkpoint["checkpoint_path"])
    guide = safe.get("guide")
    if isinstance(guide, dict):
        guide["repository"] = "."
        # Rebuild commands against the portable working-directory reference so
        # copy/paste never embeds the caller's home or temporary directory.
        try:
            refreshed = build_research_guide(
                root,
                language=str(safe.get("language") or guide.get("language") or "auto"),
                command_repo=".",
            )
            safe["guide"] = public_research_guide_payload(refreshed)
        except (OSError, ResearchGitError, ValueError):
            for step in guide.get("next_steps") or []:
                if isinstance(step, dict):
                    step["command"] = _command_for_repo(step.get("command", ""), ".")
            if isinstance(guide.get("primary_action"), dict):
                guide["primary_action"]["command"] = _command_for_repo(
                    guide["primary_action"].get("command", ""), "."
                )
            safe["guide"] = public_research_guide_payload(guide)
    safe["privacy"] = {
        "host_paths_disclosed": False,
        "matched_values_disclosed": False,
        "workspace_reference": ".",
    }
    return redact_sensitive_payload(safe)


def public_guided_research_start_payload(
    payload: dict[str, Any], *, workspace: str | Path | None = None
) -> dict[str, Any]:
    """Return a portable guided-start response without absolute host paths."""

    safe = deepcopy(payload)
    raw_workspace = workspace or safe.get("repository") or "."
    root = Path(raw_workspace).expanduser().resolve()

    def relative_value(value: Any) -> str:
        try:
            candidate = Path(str(value)).expanduser().resolve()
            return candidate.relative_to(root).as_posix() or "."
        except (OSError, TypeError, ValueError):
            return "[REDACTED_PATH]"

    safe["repository"] = "."
    checkpoint = safe.get("checkpoint")
    if isinstance(checkpoint, dict) and checkpoint.get("checkpoint_path"):
        checkpoint["checkpoint_path"] = relative_value(checkpoint["checkpoint_path"])
    guide = safe.get("guide")
    if isinstance(guide, dict):
        safe["guide"] = public_research_guide_payload(guide)
    open_action = workspace_action_contract(safe.get("open_command"))
    safe["open_action"] = open_action
    if open_action is not None:
        safe["open_command"] = open_action["command_template"]
    safe["workspace_context"] = workspace_action_context()
    safe["privacy"] = {
        "host_path_disclosed": False,
        "host_paths_disclosed": False,
        "matched_values_disclosed": False,
        "workspace_reference": ".",
    }
    return redact_sensitive_payload(safe)


def explore_research_idea(
    path: str | Path,
    *,
    idea: str | None = None,
    expectation: str | None = None,
    disconfirming_result: str | None = None,
    first_test: str | None = None,
    success_rule: str | None = None,
    name: str | None = None,
    actor: str = "human:researcher",
    language: str = "auto",
    git_user_name: str | None = None,
    git_user_email: str | None = None,
) -> dict[str, Any]:
    """Turn a user's idea into an honest, provider-free research scaffold.

    The function records only text supplied by the user. It never invents a
    hypothesis, evidence, or conclusion to make an incomplete study look done.
    Re-running it on the same repository can add the next missing framing step.
    """

    selected_language = _language(language)
    root = Path(path).expanduser().resolve()
    existing = (root / "research.yaml").is_file()
    actor_id = _human_actor(actor)
    idea_text = _optional(idea)
    expectation_text = _optional(expectation)
    disconfirming_text = _optional(disconfirming_result)
    first_test_text = _optional(first_test)
    success_rule_text = _optional(success_rule)

    if bool(expectation_text) != bool(disconfirming_text):
        raise ResearchGitError(
            "an expected result and a result that would disprove it must be "
            "provided together"
        )
    if first_test_text and not expectation_text and not existing:
        raise ResearchGitError(
            "a first test requires an expected result and a result that would "
            "disprove it"
        )
    if success_rule_text and not first_test_text:
        raise ResearchGitError("a success rule requires a first test")

    created_paths: list[str] = []
    checkpoint = None
    if existing:
        current = inspect_idea_research(root)
        if idea_text and idea_text != current["idea"]:
            raise ResearchGitError(
                "this workspace already records a different idea; use a new "
                "directory so the two research histories stay separate"
            )
        idea_text = str(current["idea"])
        repository = ResearchRepository(root)
        status = repository.status()
        if (expectation_text or first_test_text) and status["research_stage"]["paths"]:
            raise ResearchGitError(
                "the research stage already contains pending work; commit or "
                "unstage it before continuing this guided exploration"
            )
        if (expectation_text or first_test_text) and status["staged_paths"]:
            raise ResearchGitError(
                "the Git index already contains staged work; commit or unstage it "
                "before continuing this guided exploration"
            )
        question_id = str(current["question_id"])
        goal = _latest(repository.objects(), "research_goal")
        goal_id = str(goal["object_id"]) if goal is not None else None
        hypothesis_id = (
            str(current["hypothesis_id"]) if current["hypothesis_id"] else None
        )
        plan_id = str(current["plan_id"]) if current["plan_id"] else None
        if expectation_text and hypothesis_id:
            raise ResearchGitError(
                "this workspace already has a falsifiable expectation; use "
                "`xscientist research hypothesis` to add a competing one"
            )
        if first_test_text and plan_id:
            raise ResearchGitError(
                "this workspace already has a research plan; use `xscientist "
                "research plan` to add another test"
            )
    else:
        if not idea_text:
            raise ResearchGitError("idea cannot be empty")
        if root.exists() and any(root.iterdir()):
            raise ResearchGitError(
                "research destination must be absent or empty; choose a new directory"
            )
        init_repository(
            root,
            name=name,
            question=f"# Research question\n\n{idea_text}\n",
            actor=actor_id,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
            commit=False,
        )
        repository = ResearchRepository(root)
        question_object = repository.record(
            "question",
            {"question": idea_text, "source": "user_idea"},
            actor={"actor_id": actor_id, "authority": "human"},
        )
        question_id = question_object.object_id
        goal_core = {
            "question": idea_text,
            "objective": (
                "Turn the user's idea into a falsifiable study without treating "
                "a plan as evidence."
            ),
            "success_condition": (
                "Reach an evidence-bounded conclusion that can be independently "
                "reviewed."
            ),
            "authority_policy": (
                "User-supplied assumptions stay explicit and independent "
                "evaluation is required for verification."
            ),
        }
        goal_object = repository.record(
            "research_goal",
            {**goal_core, "goal_hash": canonical_content_hash(goal_core)},
            state="locked",
            relations=[{"type": "depends_on", "target": question_id}],
            actor={"actor_id": actor_id, "authority": "human"},
        )
        goal_id = goal_object.object_id
        hypothesis_id = None
        plan_id = None

    if expectation_text:
        relations = [{"type": "depends_on", "target": question_id}]
        if goal_id:
            relations.append({"type": "depends_on", "target": goal_id, "role": "goal"})
        hypothesis_object = repository.record(
            "hypothesis",
            {
                "statement": expectation_text,
                "falsifier": disconfirming_text,
                "source": "user_answer",
            },
            relations=relations,
            actor={"actor_id": actor_id, "authority": "human"},
        )
        hypothesis_id = hypothesis_object.object_id
        if hypothesis_object.created:
            created_paths.append(hypothesis_object.path.relative_to(root).as_posix())

    if first_test_text:
        if not hypothesis_id:
            raise ResearchGitError(
                "a first test requires a recorded falsifiable expectation"
            )
        plan_payload: dict[str, Any] = {
            "summary": first_test_text,
            "study_phase": "exploratory",
            "hypothesis_id": hypothesis_id,
            "discriminating_tests": [first_test_text],
            "source": "user_answer",
        }
        if success_rule_text:
            plan_payload["success_rule"] = success_rule_text
        plan_object = repository.record(
            "research_plan",
            plan_payload,
            relations=[{"type": "depends_on", "target": hypothesis_id}],
            actor={"actor_id": actor_id, "authority": "human"},
        )
        plan_id = plan_object.object_id
        if plan_object.created:
            created_paths.append(plan_object.path.relative_to(root).as_posix())

    if not existing:
        # The destination was verified empty, so every eligible path belongs to
        # this initialization and can safely enter its first exact checkpoint.
        created_paths = list(repository.status()["eligible_changes"])
    if created_paths:
        repository.stage(created_paths)
        checkpoint = repository.commit(
            stage="plan" if plan_id else "idea-framing",
            subject=(
                "record first user-supplied research plan"
                if plan_id
                else "record user-supplied research idea"
            ),
            status="draft",
            actor=actor_id,
            staged_only=True,
        )

    current = inspect_idea_research(root)
    missing: list[str] = []
    if not current["hypothesis_id"]:
        missing.extend(["expected_observation", "disconfirming_result"])
    elif not current["plan_id"]:
        missing.append("first_fair_test")
    framing_status = (
        "planned"
        if current["plan_id"]
        else ("falsifiable" if current["hypothesis_id"] else "idea_saved")
    )
    continue_command = "xscientist explore " + shlex.quote(str(root))
    if selected_language == "zh":
        continue_command += " --lang zh"
    status_command = "xscientist status " + shlex.quote(str(root))
    if selected_language == "zh":
        status_command += " --lang zh"
    guide = build_research_guide(root, language=selected_language)
    return {
        "schema_version": "xscientist.idea-exploration.v1",
        "ok": True,
        "repository": root.as_posix(),
        "language": selected_language,
        "idea": current["idea"],
        "framing": {
            "status": framing_status,
            "missing": missing,
            "question_id": current["question_id"],
            "hypothesis_id": current["hypothesis_id"],
            "plan_id": current["plan_id"],
        },
        "checkpoint": checkpoint.to_dict() if checkpoint is not None else None,
        "safety": {
            "api_key_required": False,
            "provider_used": False,
            "external_network_used": False,
            "generated_code_executed": False,
            "evidence_generated": False,
            "conclusion_generated": False,
        },
        "guide": guide,
        "continue_command": continue_command,
        "status_command": status_command,
    }


def start_guided_research(
    path: str | Path,
    *,
    question: str,
    hypothesis: str,
    falsifier: str,
    name: str | None = None,
    actor: str = "human:researcher",
    language: str = "auto",
    git_user_name: str | None = None,
    git_user_email: str | None = None,
) -> dict[str, Any]:
    """Create one repository and its first falsifiable scientific lineage."""

    question_text = _required(question, label="research question")
    hypothesis_text = _required(hypothesis, label="hypothesis")
    falsifier_text = _required(falsifier, label="falsifier")
    prospective = {
        "question": question_text,
        "hypothesis": hypothesis_text,
        "falsifier": falsifier_text,
        "name": name,
        "actor": actor,
    }
    if redact_sensitive_payload(prospective) != prospective:
        raise ResearchGitError(
            "guided research inputs must not contain credentials, host-local "
            "paths, email addresses, or other private literals"
        )
    actor_id = _human_actor(actor)
    root = Path(path).expanduser().resolve()
    init_repository(
        root,
        name=name,
        question=f"# Research question\n\n{question_text}\n",
        actor=actor_id,
        git_user_name=git_user_name,
        git_user_email=git_user_email,
        commit=False,
    )
    repository = ResearchRepository(root)
    question_object = repository.record(
        "question",
        {"question": question_text},
        actor={"actor_id": actor_id, "authority": "human"},
    )
    goal_core = {
        "question": question_text,
        "objective": "Test the falsifiable hypothesis while retaining negative evidence.",
        "success_condition": "Reach an independently reviewable evidence-bound conclusion.",
        "authority_policy": "Independent evaluation is required for verification.",
    }
    goal_object = repository.record(
        "research_goal",
        {**goal_core, "goal_hash": canonical_content_hash(goal_core)},
        state="locked",
        relations=[{"type": "depends_on", "target": question_object.object_id}],
        actor={"actor_id": actor_id, "authority": "human"},
    )
    hypothesis_object = repository.record(
        "hypothesis",
        {"statement": hypothesis_text, "falsifier": falsifier_text},
        relations=[
            {"type": "depends_on", "target": question_object.object_id},
            {"type": "depends_on", "target": goal_object.object_id, "role": "goal"},
        ],
        actor={"actor_id": actor_id, "authority": "human"},
    )
    managed_paths = {
        ".gitignore",
        "research.yaml",
        "question.md",
        ".xscientist/README.md",
        question_object.path.relative_to(root).as_posix(),
        goal_object.path.relative_to(root).as_posix(),
        hypothesis_object.path.relative_to(root).as_posix(),
    }
    eligible_paths = set(repository.status()["eligible_changes"])
    selected_paths = sorted(managed_paths & eligible_paths)
    if not selected_paths:  # pragma: no cover - records above are newly created
        raise ResearchGitError("guided research start produced no managed changes")
    repository.stage(selected_paths)
    checkpoint = repository.commit(
        stage="guided-start",
        subject="start falsifiable research question",
        status="draft",
        staged_only=True,
    )
    return {
        "schema_version": "xscientist.guided-research-start.v1",
        "repository": root.as_posix(),
        "question_id": question_object.object_id,
        "goal_id": goal_object.object_id,
        "hypothesis_id": hypothesis_object.object_id,
        "checkpoint": checkpoint.to_dict(),
        "guide": build_research_guide(root, language=language),
        "open_command": "xscientist research guide --repo " + shlex.quote(str(root)),
    }


__all__ = [
    "GUIDE_SCHEMA",
    "WORKSPACE_ACTION_CONTEXT_SCHEMA",
    "WORKSPACE_ACTION_SCHEMA",
    "WORKSPACE_PLACEHOLDER",
    "build_research_guide",
    "explore_research_idea",
    "inspect_idea_research",
    "public_exploration_payload",
    "public_guided_research_start_payload",
    "public_research_guide_payload",
    "public_workspace_action",
    "start_guided_research",
    "workspace_action_context",
    "workspace_action_contract",
]
