"""Beginner-friendly research setup and next-step guidance."""

from __future__ import annotations

import locale
import shlex
from collections import Counter
from pathlib import Path
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash

from .research_git import ResearchGitError, init_repository
from .research_vcs import ResearchRepository

GUIDE_SCHEMA = "xscientist.research-guide.v1"


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
    replacements = {
        "YOUR HYPOTHESIS": "你的假设",
        "WHAT RESULT WOULD DISPROVE IT": "什么结果会推翻它",
        "WHAT TO TEST": "要检验什么",
        "WHAT RESULT SEPARATES THE EXPLANATIONS": "什么结果能区分这些解释",
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
    rendered = command
    for source, target in replacements.items():
        rendered = rendered.replace(source, target)
    return rendered


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
        else:
            next_steps.append(
                _step(
                    selected_language,
                    code="choose_study_mode",
                    title_en="Option A — plan exploratory work",
                    title_zh="选项 A — 规划探索性研究",
                    why_en="Use this while comparing explanations or discovering which test is most informative.",
                    why_zh="当你仍在比较不同解释，或寻找最有信息量的检验时，使用这条路径。",
                    command=(
                        'xscientist research plan @latest:hypothesis "WHAT TO TEST" '
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
                        "xscientist research preregister @latest:hypothesis "
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
                    "--plan @latest:research_plan --metric NAME=VALUE --seed 1"
                ),
            )
        )
    elif not (counts["evidence"] or counts["passage_evidence"]):
        next_steps.append(
            _step(
                selected_language,
                code="bind_evidence",
                title_en="Connect the result to the hypothesis",
                title_zh="把结果连接到假设",
                why_en="A result is not scientific evidence until its source attempt and direction are explicit.",
                why_zh="只有明确结果来自哪个实验、支持还是反驳什么，它才成为科学证据。",
                command=(
                    'xscientist research evidence "WHAT THE RESULT SHOWS" '
                    "--attempt @latest:experiment_attempt --supports @latest:hypothesis"
                ),
            )
        )
    elif not counts["inference"]:
        premise_kind = "evidence" if counts["evidence"] else "passage_evidence"
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
                    f"--premise @latest:{premise_kind} "
                    '--warrant "WHY THIS EVIDENCE JUSTIFIES THE CONCLUSION"'
                ),
            )
        )
    elif not counts["review"]:
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
                    "--evaluates @latest:inference --verifier human:REVIEWER --decision hold"
                ),
            )
        )
    elif not counts["claim"]:
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
                    '--evidence @latest:inference --scope "TESTED CONDITIONS"'
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
                    "xscientist research plan @latest:hypothesis "
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
                    "--status completed --plan @latest:research_plan "
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
                    "--attempt @latest:experiment_attempt --supports @latest:hypothesis"
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
                    "--premise @latest:evidence "
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
                    "--evaluates @latest:inference --verifier human:REVIEWER "
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
                    '--evidence @latest:inference --scope "RESOLVED CONDITIONS"'
                ),
            )
        next_steps.append(step)
    elif not counts["reproduction"]:
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
                    "--reproduces @latest:claim --verifier human:REPRODUCER"
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
        "counts": dict(sorted(counts.items())),
        "next_steps": next_steps,
        "warnings": warnings,
        "program_review": {
            "gap_count": len(program_report.get("gaps") or []),
            "gaps": program_report.get("gaps") or [],
            "recommendations": program_report.get("recommended_actions") or [],
        },
    }


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
    checkpoint = repository.commit(
        stage="guided-start",
        subject="start falsifiable research question",
        status="draft",
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
    "build_research_guide",
    "explore_research_idea",
    "inspect_idea_research",
    "start_guided_research",
]
