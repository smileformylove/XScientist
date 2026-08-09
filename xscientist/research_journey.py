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
        "command": command,
    }


def build_research_guide(
    repo: str | Path,
    *,
    language: str = "auto",
) -> dict[str, Any]:
    """Explain current progress and return safe copy/paste next actions."""

    selected_language = _language(language)
    repository = ResearchRepository(repo)
    objects = repository.objects()
    counts = Counter(str(item["kind"]) for item in objects)
    relations = [
        (str(item["object_id"]), relation)
        for item in objects
        for relation in item.get("relations") or []
    ]
    next_steps: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not counts["hypothesis"]:
        next_steps.append(
            _step(
                selected_language,
                code="record_hypothesis",
                title_en="Write one falsifiable hypothesis",
                title_zh="写下一个可证伪假设",
                why_en="A scientific question becomes testable only after you state what would prove it wrong.",
                why_zh="只有同时说明什么结果会推翻它，研究问题才真正可检验。",
                command=(
                    'xscientist research hypothesis "YOUR HYPOTHESIS" '
                    '--falsifier "WHAT RESULT WOULD DISPROVE IT"'
                ),
            )
        )
    elif not counts["research_plan"]:
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
    raw_actor = _required(actor, label="actor")
    if ":" not in raw_actor:
        actor_id = "human:" + raw_actor
    elif raw_actor.startswith("human:") and raw_actor != "human:":
        actor_id = raw_actor
    else:
        raise ResearchGitError("guided research actor must be a human identity")
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
    "start_guided_research",
]
