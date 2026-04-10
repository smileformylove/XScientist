#!/usr/bin/env python3
"""
AI Scientist 自动改进系统
基于审查结果自动改进论文质量
"""
import json
import os
import os.path as osp
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai_scientist.utils.deferred_imports import load_module_attr
from ai_scientist.utils.pipeline_helpers import (
    compile_latex as shared_compile_latex,
    find_best_pdf_path,
)
from ai_scientist.utils.self_review_optimizer import (
    apply_issue_driven_rewrite,
    assess_self_review_gate,
    build_issue_ledger,
    evaluate_issue_progress,
    save_self_review_artifacts,
)
from ai_scientist.utils.experiment_todo_progress import (
    bootstrap_todo_tasks_from_round_gate,
    build_todo_progress_payload,
    evaluate_todo_progress_snapshot,
    load_experiment_todo_payload,
    save_todo_progress_artifacts,
)


def create_client(*args, **kwargs):
    return load_module_attr("ai_scientist.llm", "create_client")(*args, **kwargs)


def get_response_from_llm(*args, **kwargs):
    return load_module_attr("ai_scientist.llm", "get_response_from_llm")(
        *args, **kwargs
    )


def load_paper(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_llm_review", "load_paper")(
        *args, **kwargs
    )


def perform_review(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_llm_review", "perform_review")(
        *args, **kwargs
    )


def perform_imgs_cap_ref_review(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.perform_vlm_review",
        "perform_imgs_cap_ref_review",
    )(*args, **kwargs)


class AutoImprovementEngine:
    """自动改进引擎 - 基于审查结果改进论文"""

    def __init__(
        self,
        model: str = "glm-4-plus",
        temperature: float = 0.7,
        max_improvement_rounds: int = 3,
    ):
        """
        初始化自动改进引擎

        Args:
            model: 使用的LLM模型
            temperature: 生成温度
            max_improvement_rounds: 最大改进轮数
        """
        self.model = model
        self.temperature = temperature
        self.max_improvement_rounds = max_improvement_rounds
        self.client, self.client_model = create_client(model)

        # 改进历史
        self.improvement_history = []

    def analyze_review_results(
        self,
        text_review: Dict,
        img_review: Dict,
    ) -> Dict:
        """
        分析审查结果，提取关键改进点

        Args:
            text_review: 文本审查结果
            img_review: 图像审查结果

        Returns:
            改进点分析结果
        """
        analysis = {
            "critical_issues": [],  # 严重问题
            "major_weaknesses": [],  # 主要弱点
            "minor_suggestions": [],  # 次要建议
            "improvement_priority": [],  # 改进优先级
        }

        # 分析文本审查
        if text_review:
            scores = text_review.get("review", {}).get("scores", {})

            # 基于评分识别严重问题
            for category, score in scores.items():
                if category == "Overall":
                    continue
                if isinstance(score, (int, float)) and score <= 2:
                    issue = {
                        "category": category,
                        "score": score,
                        "description": text_review.get("review", {}).get("Weaknesses", ""),
                        "priority": "high",
                    }
                    analysis["critical_issues"].append(issue)
                elif isinstance(score, (int, float)) and score <= 3:
                    weakness = {
                        "category": category,
                        "score": score,
                        "description": text_review.get("review", {}).get("Weaknesses", ""),
                        "priority": "medium",
                    }
                    analysis["major_weaknesses"].append(weakness)

            # 提取改进建议
            questions = text_review.get("review", {}).get("Questions", [])
            if questions:
                analysis["minor_suggestions"].extend(
                    [{"type": "question", "content": q} for q in questions]
                )

            limitations = text_review.get("review", {}).get("Limitations", [])
            if limitations:
                analysis["minor_suggestions"].extend(
                    [{"type": "limitation", "content": l} for l in limitations]
                )

        # 分析图像审查
        if img_review:
            for figure_review in img_review.get("figure_reviews", []):
                if figure_review.get("overall_quality") == "poor":
                    analysis["critical_issues"].append({
                        "category": "Figure Quality",
                        "figure": figure_review.get("figure_id"),
                        "issue": figure_review.get("description"),
                        "priority": "high",
                    })

        # 按优先级排序
        all_issues = (
            analysis["critical_issues"] +
            analysis["major_weaknesses"] +
            analysis["minor_suggestions"]
        )
        analysis["improvement_priority"] = sorted(
            all_issues,
            key=lambda x: (
                0 if x.get("priority") == "high" else
                1 if x.get("priority") == "medium" else 2
            )
        )

        return analysis

    def generate_improvement_strategy(
        self,
        paper_path: str,
        review_analysis: Dict,
    ) -> Dict:
        """
        生成改进策略

        Args:
            paper_path: 论文路径
            review_analysis: 审查分析结果

        Returns:
            改进策略
        """
        # 选择本轮要处理的关键问题（最多3个）
        top_issues = review_analysis["improvement_priority"][:3]

        strategy = {
            "target_issues": top_issues,
            "improvement_actions": [],
            "expected_outcomes": [],
        }

        for issue in top_issues:
            if issue.get("priority") == "high":
                action = {
                    "issue": issue,
                    "action_type": "major_revision",
                    "description": f"重点改进 {issue.get('category')}",
                    "prompt_template": self._get_improvement_prompt(issue, "major"),
                }
            else:
                action = {
                    "issue": issue,
                    "action_type": "minor_revision",
                    "description": f"优化 {issue.get('category')}",
                    "prompt_template": self._get_improvement_prompt(issue, "minor"),
                }

            strategy["improvement_actions"].append(action)

        return strategy

    def _get_improvement_prompt(self, issue: Dict, revision_type: str) -> str:
        """生成改进提示词"""
        category = issue.get("category", "General")
        description = issue.get("description", "")

        if revision_type == "major":
            return f"""
请重点改进论文的 {category} 方面。

当前问题:
{description}

改进要求:
1. 深入分析问题的根源
2. 提供具体的改进方案
3. 修改相关内容，确保问题得到解决
4. 保持论文其他部分的连贯性

请提供修改后的完整LaTeX代码。
"""
        else:
            return f"""
请优化论文的 {category} 方面。

当前问题:
{description}

改进要求:
1. 对现有内容进行微调
2. 保持原有的结构和逻辑
3. 提升清晰度和可读性

请提供修改后的相关LaTeX片段。
"""

    def apply_improvements(
        self,
        paper_dir: str,
        strategy: Dict,
        latex_content: str,
    ) -> Tuple[str, Dict]:
        """
        应用改进策略

        Args:
            paper_dir: 论文目录
            strategy: 改进策略
            latex_content: 当前LaTeX内容

        Returns:
            (改进后的LaTeX内容, 改进记录)
        """
        improved_latex = latex_content
        improvement_log = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "sections_modified": [],
        }

        for action in strategy["improvement_actions"]:
            print(f"\n🔧 执行改进: {action['description']}")

            # 构建改进提示
            improvement_prompt = f"""
你是专业的学术编辑。请根据以下改进要求修改LaTeX论文。

{action['prompt_template']}

当前论文内容（LaTeX格式）:
```latex
{improved_latex[:15000]}
```

请直接返回改进后的完整LaTeX代码，用```latex包裹。
"""

            try:
                # 调用LLM进行改进
                response, _ = get_response_from_llm(
                    prompt=improvement_prompt,
                    client=self.client,
                    model=self.client_model,
                    system_message="你是专业的学术编辑，擅长改进科研论文。",
                    temperature=self.temperature,
                )

                # 提取LaTeX代码
                latex_match = re.search(
                    r'```latex\s*(.*?)\s*```',
                    response,
                    re.DOTALL
                )
                if not latex_match:
                    latex_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)

                if latex_match:
                    improved_latex = latex_match.group(1)
                    print(f"✅ 改进完成: {action['description']}")

                    # 记录改进
                    improvement_log["actions_taken"].append({
                        "action": action["description"],
                        "status": "success",
                    })

                    # 识别修改的section
                    sections = re.findall(
                        r'\\section\{([^}]+)\}',
                        improved_latex
                    )
                    improvement_log["sections_modified"].extend(sections)

                else:
                    print(f"⚠️  改进失败: {action['description']}")
                    improvement_log["actions_taken"].append({
                        "action": action["description"],
                        "status": "failed",
                        "reason": "无法提取LaTeX代码",
                    })

            except Exception as e:
                print(f"❌ 改进出错: {e}")
                improvement_log["actions_taken"].append({
                    "action": action["description"],
                    "status": "error",
                    "error": str(e),
                })

        return improved_latex, improvement_log

    def evaluate_improvement(
        self,
        old_review: Dict,
        new_review: Dict,
    ) -> Dict:
        """
        评估改进效果

        Args:
            old_review: 改进前的审查结果
            new_review: 改进后的审查结果

        Returns:
            改进效果评估
        """
        old_scores = old_review.get("review", {}).get("scores", {})
        new_scores = new_review.get("review", {}).get("scores", {})

        improvements = {}
        overall_improvement = 0

        for category in old_scores:
            if category in new_scores:
                old_score = old_scores[category]
                new_score = new_scores[category]

                if isinstance(old_score, (int, float)) and isinstance(new_score, (int, float)):
                    improvement = new_score - old_score
                    improvements[category] = {
                        "old": old_score,
                        "new": new_score,
                        "improvement": improvement,
                        "percentage": (improvement / old_score * 100) if old_score > 0 else 0,
                    }

                    if category == "Overall":
                        overall_improvement = improvement

        return {
            "score_improvements": improvements,
            "overall_improvement": overall_improvement,
            "significant_improvement": overall_improvement >= 1.0,
            "categories_improved": len([
                v for v in improvements.values()
                if v["improvement"] > 0
            ]),
            "categories_declined": len([
                v for v in improvements.values()
                if v["improvement"] < 0
            ]),
        }

    def should_continue_improvement(
        self,
        improvement_evaluation: Dict,
        current_round: int,
    ) -> Tuple[bool, str]:
        """
        判断是否继续改进

        Args:
            improvement_evaluation: 改进效果评估
            current_round: 当前轮数

        Returns:
            (是否继续, 原因)
        """
        # 达到最大轮数
        if current_round >= self.max_improvement_rounds:
            return False, "达到最大改进轮数"

        # 评估结果
        evaluation = improvement_evaluation

        # 显著改进，继续
        if evaluation.get("significant_improvement"):
            return True, "有显著改进，继续优化"

        # 没有改进或退化，停止
        if evaluation.get("overall_improvement", 0) <= 0:
            return False, "改进效果不明显，停止迭代"

        # 改进较小，看是否有改进空间
        if evaluation.get("overall_improvement", 0) < 0.5:
            # 如果有多个类别改进了，继续
            if evaluation.get("categories_improved", 0) >= 3:
                return True, "多方面有改进，继续优化"
            else:
                return False, "改进趋于收敛，停止迭代"

        # 默认继续
        return True, "继续改进"


def improve_paper_with_review(
    paper_dir: str,
    text_review: Dict,
    img_review: Dict,
    model: str = "glm-4-plus",
    max_rounds: int = 3,
    target_venue: Optional[str] = None,
) -> Dict:
    """
    基于审查结果改进论文（主函数）

    Args:
        paper_dir: 论文目录
        text_review: 文本审查结果
        img_review: 图像审查结果
        model: 使用的模型
        max_rounds: 最大改进轮数
        target_venue: 目标投稿 venue（用于 gate 阈值与优先级）

    Returns:
        改进结果
    """
    print("\n" + "=" * 80)
    print("🔧 自动改进系统启动 (issue-driven)")
    print("=" * 80)

    latex_dir = osp.join(paper_dir, "latex")
    latex_file = osp.join(latex_dir, "template.tex")
    if not osp.exists(latex_file):
        return {
            "status": "failed",
            "reason": "未找到LaTeX源文件",
        }

    engine = AutoImprovementEngine(
        model=model,
        max_improvement_rounds=max_rounds,
    )

    current_text_review = dict(text_review or {})
    current_img_review = dict(img_review or {})
    previous_issue_ledger = None
    improvement_rounds: List[Dict] = []
    experiment_todo_payload = load_experiment_todo_payload(paper_dir)
    experiment_todo_round_snapshots: List[Dict[str, Any]] = []
    final_experiment_todo_snapshot: Dict[str, Any] | None = None
    experiment_todo_progress_payload: Dict[str, Any] | None = None
    experiment_todo_progress_files: Dict[str, str] = {}
    improvement_artifacts_dir = osp.join(paper_dir, "improvements")
    os.makedirs(improvement_artifacts_dir, exist_ok=True)

    for round_num in range(max_rounds):
        round_index = round_num + 1
        print(f"\n{'='*80}")
        print(f"🔄 改进轮次 {round_index}/{max_rounds}")
        print(f"{'='*80}")

        round_dir = osp.join(improvement_artifacts_dir, f"round_{round_index}")
        issue_ledger = build_issue_ledger(
            text_review=current_text_review,
            img_review=current_img_review,
            max_issues=14,
            previous_ledger=previous_issue_ledger,
            target_venue=target_venue,
        )
        issue_progress_before = None
        if isinstance(previous_issue_ledger, dict):
            issue_progress_before = evaluate_issue_progress(
                previous_issues=list(previous_issue_ledger.get("issues") or []),
                current_issues=list(issue_ledger.get("issues") or []),
            )

        print(
            f"🧭 台账问题: {issue_ledger.get('issue_count', 0)} "
            f"(critical={issue_ledger.get('critical_count', 0)}, "
            f"major={issue_ledger.get('major_count', 0)}, minor={issue_ledger.get('minor_count', 0)})"
        )

        rewrite_result = apply_issue_driven_rewrite(
            paper_dir=paper_dir,
            model=model,
            ledger=issue_ledger,
            round_index=round_index,
            artifact_dir=round_dir,
            target_venue=target_venue,
            temperature=0.35,
        )
        failed_round_gate = assess_self_review_gate(
            ledger=issue_ledger,
            progress=issue_progress_before,
            rewrite_result=rewrite_result,
            round_index=round_index,
            target_venue=target_venue,
        )
        if not (experiment_todo_payload.get("tasks") or []):
            bootstrap_tasks = bootstrap_todo_tasks_from_round_gate(
                failed_round_gate,
                prefix="autoimprove",
                max_tasks=8,
            )
            if bootstrap_tasks:
                experiment_todo_payload["tasks"] = bootstrap_tasks
                experiment_todo_payload["generated_at"] = datetime.now().isoformat()
                experiment_todo_payload["bootstrap"] = True
                todo_file = Path(paper_dir) / "experiment_todo.json"
                todo_file.write_text(
                    json.dumps(experiment_todo_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                experiment_todo_payload["file"] = str(todo_file)
        failed_todo_snapshot = evaluate_todo_progress_snapshot(
            experiment_todo_payload,
            round_gate=failed_round_gate,
            issue_progress=issue_progress_before,
            round_index=round_index,
        )
        if int((failed_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0:
            experiment_todo_round_snapshots.append(failed_todo_snapshot)
        artifact_files = save_self_review_artifacts(
            review_dir=round_dir,
            ledger=issue_ledger,
            progress=issue_progress_before,
            gate=failed_round_gate,
        )
        if rewrite_result.get("status") != "success":
            print(
                f"⚠️  改稿失败，停止迭代: status={rewrite_result.get('status')}"
            )
            improvement_rounds.append(
                {
                    "round": round_index,
                    "issue_ledger": issue_ledger,
                    "issue_progress_before": issue_progress_before,
                    "rewrite_result": rewrite_result,
                    "round_gate": failed_round_gate,
                    "artifact_files": artifact_files,
                    "experiment_todo_progress": (
                        failed_todo_snapshot
                        if int((failed_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0
                        else None
                    ),
                    "stop_reason": "rewrite_failed",
                }
            )
            break

        print("\n🔍 重新审查改进后的论文...")
        pdf_file = find_best_pdf_path(paper_dir, prefer_reflections=True)
        if not pdf_file or not osp.exists(pdf_file):
            print("⚠️  PDF编译或定位失败，无法进行重新审查")
            improvement_rounds.append(
                {
                    "round": round_index,
                    "issue_ledger": issue_ledger,
                    "issue_progress_before": issue_progress_before,
                    "rewrite_result": rewrite_result,
                    "artifact_files": artifact_files,
                    "stop_reason": "missing_pdf_after_rewrite",
                }
            )
            break

        paper_content = load_paper(pdf_file)
        new_text_review = perform_review(
            paper_content, engine.client_model, engine.client
        )
        try:
            new_img_review = perform_imgs_cap_ref_review(
                engine.client, engine.client_model, pdf_file
            )
        except Exception:
            new_img_review = {}

        improvement_eval = engine.evaluate_improvement(
            current_text_review, new_text_review
        )
        next_issue_ledger = build_issue_ledger(
            text_review=new_text_review,
            img_review=new_img_review,
            max_issues=14,
            previous_ledger=issue_ledger,
            target_venue=target_venue,
        )
        issue_progress_after = evaluate_issue_progress(
            previous_issues=list(issue_ledger.get("issues") or []),
            current_issues=list(next_issue_ledger.get("issues") or []),
        )
        round_gate = assess_self_review_gate(
            ledger=next_issue_ledger,
            progress=issue_progress_after,
            rewrite_result=rewrite_result,
            round_index=round_index,
            target_venue=target_venue,
        )
        round_todo_snapshot = evaluate_todo_progress_snapshot(
            experiment_todo_payload,
            round_gate=round_gate,
            issue_progress=issue_progress_after,
            round_index=round_index,
        )
        if int((round_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0:
            experiment_todo_round_snapshots.append(round_todo_snapshot)
        artifact_files = save_self_review_artifacts(
            review_dir=round_dir,
            ledger=issue_ledger,
            progress=issue_progress_before,
            gate=round_gate,
        )

        print(f"\n改进效果:")
        print(f"  总体评分变化: {improvement_eval['overall_improvement']:+.1f}")
        print(f"  改进的类别: {improvement_eval['categories_improved']}")
        print(f"  下降的类别: {improvement_eval['categories_declined']}")
        print(
            "  问题演进: "
            f"resolved={issue_progress_after.get('resolved_issue_count', 0)}, "
            f"persistent={issue_progress_after.get('persistent_issue_count', 0)}, "
            f"new={issue_progress_after.get('new_issue_count', 0)}, "
            f"unresolved_critical={issue_progress_after.get('unresolved_critical_count', 0)}"
        )
        print(
            "  Round gate: "
            f"ready={round_gate.get('ready')}, "
            f"score={round_gate.get('score')}, "
            f"reasons={round_gate.get('reasons', [])}"
        )
        if int((round_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0:
            print(
                "  TODO闭环: "
                f"closure={round_todo_snapshot.get('closure_rate')}, "
                f"p0_closure={round_todo_snapshot.get('p0_closure_rate')}, "
                f"unresolved={round_todo_snapshot.get('counts', {}).get('unresolved_tasks')}"
            )

        should_continue, reason = engine.should_continue_improvement(
            improvement_eval, round_index
        )
        if round_gate.get("ready"):
            should_continue = False
            reason = f"round gate 达标 (score={round_gate.get('score')})"
        if (
            issue_progress_after.get("unresolved_critical_count", 0) == 0
            and issue_progress_after.get("persistent_issue_count", 0) == 0
            and round_index >= 1
        ):
            should_continue = False
            reason = "关键问题已清空且无持续问题，停止迭代"

        round_record = {
            "round": round_index,
            "issue_ledger": issue_ledger,
            "issue_progress_before": issue_progress_before,
            "rewrite_result": rewrite_result,
            "evaluation": improvement_eval,
            "issue_progress_after": issue_progress_after,
            "round_gate": round_gate,
            "artifact_files": artifact_files,
            "experiment_todo_progress": (
                round_todo_snapshot
                if int((round_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0
                else None
            ),
            "reviews": {
                "old": current_text_review,
                "new": new_text_review,
            },
            "decision": {
                "should_continue": should_continue,
                "reason": reason,
            },
        }
        improvement_rounds.append(round_record)

        current_text_review = new_text_review
        current_img_review = new_img_review
        previous_issue_ledger = next_issue_ledger

        print(f"\n决策: {reason}")
        if not should_continue:
            print("✅ 改进完成")
            break

    latest_round_gate = (
        improvement_rounds[-1].get("round_gate")
        if improvement_rounds
        and isinstance(improvement_rounds[-1].get("round_gate"), dict)
        else {}
    )
    final_issue_progress = (
        improvement_rounds[-1].get("issue_progress_after")
        if improvement_rounds
        and isinstance(improvement_rounds[-1].get("issue_progress_after"), dict)
        else None
    )
    final_experiment_todo_snapshot = evaluate_todo_progress_snapshot(
        experiment_todo_payload,
        round_gate=latest_round_gate,
        issue_progress=final_issue_progress,
        round_index=(len(improvement_rounds) or None),
    )
    if int((final_experiment_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0:
        experiment_todo_progress_payload = build_todo_progress_payload(
            experiment_todo_payload,
            round_snapshots=experiment_todo_round_snapshots,
            final_snapshot=final_experiment_todo_snapshot,
        )
        experiment_todo_progress_files = save_todo_progress_artifacts(
            paper_dir, experiment_todo_progress_payload
        )

    improvement_record = {
        "generated_at": datetime.now().isoformat(),
        "mode": "issue_driven",
        "total_rounds": len(improvement_rounds),
        "rounds": improvement_rounds,
        "final_evaluation": (
            improvement_rounds[-1].get("evaluation") if improvement_rounds else None
        ),
        "final_issue_ledger": previous_issue_ledger,
        "final_round_gate": (
            improvement_rounds[-1].get("round_gate") if improvement_rounds else None
        ),
        "experiment_todo_progress_file": experiment_todo_progress_files.get("json"),
        "experiment_todo_progress_markdown_file": experiment_todo_progress_files.get(
            "markdown"
        ),
        "experiment_todo_progress": (
            final_experiment_todo_snapshot
            if int((final_experiment_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0
            else None
        ),
    }

    record_file = osp.join(paper_dir, "improvement_record.json")
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(improvement_record, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 改进记录已保存: {record_file}")

    latest_round = improvement_rounds[-1] if improvement_rounds else {}
    return {
        "status": "success" if improvement_rounds else "failed",
        "rounds_completed": len(improvement_rounds),
        "final_evaluation": latest_round.get("evaluation"),
        "final_issue_progress": latest_round.get("issue_progress_after"),
        "final_round_gate": latest_round.get("round_gate"),
        "experiment_todo_progress_file": experiment_todo_progress_files.get("json"),
        "experiment_todo_progress_markdown_file": experiment_todo_progress_files.get(
            "markdown"
        ),
        "experiment_todo_progress": (
            final_experiment_todo_snapshot
            if int((final_experiment_todo_snapshot.get("counts") or {}).get("total_tasks") or 0) > 0
            else None
        ),
        "record_file": record_file,
    }


def _compile_latex(cwd: str, timeout: int = 30):
    """编译LaTeX文件"""
    return shared_compile_latex(cwd, pdf_file=None, timeout=timeout, verbose=False)
