#!/usr/bin/env python3
"""
AI Scientist 改进可视化和报告生成系统
生成详细的改进报告和可视化图表
"""
import json
import os
import os.path as osp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ImprovementReporter:
    """改进报告生成器"""

    def __init__(self, output_dir: str):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / "improvement_reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_improvement_report(
        self,
        paper_name: str,
        improvement_record: Dict,
        original_review: Dict,
        final_review: Dict,
    ) -> str:
        """
        生成改进报告

        Args:
            paper_name: 论文名称
            improvement_record: 改进记录
            original_review: 原始审查结果
            final_review: 最终审查结果

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"improvement_report_{timestamp}.json"

        # 构建报告
        report = {
            "paper_name": paper_name,
            "generated_at": datetime.now().isoformat(),
            "summary": self._generate_summary(improvement_record),
            "rounds": improvement_record.get("rounds", []),
            "original_review": original_review,
            "final_review": final_review,
            "comparisons": self._generate_comparisons(
                original_review, final_review
            ),
            "recommendations": self._generate_recommendations(
                improvement_record, final_review
            ),
        }

        # 保存JSON报告
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        # 生成Markdown报告
        md_report_file = self._generate_markdown_report(report)

        # 生成可视化
        self._generate_visualizations(report)

        return str(report_file)

    def _generate_summary(self, improvement_record: Dict) -> Dict:
        """生成改进摘要"""
        rounds = improvement_record.get("rounds", [])

        if not rounds:
            return {
                "total_rounds": 0,
                "status": "no_improvements",
                "message": "没有改进记录"
            }

        final_eval = rounds[-1].get("evaluation", {})

        return {
            "total_rounds": len(rounds),
            "overall_improvement": final_eval.get("overall_improvement", 0),
            "categories_improved": final_eval.get("categories_improved", 0),
            "categories_declined": final_eval.get("categories_declined", 0),
            "significant_improvement": final_eval.get("significant_improvement", False),
            "status": "success" if final_eval.get("overall_improvement", 0) > 0 else "limited",
        }

    def _generate_comparisons(
        self,
        original_review: Dict,
        final_review: Dict,
    ) -> Dict:
        """生成对比分析"""
        original_scores = original_review.get("review", {}).get("scores", {})
        final_scores = final_review.get("review", {}).get("scores", {})

        comparisons = {}
        score_changes = []

        for category in original_scores:
            if category in final_scores:
                old_score = original_scores[category]
                new_score = final_scores[category]

                if isinstance(old_score, (int, float)) and isinstance(new_score, (int, float)):
                    change = new_score - old_score
                    comparisons[category] = {
                        "original": old_score,
                        "final": new_score,
                        "change": change,
                        "percentage_change": (change / old_score * 100) if old_score > 0 else 0,
                    }
                    score_changes.append((category, change))

        # 排序：变化最大的在前
        score_changes.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "detailed": comparisons,
            "top_improvements": score_changes[:3],
            "top_declines": score_changes[-3:] if len(score_changes) > 3 else [],
        }

    def _generate_recommendations(
        self,
        improvement_record: Dict,
        final_review: Dict,
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        final_eval = improvement_record.get("final_evaluation", {})
        overall_improvement = final_eval.get("overall_improvement", 0)

        # 基于改进效果的建议
        if overall_improvement >= 2.0:
            recommendations.append(
                "✅ 论文质量显著提升，建议投稿。"
            )
        elif overall_improvement >= 1.0:
            recommendations.append(
                "✅ 论文有明显改进，建议进行最终审查后投稿。"
            )
        elif overall_improvement > 0:
            recommendations.append(
                "⚠️  论文有一定改进，建议继续优化或考虑其他投稿目标。"
            )
        else:
            recommendations.append(
                "❌ 改进效果有限，建议重新评估研究方向或方法。"
            )

        # 基于最终评分的建议
        final_scores = final_review.get("review", {}).get("scores", {})
        overall_score = final_scores.get("Overall", 0)

        if overall_score >= 8.0:
            recommendations.append(
                "🌟 总体评分优秀，可以冲击顶级会议/期刊。"
            )
        elif overall_score >= 6.0:
            recommendations.append(
                "📝 总体评分良好，适合标准会议/期刊投稿。"
            )
        else:
            recommendations.append(
                "💪 建议继续改进以提升竞争力。"
            )

        # 基于具体类别的建议
        comparisons = self._generate_comparisons(
            improvement_record.get("rounds", [{}])[0].get("reviews", {}).get("old", {}),
            final_review
        )

        for category, change in comparisons["top_declines"]:
            if change < -0.5:
                recommendations.append(
                    f"⚠️  {category} 评分下降 {abs(change):.1f}，需要关注。"
                )

        return recommendations

    def _generate_markdown_report(self, report: Dict) -> str:
        """生成Markdown格式报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = self.reports_dir / f"improvement_report_{timestamp}.md"

        with open(md_file, "w", encoding="utf-8") as f:
            # 标题
            f.write(f"# 论文改进报告\n\n")
            f.write(f"**论文名称**: {report['paper_name']}\n\n")
            f.write(f"**生成时间**: {report['generated_at']}\n\n")

            # 摘要
            f.write("## 📊 改进摘要\n\n")
            summary = report["summary"]
            f.write(f"- **总轮数**: {summary['total_rounds']}\n")
            f.write(f"- **总体改进**: {summary['overall_improvement']:+.1f}\n")
            f.write(f"- **改进类别**: {summary['categories_improved']}\n")
            f.write(f"- **下降类别**: {summary['categories_declined']}\n")
            f.write(f"- **显著改进**: {'是' if summary['significant_improvement'] else '否'}\n\n")

            # 对比分析
            f.write("## 📈 评分对比\n\n")
            f.write("| 类别 | 原始评分 | 最终评分 | 变化 |\n")
            f.write("|------|----------|----------|------|\n")

            for category, comp in report["comparisons"]["detailed"].items():
                change_sign = "+" if comp["change"] > 0 else ""
                f.write(f"| {category} | {comp['original']:.1f} | {comp['final']:.1f} | {change_sign}{comp['change']:.1f} |\n")

            f.write("\n")

            # 主要改进
            if report["comparisons"]["top_improvements"]:
                f.write("### ✅ 主要改进\n\n")
                for category, change in report["comparisons"]["top_improvements"]:
                    if change > 0:
                        f.write(f"- **{category}**: +{change:.1f}\n")
                f.write("\n")

            # 主要下降
            if report["comparisons"]["top_declines"]:
                f.write("### ⚠️  需要关注的方面\n\n")
                for category, change in report["comparisons"]["top_declines"]:
                    if change < 0:
                        f.write(f"- **{category}**: {change:.1f}\n")
                f.write("\n")

            # 建议
            f.write("## 💡 建议\n\n")
            for rec in report["recommendations"]:
                f.write(f"{rec}\n")
            f.write("\n")

            # 详细轮次
            f.write("## 🔄 详细改进轮次\n\n")
            for round_data in report.get("rounds", []):
                round_num = round_data.get("round", 0)
                f.write(f"### 第 {round_num} 轮\n\n")

                eval_data = round_data.get("evaluation", {})
                f.write(f"- **改进**: {eval_data.get('overall_improvement', 0):+.1f}\n")
                f.write(f"- **改进类别**: {eval_data.get('categories_improved', 0)}\n")
                f.write(f"- **下降类别**: {eval_data.get('categories_declined', 0)}\n\n")

            # 原始和最终审查
            f.write("## 📝 审查详情\n\n")
            f.write("### 原始审查\n\n")
            original_review = report.get("original_review", {}).get("review", {})
            if original_review:
                f.write(f"**摘要**: {original_review.get('Summary', 'N/A')}\n\n")
                f.write(f"**优点**: {original_review.get('Strengths', 'N/A')}\n\n")
                f.write(f"**缺点**: {original_review.get('Weaknesses', 'N/A')}\n\n")

            f.write("### 最终审查\n\n")
            final_review = report.get("final_review", {}).get("review", {})
            if final_review:
                f.write(f"**摘要**: {final_review.get('Summary', 'N/A')}\n\n")
                f.write(f"**优点**: {final_review.get('Strengths', 'N/A')}\n\n")
                f.write(f"**缺点**: {final_review.get('Weaknesses', 'N/A')}\n\n")

        return str(md_file)

    def _generate_visualizations(self, report: Dict):
        """生成可视化图表（使用ASCII艺术）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_file = self.reports_dir / f"improvement_chart_{timestamp}.txt"

        with open(viz_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("改进趋势可视化\n")
            f.write("=" * 80 + "\n\n")

            # 评分变化条形图
            f.write("评分变化对比:\n\n")

            comparisons = report["comparisons"]["detailed"]
            for category, comp in sorted(comparisons.items(), key=lambda x: x[1]["change"]):
                change = comp["change"]
                bar_length = int(abs(change) * 10)

                if change > 0:
                    bar = "🟢" * bar_length
                    f.write(f"{category:15s}: [{bar}] +{change:.1f}\n")
                elif change < 0:
                    bar = "🔴" * bar_length
                    f.write(f"{category:15s}: [{bar}] {change:.1f}\n")
                else:
                    f.write(f"{category:15s}: [▓▓▓] 0.0\n")

            f.write("\n")

            # 进度时间线
            f.write("改进进度时间线:\n\n")

            rounds = report.get("rounds", [])
            if rounds:
                for i, round_data in enumerate(rounds):
                    round_num = round_data.get("round", i + 1)
                    improvement = round_data.get("evaluation", {}).get("overall_improvement", 0)

                    if i == 0:
                        f.write(f"原始状态 -> ")
                    elif i == len(rounds) - 1:
                        f.write(f"第{round_num}轮 (改进:{improvement:+.1f}) -> 最终状态\n")
                    else:
                        f.write(f"第{round_num}轮 (改进:{improvement:+.1f}) -> ")

            f.write("\n")

            # 总体评价
            summary = report["summary"]
            f.write("=" * 80 + "\n")
            f.write("总体评价:\n")
            f.write("=" * 80 + "\n\n")

            overall_improvement = summary.get("overall_improvement", 0)

            if overall_improvement >= 2.0:
                f.write("🌟🌟🌟 优秀改进！论文质量显著提升。\n")
            elif overall_improvement >= 1.0:
                f.write("🌟🌟 良好改进！论文有明显提升。\n")
            elif overall_improvement > 0:
                f.write("🌟 有所改进。建议继续优化。\n")
            else:
                f.write("⚠️  改进有限。需要重新评估策略。\n")

    def generate_batch_report(
        self,
        batch_results: List[Dict],
    ) -> str:
        """
        生成批量改进报告

        Args:
            batch_results: 批量改进结果列表

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.reports_dir / f"batch_report_{timestamp}.json"

        # 统计
        total_papers = len(batch_results)
        successful = len([r for r in batch_results if r.get("status") == "success"])
        failed = total_papers - successful

        total_improvement = sum([
            r.get("final_evaluation", {}).get("overall_improvement", 0)
            for r in batch_results
            if r.get("status") == "success"
        ])

        avg_improvement = total_improvement / successful if successful > 0 else 0

        # 构建报告
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": {
                "total_papers": total_papers,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_papers if total_papers > 0 else 0,
                "total_improvement": total_improvement,
                "average_improvement": avg_improvement,
            },
            "papers": batch_results,
        }

        # 保存报告
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        return str(report_file)


def print_improvement_summary(improvement_record: Dict):
    """
    打印改进摘要到控制台

    Args:
        improvement_record: 改进记录
    """
    print("\n" + "=" * 80)
    print("📊 改进总结")
    print("=" * 80)

    if not improvement_record.get("rounds"):
        print("❌ 没有改进记录")
        return

    final_eval = improvement_record["rounds"][-1]["evaluation"]

    print(f"\n总轮数: {len(improvement_record['rounds'])}")
    print(f"总体改进: {final_eval['overall_improvement']:+.1f}")
    print(f"改进类别: {final_eval['categories_improved']}")
    print(f"下降类别: {final_eval['categories_declined']}")

    # 显示评分变化
    print("\n评分变化:")
    for category, scores in final_eval.get("score_improvements", {}).items():
        change = scores["improvement"]
        change_sign = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        print(f"  {change_sign} {category}: {scores['old']:.1f} → {scores['new']:.1f} ({change:+.1f})")

    print("\n" + "=" * 80)
