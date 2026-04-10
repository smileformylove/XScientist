#!/usr/bin/env python3
"""
自主学习知识库
存储和管理从过去经验中学习的知识
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ai_scientist.config.paths import OUTPUT_PATH


class SelfLearningKnowledgeBase:
    """自主学习知识库 - 中央知识存储库"""

    def __init__(self, research_dir: str = str(OUTPUT_PATH)):
        """
        初始化知识库

        Args:
            research_dir: 研究目录
        """
        self.research_dir = Path(research_dir)
        self.knowledge_dir = self.research_dir / "knowledge_base"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        # 知识存储
        self.success_patterns = []  # 成功模式
        self.failure_patterns = []  # 失败模式
        self.improvement_strategies = {}  # 改进策略效果
        self.review_insights = {}  # 审查洞察
        self.writing_insights = {}  # 写作洞察
        self.config_performance = {}  # 配置性能

        # 加载已有知识
        self._load_knowledge()

    def _load_knowledge(self):
        """加载已有知识"""
        # 加载成功模式
        success_file = self.knowledge_dir / "success_patterns.json"
        if success_file.exists():
            with open(success_file, "r") as f:
                self.success_patterns = json.load(f)

        # 加载失败模式
        failure_file = self.knowledge_dir / "failure_patterns.json"
        if failure_file.exists():
            with open(failure_file, "r") as f:
                self.failure_patterns = json.load(f)

        # 加载改进策略
        strategy_file = self.knowledge_dir / "improvement_strategies.json"
        if strategy_file.exists():
            with open(strategy_file, "r") as f:
                self.improvement_strategies = json.load(f)

        # 加载审查洞察
        review_file = self.knowledge_dir / "review_insights.json"
        if review_file.exists():
            with open(review_file, "r") as f:
                self.review_insights = json.load(f)

        # 加载写作洞察
        writing_file = self.knowledge_dir / "writing_insights.json"
        if writing_file.exists():
            with open(writing_file, "r") as f:
                self.writing_insights = json.load(f)

        print(f"✅ 知识库已加载:")
        print(f"   成功模式: {len(self.success_patterns)}")
        print(f"   失败模式: {len(self.failure_patterns)}")
        print(f"   改进策略: {len(self.improvement_strategies)}")

    def _save_knowledge(self):
        """保存知识"""
        # 保存成功模式
        with open(self.knowledge_dir / "success_patterns.json", "w") as f:
            json.dump(self.success_patterns, f, indent=2, ensure_ascii=False)

        # 保存失败模式
        with open(self.knowledge_dir / "failure_patterns.json", "w") as f:
            json.dump(self.failure_patterns, f, indent=2, ensure_ascii=False)

        # 保存改进策略
        with open(self.knowledge_dir / "improvement_strategies.json", "w") as f:
            json.dump(self.improvement_strategies, f, indent=2, ensure_ascii=False)

        # 保存审查洞察
        with open(self.knowledge_dir / "review_insights.json", "w") as f:
            json.dump(self.review_insights, f, indent=2, ensure_ascii=False)

        # 保存写作洞察
        with open(self.knowledge_dir / "writing_insights.json", "w") as f:
            json.dump(self.writing_insights, f, indent=2, ensure_ascii=False)

    def get_self_evolution_playbook(self) -> Dict:
        """获取由 reviewer 修复闭环沉淀出的 self-evolution playbook。"""
        playbook_file = self.knowledge_dir / "self_evolution_playbook.json"
        if not playbook_file.exists():
            return {}
        try:
            with open(playbook_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def store_paper_experience(
        self,
        paper_data: Dict,
        outcome: str,
        final_scores: Dict = None,
        reviews: List[Dict] = None,
        improvements: List[Dict] = None,
    ):
        """
        存储论文经验

        Args:
            paper_data: 论文数据
            outcome: 结果 (accepted, rejected, minor_revision, major_revision)
            final_scores: 最终评分
            reviews: 审查列表
            improvements: 改进记录
        """
        experience = {
            "paper_id": paper_data.get("paper_id", f"paper_{datetime.now().timestamp()}"),
            "idea_name": paper_data.get("idea_name", ""),
            "paper_type": paper_data.get("paper_type", ""),
            "timestamp": datetime.now().isoformat(),
            "outcome": outcome,
            "final_scores": final_scores or {},
            "reviews": reviews or [],
            "improvements": improvements or [],
            "idea": paper_data.get("idea", {}),
        }

        # 根据结果存储
        if outcome in ["accepted", "minor_revision"]:
            self.success_patterns.append(experience)
            print(f"✅ 存储成功模式: {experience['idea_name']}")
        else:
            self.failure_patterns.append(experience)
            print(f"❌ 存储失败模式: {experience['idea_name']}")

        # 更新改进策略效果
        if improvements:
            self._update_improvement_strategies(improvements, outcome)

        # 更新审查洞察
        if reviews:
            self._update_review_insights(reviews, outcome)

        # 保存知识
        self._save_knowledge()

    def _update_improvement_strategies(self, improvements: List[Dict], outcome: str):
        """更新改进策略效果"""
        success_score = 1 if outcome in ["accepted", "minor_revision"] else 0

        for improvement in improvements:
            strategy = improvement.get("strategy", "unknown")
            issue_type = improvement.get("issue_type", "general")

            key = f"{strategy}_{issue_type}"

            if key not in self.improvement_strategies:
                self.improvement_strategies[key] = {
                    "strategy": strategy,
                    "issue_type": issue_type,
                    "success_count": 0,
                    "total_count": 0,
                    "avg_improvement": 0.0,
                    "improvement_scores": [],
                }

            self.improvement_strategies[key]["total_count"] += 1
            self.improvement_strategies[key]["success_count"] += success_score

            if "improvement_score" in improvement:
                self.improvement_strategies[key]["improvement_scores"].append(
                    improvement["improvement_score"]
                )

            # 计算平均改进分数
            scores = self.improvement_strategies[key]["improvement_scores"]
            if scores:
                self.improvement_strategies[key]["avg_improvement"] = sum(scores) / len(scores)

    def _update_review_insights(self, reviews: List[Dict], outcome: str):
        """更新审查洞察"""
        for review in reviews:
            if "review" in review and "scores" in review["review"]:
                scores = review["review"]["scores"]

                for dimension, score in scores.items():
                    if dimension not in self.review_insights:
                        self.review_insights[dimension] = {
                            "scores": [],
                            "outcomes": [],
                            "threshold_for_success": None,
                        }

                    self.review_insights[dimension]["scores"].append(score)
                    self.review_insights[dimension]["outcomes"].append(
                        1 if outcome in ["accepted", "minor_revision"] else 0
                    )

    def find_similar_papers(
        self,
        current_idea: Dict,
        paper_type: str = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        查找相似的历史论文

        Args:
            current_idea: 当前想法
            paper_type: 论文类型
            top_k: 返回前k个

        Returns:
            相似论文列表
        """
        if not self.success_patterns:
            return []

        # 构建文本表示
        current_text = self._idea_to_text(current_idea)

        # 构建历史论文文本
        historical_texts = []
        historical_papers = []

        for paper in self.success_patterns:
            if paper_type and paper.get("paper_type") != paper_type:
                continue

            historical_texts.append(self._idea_to_text(paper.get("idea", {})))
            historical_papers.append(paper)

        if not historical_texts:
            return []

        # 使用TF-IDF计算相似度
        vectorizer = TfidfVectorizer()
        all_texts = [current_text] + historical_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # 计算相似度
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

        # 排序并返回top-k
        sorted_indices = np.argsort(similarities)[::-1][:top_k]

        similar_papers = []
        for idx in sorted_indices:
            if similarities[idx] > 0.1:  # 相似度阈值
                similar_papers.append({
                    **historical_papers[idx],
                    "similarity": float(similarities[idx]),
                })

        return similar_papers

    def _idea_to_text(self, idea: Dict) -> str:
        """将想法转换为文本"""
        parts = [
            idea.get("Title", ""),
            idea.get("Abstract", ""),
            idea.get("Method", ""),
            idea.get("Field", ""),
            idea.get("Task", ""),
        ]
        return " ".join(parts)

    def get_success_patterns(
        self,
        paper_type: str = None,
        min_score: float = None,
    ) -> List[Dict]:
        """
        获取成功模式

        Args:
            paper_type: 论文类型过滤
            min_score: 最低分数过滤

        Returns:
            成功模式列表
        """
        patterns = self.success_patterns

        if paper_type:
            patterns = [p for p in patterns if p.get("paper_type") == paper_type]

        if min_score:
            patterns = [
                p for p in patterns
                if p.get("final_scores", {}).get("overall", 0) >= min_score
            ]

        return patterns

    def get_common_issues(self) -> Dict[str, int]:
        """获取常见问题"""
        issue_counter = Counter()

        for paper in self.failure_patterns:
            for review in paper.get("reviews", []):
                if "review" in review and "main_issues" in review["review"]:
                    for issue in review["review"]["main_issues"]:
                        issue_counter[issue] += 1

        return dict(issue_counter.most_common(10))

    def get_effective_strategies(
        self,
        issue_type: str = None,
        min_success_rate: float = 0.6,
    ) -> List[Dict]:
        """
        获取有效策略

        Args:
            issue_type: 问题类型
            min_success_rate: 最低成功率

        Returns:
            有效策略列表
        """
        effective = []

        for key, stats in self.improvement_strategies.items():
            if issue_type and stats.get("issue_type") != issue_type:
                continue

            total = stats.get("total_count", 0)
            success_count = stats.get("success_count", 0)

            if total > 0:
                success_rate = success_count / total
                if success_rate >= min_success_rate:
                    effective.append({
                        **stats,
                        "success_rate": success_rate,
                    })

        # 按成功率排序
        effective.sort(key=lambda x: x["success_rate"], reverse=True)

        return effective

    def get_score_thresholds(self) -> Dict[str, float]:
        """获取各维度成功阈值"""
        thresholds = {}

        for dimension, data in self.review_insights.items():
            scores = data.get("scores", [])
            outcomes = data.get("outcomes", [])

            if len(scores) > 10:  # 至少10个样本
                # 找到最佳阈值
                best_threshold = 3.0
                best_accuracy = 0.0

                for threshold in np.arange(2.0, 5.0, 0.1):
                    predictions = [1 if s >= threshold else 0 for s in scores]
                    accuracy = sum(p == o for p, o in zip(predictions, outcomes)) / len(outcomes)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold

                thresholds[dimension] = round(best_threshold, 2)

        return thresholds

    def generate_learning_summary(self) -> Dict:
        """生成学习摘要"""
        self_evolution_playbook = self.get_self_evolution_playbook()
        summary = {
            "total_papers": len(self.success_patterns) + len(self.failure_patterns),
            "success_count": len(self.success_patterns),
            "failure_count": len(self.failure_patterns),
            "success_rate": len(self.success_patterns) / (len(self.success_patterns) + len(self.failure_patterns)) if (len(self.success_patterns) + len(self.failure_patterns)) > 0 else 0,
            "common_issues": self.get_common_issues(),
            "effective_strategies": self.get_effective_strategies(min_success_rate=0.7),
            "score_thresholds": self.get_score_thresholds(),
            "recent_successes": self.success_patterns[-5:] if len(self.success_patterns) >= 5 else self.success_patterns,
            "self_evolution_playbook": self_evolution_playbook,
        }

        return summary

    def prune_old_knowledge(self, days: int = 180):
        """清理旧知识"""
        cutoff = datetime.now() - timedelta(days=days)

        # 清理成功模式
        self.success_patterns = [
            p for p in self.success_patterns
            if datetime.fromisoformat(p["timestamp"]) > cutoff
        ]

        # 清理失败模式
        self.failure_patterns = [
            p for p in self.failure_patterns
            if datetime.fromisoformat(p["timestamp"]) > cutoff
        ]

        self._save_knowledge()
        print(f"✅ 清理了{days}天前的旧知识")


class PatternAnalyzer:
    """模式分析器 - 分析成功和失败模式"""

    def __init__(self, knowledge_base: SelfLearningKnowledgeBase):
        """
        初始化模式分析器

        Args:
            knowledge_base: 知识库实例
        """
        self.kb = knowledge_base

    def analyze_success_factors(self) -> Dict:
        """分析成功因素"""
        if not self.kb.success_patterns:
            return {}

        analysis = {
            "common_traits": self._extract_common_traits(self.kb.success_patterns),
            "score_patterns": self._analyze_score_patterns(),
            "writing_patterns": self._analyze_writing_patterns(),
            "improvement_patterns": self._analyze_improvement_patterns(),
        }

        return analysis

    def _extract_common_traits(self, papers: List[Dict]) -> Dict:
        """提取共同特征"""
        traits = {
            "paper_types": Counter(),
            "fields": Counter(),
            "methods": Counter(),
            "avg_scores": defaultdict(list),
        }

        for paper in papers:
            traits["paper_types"][paper.get("paper_type", "")] += 1
            traits["fields"][paper.get("idea", {}).get("Field", "")] += 1
            traits["methods"][paper.get("idea", {}).get("Method", "")] += 1

            # 收集分数
            for dim, score in paper.get("final_scores", {}).items():
                if isinstance(score, (int, float)):
                    traits["avg_scores"][dim].append(score)

        # 计算平均分
        avg_scores = {}
        for dim, scores in traits["avg_scores"].items():
            if scores:
                avg_scores[dim] = sum(scores) / len(scores)

        return {
            "top_paper_types": traits["paper_types"].most_common(3),
            "top_fields": traits["fields"].most_common(3),
            "top_methods": traits["methods"].most_common(3),
            "average_scores": avg_scores,
        }

    def _analyze_score_patterns(self) -> Dict:
        """分析分数模式"""
        if not self.kb.success_patterns:
            return {}

        # 分析成功论文的分数分布
        all_scores = defaultdict(list)

        for paper in self.kb.success_patterns:
            for dim, score in paper.get("final_scores", {}).items():
                if isinstance(score, (int, float)):
                    all_scores[dim].append(score)

        # 计算统计信息
        patterns = {}
        for dim, scores in all_scores.items():
            if scores:
                patterns[dim] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "std": np.std(scores) if len(scores) > 1 else 0,
                }

        return patterns

    def _analyze_writing_patterns(self) -> Dict:
        """分析写作模式"""
        if not self.kb.success_patterns:
            return {}

        review_round_counts: List[int] = []
        score_by_dimension: Dict[str, List[float]] = defaultdict(list)
        review_dimension_counter: Counter = Counter()
        issue_counter: Counter = Counter()
        strategy_counter: Counter = Counter()
        issue_type_counter: Counter = Counter()

        for paper in self.kb.success_patterns:
            reviews = paper.get("reviews") or []
            improvements = paper.get("improvements") or []
            final_scores = paper.get("final_scores") or {}
            if isinstance(reviews, list):
                review_round_counts.append(len(reviews))
                for review in reviews:
                    if not isinstance(review, dict):
                        continue
                    review_payload = review.get("review") or {}
                    if isinstance(review_payload.get("scores"), dict):
                        for dim, score in review_payload.get("scores", {}).items():
                            review_dimension_counter[str(dim)] += 1
                            if isinstance(score, (int, float)):
                                score_by_dimension[str(dim)].append(float(score))
                    if isinstance(review_payload.get("main_issues"), list):
                        for issue in review_payload.get("main_issues", []):
                            text = str(issue).strip()
                            if text:
                                issue_counter[text] += 1
            if isinstance(improvements, list):
                for item in improvements:
                    if not isinstance(item, dict):
                        continue
                    strategy = str(item.get("strategy") or "").strip()
                    issue_type = str(item.get("issue_type") or "").strip()
                    if strategy:
                        strategy_counter[strategy] += 1
                    if issue_type:
                        issue_type_counter[issue_type] += 1
            if isinstance(final_scores, dict):
                for dim, score in final_scores.items():
                    if isinstance(score, (int, float)):
                        score_by_dimension[str(dim)].append(float(score))

        avg_scores = {
            dim: round(sum(values) / len(values), 3)
            for dim, values in score_by_dimension.items()
            if values
        }
        avg_review_rounds = (
            round(sum(review_round_counts) / len(review_round_counts), 3)
            if review_round_counts
            else 0.0
        )

        return {
            "avg_review_rounds": avg_review_rounds,
            "top_review_dimensions": review_dimension_counter.most_common(8),
            "top_issue_patterns": issue_counter.most_common(8),
            "top_improvement_strategies": strategy_counter.most_common(8),
            "top_improvement_issue_types": issue_type_counter.most_common(8),
            "average_scores_by_dimension": avg_scores,
        }

    def _analyze_improvement_patterns(self) -> Dict:
        """分析改进模式"""
        # 收集所有改进
        all_improvements = []

        for paper in self.kb.success_patterns:
            all_improvements.extend(paper.get("improvements", []))

        if not all_improvements:
            return {}

        # 分析最有效的改进
        improvement_impact = defaultdict(list)

        for imp in all_improvements:
            if "improvement_score" in imp:
                strategy = imp.get("strategy", "unknown")
                improvement_impact[strategy].append(imp["improvement_score"])

        # 计算平均影响
        effective_improvements = {}
        for strategy, scores in improvement_impact.items():
            if scores:
                effective_improvements[strategy] = {
                    "avg_improvement": sum(scores) / len(scores),
                    "count": len(scores),
                }

        return effective_improvements

    def predict_success_probability(self, paper_data: Dict) -> float:
        """预测成功概率"""
        # 基于相似论文的成功率
        similar_papers = self.kb.find_similar_papers(
            paper_data.get("idea", {}),
            paper_data.get("paper_type"),
            top_k=10,
        )

        if not similar_papers:
            return 0.5  # 默认50%

        # 计算加权成功率
        total_weight = 0
        success_weight = 0

        for paper in similar_papers:
            weight = paper.get("similarity", 0.5)
            total_weight += weight

            if paper.get("outcome") in ["accepted", "minor_revision"]:
                success_weight += weight

        if total_weight > 0:
            return success_weight / total_weight

        return 0.5
