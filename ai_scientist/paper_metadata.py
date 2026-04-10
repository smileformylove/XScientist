#!/usr/bin/env python3
"""
论文元数据管理系统
为论文文件夹添加标准化标记，使其他Agent能够了解状态并提供指导
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum

from ai_scientist.config.paths import OUTPUT_PATH


class PaperStatus(Enum):
    """论文状态"""
    IDEATION = "ideation"           # 想法阶段
    GENERATING = "generating"       # 生成中
    DRAFT = "draft"                 # 初稿完成
    UNDER_REVIEW = "under_review"   # 审查中
    IMPROVING = "improving"         # 改进中
    COMPLETED = "completed"         # 完成
    PUBLISHED = "published"         # 已发表


class ReviewStatus(Enum):
    """审查状态"""
    PENDING = "pending"             # 待审查
    SELF_REVIEWED = "self_reviewed" # 自我审查完成
    AGENT_REVIEWED = "agent_reviewed" # Agent审查完成
    HUMAN_REVIEWED = "human_reviewed" # 人工审查完成
    ACCEPTED = "accepted"           # 已接受
    REJECTED = "rejected"           # 已拒绝


class QualityLevel(Enum):
    """质量等级"""
    POOR = "poor"                   # 差 (<3.0)
    FAIR = "fair"                   # 一般 (3.0-3.5)
    GOOD = "good"                   # 良好 (3.5-4.0)
    EXCELLENT = "excellent"         # 优秀 (4.0-4.5)
    OUTSTANDING = "outstanding"     # 卓越 (>4.5)


class PaperMetadata:
    """
    论文元数据管理器

    提供标准化接口让其他Agent了解论文状态
    """

    METADATA_FILE = ".paper_metadata.json"
    STATUS_FILE = ".status.json"
    AGENT_COMMENTS_FILE = ".agent_comments.json"

    def __init__(self, paper_dir: str):
        """
        初始化元数据管理器

        Args:
            paper_dir: 论文目录
        """
        self.paper_dir = Path(paper_dir)
        self.metadata_file = self.paper_dir / self.METADATA_FILE
        self.status_file = self.paper_dir / self.STATUS_FILE
        self.agent_comments_file = self.paper_dir / self.AGENT_COMMENTS_FILE

        # 加载或创建元数据
        self.metadata = self._load_metadata()
        self.status = self._load_status()
        self.agent_comments = self._load_agent_comments()

    def _load_metadata(self) -> Dict:
        """加载元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        else:
            return self._create_initial_metadata()

    def _create_initial_metadata(self) -> Dict:
        """创建初始元数据"""
        return {
            "paper_id": self.paper_dir.name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "paper_type": "unknown",
            "idea_name": "unknown",
            "title": "",
            "abstract": "",
            "authors": ["AI Scientist"],
            "keywords": [],
            "field": "",
            "task": "",
        }

    def _load_status(self) -> Dict:
        """加载状态"""
        if self.status_file.exists():
            with open(self.status_file, "r") as f:
                return json.load(f)
        else:
            return {
                "current_status": PaperStatus.IDEATION.value,
                "review_status": ReviewStatus.PENDING.value,
                "quality_level": None,
                "overall_score": None,
                "last_updated": datetime.now().isoformat(),
                "history": [],
            }

    def _load_agent_comments(self) -> List[Dict]:
        """加载Agent评论"""
        if self.agent_comments_file.exists():
            with open(self.agent_comments_file, "r") as f:
                return json.load(f)
        else:
            return []

    def save(self):
        """保存所有元数据"""
        # 更新时间戳
        self.metadata["updated_at"] = datetime.now().isoformat()
        self.status["last_updated"] = datetime.now().isoformat()

        # 保存文件
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        with open(self.status_file, "w") as f:
            json.dump(self.status, f, indent=2, ensure_ascii=False)

        with open(self.agent_comments_file, "w") as f:
            json.dump(self.agent_comments, f, indent=2, ensure_ascii=False)

    # ========================================
    # 状态管理
    # ========================================

    def set_status(self, status: PaperStatus, note: str = None):
        """
        设置论文状态

        Args:
            status: 新状态
            note: 备注
        """
        old_status = self.status["current_status"]

        # 添加历史记录
        self.status["history"].append({
            "from": old_status,
            "to": status.value,
            "timestamp": datetime.now().isoformat(),
            "note": note,
        })

        # 更新状态
        self.status["current_status"] = status.value
        self.save()

        print(f"📝 状态更新: {old_status} → {status.value}")

    def set_review_status(self, review_status: ReviewStatus):
        """设置审查状态"""
        self.status["review_status"] = review_status.value
        self.save()

    def set_quality(self, score: float, quality_level: QualityLevel = None):
        """
        设置质量评估

        Args:
            score: 总分 (0-5)
            quality_level: 质量等级
        """
        self.status["overall_score"] = score

        if quality_level:
            self.status["quality_level"] = quality_level.value
        else:
            # 自动确定等级
            if score < 3.0:
                self.status["quality_level"] = QualityLevel.POOR.value
            elif score < 3.5:
                self.status["quality_level"] = QualityLevel.FAIR.value
            elif score < 4.0:
                self.status["quality_level"] = QualityLevel.GOOD.value
            elif score < 4.5:
                self.status["quality_level"] = QualityLevel.EXCELLENT.value
            else:
                self.status["quality_level"] = QualityLevel.OUTSTANDING.value

        self.save()

    # ========================================
    # Agent评论管理
    # ========================================

    def add_agent_comment(
        self,
        agent_name: str,
        agent_type: str,
        comment: str,
        score: float = None,
        issues: List[str] = None,
        suggestions: List[str] = None,
        priority: str = "medium",
    ):
        """
        添加Agent评论

        Args:
            agent_name: Agent名称
            agent_type: Agent类型
            comment: 评论内容
            score: 评分 (0-5)
            issues: 问题列表
            suggestions: 建议列表
            priority: 优先级
        """
        comment_record = {
            "agent_name": agent_name,
            "agent_type": agent_type,
            "comment": comment,
            "score": score,
            "issues": issues or [],
            "suggestions": suggestions or [],
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "addressed": False,  # 是否已处理
        }

        self.agent_comments.append(comment_record)
        self.save()

        print(f"💬 已添加 {agent_name} 的评论")

    def get_unaddressed_comments(self) -> List[Dict]:
        """获取未处理的评论"""
        return [c for c in self.agent_comments if not c["addressed"]]

    def mark_comment_addressed(self, comment_index: int):
        """标记评论为已处理"""
        if 0 <= comment_index < len(self.agent_comments):
            self.agent_comments[comment_index]["addressed"] = True
            self.save()

    def get_agent_summary(self) -> Dict:
        """获取Agent评论摘要"""
        if not self.agent_comments:
            return {
                "total_comments": 0,
                "average_score": None,
                "common_issues": [],
                "top_suggestions": [],
            }

        # 统计
        total_comments = len(self.agent_comments)
        scores = [c["score"] for c in self.agent_comments if c["score"] is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        # 常见问题
        issue_counter = {}
        for comment in self.agent_comments:
            for issue in comment.get("issues", []):
                issue_counter[issue] = issue_counter.get(issue, 0) + 1

        common_issues = sorted(issue_counter.items(), key=lambda x: x[1], reverse=True)[:5]

        # Top建议
        suggestion_counter = {}
        for comment in self.agent_comments:
            for suggestion in comment.get("suggestions", []):
                suggestion_counter[suggestion] = suggestion_counter.get(suggestion, 0) + 1

        top_suggestions = sorted(suggestion_counter.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "total_comments": total_comments,
            "average_score": avg_score,
            "common_issues": [{"issue": i, "count": c} for i, c in common_issues],
            "top_suggestions": [{"suggestion": s, "count": c} for s, c in top_suggestions],
            "unaddressed_count": len(self.get_unaddressed_comments()),
        }

    # ========================================
    # 外部Agent接口
    # ========================================

    def get_info_for_agents(self) -> Dict:
        """
        获取供外部Agent使用的信息

        Returns:
            标准化的信息字典
        """
        return {
            "paper_info": {
                "paper_id": self.metadata["paper_id"],
                "title": self.metadata.get("title", ""),
                "abstract": self.metadata.get("abstract", ""),
                "paper_type": self.metadata.get("paper_type", ""),
                "field": self.metadata.get("field", ""),
                "task": self.metadata.get("task", ""),
                "keywords": self.metadata.get("keywords", []),
            },
            "status": {
                "current_status": self.status["current_status"],
                "review_status": self.status["review_status"],
                "quality_level": self.status["quality_level"],
                "overall_score": self.status["overall_score"],
                "can_be_improved": self._can_be_improved(),
            },
            "agent_summary": self.get_agent_summary(),
            "files": self._get_paper_files(),
            "history": self.status.get("history", [])[-5:],  # 最近5条历史
        }

    def _can_be_improved(self) -> bool:
        """判断是否可以改进"""
        status = self.status["current_status"]
        score = self.status.get("overall_score", 0)

        # 未完成且分数不高的论文可以改进
        return (
            status not in [PaperStatus.PUBLISHED.value, PaperStatus.COMPLETED.value]
            and score < 4.5
        )

    def _get_paper_files(self) -> Dict:
        """获取论文文件列表"""
        files = {
            "latex": [],
            "pdf": [],
            "review": [],
            "experiment": [],
            "other": [],
        }

        for file_path in self.paper_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith("."):
                rel_path = file_path.relative_to(self.paper_dir)

                if file_path.suffix == ".tex":
                    files["latex"].append(str(rel_path))
                elif file_path.suffix == ".pdf":
                    files["pdf"].append(str(rel_path))
                elif "review" in str(rel_path):
                    files["review"].append(str(rel_path))
                elif "experiment" in str(rel_path):
                    files["experiment"].append(str(rel_path))
                else:
                    files["other"].append(str(rel_path))

        return files

    def get_actionable_items(self) -> List[Dict]:
        """
        获取可执行的行动项

        供外部Agent了解需要做什么
        """
        items = []

        # 基于状态
        if self.status["current_status"] == PaperStatus.DRAFT.value:
            items.append({
                "type": "review",
                "priority": "high",
                "description": "需要审查初稿",
                "suggested_agents": ["WritingCriticAgent", "TechnicalReviewerAgent"],
            })

        # 基于未处理评论
        unaddressed = self.get_unaddressed_comments()
        if unaddressed:
            # 按优先级排序
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}
            sorted_comments = sorted(
                unaddressed,
                key=lambda c: priority_order.get(c.get("priority", "medium"), 2)
            )

            for comment in sorted_comments[:5]:  # 最多5个
                items.append({
                    "type": "address_comment",
                    "priority": comment.get("priority", "medium"),
                    "description": f"处理 {comment['agent_name']} 的评论",
                    "agent_name": comment["agent_name"],
                    "comment": comment["comment"],
                    "suggestions": comment.get("suggestions", []),
                })

        # 基于质量分数
        score = self.status.get("overall_score", 5.0)
        if score < 3.0:
            items.append({
                "type": "major_improvement",
                "priority": "critical",
                "description": "质量较低，需要大幅改进",
                "target_score": 4.0,
            })
        elif score < 4.0:
            items.append({
                "type": "minor_improvement",
                "priority": "high",
                "description": "需要改进以达到优秀水平",
                "target_score": 4.5,
            })

        return items

    # ========================================
    # 快捷方法
    # ========================================

    def update_from_generation(self, idea: Dict, paper_type: str):
        """从生成结果更新"""
        self.metadata["idea_name"] = idea.get("Name", "")
        self.metadata["title"] = idea.get("Title", "")
        self.metadata["abstract"] = idea.get("Abstract", "")
        self.metadata["paper_type"] = paper_type
        self.metadata["field"] = idea.get("Field", "")
        self.metadata["task"] = idea.get("Task", "")
        self.metadata["keywords"] = idea.get("Keywords", [])
        self.save()

    def update_from_evaluation(self, evaluation: Dict):
        """从评估结果更新"""
        overall = evaluation.get("overall", {})
        score = overall.get("score", 0)

        self.set_quality(score)
        self.set_status(PaperStatus.DRAFT, "评估完成")
        self.save()

    def get_status_summary(self) -> str:
        """获取状态摘要（用于显示）"""
        lines = [
            f"📄 论文: {self.metadata.get('title', self.metadata['idea_name'])}",
            f"📊 状态: {self.status['current_status']} ({self.status['quality_level'] or '未评估'})",
            f"⭐ 评分: {self.status.get('overall_score', 'N/A')}",
            f"💬 评论: {len(self.agent_comments)} 条",
        ]

        if self.status.get("overall_score", 5) < 4.0:
            lines.append(f"⚠️  需要改进")

        return "\n".join(lines)


class MetadataRegistry:
    """
    元数据注册表
    管理所有论文的元数据
    """

    def __init__(self, research_dir: str = str(OUTPUT_PATH)):
        """
        初始化注册表

        Args:
            research_dir: 研究目录
        """
        self.research_dir = Path(research_dir)
        self.registry_file = self.research_dir / ".metadata_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """加载注册表"""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        else:
            return {
                "papers": {},
                "last_updated": datetime.now().isoformat(),
            }

    def _save_registry(self):
        """保存注册表"""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def register_paper(self, paper_dir: str) -> PaperMetadata:
        """
        注册论文

        Args:
            paper_dir: 论文目录

        Returns:
            PaperMetadata实例
        """
        metadata = PaperMetadata(paper_dir)
        paper_id = metadata.metadata["paper_id"]

        self.registry["papers"][paper_id] = {
            "dir": paper_dir,
            "created_at": metadata.metadata["created_at"],
            "updated_at": metadata.metadata["updated_at"],
            "status": metadata.status["current_status"],
            "quality": metadata.status.get("quality_level"),
            "score": metadata.status.get("overall_score"),
        }

        self._save_registry()
        return metadata

    def get_paper_metadata(self, paper_id: str) -> Optional[PaperMetadata]:
        """获取论文元数据"""
        if paper_id not in self.registry["papers"]:
            return None

        paper_dir = self.registry["papers"][paper_id]["dir"]
        return PaperMetadata(paper_dir)

    def find_papers_by_status(
        self,
        status: PaperStatus = None,
        quality_level: QualityLevel = None,
        min_score: float = None,
        max_score: float = None,
    ) -> List[Dict]:
        """
        查找论文

        Args:
            status: 状态过滤
            quality_level: 质量等级过滤
            min_score: 最低分数
            max_score: 最高分数

        Returns:
            匹配的论文列表
        """
        results = []

        for paper_id, info in self.registry["papers"].items():
            # 状态过滤
            if status and info["status"] != status.value:
                continue

            # 质量等级过滤
            if quality_level and info.get("quality") != quality_level.value:
                continue

            # 分数过滤
            score = info.get("score")
            if min_score is not None and (score is None or score < min_score):
                continue
            if max_score is not None and (score is None or score > max_score):
                continue

            results.append({
                "paper_id": paper_id,
                **info
            })

        return results

    def get_papers_needing_review(self) -> List[Dict]:
        """获取需要审查的论文"""
        return self.find_papers_by_status(status=PaperStatus.DRAFT)

    def get_papers_needing_improvement(self) -> List[Dict]:
        """获取需要改进的论文"""
        return self.find_papers_by_status(min_score=0, max_score=4.0)

    def get_registry_summary(self) -> Dict:
        """获取注册表摘要"""
        papers = list(self.registry["papers"].values())

        # 统计
        status_count = {}
        quality_count = {}
        total_score = 0
        score_count = 0

        for paper in papers:
            # 状态统计
            status = paper.get("status", "unknown")
            status_count[status] = status_count.get(status, 0) + 1

            # 质量统计
            quality = paper.get("quality", "unknown")
            quality_count[quality] = quality_count.get(quality, 0) + 1

            # 分数统计
            score = paper.get("score")
            if score is not None:
                total_score += score
                score_count += 1

        return {
            "total_papers": len(papers),
            "status_distribution": status_count,
            "quality_distribution": quality_count,
            "average_score": total_score / score_count if score_count > 0 else None,
            "papers_needing_review": len(self.get_papers_needing_review()),
            "papers_needing_improvement": len(self.get_papers_needing_improvement()),
        }


# ========================================
# 便捷函数
# ========================================

def get_paper_metadata(paper_dir: str) -> PaperMetadata:
    """获取论文元数据（便捷函数）"""
    return PaperMetadata(paper_dir)


def create_paper_metadata(
    paper_dir: str,
    idea: Dict,
    paper_type: str,
) -> PaperMetadata:
    """创建论文元数据（便捷函数）"""
    metadata = PaperMetadata(paper_dir)
    metadata.update_from_generation(idea, paper_type)
    return metadata
