#!/usr/bin/env python3
"""
Agent发现和指导协调器
让外部Agent能够自动发现需要帮助的论文并提供指导
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum

from ai_scientist.config.paths import OUTPUT_PATH
from ai_scientist.paper_metadata import (
    PaperMetadata,
    MetadataRegistry,
    PaperStatus,
    QualityLevel,
)


class GuidancePriority(Enum):
    """指导优先级"""
    CRITICAL = "critical"  # 紧急（质量差或有严重问题）
    HIGH = "high"        # 高优先级（有明显改进空间）
    MEDIUM = "medium"    # 中等优先级（正常改进需求）
    LOW = "low"         # 低优先级（小幅优化）
    OPTIONAL = "optional"  # 可选（已经很好）


class AgentDiscoverySystem:
    """
    Agent发现系统

    帮助外部Agent发现需要指导的论文
    """

    def __init__(self, research_dir: str = str(OUTPUT_PATH)):
        """
        初始化发现系统

        Args:
            research_dir: 研究目录
        """
        self.research_dir = Path(research_dir)
        self.registry = MetadataRegistry(research_dir)

        # Agent注册表
        self.agent_registry_file = self.research_dir / ".agent_registry.json"
        self.agent_registry = self._load_agent_registry()

    def _load_agent_registry(self) -> Dict:
        """加载Agent注册表"""
        if self.agent_registry_file.exists():
            with open(self.agent_registry_file, "r") as f:
                return json.load(f)
        else:
            return {"agents": {}, "last_updated": datetime.now().isoformat()}

    def _save_agent_registry(self):
        """保存Agent注册表"""
        self.agent_registry["last_updated"] = datetime.now().isoformat()
        with open(self.agent_registry_file, "w") as f:
            json.dump(self.agent_registry, f, indent=2, ensure_ascii=False)

    def register_agent_capabilities(
        self,
        agent_name: str,
        capabilities: List[str],
        agent_type: str,
        description: str = None,
    ):
        """
        注册Agent能力

        Args:
            agent_name: Agent名称
            capabilities: 能力列表
            agent_type: Agent类型
            description: 描述
        """
        self.agent_registry["agents"][agent_name] = {
            "capabilities": capabilities,
            "type": agent_type,
            "description": description,
            "registered_at": datetime.now().isoformat(),
            "papers_reviewed": [],
            "total_contributions": 0,
        }

        self._save_agent_registry()
        print(f"✅ 已注册Agent能力: {agent_name}")

    def discover_papers_for_agent(
        self,
        agent_name: str,
        max_papers: int = 10,
        status_filter: List[str] = None,
    ) -> List[Dict]:
        """
        为Agent发现需要指导的论文

        Args:
            agent_name: Agent名称
            max_papers: 最大返回数量
            status_filter: 状态过滤

        Returns:
            论文列表（按优先级排序）
        """
        if agent_name not in self.agent_registry["agents"]:
            print(f"⚠️  Agent {agent_name} 未注册")
            return []

        agent_info = self.agent_registry["agents"][agent_name]
        capabilities = agent_info.get("capabilities", [])

        # 查找论文
        papers = []

        # 获取需要改进的论文
        papers_needing_improvement = self.registry.get_papers_needing_improvement()

        for paper_info in papers_needing_improvement:
            paper_id = paper_info["paper_id"]
            metadata = self.registry.get_paper_metadata(paper_id)

            if not metadata:
                continue

            # 获取论文信息
            info = metadata.get_info_for_agents()

            # 计算优先级
            priority = self._calculate_guidance_priority(
                paper_info=info,
                agent_capabilities=capabilities,
            )

            # 获取匹配的行动项
            actionable_items = metadata.get_actionable_items()

            papers.append({
                "paper_id": paper_id,
                "paper_dir": paper_info["dir"],
                "priority": priority,
                "info": info,
                "actionable_items": actionable_items,
            })

        # 按优先级排序
        priority_order = {
            GuidancePriority.CRITICAL: 0,
            GuidancePriority.HIGH: 1,
            GuidancePriority.MEDIUM: 2,
            GuidancePriority.LOW: 3,
            GuidancePriority.OPTIONAL: 4,
        }

        papers.sort(key=lambda p: priority_order.get(p["priority"], 5))

        # 限制数量
        return papers[:max_papers]

    def _calculate_guidance_priority(
        self,
        paper_info: Dict,
        agent_capabilities: List[str],
    ) -> GuidancePriority:
        """计算指导优先级"""
        score = paper_info["status"].get("overall_score", 5.0)
        status = paper_info["status"]["current_status"]

        # 未完成且分数低的论文优先级高
        if score < 3.0:
            return GuidancePriority.CRITICAL
        elif score < 3.5:
            return GuidancePriority.HIGH
        elif score < 4.0:
            return GuidancePriority.MEDIUM
        elif score < 4.5:
            return GuidancePriority.LOW
        else:
            return GuidancePriority.OPTIONAL

    def get_guidance_opportunities(
        self,
        agent_name: str = None,
    ) -> Dict:
        """
        获取指导机会摘要

        Args:
            agent_name: Agent名称（None表示全部）

        Returns:
            机会摘要
        """
        summary = self.registry.get_registry_summary()

        # 获取需要改进的论文详情
        papers_needing_improvement = self.registry.get_papers_needing_improvement()

        # 按质量等级分类
        by_quality = {
            "poor": [],
            "fair": [],
            "good": [],
        }

        for paper in papers_needing_improvement:
            quality = paper.get("quality", "unknown")
            if quality in by_quality:
                by_quality[quality].append(paper)

        return {
            "total_opportunities": len(papers_needing_improvement),
            "by_quality": by_quality,
            "average_score_improvement": summary.get("average_score"),
            "registry_summary": summary,
        }


class AgentGuidanceCoordinator:
    """
    Agent指导协调器

    协调Agent与论文之间的交互
    """

    def __init__(self, research_dir: str = str(OUTPUT_PATH)):
        """初始化协调器"""
        self.research_dir = Path(research_dir)
        self.discovery_system = AgentDiscoverySystem(research_dir)
        self.guidance_log_file = self.research_dir / ".agent_guidance_log.json"
        self.guidance_log = self._load_guidance_log()

    def _load_guidance_log(self) -> List[Dict]:
        """加载指导日志"""
        if self.guidance_log_file.exists():
            with open(self.guidance_log_file, "r") as f:
                return json.load(f)
        return []

    def _save_guidance_log(self):
        """保存指导日志"""
        with open(self.guidance_log_file, "w") as f:
            json.dump(self.guidance_log, f, indent=2, ensure_ascii=False)

    def provide_guidance(
        self,
        agent_name: str,
        paper_id: str,
        guidance: Dict,
    ) -> Dict:
        """
        Agent提供指导

        Args:
            agent_name: Agent名称
            paper_id: 论文ID
            guidance: 指导内容

        Returns:
            处理结果
        """
        # 获取论文元数据
        metadata = self.discovery_system.registry.get_paper_metadata(paper_id)

        if not metadata:
            return {
                "success": False,
                "error": f"Paper {paper_id} not found"
            }

        # 添加Agent评论
        metadata.add_agent_comment(
            agent_name=agent_name,
            agent_type=guidance.get("agent_type", "general"),
            comment=guidance.get("comment", ""),
            score=guidance.get("score"),
            issues=guidance.get("issues", []),
            suggestions=guidance.get("suggestions", []),
            priority=guidance.get("priority", "medium"),
        )

        # 记录日志
        log_entry = {
            "agent_name": agent_name,
            "paper_id": paper_id,
            "guidance": guidance,
            "timestamp": datetime.now().isoformat(),
        }

        self.guidance_log.append(log_entry)
        self._save_guidance_log()

        # 更新Agent注册表
        if agent_name in self.discovery_system.agent_registry["agents"]:
            agent_info = self.discovery_system.agent_registry["agents"][agent_name]
            agent_info["papers_reviewed"].append(paper_id)
            agent_info["total_contributions"] += 1
            self.discovery_system._save_agent_registry()

        return {
            "success": True,
            "paper_id": paper_id,
            "agent_name": agent_name,
            "metadata_updated": True,
        }

    def get_guidance_for_paper(self, paper_id: str) -> List[Dict]:
        """
        获取论文收到的所有指导

        Args:
            paper_id: 论文ID

        Returns:
            指导列表
        """
        metadata = self.discovery_system.registry.get_paper_metadata(paper_id)

        if not metadata:
            return []

        return metadata.agent_comments

    def get_agent_statistics(self, agent_name: str = None) -> Dict:
        """
        获取Agent统计信息

        Args:
            agent_name: Agent名称（None表示全部）

        Returns:
            统计信息
        """
        if agent_name:
            if agent_name not in self.discovery_system.agent_registry["agents"]:
                return {"error": "Agent not found"}

            agent_info = self.discovery_system.agent_registry["agents"][agent_name]

            # 从日志中统计
            agent_logs = [
                log for log in self.guidance_log
                if log["agent_name"] == agent_name
            ]

            return {
                "agent_name": agent_name,
                "papers_reviewed": agent_info.get("papers_reviewed", []),
                "total_contributions": agent_info.get("total_contributions", 0),
                "guidance_provided": len(agent_logs),
                "capabilities": agent_info.get("capabilities", []),
            }
        else:
            # 全部Agent统计
            all_stats = {}

            for agent_name in self.discovery_system.agent_registry["agents"]:
                all_stats[agent_name] = self.get_agent_statistics(agent_name)

            return {
                "total_agents": len(all_stats),
                "agents": all_stats,
            }

    def generate_guidance_report(self) -> Dict:
        """生成指导报告"""
        registry_summary = self.discovery_system.registry.get_registry_summary()
        agent_stats = self.get_agent_statistics()

        # 统计未处理的评论
        total_unaddressed = 0
        papers_with_unaddressed = 0

        for paper_id, paper_info in self.discovery_system.registry.registry["papers"].items():
            metadata = self.discovery_system.registry.get_paper_metadata(paper_id)
            if metadata:
                unaddressed = len(metadata.get_unaddressed_comments())
                if unaddressed > 0:
                    papers_with_unaddressed += 1
                    total_unaddressed += unaddressed

        return {
            "generated_at": datetime.now().isoformat(),
            "paper_registry": registry_summary,
            "agent_statistics": agent_stats,
            "unaddressed_comments": {
                "total": total_unaddressed,
                "papers_affected": papers_with_unaddressed,
            },
            "guidance_log_entries": len(self.guidance_log),
        }


class AgentGuidanceAPI:
    """
    Agent指导API

    为外部Agent提供简单的API接口
    """

    def __init__(self, research_dir: str = str(OUTPUT_PATH)):
        """初始化API"""
        self.research_dir = Path(research_dir)
        self.discovery = AgentDiscoverySystem(research_dir)
        self.coordinator = AgentGuidanceCoordinator(research_dir)

    def discover_papers(
        self,
        agent_name: str,
        agent_capabilities: List[str],
        max_papers: int = 5,
    ) -> List[Dict]:
        """
        Agent发现需要指导的论文

        Args:
            agent_name: Agent名称
            agent_capabilities: Agent能力列表
            max_papers: 最大返回数量

        Returns:
            论文列表
        """
        # 先注册Agent能力
        self.discovery.register_agent_capabilities(
            agent_name=agent_name,
            capabilities=agent_capabilities,
            agent_type="external",
            description=f"External agent: {agent_name}",
        )

        # 发现论文
        papers = self.discovery.discover_papers_for_agent(
            agent_name=agent_name,
            max_papers=max_papers,
        )

        return papers

    def submit_guidance(
        self,
        agent_name: str,
        paper_id: str,
        comment: str,
        score: float = None,
        issues: List[str] = None,
        suggestions: List[str] = None,
        priority: str = "medium",
    ) -> Dict:
        """
        提交指导

        Args:
            agent_name: Agent名称
            paper_id: 论文ID
            comment: 评论
            score: 评分
            issues: 问题列表
            suggestions: 建议列表
            priority: 优先级

        Returns:
            提交结果
        """
        guidance = {
            "agent_type": "external",
            "comment": comment,
            "score": score,
            "issues": issues or [],
            "suggestions": suggestions or [],
            "priority": priority,
        }

        return self.coordinator.provide_guidance(
            agent_name=agent_name,
            paper_id=paper_id,
            guidance=guidance,
        )

    def get_paper_info(self, paper_id: str) -> Dict:
        """
        获取论文信息

        Args:
            paper_id: 论文ID

        Returns:
            论文信息
        """
        metadata = self.discovery.registry.get_paper_metadata(paper_id)

        if not metadata:
            return {"error": "Paper not found"}

        return metadata.get_info_for_agents()

    def get_actionable_items(self, paper_id: str) -> List[Dict]:
        """
        获取可执行项

        Args:
            paper_id: 论文ID

        Returns:
            行动项列表
        """
        metadata = self.discovery.registry.get_paper_metadata(paper_id)

        if not metadata:
            return []

        return metadata.get_actionable_items()


# ========================================
# 标准化标记文件生成器
# ========================================

def create_standardized_markers(paper_dir: str):
    """
    为论文文件夹创建标准化标记

    生成让其他Agent能够轻松识别的标记文件

    Args:
        paper_dir: 论文目录
    """
    paper_path = Path(paper_dir)

    # 1. README.md - 论文概览
    readme_path = paper_path / "README.md"
    if not readme_path.exists():
        metadata = PaperMetadata(paper_dir)
        info = metadata.get_info_for_agents()

        readme_content = f"""# {info['paper_info'].get('title', 'Untitled')}

## 📄 论文信息

- **论文ID**: {info['paper_info']['paper_id']}
- **类型**: {info['paper_info'].get('paper_type', 'unknown')}
- **领域**: {info['paper_info'].get('field', 'unknown')}
- **任务**: {info['paper_info'].get('task', 'unknown')}

## 📊 当前状态

- **状态**: {info['status']['current_status']}
- **审查状态**: {info['status']['review_status']}
- **质量等级**: {info['status'].get('quality_level', '未评估')}
- **评分**: {info['status'].get('overall_score', 'N/A')}

## 💬 Agent评论

{info['agent_summary'].get('total_comments', 0)} 条评论

{info['agent_summary'].get('average_score', 'N/A')} 平均分

## 📁 文件结构

- LaTeX文件: {len(info['files'].get('latex', []))} 个
- PDF文件: {len(info['files'].get('pdf', []))} 个
- 审查文件: {len(info['files'].get('review', []))} 个

## 🎯 可执行项

{len(info.get('actionable_items', []))} 个待处理项

---

*此文件由AI Scientist自动生成，供其他Agent参考*
"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

    # 2. .agent_instructions.md - Agent指导说明
    instructions_path = paper_path / ".agent_instructions.md"
    metadata = PaperMetadata(paper_dir)
    actionable_items = metadata.get_actionable_items()

    instructions_content = f"""# Agent指导说明

## 📋 当前需要帮助的方面

"""

    if actionable_items:
        for i, item in enumerate(actionable_items, 1):
            instructions_content += f"""
### {i}. {item['description']}

- **优先级**: {item['priority']}
- **类型**: {item['type']}
- **建议Agent**: {', '.join(item.get('suggested_agents', ['Any']))}

"""
            if item.get('suggestions'):
                instructions_content += f"建议行动:\n"
                for suggestion in item['suggestions']:
                    instructions_content += f"- {suggestion}\n"
                instructions_content += "\n"
    else:
        instructions_content += "\n目前没有紧急待处理项\n"

    instructions_content += f"""
## 📝 如何提供指导

使用AgentGuidanceAPI:

```python
from ai_scientist.paper_metadata import AgentGuidanceAPI

api = AgentGuidanceAPI()

# 1. 查看论文信息
info = api.get_paper_info("{metadata.metadata['paper_id']}")

# 2. 提供指导
api.submit_guidance(
    agent_name="YourAgentName",
    paper_id="{metadata.metadata['paper_id']}",
    comment="你的评论",
    score=4.0,
    issues=["问题1", "问题2"],
    suggestions=["建议1", "建议2"],
    priority="high",
)
```

## 📊 论文详情

{metadata.get_status_summary()}

---

*此文件由AI Scientist自动生成*
"""
    with open(instructions_path, "w", encoding="utf-8") as f:
        f.write(instructions_content)

    # 3. .status_badge.txt - 状态徽章（简单文本）
    badge_path = paper_path / ".status_badge.txt"
    status = metadata.status["current_status"]
    quality = metadata.status.get("quality_level", "unknown").upper()
    score = metadata.status.get("overall_score", 0)

    badge_content = f"""
╔══════════════════════════════════════════════════════════════╗
║                    AI SCIENTIST PAPER                        ║
╠══════════════════════════════════════════════════════════════╣
║  状态: {status:<20}  质量: {quality:<10}               ║
║  评分: {score:.1f}/5.0{' ' * 44}║
╚══════════════════════════════════════════════════════════════╝
"""
    with open(badge_path, "w", encoding="utf-8") as f:
        f.write(badge_content)

    print(f"✅ 已为论文 {paper_path.name} 创建标准化标记")
