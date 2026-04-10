#!/usr/bin/env python3
"""
多层次写作架构系统
提升论文写作质量和专业性
"""
import json
import os
import re
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ai_scientist.llm import create_client, get_response_from_llm


class WritingStrategy(str, Enum):
    """写作策略类型"""
    CONFERENCE_PAPER = "conference_paper"
    WORKSHOP_PAPER = "workshop_paper"
    JOURNAL_PAPER = "journal_paper"
    EXTENDED_ABSTRACT = "extended_abstract"
    TECHNICAL_REPORT = "technical_report"


class WritingFocus(str, Enum):
    """写作焦点"""
    NOVELTY = "novelty"           # 创新性
    CONTRIBUTION = "contribution" # 贡献
    RIGOR = "rigor"              # 严谨性
    CLARITY = "clarity"          # 清晰度
    IMPACT = "impact"            # 影响力
    REPRODUCIBILITY = "reproducibility"  # 可复现性


# 论文结构模板
PAPER_STRUCTURES = {
    WritingStrategy.CONFERENCE_PAPER: {
        "name": "会议论文",
        "sections": [
            "abstract",
            "introduction",
            "related_work",
            "method",
            "experiments",
            "results",
            "discussion",
            "conclusion",
            "references",
        ],
        "page_limit": 8,
        "emphasis": [WritingFocus.NOVELTY, WritingFocus.CONTRIBUTION],
        "style": "formal_academic",
    },

    WritingStrategy.WORKSHOP_PAPER: {
        "name": "工作坊论文",
        "sections": [
            "abstract",
            "introduction",
            "problem_statement",
            "approach",
            "preliminary_results",
            "lessons_learned",
            "future_work",
            "references",
        ],
        "page_limit": 4,
        "emphasis": [WritingFocus.CLARITY, WritingFocus.IMPACT],
        "style": "practical",
    },

    WritingStrategy.JOURNAL_PAPER: {
        "name": "期刊论文",
        "sections": [
            "abstract",
            "introduction",
            "literature_review",
            "theoretical_analysis",
            "method",
            "experiments",
            "results",
            "discussion",
            "limitation",
            "conclusion",
            "appendix",
            "references",
        ],
        "page_limit": 12,
        "emphasis": [
            WritingFocus.RIGOR,
            WritingFocus.REPRODUCIBILITY,
            WritingFocus.CONTRIBUTION,
        ],
        "style": "comprehensive",
    },

    WritingStrategy.EXTENDED_ABSTRACT: {
        "name": "扩展摘要",
        "sections": [
            "abstract",
            "introduction",
            "key_idea",
            "preliminary_results",
            "conclusion",
        ],
        "page_limit": 2,
        "emphasis": [WritingFocus.NOVELTY, WritingFocus.CLARITY],
        "style": "concise",
    },

    WritingStrategy.TECHNICAL_REPORT: {
        "name": "技术报告",
        "sections": [
            "executive_summary",
            "introduction",
            "background",
            "methodology",
            "implementation",
            "experiments",
            "analysis",
            "conclusions",
            "references",
            "appendices",
        ],
        "page_limit": 20,
        "emphasis": [
            WritingFocus.RIGOR,
            WritingFocus.REPRODUCIBILITY,
            WritingFocus.CLARITY,
        ],
        "style": "detailed",
    },
}


# 学术反思框架
ACADEMIC_REFLECTION_FRAMEWORK = {
    "innovation_analysis": {
        "name": "创新性分析",
        "questions": [
            "研究的主要创新点是什么？",
            "与现有工作的关键区别在哪里？",
            "创新点是否具有突破性？",
            "如何量化创新的程度？",
        ],
        "scoring_criteria": {
            "breakthrough": 4,
            "significant": 3,
            "incremental": 2,
            "marginal": 1,
        },
    },

    "rigor_analysis": {
        "name": "严谨性分析",
        "questions": [
            "实验设计是否充分？",
            "基线对比是否全面？",
            "统计显著性是否得到验证？",
            "消融实验是否充分？",
            "结论是否有充分证据支持？",
        ],
        "scoring_criteria": {
            "very_rigorous": 4,
            "rigorous": 3,
            "moderately_rigorous": 2,
            "limited_rigor": 1,
        },
    },

    "depth_analysis": {
        "name": "深度分析",
        "questions": [
            "问题分析是否深入？",
            "是否探讨了根本原因？",
            "理论分析是否充分？",
            "结果分析是否透彻？",
            "是否识别了边界条件？",
        ],
        "scoring_criteria": {
            "very_deep": 4,
            "deep": 3,
            "moderate": 2,
            "shallow": 1,
        },
    },

    "clarity_analysis": {
        "name": "清晰度分析",
        "questions": [
            "论文结构是否清晰？",
            "表达是否准确？",
            "图表是否有助于理解？",
            "逻辑流是否连贯？",
            "目标读者是否能理解？",
        ],
        "scoring_criteria": {
            "very_clear": 4,
            "clear": 3,
            "moderately_clear": 2,
            "confusing": 1,
        },
    },

    "impact_analysis": {
        "name": "影响力分析",
        "questions": [
            "研究是否有实际应用价值？",
            "是否解决了重要问题？",
            "结果是否具有普遍性？",
            "是否能启发后续研究？",
            "社会/经济效益如何？",
        ],
        "scoring_criteria": {
            "high_impact": 4,
            "moderate_impact": 3,
            "limited_impact": 2,
            "niche_impact": 1,
        },
    },
}


class EnhancedWritingEngine:
    """增强的写作引擎"""

    def __init__(
        self,
        strategy: WritingStrategy,
        model: str = "claude-3-5-sonnet",
    ):
        """
        初始化增强写作引擎

        Args:
            strategy: 写作策略
            model: 使用的模型
        """
        self.strategy = strategy
        self.model = model
        self.client, self.client_model = create_client(model)
        self.config = PAPER_STRUCTURES[strategy]

    def generate_section_outline(
        self,
        idea: Dict,
        experiment_results: Dict,
    ) -> Dict:
        """
        生成章节大纲

        Args:
            idea: 研究想法
            experiment_results: 实验结果

        Returns:
            章节大纲
        """
        print(f"\n📝 生成 {self.config['name']} 章节大纲...")

        # 构建提示词
        prompt = f"""
你是资深的学术写作专家，专门撰写{self.config['name']}。

**研究信息**:
标题: {idea.get('Title', '')}
摘要: {idea.get('Abstract', '')}
假设: {idea.get('Hypothesis', '')}
实验: {idea.get('Experiments', '')}

**论文类型**: {self.config['name']}
**页数限制**: {self.config['page_limit']}页
**重点**: {', '.join([f.value for f in self.config['emphasis']])}
**风格**: {self.config['style']}

**要求章节**: {', '.join(self.config['sections'])}

**实验结果摘要**:
{json.dumps(experiment_results.get('summary', {}), indent=2, ensure_ascii=False)}

请为每个章节生成详细的大纲，包括:
1. 章节目标（1-2句话）
2. 关键内容要点（3-5个bullet points）
3. 预期长度（占论文的比例）

以JSON格式返回，格式如下:
{{
  "sections": {{
    "abstract": {{"goal": "...", "points": [...], "length_ratio": 0.05}},
    "introduction": {{"goal": "...", "points": [...], "length_ratio": 0.15}},
    ...
  }},
  "overall_narrative": "论文的整体叙事逻辑",
  "key_contributions": ["主要贡献1", "主要贡献2", ...],
  "target_audience": "目标读者群体"
}}
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message=f"你是资深的学术写作专家，专门撰写{self.config['name']}。",
                temperature=0.5,
            )

            # 提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                outline = json.loads(json_match.group(1))
                print("✅ 章节大纲生成完成")
                return outline
            else:
                print("⚠️  无法提取JSON，返回原始响应")
                return {"raw_response": response}

        except Exception as e:
            print(f"❌ 生成大纲失败: {e}")
            return {}

    def write_section(
        self,
        section_name: str,
        section_outline: Dict,
        context: Dict,
        previous_sections: List[str] = None,
    ) -> str:
        """
        编写单个章节

        Args:
            section_name: 章节名称
            section_outline: 章节大纲
            context: 上下文信息
            previous_sections: 前面的章节内容

        Returns:
            章节内容
        """
        print(f"\n✍️  编写章节: {section_name}")

        # 构建上下文
        context_str = self._build_context(context, previous_sections)

        # 构建提示词
        prompt = f"""
你是资深的学术写作专家。请撰写论文的"{section_name}"章节。

**章节信息**:
目标: {section_outline.get('goal', '')}
关键要点:
{chr(10).join([f"- {p}" for p in section_outline.get('points', [])])}

**论文上下文**:
{context_str}

**写作要求**:
1. 符合{self.config['name']}的学术规范
2. 侧重于: {', '.join([f.value for f in self.config['emphasis']])}
3. 风格: {self.config['style']}
4. 保持与前后章节的逻辑连贯性

**LaTeX格式**:
请使用标准LaTeX格式，包含适当的section、subsection、figure、table引用。

请直接返回完整的LaTeX代码，用```latex包裹。
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message="你是资深的学术写作专家，擅长撰写高质量学术论文。",
                temperature=0.7,
            )

            # 提取LaTeX
            latex_match = re.search(r'```latex\s*(.*?)\s*```', response, re.DOTALL)
            if not latex_match:
                latex_match = re.search(r'\\section\{.*', response, re.DOTALL)

            if latex_match:
                content = latex_match.group(1) if latex_match.lastindex else latex_match.group(0)
                print(f"✅ {section_name} 编写完成")
                return content
            else:
                print(f"⚠️  {section_name} 未能提取LaTeX代码")
                return response

        except Exception as e:
            print(f"❌ {section_name} 编写失败: {e}")
            return ""

    def perform_academic_reflection(
        self,
        paper_content: str,
        focus_areas: List[WritingFocus] = None,
    ) -> Dict:
        """
        执行学术反思

        Args:
            paper_content: 论文内容
            focus_areas: 关注的焦点区域

        Returns:
            反思结果
        """
        if focus_areas is None:
            focus_areas = self.config["emphasis"]

        print(f"\n🔍 执行学术反思...")

        reflection_results = {}

        for focus in focus_areas:
            framework_key = focus.value + "_analysis"
            if framework_key not in ACADEMIC_REFLECTION_FRAMEWORK:
                continue

            framework = ACADEMIC_REFLECTION_FRAMEWORK[framework_key]

            print(f"  分析: {framework['name']}")

            prompt = f"""
请从"{framework['name']}"维度分析以下论文。

**分析维度**: {framework['name']}

**分析问题**:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(framework['questions'])])}

**评分标准**:
{json.dumps(framework['scoring_criteria'], indent=2, ensure_ascii=False)}

**论文内容**（LaTeX格式，前2000字）:
```latex
{paper_content[:10000]}
```

请提供:
1. 评分（按照评分标准）
2. 具体分析（针对每个问题）
3. 改进建议（3-5条）

以JSON格式返回。
"""

            try:
                response, _ = get_response_from_llm(
                    prompt=prompt,
                    client=self.client,
                    model=self.client_model,
                    system_message="你是资深的学术审稿人，具有丰富的论文评审经验。",
                    temperature=0.3,
                )

                # 提取JSON
                json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)

                if json_match:
                    result = json.loads(json_match.group(1))
                    reflection_results[framework['name']] = result
                    print(f"    评分: {result.get('score', 'N/A')}")

            except Exception as e:
                print(f"    ⚠️  {framework['name']} 分析失败: {e}")

        return reflection_results

    def improve_section(
        self,
        section_name: str,
        section_content: str,
        feedback: Dict,
        context: Dict,
    ) -> str:
        """
        基于反馈改进章节

        Args:
            section_name: 章节名称
            section_content: 章节内容
            feedback: 反馈意见
            context: 上下文

        Returns:
            改进后的章节内容
        """
        print(f"\n🔧 改进章节: {section_name}")

        prompt = f"""
你是资深的学术写作专家。请根据反馈意见改进以下章节。

**章节**: {section_name}

**当前内容**:
```latex
{section_content}
```

**反馈意见**:
{json.dumps(feedback, indent=2, ensure_ascii=False)}

**改进要求**:
1. 具体回应每条反馈
2. 保持LaTeX格式正确
3. 保持与论文整体风格一致
4. 提升学术表达质量

请返回改进后的完整LaTeX代码，用```latex包裹。
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message="你是资深的学术写作专家，擅长改进学术论文质量。",
                temperature=0.6,
            )

            # 提取LaTeX
            latex_match = re.search(r'```latex\s*(.*?)\s*```', response, re.DOTALL)
            if latex_match:
                improved_content = latex_match.group(1)
                print(f"✅ {section_name} 改进完成")
                return improved_content
            else:
                print(f"⚠️  {section_name} 改进失败，返回原内容")
                return section_content

        except Exception as e:
            print(f"❌ {section_name} 改进失败: {e}")
            return section_content

    def _build_context(self, context: Dict, previous_sections: List[str] = None) -> str:
        """构建上下文字符串"""
        parts = []

        # 研究信息
        if "idea" in context:
            idea = context["idea"]
            parts.append(f"**研究标题**: {idea.get('Title', '')}")
            parts.append(f"**研究假设**: {idea.get('Hypothesis', '')}")

        # 实验结果
        if "experiments" in context:
            parts.append(f"**主要发现**: {context['experiments'].get('summary', {})}")

        # 前面章节
        if previous_sections:
            parts.append("**前面章节**:")
            for sec in previous_sections[-2:]:  # 只包含最近2个章节
                parts.append(f"- {sec}")

        return "\n\n".join(parts)


class WritingQualityAssessor:
    """写作质量评估器"""

    @staticmethod
    def assess_paper_quality(paper_dir: str) -> Dict:
        """
        评估论文质量

        Args:
            paper_dir: 论文目录

        Returns:
            质量评估结果
        """
        # 读取论文信息
        idea_file = Path(paper_dir) / "idea.json"
        latex_file = Path(paper_dir) / "latex" / "template.tex"

        if not idea_file.exists() or not latex_file.exists():
            return {"error": "缺少必要文件"}

        with open(idea_file, "r") as f:
            idea = json.load(f)

        with open(latex_file, "r", encoding="utf-8", errors="ignore") as f:
            latex_content = f.read()

        # 评估维度
        assessments = {
            "structure": WritingQualityAssessor._assess_structure(latex_content),
            "content": WritingQualityAssessor._assess_content(latex_content, idea),
            "innovation": WritingQualityAssessor._assess_innovation(latex_content, idea),
            "rigor": WritingQualityAssessor._assess_rigor(latex_content),
            "clarity": WritingQualityAssessor._assess_clarity(latex_content),
        }

        # 计算总体质量
        total_score = sum(a.get("score", 0) for a in assessments.values())
        overall_quality = total_score / len(assessments)

        return {
            "overall_quality": overall_quality,
            "assessments": assessments,
            "recommendations": WritingQualityAssessor._generate_recommendations(assessments),
        }

    @staticmethod
    def _assess_structure(latex_content: str) -> Dict:
        """评估论文结构"""
        score = 0
        issues = []

        # 检查必需章节
        required_sections = [
            r"\\section\{.*Abstract.*\}",
            r"\\section\{.*Introduction.*\}",
            r"\\section\{.*Method.*\}",
            r"\\section\{.*Experiment.*\}",
            r"\\section\{.*Conclusion.*\}",
        ]

        found_sections = 0
        for section_pattern in required_sections:
            if re.search(section_pattern, latex_content, re.IGNORECASE):
                found_sections += 1
            else:
                issues.append(f"缺少章节: {section_pattern}")

        score = (found_sections / len(required_sections)) * 4

        # 检查引用
        if "\\cite{" in latex_content:
            score += 0.5
        else:
            issues.append("缺少引用")

        # 检查图表
        if "\\begin{figure}" in latex_content or "\\begin{table}" in latex_content:
            score += 0.5
        else:
            issues.append("缺少图表")

        return {
            "score": min(score, 4),
            "issues": issues,
            "found_sections": found_sections,
        }

    @staticmethod
    def _assess_content(latex_content: str, idea: Dict) -> Dict:
        """评估内容质量"""
        score = 2  # 基础分
        issues = []

        # 检查字数（估算）
        word_count = len(latex_content.split())
        if word_count < 1000:
            issues.append("内容过短")
            score -= 1
        elif word_count > 5000:
            score += 1

        # 检查是否包含研究要素
        required_elements = [
            ("假设", idea.get("Hypothesis", "")),
            ("实验", idea.get("Experiments", "")),
        ]

        for elem_name, elem_content in required_elements:
            if elem_content and elem_content.lower() in latex_content.lower():
                score += 0.5
            else:
                issues.append(f"缺少{elem_name}相关内容")

        return {
            "score": min(score, 4),
            "issues": issues,
            "word_count": word_count,
        }

    @staticmethod
    def _assess_innovation(latex_content: str, idea: Dict) -> Dict:
        """评估创新性"""
        score = 2  # 基础分
        innovations = []

        # 检查创新性关键词
        innovation_keywords = [
            "novel", "new", "innovative", "first", "state-of-the-art",
            "advance", "breakthrough", "unprecedented", "unique"
        ]

        keyword_count = sum(
            1 for kw in innovation_keywords
            if kw in latex_content.lower()
        )

        if keyword_count >= 5:
            score += 1
            innovations.append(f"包含{keyword_count}个创新性关键词")
        else:
            innovations.append("创新性关键词较少")

        # 检查与现有工作的区别
        if "different from" in latex_content.lower() or "unlike" in latex_content.lower():
            score += 0.5
            innovations.append("明确说明了与现有工作的区别")
        else:
            innovations.append("未明确说明与现有工作的区别")

        return {
            "score": min(score, 4),
            "innovations": innovations,
        }

    @staticmethod
    def _assess_rigor(latex_content: str) -> Dict:
        """评估严谨性"""
        score = 2  # 基础分
        rigor_indicators = []

        # 检查严谨性指标
        rigor_keywords = [
            "baseline", "comparison", "statistical", "significant",
            "ablation", "robust", "reproducible", "consistent"
        ]

        keyword_count = sum(
            1 for kw in rigor_keywords
            if kw in latex_content.lower()
        )

        if keyword_count >= 4:
            score += 1
            rigor_indicators.append(f"包含{keyword_count}个严谨性指标")
        else:
            rigor_indicators.append("严谨性指标不足")

        # 检查数值结果
        if re.search(r'\d+\.\d+', latex_content):
            score += 0.5
            rigor_indicators.append("包含定量结果")
        else:
            rigor_indicators.append("缺乏定量结果")

        # 检查实验设置
        if "setup" in latex_content.lower() or "implementation" in latex_content.lower():
            score += 0.5
            rigor_indicators.append("描述了实验设置")

        return {
            "score": min(score, 4),
            "indicators": rigor_indicators,
        }

    @staticmethod
    def _assess_clarity(latex_content: str) -> Dict:
        """评估清晰度"""
        score = 3  # 基础分
        clarity_issues = []

        # 检查句子长度
        sentences = re.split(r'[.!?]+', latex_content)
        long_sentences = sum(
            1 for s in sentences
            if len(s.split()) > 30
        )

        if long_sentences > len(sentences) * 0.2:
            clarity_issues.append(f"有{long_sentences}个长句子")
            score -= 1
        else:
            clarity_issues.append("句子长度适中")

        # 检查段落结构
        paragraphs = latex_content.split('\n\n')
        if len(paragraphs) > 50:
            score += 0.5
            clarity_issues.append("段落结构清晰")
        else:
            clarity_issues.append("段落可能需要优化")

        # 检查LaTeX语法
        latex_errors = []
        if latex_content.count('\\begin{') != latex_content.count('\\end{'):
            latex_errors.append("begin/end不匹配")
            score -= 1

        if latex_errors:
            clarity_issues.extend(latex_errors)
        else:
            clarity_issues.append("LaTeX语法正确")
            score += 0.5

        return {
            "score": max(min(score, 4), 0),
            "issues": clarity_issues,
        }

    @staticmethod
    def _generate_recommendations(assessments: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于各维度评分生成建议
        for dimension, assessment in assessments.items():
            score = assessment.get("score", 0)

            if score < 2:
                recommendations.append(f"重点改进{dimension}维度（当前评分: {score}/4）")
            elif score < 3:
                recommendations.append(f"优化{dimension}维度（当前评分: {score}/4）")

        # 通用建议
        overall_avg = sum(a.get("score", 0) for a in assessments.values()) / len(assessments)

        if overall_avg >= 3.5:
            recommendations.append("✅ 论文质量优秀，可以考虑投稿")
        elif overall_avg >= 2.5:
            recommendations.append("📝 论文质量良好，建议进一步优化")
        else:
            recommendations.append("💪 论文需要重点改进")

        return recommendations


def get_writing_strategy(paper_type: str) -> WritingStrategy:
    """根据论文类型获取写作策略"""
    strategy_map = {
        "icbinb": WritingStrategy.WORKSHOP_PAPER,
        "normal": WritingStrategy.CONFERENCE_PAPER,
        "journal": WritingStrategy.JOURNAL_PAPER,
        "extended": WritingStrategy.EXTENDED_ABSTRACT,
    }

    return strategy_map.get(paper_type, WritingStrategy.CONFERENCE_PAPER)
