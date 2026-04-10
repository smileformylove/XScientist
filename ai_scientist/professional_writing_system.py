#!/usr/bin/env python3
"""
专业学术写作系统
提供顶会级别的论文写作能力
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.writing_prompt_profiles import (
    DEFAULT_WRITING_PROFILE,
    normalize_writing_profile,
    render_writing_profile_self_checks,
    render_writing_profile_system_guidance,
)


# ========================================
# 论文模板库
# ========================================

PAPER_TEMPLATES = {
    "neurips": {
        "name": "NeurIPS",
        "page_limit": 8,
        "style": "theoretical_ml",
        "sections": {
            "abstract": {
                "order": 1,
                "length": "150-250 words",
                "structure": [
                    "Background and motivation (1-2 sentences)",
                    "Problem statement (1 sentence)",
                    "Key approach/method (2-3 sentences)",
                    "Main results (1-2 sentences)",
                    "Impact and implications (1 sentence)",
                ],
                "tips": [
                    "Be concise and precise",
                    "Avoid citations in abstract",
                    "Use present tense for established facts",
                    "Use past tense for your findings",
                    "Emphasize novelty and contribution",
                ]
            },
            "introduction": {
                "order": 2,
                "length": "1-1.5 pages",
                "structure": [
                    "Broad context and motivation",
                    "Specific problem definition",
                    "Current limitations/gaps",
                    "Our approach and key insights",
                    "Contributions (bulleted list)",
                    "Paper organization",
                ],
                "tips": [
                    "Start with accessible motivation",
                    "Clearly define the problem",
                    "Build intuitive understanding before technical details",
                    "Use concrete examples when possible",
                    "End with clear contribution statement",
                ]
            },
            "related_work": {
                "order": 3,
                "length": "1-1.5 pages",
                "structure": [
                    "Organize by themes or approaches",
                    "Compare and contrast key methods",
                    "Highlight limitations of prior work",
                    "Position your contribution uniquely",
                ],
                "tips": [
                    "Group related papers thematically",
                    "Focus on insights, not just listing",
                    "Be respectful but clear about limitations",
                    "Position your work as filling gaps",
                    "Consider using a comparison table",
                ]
            },
            "method": {
                "order": 4,
                "length": "2-3 pages",
                "structure": [
                    "Problem formulation",
                    "Overall approach/architecture",
                    "Detailed algorithm or model",
                    "Theoretical analysis (if applicable)",
                    "Implementation details",
                ],
                "tips": [
                    "Use clear notation and define it early",
                    "Include algorithmic pseudocode",
                    "Provide theoretical justifications",
                    "Use figures to illustrate architecture",
                    "Be detailed enough for reproducibility",
                ]
            },
            "experiments": {
                "order": 5,
                "length": "2-3 pages",
                "structure": [
                    "Experimental setup",
                    "Datasets",
                    "Baselines",
                    "Evaluation metrics",
                    "Main results",
                    "Ablation studies",
                    "Additional analyses",
                ],
                "tips": [
                    "Describe setup thoroughly for reproducibility",
                    "Use strong and diverse baselines",
                    "Report statistical significance",
                    "Visualize results clearly",
                    "Ablation studies are crucial",
                ]
            },
            "results": {
                "order": 6,
                "length": "1-1.5 pages",
                "structure": [
                    "Overview of findings",
                    "Detailed results analysis",
                    "Comparison with baselines",
                    "Ablation study results",
                    "Error analysis or case studies",
                ],
                "tips": [
                    "Start with key findings",
                    "Connect results to hypotheses",
                    "Explain why your method works",
                    "Discuss limitations honestly",
                    "Use visualizations effectively",
                ]
            },
            "discussion": {
                "order": 7,
                "length": "0.5-1 page",
                "structure": [
                    "Broader implications",
                    "Limitations",
                    "Future work directions",
                ],
                "tips": [
                    "Connect to bigger picture",
                    "Be honest about limitations",
                    "Suggest concrete future directions",
                    "Avoid overclaiming",
                ]
            },
            "conclusion": {
                "order": 8,
                "length": "0.25-0.5 page",
                "structure": [
                    "Summary of contributions",
                    "Broader impact statement",
                ],
                "tips": [
                    "Keep it brief and impactful",
                    "Include broader impact (NeurIPS requirement)",
                    "Don't just repeat the abstract",
                ]
            },
        }
    },

    "iclr": {
        "name": "ICLR",
        "page_limit": 8,
        "style": "representational_learning",
        "sections": {
            "abstract": {
                "order": 1,
                "length": "150-250 words",
                "structure": [
                    "Context and problem",
                    "Key insight or intuition",
                    "Method overview",
                    "Main contributions",
                    "Results and impact",
                ],
                "tips": [
                    "Emphasize the learning aspect",
                    "Highlight representational insights",
                    "Focus on intuition",
                ]
            },
            "introduction": {
                "order": 2,
                "length": "1-1.5 pages",
                "structure": [
                    "Background and motivation",
                    "Problem formulation",
                    "Key challenges",
                    "Our insights and approach",
                    "Contributions",
                ],
                "tips": [
                    "Focus on learning representations",
                    "Emphasize theoretical insights",
                    "Build intuition gradually",
                ]
            },
            "background": {
                "order": 3,
                "length": "0.5-1 page",
                "structure": [
                    "Essential preliminaries",
                    "Key concepts and notation",
                    "Relevant prior work summary",
                ],
                "tips": [
                    "Keep it focused and essential",
                    "Define notation clearly",
                    "Avoid redundancy with related work",
                ]
            },
            "method": {
                "order": 4,
                "length": "2.5-3.5 pages",
                "structure": [
                    "Problem formalization",
                    "Proposed approach",
                    "Algorithmic details",
                    "Theoretical foundation",
                    "Analysis",
                ],
                "tips": [
                    "Strong theoretical emphasis",
                    "Clear mathematical formulation",
                    "Include convergence or complexity analysis",
                    "Connect representation to learning",
                ]
            },
            "experiments": {
                "order": 5,
                "length": "2-3 pages",
                "structure": [
                    "Setup",
                    "Main results",
                    "Ablation studies",
                    "Analysis",
                ],
                "tips": [
                    "Multiple datasets preferred",
                    "Strong baselines",
                    "Ablation studies essential",
                    "Analyze learned representations",
                ]
            },
            "related_work": {
                "order": 6,
                "length": "1 page",
                "structure": [
                    "Thematic organization",
                    "Detailed comparison",
                    "Positioning",
                ],
                "tips": [
                    "Can be after experiments",
                    "Focus on learning methods",
                    "Compare representational approaches",
                ]
            },
            "conclusion": {
                "order": 7,
                "length": "0.5 page",
                "structure": [
                    "Summary",
                    "Limitations",
                    "Future work",
                    "Broader impact",
                ],
                "tips": [
                    "Include broader impact statement",
                    "Discuss societal implications",
                ]
            },
        }
    },

    "cvpr": {
        "name": "CVPR",
        "page_limit": 8,
        "style": "computer_vision",
        "sections": {
            "abstract": {
                "order": 1,
                "length": "150-250 words",
                "structure": [
                    "Visual task motivation",
                    "Technical challenge",
                    "Proposed solution",
                    "Key innovation",
                    "Performance gains",
                ],
                "tips": [
                    "Emphasize visual aspects",
                    "Mention computational efficiency",
                    "Highlight real-world applicability",
                ]
            },
            "introduction": {
                "order": 2,
                "length": "1 page",
                "structure": [
                    "Visual task context",
                    "Problem definition",
                    "Current limitations",
                    "Our approach",
                    "Contributions",
                ],
                "tips": [
                    "Include visual motivation",
                    "Be specific about visual challenges",
                    "Emphasize efficiency if applicable",
                ]
            },
            "related_work": {
                "order": 3,
                "length": "1-1.5 pages",
                "structure": [
                    "Task-specific methods",
                    "General vision methods",
                    "Comparison",
                ],
                "tips": [
                    "Cover both task and general methods",
                    "Include recent CVPR/ICCV/ECCV work",
                    "Use comparison table",
                ]
            },
            "method": {
                "order": 4,
                "length": "2.5-3.5 pages",
                "structure": [
                    "Overview",
                    "Network architecture",
                    "Key components",
                    "Loss functions",
                    "Training procedure",
                ],
                "tips": [
                    "Include architecture diagrams",
                    "Detailed layer specifications",
                    "Visualize key operations",
                    "Computational complexity analysis",
                ]
            },
            "experiments": {
                "order": 5,
                "length": "2-3 pages",
                "structure": [
                    "Datasets",
                    "Implementation details",
                    "State-of-the-art comparison",
                    "Ablation studies",
                    "Qualitative results",
                ],
                "tips": [
                    "Multiple standard datasets",
                    "Qualitative visualizations",
                    "Timing/memory analysis",
                    "Per-class breakdown if classification",
                ]
            },
            "conclusion": {
                "order": 6,
                "length": "0.25 page",
                "structure": [
                    "Summary",
                    "Future work",
                ],
                "tips": [
                    "Keep very brief",
                    "Focus on practical impact",
                ]
            },
        }
    },

    "icbinb": {
        "name": "ICBINB (ICLR Workshop)",
        "page_limit": 4,
        "style": "workshop",
        "sections": {
            "abstract": {
                "order": 1,
                "length": "100-150 words",
                "structure": [
                    "Problem and motivation",
                    "Approach",
                    "Key result",
                ],
                "tips": [
                    "Very concise",
                    "Focus on key contribution",
                ]
            },
            "introduction": {
                "order": 2,
                "length": "0.75-1 page",
                "structure": [
                    "Motivation",
                    "Problem",
                    "Approach overview",
                    "Summary of results",
                ],
                "tips": [
                    "Get to the point quickly",
                    "Combine related work here",
                ]
            },
            "method": {
                "order": 3,
                "length": "1.5-2 pages",
                "structure": [
                    "Approach",
                    "Key details",
                ],
                "tips": [
                    "Focus on essentials",
                    "Use figures efficiently",
                ]
            },
            "experiments": {
                "order": 4,
                "length": "1-1.5 pages",
                "structure": [
                    "Setup",
                    "Results",
                ],
                "tips": [
                    "Combine setup with results",
                    "Focus on key findings",
                    "Essential ablations only",
                ]
            },
            "conclusion": {
                "order": 5,
                "length": "0.25 page",
                "structure": [
                    "Summary",
                ],
                "tips": [
                    "Just a few sentences",
                ]
            },
        }
    },

    "journal": {
        "name": "Journal (JMLR/TPAMI/Pattern Recognition)",
        "page_limit": 12,
        "style": "comprehensive",
        "sections": {
            "abstract": {
                "order": 1,
                "length": "200-300 words",
                "structure": [
                    "Extended background",
                    "Problem statement",
                    "Comprehensive approach summary",
                    "Detailed findings",
                    "Implications",
                ],
                "tips": [
                    "Can be more detailed",
                    "Include broader context",
                ]
            },
            "introduction": {
                "order": 2,
                "length": "2-3 pages",
                "structure": [
                    "Comprehensive background",
                    "Problem motivation",
                    "Literature preview",
                    "Our contributions",
                    "Paper roadmap",
                ],
                "tips": [
                    "Thorough literature context",
                    "Clear significance statement",
                    "Detailed roadmap",
                ]
            },
            "related_work": {
                "order": 3,
                "length": "2-3 pages",
                "structure": [
                    "Comprehensive literature review",
                    "Historical development",
                    "Taxonomy of approaches",
                    "Detailed comparisons",
                    "Gap analysis",
                ],
                "tips": [
                    "Comprehensive coverage",
                    "Historical perspective",
                    "Critical analysis",
                    "Clear positioning",
                ]
            },
            "background": {
                "order": 4,
                "length": "1-2 pages",
                "structure": [
                    "Theoretical foundations",
                    "Essential concepts",
                    "Notation and definitions",
                ],
                "tips": [
                    "Self-contained background",
                    "Mathematical foundations",
                ]
            },
            "method": {
                "order": 5,
                "length": "3-4 pages",
                "structure": [
                    "Problem formulation",
                    "Detailed approach",
                    "Theoretical analysis",
                    "Algorithmic specification",
                    "Complexity analysis",
                ],
                "tips": [
                    "Rigorous treatment",
                    "Complete specifications",
                    "Strong theory",
                    "Reproducibility focus",
                ]
            },
            "experiments": {
                "order": 6,
                "length": "3-4 pages",
                "structure": [
                    "Comprehensive setup",
                    "Multiple datasets",
                    "Extensive baselines",
                    "Detailed results",
                    "Ablation studies",
                    "Sensitivity analysis",
                    "Case studies",
                ],
                "tips": [
                    "Thorough evaluation",
                    "Multiple scenarios",
                    "Statistical rigor",
                    "Comprehensive analysis",
                ]
            },
            "discussion": {
                "order": 7,
                "length": "1-1.5 pages",
                "structure": [
                    "Interpretation",
                    "Implications",
                    "Limitations",
                    "Future directions",
                ],
                "tips": [
                    "Deep interpretation",
                    "Broader implications",
                    "Honest limitations",
                    "Concrete future work",
                ]
            },
            "conclusion": {
                "order": 8,
                "length": "0.5 page",
                "structure": [
                    "Summary",
                    "Broader impact",
                ],
                "tips": [
                    "Comprehensive summary",
                    "Societal implications",
                ]
            },
        }
    },
    "nature": {
        "name": "Nature-style Research Article",
        "page_limit": 8,
        "style": "high_impact_science",
        "sections": {
            "abstract": {
                "order": 1,
                "length": "120-180 words",
                "structure": [
                    "Big-picture problem and significance",
                    "Core discovery or insight",
                    "Key evidence/results",
                    "Broader implication",
                ],
                "tips": [
                    "Lead with why the problem matters broadly",
                    "Keep the claim strong but evidence-backed",
                    "Use minimal jargon",
                ]
            },
            "introduction": {
                "order": 2,
                "length": "1-1.5 pages",
                "structure": [
                    "Broad scientific context",
                    "Gap or unresolved challenge",
                    "Why it matters beyond a niche subfield",
                    "Our key advance",
                ],
                "tips": [
                    "Build significance early",
                    "Avoid overloading with narrow literature",
                    "End with a crisp statement of advance",
                ]
            },
            "results": {
                "order": 3,
                "length": "3-4 pages",
                "structure": [
                    "Primary empirical result",
                    "Key comparison or validation",
                    "Mechanistic or explanatory analysis",
                    "Robustness evidence",
                ],
                "tips": [
                    "Center the narrative on the strongest evidence",
                    "Use figures/tables to support every major claim",
                    "Prefer crisp story over exhaustive enumeration",
                ]
            },
            "discussion": {
                "order": 4,
                "length": "1-1.5 pages",
                "structure": [
                    "Interpretation of main findings",
                    "Implications for the field",
                    "Limitations and boundaries",
                    "Future directions",
                ],
                "tips": [
                    "Be ambitious about impact but honest about scope",
                    "Explicitly discuss limitations",
                    "Connect to broader scientific significance",
                ]
            },
            "methods": {
                "order": 5,
                "length": "2-3 pages",
                "structure": [
                    "Experimental setup",
                    "Implementation details",
                    "Evaluation protocol",
                    "Statistical procedures",
                ],
                "tips": [
                    "Make reproducibility straightforward",
                    "State statistical and robustness protocols explicitly",
                ]
            },
        }
    },
}


# ========================================
# 学术写作标准
# ========================================

ACADEMIC_WRITING_STANDARDS = {
    "formal_language": {
        "use": [
            "Precise technical terminology",
            "Complete sentences",
            "Active voice for clarity",
            "Third person or first person plural ('we')",
            "Present tense for general truths",
            "Past tense for methods and results",
        ],
        "avoid": [
            "Colloquial expressions",
            "Subjective language without evidence",
            "Overstatements ('always', 'never', 'prove')",
            "First person singular ('I')",
            "Rhetorical questions",
            "Emotional language",
        ]
    },
    "structure_guidelines": {
        "paragraph_structure": [
            "Topic sentence introducing main idea",
            "Supporting sentences with evidence",
            "Analysis/interpretation",
            "Transition/concluding sentence",
        ],
        "section_transitions": [
            "Connect to previous section",
            "Introduce current section",
            "Preview next section",
        ],
        "logical_flow": [
            "Clear problem-solution structure",
            "Coherent argument progression",
            "Explicit justifications for choices",
        ]
    },
    "writing_style": {
        "clarity": [
            "Define notation before use",
            "Explain technical concepts",
            "Use examples for complex ideas",
            "Avoid unnecessary jargon",
        ],
        "precision": [
            "Quantitative statements with numbers",
            "Specific rather than general",
            "Qualify statements appropriately",
            "Report uncertainties",
        ],
        "conciseness": [
            "Remove redundancy",
            "Prefer simple over complex",
            "Avoid wordy phrases",
            "Combine related ideas",
        ]
    },
    "technical_writing": {
        "mathematics": [
            "Define all symbols",
            "Number important equations",
            "Explain equation meanings",
            "Check dimensional consistency",
        ],
        "figures": [
            "Informative captions",
            "Clear labels and legends",
            "High resolution",
            "Referenced in text",
        ],
        "tables": [
            "Descriptive captions",
            "Clear column headers",
            "Appropriate precision",
            "Referred to in text",
        ],
        "algorithms": [
            "Clear pseudocode",
            "Consistent notation",
            "Explained in text",
        ]
    },
    "common_mistakes": {
        "avoid": [
            "Unsubstantiated claims",
            "Missing citations",
            "Ambiguous references",
            "Inconsistent notation",
            "Unclear antecedents",
            "Missing conclusions",
            "Weak transitions",
            "Passive voice overuse",
        ]
    }
}


# ========================================
# 专业章节写作器
# ========================================

class ExpertSectionWriter:
    """专家级章节写作器"""

    def __init__(
        self,
        template: str = "neurips",
        model: str = "claude-3-5-sonnet",
        writing_profile: str = DEFAULT_WRITING_PROFILE,
    ):
        """
        初始化章节写作器

        Args:
            template: 论文模板类型
            model: 使用的LLM模型
        """
        self.template_name = template
        self.template = PAPER_TEMPLATES.get(template, PAPER_TEMPLATES["neurips"])
        self.model = model
        try:
            self.writing_profile = normalize_writing_profile(writing_profile)
        except ValueError as exc:
            print(
                f"⚠️  Invalid writing profile '{writing_profile}': {exc}. "
                f"Falling back to {DEFAULT_WRITING_PROFILE}."
            )
            self.writing_profile = DEFAULT_WRITING_PROFILE
        self.profile_system_guidance = render_writing_profile_system_guidance(
            self.writing_profile
        )
        self.profile_self_checks = render_writing_profile_self_checks(
            self.writing_profile
        )
        self.client, self.client_model = create_client(model)

    def generate_detailed_outline(
        self,
        idea: Dict,
        experiment_results: Dict = None,
    ) -> Dict:
        """
        生成详细的章节大纲

        Args:
            idea: 研究想法
            experiment_results: 实验结果（可选）

        Returns:
            详细大纲
        """
        print(f"\n📝 生成详细大纲 ({self.template['name']} 模板)...")

        prompt = f"""
请根据以下研究想法，生成详细的论文大纲。

**研究想法**:
标题: {idea.get('Title', '')}
摘要: {idea.get('Abstract', '')}
假设: {idea.get('Hypothesis', '')}
方法: {idea.get('Method', '')}

**论文模板**: {self.template['name']}
**页数限制**: {self.template['page_limit']} pages

**章节结构**:
{json.dumps(self.template['sections'], indent=2, ensure_ascii=False)}

请生成详细的章节大纲，包括:
1. 每个章节的核心内容
2. 关键论点
3. 预期的贡献点
4. 图表建议
5. 篇幅分配

以JSON格式返回。
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message=(
                    f"你是资深的学术写作专家，擅长撰写{self.template['name']}级别的论文。\n\n"
                    f"{self.profile_system_guidance}"
                ),
                temperature=0.5,
            )

            # 解析响应
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                outline = json.loads(json_match.group(1))
                print("✅ 详细大纲生成完成")
                return outline
            else:
                return {"raw_response": response}

        except Exception as e:
            print(f"❌ 大纲生成失败: {e}")
            return {}

    def write_section(
        self,
        section_name: str,
        idea: Dict,
        context: Dict,
        feedback: str = None,
    ) -> str:
        """
        编写单个章节

        Args:
            section_name: 章节名称
            idea: 研究想法
            context: 上下文信息（包括其他章节内容、实验结果等）
            feedback: 反馈意见（可选）

        Returns:
            章节内容
        """
        print(f"\n✍️  编写章节: {section_name}")

        # 获取章节模板
        section_template = self.template.get("sections", {}).get(section_name)
        if not section_template:
            print(f"⚠️  未找到章节模板: {section_name}")
            return ""

        # 构建提示
        prompt = self._build_section_writing_prompt(
            section_name,
            section_template,
            idea,
            context,
            feedback,
        )

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message=f"""
你是资深的学术写作专家，专门撰写{self.template['name']}论文的{section_name}章节。

**写作要求**:
1. 严格遵守{self.template['name']}的写作风格和结构要求
2. 使用正式、精确的学术语言
3. 确保逻辑连贯、论证有力
4. 包含必要的引用和图表引用
5. 篇幅控制在{section_template.get('length', '1-2 pages')}

**学术写作标准**:
{json.dumps(ACADEMIC_WRITING_STANDARDS, indent=2, ensure_ascii=False)}

{self.profile_system_guidance}
""",
                temperature=0.7,
            )

            print(f"✅ {section_name} 章节编写完成")
            return response

        except Exception as e:
            print(f"❌ {section_name} 章节编写失败: {e}")
            return ""

    def _build_section_writing_prompt(
        self,
        section_name: str,
        section_template: Dict,
        idea: Dict,
        context: Dict,
        feedback: str = None,
    ) -> str:
        """构建章节写作提示"""

        prompt = f"""
请为以下研究想法撰写 **{section_name}** 章节。

**研究想法**:
标题: {idea.get('Title', '')}
摘要: {idea.get('Abstract', '')}
假设: {idea.get('Hypothesis', '')}

**章节要求**:
- 篇幅: {section_template.get('length', '1-2 pages')}
- 结构: {json.dumps(section_template.get('structure', []), indent=2, ensure_ascii=False)}
- 写作要点: {json.dumps(section_template.get('tips', []), indent=2, ensure_ascii=False)}

"""

        # 添加上下文
        if context.get("previous_sections"):
            prompt += f"\n**前面章节内容**:\n{context['previous_sections']}\n"

        if context.get("experiment_results"):
            prompt += f"\n**实验结果**:\n{json.dumps(context['experiment_results'], indent=2, ensure_ascii=False)}\n"

        if context.get("outline"):
            prompt += f"\n**详细大纲**:\n{json.dumps(context['outline'], indent=2, ensure_ascii=False)}\n"

        # 添加反馈
        if feedback:
            prompt += f"\n**改进反馈**:\n{feedback}\n"

        prompt += """
请撰写完整的章节内容，要求:
1. 符合学术写作规范
2. 包含必要的LaTeX格式
3. 逻辑清晰、论证严密
4. 适合直接用于论文

直接返回LaTeX内容，不要使用代码块。
"""
        prompt += f"\n\n{self.profile_self_checks}\n"

        return prompt

    def refine_section(
        self,
        section_name: str,
        current_content: str,
        feedback: str,
        context: Dict,
    ) -> str:
        """
        改进章节内容

        Args:
            section_name: 章节名称
            current_content: 当前内容
            feedback: 反馈意见
            context: 上下文

        Returns:
            改进后的内容
        """
        print(f"\n🔧 改进章节: {section_name}")

        prompt = f"""
请根据反馈意见改进以下章节内容。

**章节**: {section_name}

**当前内容**:
```latex
{current_content}
```

**反馈意见**:
{feedback}

**改进要求**:
1. 直接应用反馈中的改进建议
2. 保持学术写作标准
3. 确保与前后章节连贯
4. 保持LaTeX格式正确

请返回改进后的完整LaTeX内容。
{self.profile_self_checks}
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message="你是资深的学术写作专家，擅长根据反馈改进论文内容。",
                temperature=0.6,
            )

            print(f"✅ {section_name} 改进完成")
            return response

        except Exception as e:
            print(f"❌ {section_name} 改进失败: {e}")
            return current_content

    def write_full_paper(
        self,
        idea: Dict,
        experiment_results: Dict = None,
        iterative: bool = True,
    ) -> str:
        """
        编写完整论文

        Args:
            idea: 研究想法
            experiment_results: 实验结果
            iterative: 是否迭代式写作（每个章节都基于前面章节）

        Returns:
            完整论文LaTeX
        """
        print(f"\n📚 开始编写完整论文 ({self.template['name']} 模板)...")

        # 生成详细大纲
        outline = self.generate_detailed_outline(idea, experiment_results)
        context = {"outline": outline, "experiment_results": experiment_results}

        # 按顺序编写章节
        sections_content = {}
        section_order = sorted(
            self.template['sections'].items(),
            key=lambda x: x[1]['order']
        )

        for section_name, section_info in section_order:
            print(f"\n{'='*60}")
            print(f"编写章节: {section_name}")
            print(f"{'='*60}")

            # 添加前面章节的上下文
            if iterative:
                previous = "\n\n".join([
                    f"\\section{{{name}}}\n{content}"
                    for name, content in sections_content.items()
                ])
                context["previous_sections"] = previous[:2000]  # 限制长度

            # 编写章节
            content = self.write_section(
                section_name=section_name,
                idea=idea,
                context=context,
            )

            sections_content[section_name] = content

        # 组装完整论文
        full_paper = self._assemble_paper(sections_content, idea)

        print(f"\n✅ 完整论文编写完成")
        return full_paper

    def _assemble_paper(
        self,
        sections_content: Dict[str, str],
        idea: Dict,
    ) -> str:
        """组装完整论文"""

        # 论文开头
        header = f"""\\documentstyle{{{self.template.get('style', 'article')}}}

\\title{{{idea.get('Title', 'Untitled')}}}

\\author{{Author Name}}

\\begin{{document}}

\\maketitle

"""

        # 组装章节
        body = ""
        section_order = sorted(
            self.template['sections'].items(),
            key=lambda x: x[1]['order']
        )

        for section_name, _ in section_order:
            if section_name in sections_content:
                body += f"\\section{{{section_name.capitalize()}}}\n"
                body += sections_content[section_name]
                body += "\n\n"

        # 论文结尾
        footer = """
\\end{document}
"""

        return header + body + footer


# ========================================
# 专业论文评估器
# ========================================

class ProfessionalPaperEvaluator:
    """专业论文质量评估器"""

    def __init__(self, template: str = "neurips", model: str = "gpt-4o"):
        """
        初始化评估器

        Args:
            template: 论文模板
            model: 评估模型
        """
        self.template_name = template
        self.template = PAPER_TEMPLATES.get(template, PAPER_TEMPLATES["neurips"])
        self.model = model
        self.client, self.client_model = create_client(model)

    def evaluate_paper_quality(
        self,
        paper_content: str,
        idea: Dict = None,
    ) -> Dict:
        """
        全面评估论文质量

        Args:
            paper_content: 论文内容
            idea: 研究想法

        Returns:
            评估结果
        """
        print(f"\n📊 评估论文质量 ({self.template['name']} 标准)...")

        evaluation = {}

        # 多维度评估
        dimensions = [
            "structure",
            "content",
            "innovation",
            "rigor",
            "clarity",
            "professionalism",
        ]

        for dimension in dimensions:
            print(f"  评估维度: {dimension}")
            result = self._evaluate_dimension(
                dimension,
                paper_content,
                idea,
            )
            evaluation[dimension] = result

        # 计算总分
        scores = [
            d.get("score", 0)
            for d in evaluation.values()
            if isinstance(d.get("score"), (int, float))
        ]
        overall_score = sum(scores) / len(scores) if scores else 0

        evaluation["overall"] = {
            "score": overall_score,
            "level": self._get_quality_level(overall_score),
            "strengths": self._identify_strengths(evaluation),
            "weaknesses": self._identify_weaknesses(evaluation),
            "recommendations": self._generate_recommendations(evaluation),
        }

        print(f"\n✅ 评估完成")
        print(f"   总分: {overall_score:.1f}/5")
        print(f"   等级: {evaluation['overall']['level']}")

        return evaluation

    def _evaluate_dimension(
        self,
        dimension: str,
        paper_content: str,
        idea: Dict,
    ) -> Dict:
        """评估单个维度"""

        criteria = self._get_evaluation_criteria(dimension)

        prompt = f"""
请从 **{dimension}** 维度评估以下论文。

**论文内容**:
{paper_content[:5000]}

**评估标准**:
{json.dumps(criteria, indent=2, ensure_ascii=False)}

请提供:
1. 评分 (1-5分)
2. 具体分析
3. 优点
4. 缺点
5. 改进建议

以JSON格式返回。
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message=f"你是资深的论文审稿人，精通{self.template['name']}的评审标准。",
                temperature=0.3,
            )

            # 解析响应
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                return json.loads(json_match.group(1))
            else:
                return {"raw_response": response, "score": 3}

        except Exception as e:
            print(f"    ⚠️  {dimension} 评估失败: {e}")
            return {"score": 3, "error": str(e)}

    def _get_evaluation_criteria(self, dimension: str) -> Dict:
        """获取评估标准"""

        criteria = {
            "structure": {
                "description": "论文结构和组织",
                "criteria": [
                    "符合{self.template['name']}的章节结构",
                    "各章节比例合理",
                    "逻辑流程清晰",
                    "过渡自然",
                    "图表位置恰当",
                ]
            },
            "content": {
                "description": "内容质量和深度",
                "criteria": [
                    "内容全面覆盖主题",
                    "技术细节充分",
                    "论证有力",
                    "实验结果详实",
                    "分析深入",
                ]
            },
            "innovation": {
                "description": "创新性和贡献",
                "criteria": [
                    "创新点明确",
                    "与现有工作的区别清晰",
                    "贡献具有重要性",
                    "方法新颖",
                    "结果有突破",
                ]
            },
            "rigor": {
                "description": "研究严谨性",
                "criteria": [
                    "实验设计合理",
                    "基线选择恰当",
                    "统计分析正确",
                    "可复现性强",
                    "结论基于证据",
                ]
            },
            "clarity": {
                "description": "表达清晰度",
                "criteria": [
                    "语言简洁明了",
                    "术语定义清楚",
                    "数学表达规范",
                    "图表清晰易懂",
                    "读者友好",
                ]
            },
            "professionalism": {
                "description": "专业性和规范性",
                "criteria": [
                    "格式规范",
                    "引用正确完整",
                    "无语法错误",
                    "学术语言得体",
                    "符合期刊/会议规范",
                ]
            },
        }

        return criteria.get(dimension, {})

    def _get_quality_level(self, score: float) -> str:
        """获取质量等级"""
        if score >= 4.5:
            return "优秀 (Excellent) - 顶级会议水平"
        elif score >= 4.0:
            return "良好 (Very Good) - 可能被接收"
        elif score >= 3.5:
            return "中等 (Good) - 需要 minor revision"
        elif score >= 3.0:
            return "一般 (Fair) - 需要 major revision"
        else:
            return "较差 (Poor) - 需要重大改进"

    def _identify_strengths(self, evaluation: Dict) -> List[str]:
        """识别优点"""
        strengths = []
        for dim, result in evaluation.items():
            if isinstance(result, dict) and result.get("score", 0) >= 4:
                if result.get("优点"):
                    strengths.extend(result["优点"][:2])
        return strengths[:5]

    def _identify_weaknesses(self, evaluation: Dict) -> List[str]:
        """识别缺点"""
        weaknesses = []
        for dim, result in evaluation.items():
            if isinstance(result, dict) and result.get("score", 0) <= 3:
                if result.get("缺点"):
                    weaknesses.extend(result["缺点"][:2])
        return weaknesses[:5]

    def _generate_recommendations(self, evaluation: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        for dim, result in evaluation.items():
            if isinstance(result, dict):
                if result.get("改进建议"):
                    recommendations.extend(result["改进建议"][:1])
        return recommendations[:5]


# ========================================
# 辅助函数
# ========================================

def get_template_info(template: str) -> Dict:
    """获取模板信息"""
    return PAPER_TEMPLATES.get(template, PAPER_TEMPLATES["neurips"])


def list_templates() -> List[str]:
    """列出所有可用模板"""
    return list(PAPER_TEMPLATES.keys())


def recommend_template(idea: Dict) -> str:
    """推荐合适的模板"""
    # 简单推荐逻辑
    field = idea.get("Field", "").lower()
    task = idea.get("Task", "").lower()
    summary = " ".join(
        str(idea.get(key, ""))
        for key in ["Title", "Abstract", "Short Hypothesis", "Hypothesis", "Impact", "Field", "Task"]
    ).lower()

    if any(marker in summary for marker in ["real-world", "societal", "climate", "medical", "biology", "agriculture", "broad impact", "major challenge"]):
        return "nature"
    elif "vision" in field or "image" in task:
        return "cvpr"
    elif "representation" in task or "learning" in field:
        return "iclr"
    elif "theory" in task or "theoretical" in idea.get("Abstract", ""):
        return "neurips"
    else:
        return "neurips"  # 默认
