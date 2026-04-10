#!/usr/bin/env python3
"""
创新性评估和突破性想法生成系统
提升研究的创新性和突破性
"""
import json
import os
from typing import Dict, List, Optional
from datetime import datetime

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.tools.semantic_scholar import search_for_papers


class InnovationEvaluator:
    """创新性评估器"""

    def __init__(self, model: str = "claude-3-5-sonnet"):
        """
        初始化创新性评估器

        Args:
            model: 使用的模型
        """
        self.model = model
        self.client, self.client_model = create_client(model)

    def evaluate_idea_novelty(
        self,
        idea: Dict,
        related_papers: List[Dict] = None,
    ) -> Dict:
        """
        评估想法的新颖性

        Args:
            idea: 研究想法
            related_papers: 相关论文列表

        Returns:
            新颖性评估结果
        """
        print("\n🔬 评估想法新颖性...")

        # 如果没有提供相关论文，自动搜索
        if related_papers is None:
            related_papers = self._search_related_papers(idea)

        # 构建评估提示
        prompt = self._build_novelty_evaluation_prompt(idea, related_papers)

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message="你是资深的学术研究专家，擅长评估研究想法的创新性和新颖性。",
                temperature=0.3,
            )

            # 解析响应
            evaluation = self._parse_evaluation_response(response)

            # 添加相关论文信息
            evaluation["related_papers"] = related_papers[:5] if related_papers else []

            return evaluation

        except Exception as e:
            print(f"❌ 新颖性评估失败: {e}")
            return {"error": str(e)}

    def _search_related_papers(self, idea: Dict) -> List[Dict]:
        """搜索相关论文"""
        print("  📚 搜索相关论文...")

        # 提取关键词
        title = idea.get("Title", "")
        abstract = idea.get("Abstract", "")

        keywords = self._extract_keywords(title + " " + abstract)

        # 搜索论文
        papers = []
        for keyword in keywords[:3]:  # 只搜索前3个关键词
            try:
                results = search_for_papers(
                    query=keyword,
                    limit=5,
                    year_range="2020-2024"
                )
                papers.extend(results)
            except:
                pass

        # 去重
        seen = set()
        unique_papers = []
        for paper in papers:
            paper_id = paper.get("paperId", "")
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                unique_papers.append(paper)

        print(f"    找到 {len(unique_papers)} 篇相关论文")
        return unique_papers

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取（可以改进为使用NLP）
        words = text.lower().split()
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "as", "is", "was", "are"
        }

        keywords = []
        for word in words:
            if len(word) > 4 and word not in stopwords:
                keywords.append(word)

        # 返回最常见的词
        from collections import Counter
        counter = Counter(keywords)
        return [kw for kw, _ in counter.most_common(10)]

    def _build_novelty_evaluation_prompt(self, idea: Dict, related_papers: List[Dict]) -> str:
        """构建新颖性评估提示"""
        # 构建相关论文摘要
        papers_summary = ""
        if related_papers:
            papers_summary = "\n\n**相关论文**:\n"
            for i, paper in enumerate(related_papers[:5]):
                papers_summary += f"\n{i+1}. {paper.get('title', 'N/A')}\n"
                papers_summary += f"   摘要: {paper.get('abstract', 'N/A')[:200]}...\n"

        prompt = f"""
请评估以下研究想法的新颖性和创新性。

**研究想法**:
标题: {idea.get('Title', '')}
摘要: {idea.get('Abstract', '')}
假设: {idea.get('Hypothesis', '')}
方法: {idea.get('Method', '')}

{papers_summary}

请从以下维度进行评估:

1. **新颖性评分** (1-5分):
   - 5: 突破性创新，开创全新方向
   - 4: 显著创新，在现有方向上有重大突破
   - 3: 中等创新，有实质性改进
   - 2: 小幅创新，渐进式改进
   - 1: 缺乏创新，与现有工作雷同

2. **差异化分析**:
   - 与现有工作的主要区别
   - 独特贡献
   - 创新点

3. **可行性评估** (1-5分):
   - 技术可行性
   - 资源需求
   - 风险评估

4. **影响力预测** (1-5分):
   - 学术影响
   - 实际应用价值
   - 领域推动作用

5. **改进建议**:
   - 如何提升创新性
   - 如何增强可行性
   - 如何扩大影响力

请以JSON格式返回评估结果。
"""

        return prompt

    def _parse_evaluation_response(self, response: str) -> Dict:
        """解析评估响应"""
        import re

        # 尝试提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass

        # 如果无法解析，返回结构化响应
        return {
            "raw_response": response,
            "novelty_score": 3,  # 默认评分
            "feasibility_score": 3,
            "impact_score": 3,
        }


class BreakthroughIdeaGenerator:
    """突破性想法生成器"""

    def __init__(self, model: str = "o1-2024-12-17"):
        """
        初始化突破性想法生成器

        Args:
            model: 使用的模型（优先使用推理模型）
        """
        self.model = model
        self.client, self.client_model = create_client(model)

    def generate_breakthrough_ideas(
        self,
        research_area: str,
        num_ideas: int = 3,
        breakthrough_mode: bool = True,
    ) -> List[Dict]:
        """
        生成突破性研究想法

        Args:
            research_area: 研究领域
            num_ideas: 生成想法数量
            breakthrough_mode: 是否启用突破性模式

        Returns:
            想法列表
        """
        print(f"\n🚀 生成突破性想法...")
        print(f"   研究领域: {research_area}")
        print(f"   想法数量: {num_ideas}")
        print(f"   突破性模式: {'启用' if breakthrough_mode else '禁用'}")

        ideas = []

        # 突破性提示模板
        if breakthrough_mode:
            prompts = self._get_breakthrough_prompts(research_area)
        else:
            prompts = self._get_standard_prompts(research_area)

        for i, prompt_template in enumerate(prompts[:num_ideas]):
            print(f"\n  生成想法 {i+1}/{num_ideas}...")

            try:
                response, _ = get_response_from_llm(
                    prompt=prompt_template,
                    client=self.client,
                    model=self.client_model,
                    system_message="你是资深的科研专家，擅长提出突破性的研究想法。",
                    temperature=0.8,  # 较高温度以增加多样性
                )

                # 解析想法
                idea = self._parse_idea(response, i)
                ideas.append(idea)

                print(f"    ✅ {idea.get('Title', 'Untitled')}")

            except Exception as e:
                print(f"    ❌ 生成失败: {e}")

        return ideas

    def _get_breakthrough_prompts(self, research_area: str) -> List[str]:
        """获取突破性提示模板"""
        return [
            f"""
请从**跨学科角度**为"{research_area}"领域提出一个突破性的研究想法。

要求:
1. 结合2-3个不同领域的思想
2. 挑战该领域的基本假设
3. 提出全新的研究范式或方法
4. 具有实际可行性

请提供详细的研究提案，包括:
- 创新的标题
- 清晰的摘要（100-150字）
- 核心假设（挑战现有假设）
- 跨学科方法论
- 预期突破点
- 潜在影响

以JSON格式返回。
""",

            f"""
请为"{research_area}"领域提出一个**解决根本性问题**的研究想法。

要求:
1. 识别该领域的一个根本性瓶颈或局限
2. 分析根本原因（而非表面现象）
3. 提出根本性的解决方案
4. 验证方法的可行性

请提供详细的研究提案，包括:
- 标题
- 问题分析（根本原因）
- 创新解决方案
- 验证策略
- 预期贡献

以JSON格式返回。
""",

            f"""
请为"{research_area}"领域提出一个**反直觉但可验证**的研究想法。

要求:
1. 基于该领域的"常识"提出反向观点
2. 提供理论依据
3. 设计严谨的验证实验
4. 预测可能的反对意见并准备反驳

请提供详细的研究提案，包括:
- 标题（体现反直觉性）
- 常识观点 vs 反直觉观点
- 理论支撑
- 实验设计
- 讨论和应对

以JSON格式返回。
""",

            f"""
请为"{research_area}"领域提出一个**方法论创新**的研究想法。

要求:
1. 创造新的研究方法或工具
2. 解决现有方法的重大局限
3. 具有广泛的适用性
4. 可以显著提升研究效率

请提供详细的研究提案，包括:
- 标题
- 现有方法的局限分析
- 新方法的核心思想
- 技术实现
- 应用案例

以JSON格式返回。
""",
        ]

    def _get_standard_prompts(self, research_area: str) -> List[str]:
        """获取标准提示模板"""
        return [
            f"""
请为"{research_area}"领域提出一个创新的研究想法。

要求:
1. 有明确的创新点
2. 可行性强
3. 有实际应用价值
4. 研究设计严谨

请提供详细的研究提案，包括:
- 标题
- 摘要
- 研究假设
- 实验设计
- 预期结果

以JSON格式返回。
"""
        ] * 3

    def _parse_idea(self, response: str, index: int) -> Dict:
        """解析想法响应"""
        import re

        # 尝试提取JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if not json_match:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                idea = json.loads(json_match.group(1))
                idea["generated_at"] = datetime.now().isoformat()
                idea["idea_index"] = index
                return idea
            except:
                pass

        # 如果无法解析，创建基础结构
        return {
            "Title": f"Research Idea {index + 1}",
            "Abstract": response[:500],
            "Hypothesis": "See full response",
            "Experiments": "To be determined",
            "generated_at": datetime.now().isoformat(),
            "idea_index": index,
            "raw_response": response,
        }


class DepthEnhancer:
    """研究深度增强器"""

    def __init__(self, model: str = "claude-3-5-sonnet"):
        """
        初始化深度增强器

        Args:
            model: 使用的模型
        """
        self.model = model
        self.client, self.client_model = create_client(model)

    def enhance_research_depth(
        self,
        idea: Dict,
        current_experiments: Dict,
    ) -> Dict:
        """
        增强研究深度

        Args:
            idea: 研究想法
            current_experiments: 当前实验设计

        Returns:
            增强后的实验设计
        """
        print("\n🔍 增强研究深度...")

        prompt = f"""
请分析以下研究设计，并提出增强研究深度的具体建议。

**研究想法**:
标题: {idea.get('Title', '')}
假设: {idea.get('Hypothesis', '')}

**当前实验设计**:
{json.dumps(current_experiments, indent=2, ensure_ascii=False)}

请从以下维度提出增强建议:

1. **理论分析深度**:
   - 是否需要更深入的理论分析？
   - 可以添加哪些理论推导或证明？
   - 边界条件分析

2. **实验设计深度**:
   - 是否需要更多的基线对比？
   - 消融实验是否充分？
   - 是否需要跨数据集验证？
   - 统计显著性检验

3. **结果分析深度**:
   - 是否需要深入分析失败案例？
   - 是否需要可视化内部表示？
   - 是否需要相关性分析？

4. **泛化性研究**:
   - 分布外测试
   - 噪声鲁棒性
   - 不同规模的数据集

5. **实际应用验证**:
   - 真实场景测试
   - 用户研究
   - 性能基准

请提供:
1. 每个维度的具体建议
2. 优先级排序（高/中/低）
3. 预计增加的研究时间

以JSON格式返回。
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message="你是资深的科研方法论专家，擅长设计深入的研究实验。",
                temperature=0.5,
            )

            # 解析响应
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                enhancement = json.loads(json_match.group(1))
                print("✅ 研究深度增强建议生成完成")
                return enhancement
            else:
                return {"raw_response": response}

        except Exception as e:
            print(f"❌ 深度增强失败: {e}")
            return {}

    def generate_comparative_studies(
        self,
        idea: Dict,
        sota_methods: List[str] = None,
    ) -> List[Dict]:
        """
        生成对比研究方案

        Args:
            idea: 研究想法
            sota_methods: SOTA方法列表

        Returns:
            对比研究方案列表
        """
        print("\n📊 生成对比研究方案...")

        if sota_methods is None:
            sota_methods = [
                "Standard Baseline",
                "Recent SOTA Method 1",
                "Recent SOTA Method 2",
            ]

        prompt = f"""
请为以下研究设计详细的对比实验方案。

**研究想法**:
标题: {idea.get('Title', '')}
方法: {idea.get('Method', '')}

**对比方法**:
{chr(10).join([f"- {m}" for m in sota_methods])}

请为每个对比方法设计:

1. **公平性考虑**:
   - 如何确保公平对比？
   - 需要控制哪些变量？

2. **评估指标**:
   - 主要指标
   - 次要指标
   - 效率指标（时间、内存）

3. **实验设置**:
   - 数据集划分
   - 超参数设置
   - 实现细节

4. **预期结果**:
   - 在哪些指标上预期胜出？
   - 在哪些指标上可能相当？
   - 什么情况下会失败？

请为每个对比方法提供完整的实验设计，以JSON格式返回。
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message="你是资深的实验设计专家，擅长设计公平和全面的对比实验。",
                temperature=0.5,
            )

            # 解析响应
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                studies = json.loads(json_match.group(1))
                print("✅ 对比研究方案生成完成")
                return studies.get("comparative_studies", [])
            else:
                return []

        except Exception as e:
            print(f"❌ 对比研究生成失败: {e}")
            return []
