#!/usr/bin/env python3
"""
增强的基线对比系统
自动检测和配置SOTA基线方法
"""
import json
import os
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from ai_scientist.config.paths import OUTPUT_PATH
from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.tools.semantic_scholar import search_for_papers


class BaselineManager:
    """基线管理器"""

    def __init__(self, research_dir: str = str(OUTPUT_PATH)):
        """
        初始化基线管理器

        Args:
            research_dir: 研究目录
        """
        self.research_dir = Path(research_dir)
        self.baselines_dir = self.research_dir / "baselines"
        self.baselines_dir.mkdir(parents=True, exist_ok=True)

        # SOTA方法库
        self.sota_library = self._load_sota_library()

    def _load_sota_library(self) -> Dict:
        """加载SOTA方法库"""
        library_file = self.baselines_dir / "sota_library.json"

        if library_file.exists():
            with open(library_file, "r") as f:
                return json.load(f)
        else:
            # 创建初始库
            return self._create_initial_sota_library()

    def _create_initial_sota_library(self) -> Dict:
        """创建初始SOTA方法库"""
        library = {
            "computer_vision": {
                "image_classification": [
                    {
                        "name": "Vision Transformer (ViT)",
                        "year": 2021,
                        "paper": "An Image is Worth 16x16 Words",
                        "github": "https://github.com/google-research/vision_transformer",
                        "frameworks": ["pytorch", "tensorflow"],
                        "category": "transformer",
                    },
                    {
                        "name": "ResNet-50",
                        "year": 2015,
                        "paper": "Deep Residual Learning for Image Recognition",
                        "github": "https://github.com/pytorch/vision",
                        "frameworks": ["pytorch"],
                        "category": "cnn",
                    },
                    {
                        "name": "EfficientNet-V2",
                        "year": 2021,
                        "paper": "EfficientNetV2: Smaller Models and Faster Training",
                        "github": "https://github.com/google/automl",
                        "frameworks": ["pytorch", "tensorflow"],
                        "category": "efficient",
                    },
                ],
                "object_detection": [
                    {
                        "name": "YOLOv8",
                        "year": 2023,
                        "paper": "Ultralytics YOLOv8",
                        "github": "https://github.com/ultralytics/ultralytics",
                        "frameworks": ["pytorch"],
                        "category": "realtime",
                    },
                    {
                        "name": "Faster R-CNN",
                        "year": 2015,
                        "paper": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
                        "github": "https://github.com/pytorch/vision",
                        "frameworks": ["pytorch"],
                        "category": "two_stage",
                    },
                ],
            },
            "natural_language_processing": {
                "text_classification": [
                    {
                        "name": "BERT",
                        "year": 2018,
                        "paper": "BERT: Pre-training of Deep Bidirectional Transformers",
                        "github": "https://github.com/huggingface/transformers",
                        "frameworks": ["pytorch", "tensorflow"],
                        "category": "transformer",
                    },
                    {
                        "name": "RoBERTa",
                        "year": 2019,
                        "paper": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
                        "github": "https://github.com/huggingface/transformers",
                        "frameworks": ["pytorch", "tensorflow"],
                        "category": "transformer",
                    },
                ],
                "language_modeling": [
                    {
                        "name": "GPT-3.5",
                        "year": 2022,
                        "paper": "Language Models are Few-Shot Learners",
                        "api": "OpenAI API",
                        "category": "api_based",
                    },
                    {
                        "name": "LLaMA 2",
                        "year": 2023,
                        "paper": "LLaMA 2: Open Foundation and Fine-Tuned Chat Models",
                        "github": "https://github.com/facebookresearch/llama",
                        "frameworks": ["pytorch"],
                        "category": "open_source",
                    },
                ],
            },
            "machine_learning": {
                "optimization": [
                    {
                        "name": "AdamW",
                        "year": 2017,
                        "paper": "Decoupled Weight Decay Regularization",
                        "implementation": "torch.optim.AdamW",
                        "category": "optimizer",
                    },
                    {
                        "name": "SGD with Momentum",
                        "year": 1986,
                        "paper": "Learning representations by back-propagating errors",
                        "implementation": "torch.optim.SGD",
                        "category": "optimizer",
                    },
                ],
                "regularization": [
                    {
                        "name": "Dropout",
                        "year": 2014,
                        "paper": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
                        "implementation": "torch.nn.Dropout",
                        "category": "regularization",
                    },
                    {
                        "name": "Layer Normalization",
                        "year": 2016,
                        "paper": "Layer Normalization",
                        "implementation": "torch.nn.LayerNorm",
                        "category": "normalization",
                    },
                ],
            },
        }

        # 保存到文件
        library_file = self.baselines_dir / "sota_library.json"
        with open(library_file, "w") as f:
            json.dump(library, f, indent=2)

        return library

    def suggest_baselines(
        self,
        idea: Dict,
        num_baselines: int = 5,
    ) -> List[Dict]:
        """
        为研究想法推荐基线方法

        Args:
            idea: 研究想法
            num_baselines: 推荐基线数量

        Returns:
            推荐的基线方法列表
        """
        print(f"\n📚 为研究想法推荐基线方法...")

        # 分析研究领域
        title = idea.get("Title", "")
        abstract = idea.get("Abstract", "")

        # 使用LLM分析并推荐
        prompt = f"""
请分析以下研究想法，并推荐合适的基线方法。

**研究想法**:
标题: {title}
摘要: {abstract}
领域: {idea.get('Field', 'Machine Learning')}

**SOTA方法库**:
{json.dumps(self.sota_library, indent=2, ensure_ascii=False)}

请推荐{num_baselines}个最合适的基线方法，包括:
1. 经典基线方法
2. 最新SOTA方法
3. 轻量级/高效方法
4. 不同架构类型的方法

对于每个基线方法，提供:
- 方法名称
- 推荐理由
- 实现难度
- 预期性能水平
- 实现资源（GitHub链接、库等）

以JSON格式返回。
"""

        try:
            client, client_model = create_client("claude-3-5-sonnet")
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=client,
                model=client_model,
                system_message="你是资深的机器学习研究员，熟悉各种SOTA方法和基线。",
                temperature=0.3,
            )

            # 解析响应
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                baselines = json.loads(json_match.group(1))
                print(f"✅ 推荐了 {len(baselines.get('baselines', []))} 个基线方法")
                return baselines.get("baselines", [])
            else:
                return []

        except Exception as e:
            print(f"❌ 基线推荐失败: {e}")
            return []

    def configure_baseline(
        self,
        method_name: str,
        idea: Dict,
        dataset_info: Dict,
    ) -> Dict:
        """
        配置基线方法的实现

        Args:
            method_name: 方法名称
            idea: 研究想法
            dataset_info: 数据集信息

        Returns:
            配置方案
        """
        print(f"\n⚙️  配置基线: {method_name}")

        prompt = f"""
请为以下基线方法提供详细的实现配置。

**基线方法**: {method_name}

**研究背景**:
标题: {idea.get('Title', '')}
任务: {idea.get('Task', '')}

**数据集信息**:
{json.dumps(dataset_info, indent=2, ensure_ascii=False)}

请提供:

1. **数据预处理**:
   - 输入格式要求
   - 归一化方法
   - 数据增强策略

2. **模型配置**:
   - 网络架构
   - 超参数设置
   - 初始化方法

3. **训练配置**:
   - 优化器
   - 学习率调度
   - Batch size
   - 训练轮数

4. **评估配置**:
   - 评估指标
   - 评估频率
   - 早停策略

5. **实现代码框架**:
   - Python代码框架
   - 关键模块

请提供详细的配置方案，以JSON格式返回。
"""

        try:
            client, client_model = create_client("claude-3-5-sonnet")
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=client,
                model=client_model,
                system_message="你是资深的机器学习工程师，擅长实现各种深度学习方法。",
                temperature=0.3,
            )

            # 解析响应
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                config = json.loads(json_match.group(1))
                print(f"✅ {method_name} 配置完成")
                return config
            else:
                return {"raw_response": response}

        except Exception as e:
            print(f"❌ {method_name} 配置失败: {e}")
            return {}


class MultiDimensionalEvaluator:
    """多维度评估器"""

    def __init__(self, model: str = "gpt-4o"):
        """
        初始化多维度评估器

        Args:
            model: 使用的模型
        """
        self.model = model
        self.client, self.client_model = create_client(model)

    def evaluate_results(
        self,
        experiment_results: Dict,
        baselines: List[str],
    ) -> Dict:
        """
        多维度评估实验结果

        Args:
            experiment_results: 实验结果
            baselines: 基线方法列表

        Returns:
            多维度评估结果
        """
        print("\n📊 执行多维度评估...")

        # 定义评估维度
        dimensions = [
            "primary_metric",      # 主要指标
            "secondary_metrics",   # 次要指标
            "efficiency",          # 效率（时间、内存）
            "robustness",          # 鲁棒性
            "generalization",      # 泛化性
            "practicality",        # 实用性
        ]

        evaluation_results = {}

        for dimension in dimensions:
            print(f"  评估: {dimension}")
            result = self._evaluate_dimension(
                dimension,
                experiment_results,
                baselines,
            )
            evaluation_results[dimension] = result

        # 生成综合评估
        summary = self._generate_summary(evaluation_results)

        return {
            "dimensions": evaluation_results,
            "summary": summary,
        }

    def _evaluate_dimension(
        self,
        dimension: str,
        results: Dict,
        baselines: List[str],
    ) -> Dict:
        """评估单个维度"""
        prompt = f"""
请从"{dimension}"维度评估以下实验结果。

**实验结果**:
{json.dumps(results, indent=2, ensure_ascii=False)}

**基线方法**:
{', '.join(baselines)}

请提供:
1. 评分（1-5分）
2. 具体分析
3. 与基线的对比
4. 改进建议

以JSON格式返回。
"""

        try:
            response, _ = get_response_from_llm(
                prompt=prompt,
                client=self.client,
                model=self.client_model,
                system_message="你是资深的机器学习研究员，擅长全面评估实验结果。",
                temperature=0.3,
            )

            # 解析响应
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

            if json_match:
                return json.loads(json_match.group(1))
            else:
                return {"raw_response": response}

        except Exception as e:
            print(f"    ⚠️  {dimension} 评估失败: {e}")
            return {}

    def _generate_summary(self, evaluation_results: Dict) -> Dict:
        """生成综合评估摘要"""
        # 计算平均分
        scores = []
        for dim, result in evaluation_results.items():
            score = result.get("score", 0)
            if isinstance(score, (int, float)):
                scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0

        # 识别强项和弱项
        strong_points = [
            dim for dim, result in evaluation_results.items()
            if result.get("score", 0) >= 4
        ]

        weak_points = [
            dim for dim, result in evaluation_results.items()
            if result.get("score", 0) <= 2
        ]

        return {
            "average_score": avg_score,
            "strong_points": strong_points,
            "weak_points": weak_points,
            "overall_assessment": self._get_overall_assessment(avg_score),
        }

    def _get_overall_assessment(self, score: float) -> str:
        """获取总体评估"""
        if score >= 4.0:
            return "优秀"
        elif score >= 3.0:
            return "良好"
        elif score >= 2.0:
            return "一般"
        else:
            return "需要改进"


class StatisticalAnalyzer:
    """统计分析器"""

    @staticmethod
    def perform_significance_test(
        results_a: List[float],
        results_b: List[float],
        test_type: str = "t_test",
    ) -> Dict:
        """
        执行统计显著性检验

        Args:
            results_a: 方法A的结果
            results_b: 方法B的结果
            test_type: 检验类型 (t_test, wilcoxon)

        Returns:
            检验结果
        """
        import numpy as np
        from scipy import stats

        print("\n📈 执行统计显著性检验...")

        if test_type == "t_test":
            # t检验
            t_stat, p_value = stats.ttest_ind(results_a, results_b)
            test_name = "Independent t-test"

        elif test_type == "wilcoxon":
            # Wilcoxon秩和检验
            t_stat, p_value = stats.wilcoxon(results_a, results_b)
            test_name = "Wilcoxon rank-sum test"

        else:
            return {"error": f"Unknown test type: {test_type}"}

        # 判断显著性
        alpha = 0.05
        is_significant = p_value < alpha

        # 计算效应量 (Cohen's d)
        cohen_d = (np.mean(results_a) - np.mean(results_b)) / np.sqrt(
            (np.std(results_a) ** 2 + np.std(results_b) ** 2) / 2
        )

        return {
            "test_name": test_name,
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": is_significant,
            "alpha": alpha,
            "cohen_d": float(cohen_d),
            "effect_size": StatisticalAnalyzer._interpret_effect_size(cohen_d),
            "mean_a": float(np.mean(results_a)),
            "mean_b": float(np.mean(results_b)),
            "std_a": float(np.std(results_a)),
            "std_b": float(np.std(results_b)),
        }

    @staticmethod
    def _interpret_effect_size(cohen_d: float) -> str:
        """解释效应量"""
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    @staticmethod
    def generate_confidence_interval(
        data: List[float],
        confidence: float = 0.95,
    ) -> Dict:
        """
        生成置信区间

        Args:
            data: 数据
            confidence: 置信水平

        Returns:
            置信区间
        """
        import numpy as np
        from scipy import stats

        mean = np.mean(data)
        std_err = stats.sem(data)
        n = len(data)

        # t分布
        t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_critical * std_err

        return {
            "mean": float(mean),
            "std": float(np.std(data)),
            "std_err": float(std_err),
            "confidence": confidence,
            "lower_bound": float(mean - margin),
            "upper_bound": float(mean + margin),
            "margin": float(margin),
        }
