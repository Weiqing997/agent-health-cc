from crewai.tools import BaseTool
from crewai import LLM
from typing import Dict, Any
import os


class CalorieCalculatorTool(BaseTool):
    """用于计算食物卡路里和营养素的工具"""

    name: str = "calorie_calculator"
    description: str = "计算食物的估计卡路里和营养值"

    def _run(self, food_name: str, portion_size: str = "medium") -> Dict[str, Any]:
        """计算给定食物的营养信息"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        llm = LLM(
            model="dashscope/qwen-plus",
            api_key=api_key,
        )

        prompt = f"""
        计算以下食物的营养信息：
        食物：{food_name}
        份量：{portion_size}

        请提供：
        - 卡路里（千卡）
        - 蛋白质（克）
        - 碳水化合物（克）
        - 脂肪（克）
        - 纤维（克）

        请以 JSON 格式回复。
        """

        response = llm.chat(prompt)

        return {
            "food": food_name,
            "portion": portion_size,
            "analysis": response.content
        }
