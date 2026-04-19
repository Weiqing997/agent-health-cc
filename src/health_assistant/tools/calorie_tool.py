from crewai import Tool
from crewai import LLM
from typing import Dict, Any
import os


class CalorieCalculatorTool(Tool):
    """Tool for calculating calories and nutrition."""

    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        self.llm = LLM(
            model="dashscope/qwen-plus",
            api_key=api_key,
        )

        super().__init__(
            name="calorie_calculator",
            description="Calculates estimated calories and nutritional values for food items",
            func=self._calculate
        )

    def _calculate(self, food_name: str, portion_size: str = "medium") -> Dict[str, Any]:
        """Calculate nutrition for a given food."""
        prompt = f"""
        Calculate the nutritional information for:
        Food: {food_name}
        Portion: {portion_size}

        Provide:
        - Calories (kcal)
        - Protein (g)
        - Carbohydrates (g)
        - Fat (g)
        - Fiber (g)

        Respond in JSON format.
        """

        response = self.llm.chat(prompt)

        return {
            "food": food_name,
            "portion": portion_size,
            "analysis": response.content
        }
