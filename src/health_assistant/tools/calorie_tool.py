from crewai.tools import BaseTool
from crewai import LLM
from typing import Dict, Any
import os


class CalorieCalculatorTool(BaseTool):
    """Tool for calculating calories and nutrition."""

    name: str = "calorie_calculator"
    description: str = "Calculates estimated calories and nutritional values for food items"

    def _run(self, food_name: str, portion_size: str = "medium") -> Dict[str, Any]:
        """Calculate nutrition for a given food."""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        llm = LLM(
            model="dashscope/qwen-plus",
            api_key=api_key,
        )

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

        response = llm.chat(prompt)

        return {
            "food": food_name,
            "portion": portion_size,
            "analysis": response.content
        }
