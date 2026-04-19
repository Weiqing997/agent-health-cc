from crewai import Agent
from crewai import LLM
from typing import Dict, Any, List
import os


class NutritionAnalysisAgent:
    """Agent responsible for calculating nutrition information."""

    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        # Use dashscope provider for Qwen
        self.llm = LLM(
            model="dashscope/qwen-plus",
            api_key=api_key,
        )

    def create_agent(self) -> Agent:
        return Agent(
            role="Nutrition Data Analyst",
            goal="Provide accurate calorie and nutritional information based on identified food items",
            backstory="""
                You are a certified nutritionist and dietitian with extensive knowledge of
                food composition, caloric content, and nutritional values. You can quickly
                lookup and calculate nutritional information for any food item.
                You always provide accurate, science-based nutritional data.
            """,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def analyze_nutrition(self, identified_foods: str) -> Dict[str, Any]:
        """Calculate nutrition information for identified foods."""
        prompt = f"""Based on the following identified food items, provide detailed nutritional analysis:

{identified_foods}

For each food item, estimate:
1. Calories (kcal)
2. Protein (grams)
3. Carbohydrates (grams)
4. Fat (grams)
5. Fiber (grams)
6. Any notable nutritional highlights or concerns

Also provide:
- Total calories for all items combined
- Macronutrient breakdown
- General health assessment (healthy/moderate/unhealthy choice)

Use standard portion sizes and common cooking methods for your calculations."""

        response = self.llm.chat(prompt)

        return {
            "nutrition_analysis": response.content,
            "identified_foods": identified_foods
        }
