from crewai import Agent
from crewai import LLM
from typing import Dict, Any
import base64
import os


class FoodVisionAgent:
    """Agent responsible for identifying food items from images."""

    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        # Use dashscope provider for Qwen-VL
        self.llm = LLM(
            model="dashscope/qwen-vl-plus",
            api_key=api_key,
        )

    def create_agent(self) -> Agent:
        return Agent(
            role="Food Vision Analyst",
            goal="Accurately identify food items from uploaded images and estimate their portion sizes",
            backstory="""
                You are an expert nutritionist with 15 years of experience in food identification.
                You have a keen eye for detail and can accurately identify various food items,
                their cooking methods, and estimate portion sizes from photos.
                You are methodical and always provide detailed descriptions of what you observe.
            """,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze a food image and identify items."""
        with open(image_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")

        prompt = """Please analyze this food image and identify all food items present.

For each food item, provide:
1. Name of the food
2. Cooking method (grilled, fried, raw, steamed, etc.)
3. Estimated portion size (small/medium/large with approximate weight in grams)
4. Key ingredients visible
5. Estimated number of servings

Be as specific as possible in your identification."""

        response = self.llm.chat([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ])

        return {"identified_foods": response.content}
