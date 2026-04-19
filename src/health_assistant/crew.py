from crewai import Crew, Task, Process
from typing import Dict, Any, List
import os

from .agents.food_vision_agent import FoodVisionAgent
from .agents.nutrition_agent import NutritionAnalysisAgent


class HealthAssistantCrew:
    """Main crew for the health assistant, coordinating multiple agents."""

    def __init__(self):
        self.food_vision_agent = FoodVisionAgent()
        self.nutrition_agent = NutritionAnalysisAgent()

    def _create_vision_task(self, image_path: str) -> Task:
        """Create the food identification task."""
        return Task(
            description=f"""
            Analyze the uploaded food image and identify all food items present.

            Image path: {image_path}

            For each item, describe:
            1. Name of the food
            2. Cooking method (if applicable)
            3. Estimated portion size
            4. Key ingredients visible

            Return your analysis in a structured format.
            """,
            agent=self.food_vision_agent.create_agent(),
            expected_output="A detailed list of all identified food items with descriptions",
        )

    def _create_nutrition_task(self, context: List[Any]) -> Task:
        """Create the nutrition analysis task."""
        return Task(
            description="""
            Based on the identified food items, calculate detailed nutritional information.

            Calculate for each item:
            1. Estimated calories
            2. Macronutrients (protein, carbs, fat)
            3. Total calorie count

            Provide a complete nutritional breakdown.
            """,
            agent=self.nutrition_agent.create_agent(),
            expected_output="Complete nutritional breakdown with calorie estimates",
            context=context,
        )

    def analyze_food_image(self, image_path: str) -> Dict[str, Any]:
        """Run the crew to analyze a food image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Create tasks
        vision_task = self._create_vision_task(image_path)
        nutrition_task = self._create_nutrition_task(context=[vision_task])

        # Create and run the crew
        crew = Crew(
            agents=[
                self.food_vision_agent.create_agent(),
                self.nutrition_agent.create_agent(),
            ],
            tasks=[vision_task, nutrition_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        return {
            "result": result,
            "image_path": image_path
        }
