from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import Dict, Any
import os


@CrewBase
class HealthAssistantCrew:
    """Health Assistant crew using @CrewBase YAML pattern."""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def food_vision_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['food_vision_agent'],
            verbose=True
        )

    @agent
    def nutrition_analysis_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['nutrition_analysis_agent'],
            verbose=True
        )

    @task
    def food_identification_task(self) -> Task:
        return Task(
            config=self.tasks_config['food_identification_task'],
        )

    @task
    def food_nutrition_task(self) -> Task:
        return Task(
            config=self.tasks_config['food_nutrition_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Health Assistant crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def analyze_food_image(self, image_path: str, api_key: str = None) -> Dict[str, Any]:
        """Run the crew to analyze a food image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

        # Get task instances
        vision_task = self.food_identification_task()
        nutrition_task = self.food_nutrition_task()

        # Update vision task description with image path
        vision_task.description = f"""
        Analyze the uploaded food image and identify all food items present.

        Image path: {image_path}

        For each item, describe:
        1. Name of the food
        2. Cooking method (if applicable)
        3. Estimated portion size
        4. Key ingredients visible

        Return your analysis in a structured format.
        """

        # Set task context for sequential processing
        nutrition_task.context = [vision_task]

        # Create crew with updated tasks
        analysis_crew = Crew(
            agents=self.agents,
            tasks=[vision_task, nutrition_task],
            process=Process.sequential,
            verbose=True,
        )

        result = analysis_crew.kickoff()
        return {"result": result, "image_path": image_path}
