from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import Dict, Any
import os


@CrewBase
class HealthAssistantCrew:
    """使用 @CrewBase YAML 配置模式的健康助手团队"""

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
        """创建健康助手团队"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def analyze_food_image(self, image_path: str, api_key: str = None) -> Dict[str, Any]:
        """运行团队分析食物图片"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片未找到: {image_path}")

        if not api_key:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量")

        # 获取任务实例
        vision_task = self.food_identification_task()
        nutrition_task = self.food_nutrition_task()

        # 更新图像识别任务描述，包含图片路径
        vision_task.description = f"""
        分析上传的食物图片，识别出所有存在的食物项目。

        图片路径: {image_path}

        对于每种食物，请描述：
        1. 食物名称
        2. 烹饪方式（如适用）
        3. 估计份量大小
        4. 可见的关键配料

        请以结构化格式返回分析结果。
        """

        # 设置任务上下文以实现顺序处理
        nutrition_task.context = [vision_task]

        # 使用更新后的任务创建团队
        analysis_crew = Crew(
            agents=self.agents,
            tasks=[vision_task, nutrition_task],
            process=Process.sequential,
            verbose=True,
        )

        result = analysis_crew.kickoff()
        return {"result": result, "image_path": image_path}
