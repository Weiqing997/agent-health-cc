"""
健康助手 CrewAI 系统使用示例
"""
import os
import sys
from pathlib import Path

# 将 src 添加到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from health_assistant.crew import HealthAssistantCrew


def example():
    load_dotenv()

    # 检查 API 密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误：未设置 DASHSCOPE_API_KEY。请在 .env 文件中设置")
        return

    # 初始化团队
    crew = HealthAssistantCrew()

    # 使用示例 - 请替换为你的图片路径
    image_path = "path/to/your/food/image.jpg"

    if os.path.exists(image_path):
        result = crew.analyze_food_image(image_path)
        print("分析完成！")
        print(result)
    else:
        print(f"示例图片未找到: {image_path}")
        print("请提供有效的图片路径进行测试。")


if __name__ == "__main__":
    example()
