"""
Quick example of using the Health Assistant CrewAI system.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from health_assistant.crew import HealthAssistantCrew


def example():
    load_dotenv()

    # Check for API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("Error: DASHSCOPE_API_KEY not found. Please set it in .env file")
        return

    # Initialize the crew
    crew = HealthAssistantCrew()

    # Example usage - replace with your image path
    image_path = "path/to/your/food/image.jpg"

    if os.path.exists(image_path):
        result = crew.analyze_food_image(image_path)
        print("Analysis Complete!")
        print(result)
    else:
        print(f"Example image not found at: {image_path}")
        print("Please provide a valid image path to test.")


if __name__ == "__main__":
    example()
