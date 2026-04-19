import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path.parent))

from dotenv import load_dotenv
from health_assistant.crew import HealthAssistantCrew


def main():
    load_dotenv()

    # Check for API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("Error: DASHSCOPE_API_KEY not found in environment")
        print("Please create a .env file with your Alibaba DashScope API key")
        return

    print("=" * 60)
    print("Personal Health Assistant - Calorie Analyzer")
    print("   Powered by Alibaba Qwen-VL")
    print("=" * 60)
    print()

    # Get image path from user
    image_path = input("Please enter the path to your food image: ").strip()

    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    print(f"\nAnalyzing image: {image_path}")
    print("-" * 40)

    try:
        crew = HealthAssistantCrew()
        result = crew.analyze_food_image(image_path)

        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        print(result.get("result", "No result"))

    except Exception as e:
        import traceback
        print(f"Error during analysis: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
