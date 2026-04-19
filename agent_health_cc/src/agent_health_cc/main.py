import os
import sys
from pathlib import Path

# 将 src 添加到路径
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path.parent))

from dotenv import load_dotenv
from agent_health_cc.crew import HealthAssistantCrew


def main():
    load_dotenv()

    # 检查 API 密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误：未在环境中找到 DASHSCOPE_API_KEY")
        print("请创建包含阿里云 DashScope API 密钥的 .env 文件")
        return

    print("=" * 60)
    print("个人健康助手 - 卡路里分析器")
    print("   由阿里云 Qwen-VL 提供支持")
    print("=" * 60)
    print()

    # 从用户获取图片路径
    image_path = input("请输入食物图片的路径：").strip()

    if not os.path.exists(image_path):
        print(f"错误：文件未找到: {image_path}")
        return

    print(f"\n正在分析图片: {image_path}")
    print("-" * 40)

    try:
        crew = HealthAssistantCrew()
        result = crew.analyze_food_image(image_path)

        print("\n" + "=" * 60)
        print("分析结果")
        print("=" * 60)
        print(result.get("result", "无结果"))

    except Exception as e:
        import traceback
        print(f"分析过程中出错: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
