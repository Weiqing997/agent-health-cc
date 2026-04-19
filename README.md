# 个人健康助手 - CrewAI 多智能体系统

一个基于 CrewAI 框架构建的多智能体健康助手，可以分析食物图片来识别食物项目并计算包括卡路里在内的营养信息。

## 功能特点

- **食物视觉智能体**：分析图片识别食物项目、烹饪方式和份量大小
- **营养分析智能体**：计算卡路里、宏量营养素并提供营养分解
- **多智能体协作**：智能体以顺序工作流程协同工作
- **阿里云 Qwen 驱动**：使用 Qwen-VL 进行视觉识别，Qwen-plus 处理文本

## 安装设置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，添加您的阿里云 DashScope API 密钥
```

3. 获取阿里云 DashScope API 密钥：
   - 访问 [阿里云 DashScope](https://dashscope.console.aliyun.com/)
   - 启用服务并获取您的 API 密钥
   - 确保您有足够的 Qwen-VL 和 Qwen-plus 模型配额

4. 运行助手：
```bash
python -m src.health_assistant.main
```

## 项目结构

```
health_assistant/
├── src/health_assistant/
│   ├── __init__.py
│   ├── crew.py              # 团队主配置（使用 @CrewBase）
│   ├── agents/
│   │   └── __init__.py      # 智能体在 YAML 中定义
│   ├── tools/
│   │   ├── __init__.py
│   │   └── calorie_tool.py  # 卡路里计算工具
│   └── config/
│       ├── agents.yaml      # 智能体配置
│       └── tasks.yaml       # 任务配置
├── requirements.txt
└── .env.example
```

## 使用方法

```python
from health_assistant.crew import HealthAssistantCrew

crew = HealthAssistantCrew()
result = crew.analyze_food_image("path/to/food/image.jpg")
print(result)
```

## 依赖要求

- Python 3.10+
- 阿里云 DashScope API 密钥
  - **Qwen-VL-Plus** 用于视觉分析
  - **Qwen-Plus** 用于文本处理

## 使用的模型

| 任务 | 模型 | 说明 |
|------|-------|-------------|
| 食物图片分析 | qwen-vl-plus | 用于识别食物项目的视觉语言模型 |
| 营养计算 | qwen-plus | 用于营养数据分析的文本模型 |
