# Personal Health Assistant - CrewAI Multi-Agent System

A multi-agent health assistant built with CrewAI framework that can analyze food images to identify items and calculate nutritional information including calories.

## Features

- **Food Vision Agent**: Analyzes images to identify food items, cooking methods, and portion sizes
- **Nutrition Analysis Agent**: Calculates calories, macronutrients, and provides nutritional breakdown
- **Multi-Agent Collaboration**: Agents work together in a sequential workflow
- **Powered by Alibaba Qwen**: Using Qwen-VL for vision and Qwen-plus for text

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your Alibaba DashScope API key
```

3. Get Alibaba DashScope API Key:
   - Visit [Alibaba Cloud DashScope](https://dashscope.console.aliyun.com/)
   - Enable the service and get your API key
   - Ensure you have sufficient quota for Qwen-VL and Qwen-plus models

4. Run the assistant:
```bash
python -m src.health_assistant.main
```

## Project Structure

```
health_assistant/
├── src/health_assistant/
│   ├── __init__.py
│   ├── crew.py              # Main crew configuration
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── food_vision_agent.py    # Vision agent using Qwen-VL
│   │   └── nutrition_agent.py       # Nutrition agent using Qwen-plus
│   ├── tools/
│   │   ├── __init__.py
│   │   └── calorie_tool.py          # Calorie calculation tool
│   ├── llm/                   # Custom LLM wrappers
│   │   ├── __init__.py
│   │   ├── qwen_vl.py              # Qwen-VL wrapper
│   │   └── qwen.py                # Qwen text model wrapper
│   └── config/
│       ├── agents.yaml
│       └── tasks.yaml
├── requirements.txt
└── .env.example
```

## Usage

```python
from health_assistant.crew import HealthAssistantCrew

crew = HealthAssistantCrew()
result = crew.analyze_food_image("path/to/food/image.jpg")
print(result)
```

## Requirements

- Python 3.10+
- Alibaba DashScope API key
  - **Qwen-VL-Plus** for vision analysis
  - **Qwen-Plus** for text processing

## Models Used

| Task | Model | Description |
|------|-------|-------------|
| Food Image Analysis | qwen-vl-plus | Vision-language model for identifying food items |
| Nutrition Calculation | qwen-plus | Text model for nutritional data analysis |
