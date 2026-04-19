"""
Qwen-VL chat model wrapper for langchain/crewai compatibility.
"""
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import dashscope
from dashscope import MultiModalConversation


class ChatQwenVL(BaseChatModel):
    """Langchain compatible wrapper for Qwen-VL (vision model)."""

    model: str = "qwen-vl-plus"
    temperature: float = 0.3
    api_key: Optional[str] = None

    def __init__(self, model: str = "qwen-vl-plus", temperature: float = 0.3, **kwargs):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.temperature = temperature
        self._api_key = kwargs.get("api_key") or dashscope.api_key

    @property
    def _llm_type(self) -> str:
        return "qwen-vl"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert langchain messages to dashscope format."""
        converted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = []
                for item in msg.content:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            # Extract base64 or URL
                            img_url = item["image_url"].get("url", "")
                            if img_url.startswith("data:"):
                                content.append({
                                    "image": img_url
                                })
                            else:
                                content.append({
                                    "image": img_url
                                })
                        elif item.get("type") == "text":
                            content.append({
                                "text": item["text"]
                            })
                    else:
                        content.append({"text": str(item)})

                converted.append({
                    "role": "user",
                    "content": content
                })
            elif isinstance(msg, AIMessage):
                converted.append({
                    "role": "assistant",
                    "content": [{"text": msg.content}] if msg.content else [{"text": ""}]
                })
            elif isinstance(msg, SystemMessage):
                converted.append({
                    "role": "system",
                    "content": [{"text": msg.content}]
                })
        return converted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """Generate response using Qwen-VL."""
        if not self._api_key:
            raise ValueError("API key not set. Please set DASHSCOPE_API_KEY.")

        dashscope.api_key = self._api_key

        # Convert messages
        inputs = self._convert_messages(messages)

        # Call Qwen-VL
        response = MultiModalConversation.call(
            model=self.model,
            messages=[inputs] if inputs else [{}]
        )

        if response.status_code == 200:
            content = response.output.choices[0].message.content
            text = content[0].get("text", "") if isinstance(content, list) else str(content)
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            )
        else:
            raise Exception(f"API call failed: {response.message}")

    def _llm_type(self) -> str:
        return "qwen_vl"
