"""
Qwen chat model wrapper for langchain/crewai compatibility.
"""
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import dashscope
from dashscope import Generation


class ChatQwen(BaseChatModel):
    """Langchain compatible wrapper for Qwen (text model)."""

    model: str = "qwen-plus"
    temperature: float = 0.2
    api_key: Optional[str] = None

    def __init__(self, model: str = "qwen-plus", temperature: float = 0.2, **kwargs):
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.temperature = temperature
        self._api_key = kwargs.get("api_key") or dashscope.api_key

    @property
    def _llm_type(self) -> str:
        return "qwen"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert langchain messages to dashscope format."""
        converted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                converted.append({
                    "role": "user",
                    "content": msg.content
                })
            elif isinstance(msg, AIMessage):
                converted.append({
                    "role": "assistant",
                    "content": msg.content
                })
            elif isinstance(msg, SystemMessage):
                converted.append({
                    "role": "system",
                    "content": msg.content
                })
        return converted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """Generate response using Qwen."""
        if not self._api_key:
            raise ValueError("API key not set. Please set DASHSCOPE_API_KEY.")

        dashscope.api_key = self._api_key

        # Convert messages to single prompt
        converted = self._convert_messages(messages)
        prompt = "\n".join([
            f"{m['role']}: {m['content']}" for m in converted
        ])

        # Call Qwen
        response = Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            stop=stop,
        )

        if response.status_code == 200:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=response.output.text))]
            )
        else:
            raise Exception(f"API call failed: {response.message}")

    def _llm_type(self) -> str:
        return "qwen"
