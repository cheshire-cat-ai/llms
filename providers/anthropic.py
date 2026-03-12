from typing import List, TYPE_CHECKING
from pydantic import BaseModel, Field

from cat import log
from cat.services.model_providers.base import ModelProvider
from cat.protocols.model_context.type_wrappers import TextContent, ImageContent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from cat.types import Message
    from cat.mad_hatter.decorators import Tool


class Anthropic(ModelProvider):
    """Anthropic Claude models."""

    slug = "anthropic"
    description = "Anthropic Claude models via the Anthropic API."

    class Settings(BaseModel):
        anthropic_key: str = Field(
            default="",
            title="Anthropic API Key",
            description="Your Anthropic API key.",
        )

    async def setup(self):
        from anthropic import AsyncAnthropic

        settings = await self.load_settings()
        api_key = settings.anthropic_key
        if not api_key:
            self.client = None
            self._llms_cache: List[str] = []
            return

        self.client = AsyncAnthropic(api_key=api_key)
        try:
            models = await self.client.models.list()
            self._llms_cache = [m.id for m in models.data]
        except Exception as e:
            log.error(f"Anthropic: failed to fetch model list: {e}")
            self._llms_cache = []

    async def list_llms(self) -> List[str]:
        return getattr(self, "_llms_cache", [])

    async def list_embedders(self) -> List[str]:
        return []

    def _build_messages(self, messages: list["Message"]) -> list[dict]:
        """Convert Cat messages to Anthropic format (system prompt excluded)."""
        result = []
        for msg in messages:
            if msg.role == "tool":
                result.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.text,
                    }],
                })
            elif msg.role == "assistant" and msg.tool_calls:
                content = []
                if msg.text:
                    content.append({"type": "text", "text": msg.text})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
                result.append({"role": "assistant", "content": content})
            else:
                content = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        content.append({"type": "text", "text": block.text})
                    elif isinstance(block, ImageContent):
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": block.mimeType,
                                "data": block.data,
                            },
                        })
                result.append({"role": msg.role, "content": content})
        return result

    def _build_tools(self, tools: list["Tool"]) -> list[dict]:
        """Convert Cat tools to Anthropic tool format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]

    def _parse_response(self, response) -> "Message":
        """Convert Anthropic response to Cat Message."""
        from cat.types import Message

        text = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "args": block.input,
                })

        return Message(
            role="assistant",
            content=[TextContent(type="text", text=text)],
            tool_calls=tool_calls,
        )

    async def llm(
        self,
        model: str,
        messages: list["Message"],
        system_prompt: str = "",
        tools: list["Tool"] = [],
        on_token: "Callable[[str], Awaitable[None]] | None" = None,
    ) -> "Message":
        from cat.types import Message

        anthropic_messages = self._build_messages(messages)
        anthropic_tools = self._build_tools(tools) if tools else []

        kwargs = {
            "model": model,
            "max_tokens": 8096,
            "messages": anthropic_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if on_token:
            full_text = ""

            async with self.client.messages.stream(**kwargs) as stream:
                async for text_delta in stream.text_stream:
                    full_text += text_delta
                    await on_token(text_delta)
                final = await stream.get_final_message()

            tool_calls = [
                {"id": block.id, "name": block.name, "args": block.input}
                for block in final.content
                if block.type == "tool_use"
            ]

            return Message(
                role="assistant",
                content=[TextContent(type="text", text=full_text)],
                tool_calls=tool_calls,
            )

        response = await self.client.messages.create(**kwargs)
        return self._parse_response(response)
