import json
from typing import List, TYPE_CHECKING

from cat.services.model_providers.base import ModelProvider
from cat.protocols.model_context.type_wrappers import TextContent, ImageContent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from openai import AsyncOpenAI
    from cat.types import Message
    from cat.mad_hatter.decorators import Tool


class OpenAICompatibleProvider(ModelProvider):
    """
    Base class for OpenAI-compatible providers.

    Subclasses only need to:
    - Set slug and description class attributes
    - Set self.client = AsyncOpenAI(...) in setup()

    All format conversion methods are public and overridable.
    """

    service_type = "model_providers"

    async def list_llms(self) -> List[str]:
        return getattr(self, "_llms_cache", [])

    async def list_embedders(self) -> List[str]:
        return getattr(self, "_embedders_cache", [])

    async def build_messages(self, messages: list["Message"], system_prompt: str) -> list[dict]:
        """Convert Cat messages to OpenAI message format."""
        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        for msg in messages:
            result.append(await self.convert_message(msg))
        return result

    async def convert_message(self, msg: "Message") -> dict:
        """Convert a single Cat Message to OpenAI format."""
        if msg.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.text,
            }

        if msg.role == "assistant" and msg.tool_calls:
            return {
                "role": "assistant",
                "content": msg.text or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }

        # user or plain assistant message
        content = []
        for block in msg.content:
            if isinstance(block, TextContent):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.mimeType};base64,{block.data}"},
                })

        # Flatten to string if only one text block
        if len(content) == 1 and content[0]["type"] == "text":
            return {"role": msg.role, "content": content[0]["text"]}
        return {"role": msg.role, "content": content}

    def build_tools(self, tools: list["Tool"]) -> list[dict]:
        """Convert Cat tools to OpenAI tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    def parse_response(self, response) -> "Message":
        """Convert OpenAI response to Cat Message."""
        from cat.types import Message

        choice = response.choices[0]
        oai_msg = choice.message

        text = oai_msg.content or ""
        tool_calls = []

        if oai_msg.tool_calls:
            for tc in oai_msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                })

        return Message(
            role="assistant",
            content=[TextContent(type="text", text=text)],
            tool_calls=tool_calls,
        )

    async def stream_completion(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        on_token: "Callable[[str], Awaitable[None]]",
    ) -> "Message":
        """Stream completion and return complete Message."""
        from cat.types import Message

        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        full_text = ""
        tool_calls_acc: dict[int, dict] = {}

        stream = await self.client.chat.completions.create(stream=True, **kwargs)
        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            if delta.content:
                full_text += delta.content
                await on_token(delta.content)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "args_str": ""}
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["args_str"] += tc_delta.function.arguments

        tool_calls = [
            {"id": tc["id"], "name": tc["name"], "args": json.loads(tc["args_str"] or "{}")}
            for tc in tool_calls_acc.values()
        ]

        return Message(
            role="assistant",
            content=[TextContent(type="text", text=full_text)],
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
        oai_messages = await self.build_messages(messages, system_prompt)
        oai_tools = self.build_tools(tools) if tools else []

        if on_token:
            return await self.stream_completion(model, oai_messages, oai_tools, on_token)

        kwargs = {"model": model, "messages": oai_messages}
        if oai_tools:
            kwargs["tools"] = oai_tools

        response = await self.client.chat.completions.create(**kwargs)
        return self.parse_response(response)

    async def embed(self, model: str, text: str) -> list[float]:
        """Embed text using OpenAI embeddings API."""
        response = await self.client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
