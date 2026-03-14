import re
from pydantic import BaseModel, Field

from ..adapters import OpenAICompatibleProvider


class OpenAI(OpenAICompatibleProvider):
    """OpenAI models."""

    slug = "openai"
    description = "OpenAI models via the OpenAI API."

    _llm_pattern = re.compile(r"gpt-\d|o\d+-")

    class Settings(BaseModel):
        openai_key: str = Field(
            default="",
            title="OpenAI API Key",
            description="Your OpenAI API key.",
        )

    async def setup(self):
        from openai import AsyncOpenAI

        settings = await self.load_settings()
        api_key = settings.openai_key
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None

    def _is_llm(self, model_id: str) -> bool:
        return bool(self._llm_pattern.search(model_id))
