from typing import List
from pydantic import BaseModel, Field

from cat import log
from ..adapters import OpenAICompatibleProvider


class OpenAI(OpenAICompatibleProvider):
    """OpenAI models."""

    slug = "openai"
    description = "OpenAI models via the OpenAI API."

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
        if not api_key:
            self.client = None
            self._llms_cache: List[str] = []
            self._embedders_cache: List[str] = []
            return

        self.client = AsyncOpenAI(api_key=api_key)
        await self._refresh_model_lists()

    async def _refresh_model_lists(self):
        import re

        try:
            models = await self.client.models.list()
            all_ids = [m.id for m in models.data]
            llm_pattern = re.compile(r"gpt-\d|o\d+-")
            self._llms_cache = [m for m in all_ids if llm_pattern.search(m)]
            self._embedders_cache = [m for m in all_ids if "embed" in m]
        except Exception as e:
            log.error(f"OpenAI: failed to fetch model list: {e}")
            self._llms_cache = []
            self._embedders_cache = []

