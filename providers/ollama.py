from typing import List
from pydantic import BaseModel, Field

from cat import log
from ..adapters import OpenAICompatibleProvider


class Ollama(OpenAICompatibleProvider):
    """Ollama models."""

    slug = "ollama"
    description = "Locally running Ollama models."

    class Settings(BaseModel):
        host: str = Field(
            default="http://localhost:11434",
            title="Ollama Host",
            description="The host URL for the Ollama API.",
        )
        key: str = Field(
            default="ollama",
            title="Ollama API Key",
            description="The API key for authenticating with the Ollama API.",
        )

    async def setup(self):
        from openai import AsyncOpenAI

        settings = await self.load_settings()
        host = settings.host
        key = settings.key

        self.client = AsyncOpenAI(
            base_url=f"{host.rstrip('/')}/v1",
            api_key=key or "ollama",
        )
        await self._refresh_model_lists()

    async def _refresh_model_lists(self):
        try:
            models = await self.client.models.list()
            all_ids: List[str] = [m.id for m in models.data]
            self._llms_cache = [m for m in all_ids if "embed" not in m]
            self._embedders_cache = [m for m in all_ids if "embed" in m]
        except Exception as e:
            log.error(f"Ollama: failed to fetch model list: {e}")
            self._llms_cache = []
            self._embedders_cache = []
