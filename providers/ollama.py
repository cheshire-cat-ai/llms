from pydantic import BaseModel, Field

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
        self.client = AsyncOpenAI(
            base_url=f"{settings.host.rstrip('/')}/v1",
            api_key=settings.key or "ollama",
        )
