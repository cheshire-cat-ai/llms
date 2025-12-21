from typing import List
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from cat.base import ModelProvider

class Ollama(ModelProvider):
    """Ollama models."""

    async def setup(self):
        """Load configuration from settings."""
        settings = await self.plugin.load_settings()
        self.host = settings.get("host", None)
        self.key = settings.get("key", None)

    def list_llms(self) -> List[str]:
        """Return list of available LLM slugs."""
        # TODOV2: pull dynamically from Ollama API
        return [
            "gpt-oss"
        ]

    def list_embedders(self) -> List[str]:
        """Return list of available embedder slugs."""
        # TODOV2: pull dynamically from Ollama API
        return [
            "embeddinggemma:300m"
        ]

    async def get_llm(self, slug: str) -> BaseChatModel:
        """Create and return Ollama LLM instance."""
        return ChatOllama(
            base_url=self.host,
            model=slug,
            api_key=self.key,
            temperature=0.1,
        )

    async def get_embedder(self, slug: str) -> Embeddings:
        """Create and return Ollama embedder instance."""
        return OllamaEmbeddings(
            base_url=self.host,
            model=slug,
        )
    
    async def settings_model(self):
        """Return settings model."""

        class OllamaSettings(BaseModel):
            host: str = Field(
                default="",
                title="Ollama Host",
                description="The host URL for the Ollama API.",
            )
            key: str = Field(
                default="",
                title="Ollama API Key",
                description="The API key for authenticating with the Ollama API.",
            )

        return OllamaSettings