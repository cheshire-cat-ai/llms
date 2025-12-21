from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from cat.base import ModelProvider

class OpenAI(ModelProvider):
    """OpenAI models."""

    async def setup(self):
        """Load API key from settings."""
        settings = await self.plugin.load_settings()
        self.api_key = settings.get("openai_key", None)

    def list_llms(self) -> List[str]:
        """Return list of available LLM slugs."""
        if not self.api_key:
            return []

        # TODOV2: pull dynamically from OpenAI API
        return [
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5",
            "gpt-4.1",
            "gpt-4",
            "gpt-4o"
        ]

    def list_embedders(self) -> List[str]:
        """Return list of available embedder slugs."""
        if not self.api_key:
            return []

        # TODOV2: pull dynamically from OpenAI API
        return [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

    async def get_llm(self, slug: str) -> BaseChatModel:
        """Create and return OpenAI LLM instance."""
        return ChatOpenAI(
            model=slug,
            api_key=self.api_key,
            temperature=0.1,
            streaming=True
        )

    async def get_embedder(self, slug: str) -> Embeddings:
        """Create and return OpenAI embedder instance."""
        return OpenAIEmbeddings(
            model=slug,
            api_key=self.api_key,
        )

    async def settings_model(self):
        """Return settings model."""
        class OpenAISettings(BaseModel):
            openai_key: str = Field(
                default="",
                title="OpenAI API Key",
                description="Your OpenAI API key.",
            )

        return OpenAISettings



