
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from cat.provider import BaseModelProvider

class OpenAI(BaseModelProvider):
    """OpenAI models."""
    
    async def setup(self, cat):
        
        settings = await cat.plugin.load_settings()
        self.api_key = settings.get("openai_key", None)
        
        self.llms = {}
        self.embedders = {}

        if self.api_key:
            # TODOV2: pull dynamically from OpenAI API
            llm_names = [
                "gpt-5-nano",
                "gpt-5-mini",
                "gpt-5",
                "gpt-4.1",
                "gpt-4",
                "gpt-4o"
            ]

            for m in llm_names:
                # "openai:gpt-5"
                self.llms[f"{self.slug}:{m}"] = ChatOpenAI(
                    model = m,
                    api_key = self.api_key,
                    temperature = 0.1,
                    streaming = True
                )

            # TODOV2: pull dynamically from OpenAI API
            embedders_names = [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]
            for em in embedders_names:
                self.embedders[f"{self.slug}:{em}"] = OpenAIEmbeddings(
                    model = em,
                    api_key = self.api_key,
                )



