
from langchain_ollama import ChatOllama, OllamaEmbeddings

from cat.provider import BaseModelProvider

class Ollama(BaseModelProvider):
    """Ollama models."""
    
    async def setup(self, cat):
        
        settings = await cat.plugin.load_settings()
        self.host = settings.get("ollama", {}).get("host", None)
        self.key = settings.get("ollama", {}).get("key", None)
        
        self.llms = {}
        self.embedders = {}

        # TODOV2: pull dynamically from Ollama API
        llm_names = [
            "gpt-oss"
        ]

        for m in llm_names:
            # "ollama:gpt-oss"
            self.llms[f"{self.slug}:{m}"] = ChatOllama(
                base_url= self.host,
                model = m,
                api_key = self.key,
                temperature = 0.2,
            )

        embedder_names = [
            "embeddinggemma:300m"
        ]
        for em in embedder_names:
            self.embedders[f"{self.slug}:{em}"] = OllamaEmbeddings(
                base_url= self.host,
                model = em,
            )
