import os
import json
import httpx
from typing import List
from langchain_core.embeddings import Embeddings


class CustomOllamaEmbeddings(Embeddings):
    """Use Ollama to serve embedding models."""

    def __init__(self, base_url, model):
        self.url = os.path.join(base_url, "api/embed")
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model , "input": texts})
        ret = httpx.post(self.url, data=payload, timeout=None)
        ret.raise_for_status()
        return ret.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        payload = json.dumps({"model": self.model , "input": text})
        ret = httpx.post(self.url, data=payload, timeout=None)
        ret.raise_for_status()
        return ret.json()["embeddings"][0]
    

class CustomOpenAIEmbeddings(Embeddings):
    """Use LLAMA2 as embedder by calling a self-hosted lama-cpp-python instance."""

    def __init__(self, url):
        self.url = os.path.join(url, "v1/embeddings")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({"input": texts})
        ret = httpx.post(self.url, data=payload, timeout=None)
        ret.raise_for_status()
        return [e["embedding"] for e in ret.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        payload = json.dumps({"input": text})
        ret = httpx.post(self.url, data=payload, timeout=None)
        ret.raise_for_status()
        return ret.json()["data"][0]["embedding"]