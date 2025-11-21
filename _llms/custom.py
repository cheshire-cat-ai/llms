
from langchain_openai.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama


class CustomOpenAI(ChatOpenAI):
    url: str
    # key: str TODOV2: much requested

    def __init__(self, **kwargs):
        super().__init__(model_kwargs={}, base_url=kwargs["url"], **kwargs)


class CustomOllama(ChatOllama):
    def __init__(self, **kwargs) -> None:
        if kwargs["base_url"].endswith("/"):
            kwargs["base_url"] = kwargs["base_url"][:-1]
        super().__init__(**kwargs)