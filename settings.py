
from pydantic import BaseModel

from cat import plugin
from cat.log import log

class LangChainModelsPackSettings(BaseModel):
    openai_key: str = ""
    anthropic_key: str = ""
    ollama_key: str = "stocaz"

@plugin
def settings_model():
    return LangChainModelsPackSettings