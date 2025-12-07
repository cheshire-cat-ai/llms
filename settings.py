
from pydantic import BaseModel

from cat import plugin, hook, log

class Ollama(BaseModel):
    host: str = "http://localhost:11434"
    key: str = ""

class LangChainModelsPackSettings(BaseModel):
    openai_key: str = ""
    anthropic_key: str = ""
    ollama: Ollama = Ollama()

@plugin
def settings_model():
    return LangChainModelsPackSettings
