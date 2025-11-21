from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from cat import hook, plugin
from cat.env import get_env
from cat.log import log

#    openai_key: str = ""
#    anthropic_key: str = ""
#    ollama_key: str = "stocaz"

@hook
async def factory_allowed_llms(models, cat):

    settings = await cat.plugin.load_settings()

    if "openai_key" in settings:
        vendor = "openai"
        
        # build an object for each model
        for m in ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-4.1", "gpt-4", "gpt-4o"]:
            slug = f"{vendor}:{m}"  # "openai:gpt-5"
            models[slug] = ChatOpenAI(
                model = m,
                api_key = settings["openai_key"],
                temperature = 0.2,
                streaming = True
            )

    if "anthropic_key" in settings:
        vendor = "anthropic"
        
        for m in ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"]:
            slug = f"{vendor}:{m}"  # "anthropic:claude-sonnet-4-5"
            models[slug] = ChatAnthropic(
                model_name = m,
                api_key = settings["anthropic_key"],
                temperature = 0.2,
                streaming = True
            )

    return models
