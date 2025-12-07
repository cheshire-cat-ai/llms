from langchain_anthropic import ChatAnthropic
    

    # if "anthropic_key" in settings:
    #     vendor = "anthropic"
        
    #     for m in ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"]:
    #         slug = f"{vendor}:{m}"  # "anthropic:claude-sonnet-4-5"
    #         models[slug] = ChatAnthropic(
    #             model_name = m,
    #             api_key = settings["anthropic_key"],
    #             temperature = 0.2,
    #             streaming = True
    #         )