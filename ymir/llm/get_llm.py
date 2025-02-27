from ymir.llm import (
    get_openai_llm,
    get_google_llm,
    get_deepseek_llm,
    get_ollama_llm,
    OPENAI_CHAT_MODELS,
    GOOGLE_CHAT_MODELS,
    DEEPSEEK_CHAT_MODELS,
    OLLAMA_CHAT_MODELS,
    get_deepseek_config_components,
    get_google_config_components,
    get_ollama_config_components,
    get_openai_config_components,
)
from typing import Union, Dict, List
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from loguru import logger


def get_llm(
    model_name: str, **kwargs
) -> Union[ChatOpenAI, ChatGoogleGenerativeAI, ChatDeepSeek, ChatOllama]:
    """
    Get an LLM instance based on the model name.

    Args:
        model_name (str): The name of the model to get.
        **kwargs: Additional keyword arguments to pass to the LLM constructor.
    """
    if model_name in OPENAI_CHAT_MODELS:
        logger.info(f"Using OpenAI model: {model_name}")
        return get_openai_llm(model_name, **kwargs)
    elif model_name in GOOGLE_CHAT_MODELS:
        logger.info(f"Using Google model: {model_name}")
        return get_google_llm(model_name, **kwargs)
    elif model_name in DEEPSEEK_CHAT_MODELS:
        logger.info(f"Using DeepSeek model: {model_name}")
        return get_deepseek_llm(model_name, **kwargs)
    else:
        logger.info(f"Using Ollama model: {model_name}")
        return get_ollama_llm(model_name, **kwargs)


def get_supported_configurations(model_name: str = None) -> Dict[str, Dict]:
    """
    Get the supported configurations for a model.

    Args:
        model_name (str): The name of the model to get configurations for.

    Returns:
        Dict[str, Dict]: A dictionary of configuration options with their settings.
    """
    if model_name is None:
        # Return all providers and their models
        return get_all_providers_and_models()

    if model_name in OPENAI_CHAT_MODELS:
        return get_openai_config_components(model_name)
    elif model_name in GOOGLE_CHAT_MODELS:
        return get_google_config_components(model_name)
    elif model_name in DEEPSEEK_CHAT_MODELS:
        return get_deepseek_config_components(model_name)
    else:
        return get_ollama_config_components(model_name)


def get_all_providers_and_models() -> List[Dict]:
    """
    Get all available providers and their models.

    Returns:
        List[Dict]: A list of dictionaries containing provider information and their models.
    """
    return [
        {
            "provider": "OpenAI",
            "models": OPENAI_CHAT_MODELS,
        },
        {
            "provider": "Google",
            "models": GOOGLE_CHAT_MODELS,
        },
        {
            "provider": "DeepSeek",
            "models": DEEPSEEK_CHAT_MODELS,
        },
        {
            "provider": "Ollama",
            "models": OLLAMA_CHAT_MODELS,
        },
    ]
