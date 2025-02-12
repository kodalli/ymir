from ymir.llm import (
    get_openai_llm,
    get_google_llm,
    get_deepseek_llm,
    get_ollama_llm,
    OPENAI_CHAT_MODELS,
    GOOGLE_CHAT_MODELS,
    DEEPSEEK_CHAT_MODELS,
)
from typing import Union
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from loguru import logger


def get_llm(
    model_name: str, **kwargs
) -> Union[ChatOpenAI, ChatGoogleGenerativeAI, ChatDeepSeek, ChatOllama]:
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
