from typing import Dict, Optional
from langchain_deepseek import ChatDeepSeek

DEEPSEEK_CHAT_MODELS = [
    "deepseek-chat",
    "deepseek-reasoner",
]


def get_deepseek_config_components(
    model_name: Optional[str] = None,
) -> Dict[str, Dict]:
    """Returns configuration options for DeepSeek model configuration."""
    return {
        "temperature": {
            "type": "slider",
            "min": 0.0,
            "max": 2.0,
            "default": 0.7,
            "step": 0.1,
            "label": "Temperature",
        },
        "max_tokens": {
            "type": "slider",
            "min": 50,
            "max": 4000,
            "default": 1000,
            "step": 50,
            "label": "Max Tokens",
        },
        "top_p": {
            "type": "slider",
            "min": 0.0,
            "max": 1.0,
            "default": 0.9,
            "step": 0.05,
            "label": "Top P",
        },
    }


def get_deepseek_llm(model_name: str, config: Dict = None) -> ChatDeepSeek:
    if config is None:
        config = {}
    return ChatDeepSeek(model=model_name, **config)
