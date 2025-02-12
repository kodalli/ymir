from typing import Literal, Dict
from langchain_deepseek import ChatDeepSeek
import gradio as gr

DEEPSEEK_CHAT_MODELS = Literal[
    "deepseek-chat",
    "deepseek-reasoner",
]


def get_deepseek_config_components() -> Dict[str, gr.Component]:
    """Returns Gradio components for DeepSeek model configuration."""
    return {
        "temperature": gr.Slider(
            minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"
        ),
        "max_tokens": gr.Slider(
            minimum=50, maximum=4000, value=1000, step=50, label="Max Tokens"
        ),
        "top_p": gr.Slider(
            minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top P"
        ),
    }


def get_deepseek_llm(
    model_name: DEEPSEEK_CHAT_MODELS, config: Dict = None
) -> ChatDeepSeek:
    if config is None:
        config = {}
    return ChatDeepSeek(model=model_name, **config)
