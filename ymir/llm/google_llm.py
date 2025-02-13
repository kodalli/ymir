from typing import Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
import gradio as gr

GOOGLE_CHAT_MODELS = [
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-exp",
]


def get_google_config_components(
    model_name: Optional[str] = None,
) -> Dict[str, gr.Component]:
    """Returns Gradio components for Google model configuration."""
    return {
        "temperature": gr.Slider(
            minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Temperature"
        ),
        "top_p": gr.Slider(
            minimum=0.0, maximum=1.0, value=0.95, step=0.05, label="Top P"
        ),
        "top_k": gr.Slider(minimum=1, maximum=40, value=40, step=1, label="Top K"),
    }


def get_google_llm(model_name: str, config: Dict = None) -> ChatGoogleGenerativeAI:
    if config is None:
        config = {}
    return ChatGoogleGenerativeAI(model=model_name, **config)
