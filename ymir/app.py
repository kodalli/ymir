import gradio as gr
from ymir.llm import (
    get_llm,
    get_supported_configurations,
    OLLAMA_CHAT_MODELS,
    OPENAI_CHAT_MODELS,
    GOOGLE_CHAT_MODELS,
    DEEPSEEK_CHAT_MODELS,
)
from ymir.rlhf import RLHFDatasetBuilder
from langchain_core.messages import convert_to_openai_messages
from loguru import logger
import re

rlhf_builder = RLHFDatasetBuilder()


def to_langchain_messages(message, history):
    messages = []
    for m in history:
        messages.append((m["role"], m["content"]))
    messages.append(("user", message))
    return messages


def convert_reasoning_to_markdown(message):
    match = re.search(r"<think>.*?</think>", message, flags=re.DOTALL)
    if match:
        think_content = match.group(0).strip()
        think_md = "```think\n" + think_content + "\n```\n"
        message = re.sub(r"<think>.*?</think>", think_md, message, flags=re.DOTALL)
    return message


def generate_response(llm, message, history):
    langchain_messages = to_langchain_messages(message, history)
    response = llm.invoke(langchain_messages)
    openai_message = convert_to_openai_messages(response)
    final_message = convert_reasoning_to_markdown(openai_message["content"])
    return final_message


history_1 = [
    gr.ChatMessage(role="assistant", content="How can I help you?"),
    gr.ChatMessage(role="user", content="What is the capital of France?"),
    gr.ChatMessage(role="assistant", content="The capital of France is Paris."),
]

history_2 = [
    gr.ChatMessage(role="assistant", content="How can I help you?"),
    gr.ChatMessage(role="user", content="What is the capital of France?"),
    gr.ChatMessage(role="assistant", content="The capital of France is Paris."),
]

provider_map = {
    "Ollama": OLLAMA_CHAT_MODELS,
    "OpenAI": OPENAI_CHAT_MODELS,
    "Google": GOOGLE_CHAT_MODELS,
    "DeepSeek": DEEPSEEK_CHAT_MODELS,
}


def change_llm_dropdown_1(provider):
    logger.debug(f"Changing LLM 1 provider to {provider}")
    return gr.Dropdown(choices=provider_map[provider], label="LLM 1")


def change_llm_dropdown_2(provider):
    logger.debug(f"Changing LLM 2 provider to {provider}")
    return gr.Dropdown(choices=provider_map[provider], label="LLM 2")


def generate_response_1(message, history):
    return generate_response(llm_1, message, history)


def generate_response_2(message, history):
    return generate_response(llm_2, message, history)


with gr.Blocks() as demo:
    gr.Markdown("# Ymir")
    gr.Markdown(
        "**RLHF Dataset Creation:** Compare two model outputs side by side and rate which one is better."
    )

    with gr.Row():
        chatbot_1 = gr.Chatbot(
            history_1, type="messages", min_height=800, label="LLM 1"
        )
        chatbot_2 = gr.Chatbot(
            history_2, type="messages", min_height=800, label="LLM 2"
        )

    input_box = gr.Textbox(placeholder="Enter your message here", label="Input")

    with gr.Sidebar(width=500):
        gr.Markdown("## Settings")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    llm_provider_1 = gr.Dropdown(
                        choices=provider_map.keys(),
                        value="OpenAI",
                        label="LLM 1 Provider",
                    )
                    llm_dropdown_1 = gr.Dropdown(
                        choices=OPENAI_CHAT_MODELS, label="LLM 1"
                    )

                @gr.render(inputs=llm_dropdown_1)
                def update_llm_1(model_name):
                    global llm_1, llm_config_1
                    llm_1 = get_llm(model_name)
                    llm_config_1 = {}
                    llm_config_map_1 = get_supported_configurations(model_name)
                    for k, v in llm_config_map_1.items():
                        v.change(
                            fn=lambda x: llm_config_1.update({k: x}),
                            inputs=[v],
                            outputs=[],
                        )

            with gr.Column():
                with gr.Row():
                    llm_provider_2 = gr.Dropdown(
                        choices=provider_map.keys(),
                        value="DeepSeek",
                        label="LLM 2 Provider",
                    )
                    llm_dropdown_2 = gr.Dropdown(
                        choices=DEEPSEEK_CHAT_MODELS, label="LLM 2"
                    )

                @gr.render(inputs=llm_dropdown_2)
                def update_llm_2(model_name):
                    global llm_2, llm_config_2
                    llm_2 = get_llm(model_name)
                    llm_config_2 = {}
                    llm_config_map_2 = get_supported_configurations(model_name)
                    for k, v in llm_config_map_2.items():
                        v.change(
                            fn=lambda x: llm_config_2.update({k: x}),
                            inputs=[v],
                            outputs=[],
                        )

    chat_interface_1 = gr.ChatInterface(
        fn=generate_response_1, chatbot=chatbot_1, textbox=input_box, type="messages"
    )
    chat_interface_2 = gr.ChatInterface(
        fn=generate_response_2, chatbot=chatbot_2, textbox=input_box, type="messages"
    )

    llm_provider_1.change(
        fn=change_llm_dropdown_1,
        inputs=[llm_provider_1],
        outputs=[llm_dropdown_1],
    )
    llm_provider_2.change(
        fn=change_llm_dropdown_2,
        inputs=[llm_provider_2],
        outputs=[llm_dropdown_2],
    )

demo.launch()
