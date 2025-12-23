"""Simulates a user/patient based on a persona and background."""

import json
from typing import Any
from loguru import logger

from ymir.core import Message, MessageRole
from ymir.llm import OllamaLLM


class UserSimulator:
    """Simulates a user (e.g., a patient) based on a persona and background."""

    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.7,
    ):
        self.model = model
        self.temperature = temperature
        self.llm = OllamaLLM(model=model, temperature=temperature)

    def _build_system_prompt(self, background: str, goal: str) -> str:
        """Build the system prompt for the user simulator."""
        return f"""You are a patient participating in a medical simulation. 

YOUR BACKGROUND:
{background}

YOUR GOAL:
{goal}

INSTRUCTIONS:
- Stay in character at all times.
- Respond to the medical assistant naturally.
- Provide information from your background when asked.
- If the assistant asks for something not in your background, you can make up a realistic detail consistent with your persona.
- Keep your responses concise and conversational, like a real person would.
- Do NOT mention that you are an AI or a simulation.
- Your response should be just your message to the assistant."""

    async def generate_response(
        self,
        background: str,
        goal: str,
        history: list[Message],
    ) -> str:
        """Generate the next user message based on conversation history."""
        system_prompt = self._build_system_prompt(background, goal)
        
        # Convert history to LLM format
        llm_messages = []
        for msg in history:
            if msg.role == MessageRole.USER:
                llm_messages.append({"role": "assistant", "content": msg.content}) # User is assistant to the simulator
            elif msg.role == MessageRole.ASSISTANT:
                llm_messages.append({"role": "user", "content": msg.content}) # Assistant is user to the simulator
        
        try:
            response = await self.llm.agenerate(
                messages=llm_messages,
                system=system_prompt,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"User simulator error: {e}")
            return "I'm sorry, can you repeat that?"

