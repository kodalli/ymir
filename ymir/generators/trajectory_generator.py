"""Multi-turn trajectory generator for agentic training data."""

import json
from typing import Any

from loguru import logger

from ymir.core import (
    Message,
    MessageRole,
    ToolCall,
    Trajectory,
    TOOL_CALL_START,
    TOOL_CALL_END,
)
from ymir.core.constants import DEFAULT_AGENT_SYSTEM_PROMPT
from ymir.functions import ScenarioTemplate
from ymir.llm import OllamaLLM

from .observation_simulator import ObservationSimulator
from .user_simulator import UserSimulator


class TrajectoryGenerator:
    """Generates multi-turn tool-calling trajectories for training."""

    def __init__(
        self,
        model: str = "llama3.2",
        max_turns: int = 10,
        temperature: float = 0.7,
    ):
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.llm: OllamaLLM | None = None
        self.observation_simulator: ObservationSimulator | None = None
        self.user_simulator: UserSimulator | None = None

    def _ensure_llm(self) -> None:
        """Ensure LLM is initialized."""
        if self.llm is None:
            self.llm = OllamaLLM(
                model=self.model,
                temperature=self.temperature,
            )
            self.observation_simulator = ObservationSimulator(llm=self.llm)
            self.user_simulator = UserSimulator(
                model=self.model,
                temperature=self.temperature,
            )

    def _build_system_prompt(self, scenario: ScenarioTemplate) -> str:
        """Build system prompt with tool definitions."""
        return self._build_system_prompt_with_functions(scenario, scenario.functions)

    def _build_system_prompt_with_functions(
        self, scenario: ScenarioTemplate, functions: list
    ) -> str:
        """Build system prompt with specific tool definitions."""
        tools_text = self._format_tools_for_prompt(functions)

        base_prompt = scenario.system_prompt or DEFAULT_AGENT_SYSTEM_PROMPT

        return f"""{base_prompt}

Available tools:
{tools_text}"""

    def _format_tools_for_prompt(self, functions: list) -> str:
        """Format tool definitions for inclusion in prompt."""
        lines = []
        for func in functions:
            params = func.parameters.get("properties", {})
            required = func.parameters.get("required", [])

            param_strs = []
            for name, prop in params.items():
                req_marker = " (required)" if name in required else ""
                param_strs.append(f"    - {name}: {prop.get('type', 'any')}{req_marker}")

            lines.append(f"- {func.name}: {func.description}")
            if param_strs:
                lines.append("  Parameters:")
                lines.extend(param_strs)

        return "\n".join(lines)

    def _build_messages_for_llm(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to LLM-compatible format."""
        result = []
        for msg in messages:
            if msg.role == MessageRole.USER:
                result.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                result.append({"role": "assistant", "content": msg.content})
            elif msg.role == MessageRole.TOOL_RESULT:
                # Format tool results as assistant-style messages
                result.append({
                    "role": "user",
                    "content": f"Tool result from {msg.tool_result.tool_name if msg.tool_result else 'unknown'}:\n{msg.content}",
                })
        return result

    def _filter_functions(
        self, scenario: ScenarioTemplate, enabled_tools: list[str] | None
    ) -> list:
        """Filter scenario functions to only include enabled tools."""
        if enabled_tools is None:
            return scenario.functions
        return [f for f in scenario.functions if f.name in enabled_tools]

    async def generate(
        self,
        scenario: ScenarioTemplate,
        user_query: str,
        user_background: str | None = None,
        user_goal: str | None = None,
        enabled_tools: list[str] | None = None,
    ) -> Trajectory:
        """
        Generate a multi-turn trajectory.
        If user_background and user_goal are provided, it simulates a conversation.
        Otherwise, it generates a single-query response trajectory.

        Args:
            enabled_tools: Optional list of tool names to enable. If None, all tools are enabled.
        """
        if user_background and user_goal:
            return await self.generate_simulated(
                scenario, user_background, user_goal, user_query, enabled_tools
            )

        return await self._generate_single_turn(scenario, user_query, enabled_tools)

    async def _generate_single_turn(
        self,
        scenario: ScenarioTemplate,
        user_query: str,
        enabled_tools: list[str] | None = None,
    ) -> Trajectory:
        """
        Original generation flow: one user query, agent acts until done.
        """
        self._ensure_llm()

        messages: list[Message] = []
        functions = self._filter_functions(scenario, enabled_tools)
        tools = [f.to_openai_format() for f in functions]
        system_prompt = self._build_system_prompt_with_functions(scenario, functions)

        # Set mock data for observation simulator
        if self.observation_simulator and scenario.mock_responses:
            self.observation_simulator.set_mock_data(scenario.mock_responses)

        # Add initial user message
        messages.append(Message(role=MessageRole.USER, content=user_query))

        turn = 0
        while turn < self.max_turns:
            # Build conversation for LLM
            llm_messages = self._build_messages_for_llm(messages)

            # Generate LLM response
            try:
                response = await self.llm.agenerate(
                    messages=llm_messages,
                    system=system_prompt,
                    tools=tools,
                )
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                break

            # Parse response for tool calls
            tool_calls = ToolCall.parse(response)

            if tool_calls:
                # Add assistant message with tool calls
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response,
                        tool_calls=tool_calls,
                    )
                )

                # Simulate tool execution for each tool call
                for tool_call in tool_calls:
                    result = await self.observation_simulator.execute(
                        tool_call, scenario.mock_responses
                    )
                    messages.append(
                        Message(
                            role=MessageRole.TOOL_RESULT,
                            content=json.dumps(result.result, indent=2),
                            tool_result=result,
                        )
                    )
            else:
                # Final response - no more tool calls
                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response,
                    )
                )
                break

            turn += 1

        return Trajectory(
            scenario_id=scenario.id,
            scenario_description=scenario.description,
            tools=tools,
            system_prompt=system_prompt,
            messages=messages,
            source="generated",
        )

    async def generate_simulated(
        self,
        scenario: ScenarioTemplate,
        user_background: str,
        user_goal: str,
        initial_query: str | None = None,
        enabled_tools: list[str] | None = None,
    ) -> Trajectory:
        """
        Simulate a full interaction between a user (patient) and an agent.
        """
        self._ensure_llm()
        messages: list[Message] = []
        functions = self._filter_functions(scenario, enabled_tools)
        tools = [f.to_openai_format() for f in functions]
        system_prompt = self._build_system_prompt_with_functions(scenario, functions)

        if self.observation_simulator and scenario.mock_responses:
            self.observation_simulator.set_mock_data(scenario.mock_responses)

        # 1. Start with initial query or generate one from background
        if initial_query:
            current_user_msg = initial_query
        else:
            current_user_msg = await self.user_simulator.generate_response(
                user_background, user_goal, []
            )

        conv_turn = 0
        max_conv_turns = 5 # Prevent infinite loops

        while conv_turn < max_conv_turns:
            # --- AGENT TURN ---
            messages.append(Message(role=MessageRole.USER, content=current_user_msg))
            
            # Agent reasoning loop (tool calls)
            agent_done = False
            agent_loop_turn = 0
            while not agent_done and agent_loop_turn < self.max_turns:
                llm_messages = self._build_messages_for_llm(messages)
                response = await self.llm.agenerate(
                    messages=llm_messages, system=system_prompt, tools=tools
                )
                
                tool_calls = ToolCall.parse(response)
                if tool_calls:
                    messages.append(Message(role=MessageRole.ASSISTANT, content=response, tool_calls=tool_calls))
                    for tool_call in tool_calls:
                        result = await self.observation_simulator.execute(tool_call, scenario.mock_responses)
                        messages.append(Message(role=MessageRole.TOOL_RESULT, content=json.dumps(result.result, indent=2), tool_result=result))
                else:
                    messages.append(Message(role=MessageRole.ASSISTANT, content=response))
                    agent_done = True
                agent_loop_turn += 1

            # --- USER TURN ---
            # Generate user's next response to the agent's final message
            current_user_msg = await self.user_simulator.generate_response(
                user_background, user_goal, messages
            )
            
            # If the agent has provided a final confirmation or the task seems done, we might break
            # For now, let's just use max_conv_turns
            conv_turn += 1
            
            # Heuristic to detect if conversation is over (e.g. user says "Thanks")
            if any(word in current_user_msg.lower() for word in ["thank you", "thanks", "that's all", "goodbye"]):
                messages.append(Message(role=MessageRole.USER, content=current_user_msg))
                break

        return Trajectory(
            scenario_id=scenario.id,
            scenario_description=scenario.description,
            tools=tools,
            system_prompt=system_prompt,
            messages=messages,
            source="simulated",
        )

    def generate_sync(
        self,
        scenario: ScenarioTemplate,
        user_query: str,
    ) -> Trajectory:
        """Synchronous version of generate."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.generate(scenario, user_query))

    async def generate_batch(
        self,
        scenario: ScenarioTemplate,
        queries: list[str],
        parallel: int = 3,
    ) -> list[Trajectory]:
        """Generate multiple trajectories, with limited parallelism."""
        import asyncio

        results = []
        for i in range(0, len(queries), parallel):
            batch = queries[i : i + parallel]
            tasks = [self.generate(scenario, query) for query in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch generation error: {result}")
                else:
                    results.append(result)

        return results
