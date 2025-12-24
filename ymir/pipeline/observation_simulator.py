"""Simulates tool execution and generates realistic responses."""

import json
from typing import Any

from loguru import logger

from ymir.core import ToolCall, ToolResult
from .llm import OllamaLLM


class ObservationSimulator:
    """Simulates tool execution and generates realistic responses."""

    def __init__(self, llm: OllamaLLM | None = None):
        self.llm = llm
        self._mock_data: dict[str, Any] = {}

    def set_mock_data(self, mock_data: dict[str, Any]) -> None:
        """Set mock data for tool responses."""
        self._mock_data = mock_data

    async def execute(
        self,
        tool_call: ToolCall,
        mock_data: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Execute a tool call and return simulated result.

        Uses mock data if provided, otherwise generates plausible responses.
        """
        # Merge mock data sources
        all_mock_data = {**self._mock_data, **(mock_data or {})}

        # Check for mock data first
        if tool_call.name in all_mock_data:
            mock_result = all_mock_data[tool_call.name]

            # If mock is a callable, call it with arguments
            if callable(mock_result):
                result = mock_result(tool_call.arguments)
            else:
                result = mock_result

            return ToolResult(
                tool_name=tool_call.name,
                result=result if isinstance(result, dict) else {"result": result},
                success=True,
            )

        # Generate plausible response using LLM if available
        if self.llm:
            return await self._generate_plausible_response(tool_call)

        # Fallback: return a generic success response
        return ToolResult(
            tool_name=tool_call.name,
            result={
                "status": "success",
                "message": f"Tool {tool_call.name} executed successfully",
                "data": tool_call.arguments,
            },
            success=True,
        )

    async def _generate_plausible_response(self, tool_call: ToolCall) -> ToolResult:
        """Use LLM to generate a plausible tool response."""
        prompt = f"""Generate a realistic JSON response for this function call.
The response should be what the function would actually return in a real system.

Function: {tool_call.name}
Arguments: {json.dumps(tool_call.arguments, indent=2)}

Respond with ONLY valid JSON (no markdown, no explanation). The response should be realistic and useful for continuing a conversation."""

        try:
            response = await self.llm.agenerate(
                messages=[{"role": "user", "content": prompt}],
            )

            # Try to parse as JSON
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])

            result = json.loads(response)
            return ToolResult(
                tool_name=tool_call.name,
                result=result if isinstance(result, dict) else {"result": result},
                success=True,
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return ToolResult(
                tool_name=tool_call.name,
                result={"raw_response": response, "parse_error": str(e)},
                success=False,
            )
        except Exception as e:
            logger.error(f"Error generating tool response: {e}")
            return ToolResult(
                tool_name=tool_call.name,
                result={"error": str(e)},
                success=False,
            )

    def execute_sync(
        self,
        tool_call: ToolCall,
        mock_data: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Synchronous version of execute."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.execute(tool_call, mock_data))
