"""Constants for the agentic dataset generator."""

# Tool call format tags - matches the format used in llm-agent-training
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

# Quality thresholds for trajectory scoring
MIN_TURNS_FOR_QUALITY = 2
MAX_TURNS_FOR_QUALITY = 10
IDEAL_TURN_RANGE = (2, 5)

# Default system prompts
DEFAULT_AGENT_SYSTEM_PROMPT = """You are a helpful assistant with access to tools. When you need to use a tool, respond with a tool call in the following format:

<tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>

After receiving tool results, continue reasoning and call more tools if needed, or provide your final response to the user."""
