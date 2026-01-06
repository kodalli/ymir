"""LLM-based rule resolver for natural language constraints.

Resolves <rule>...</rule> tags in actor templates using an LLM to interpret
natural language constraints and generate concrete values.

Example rules:
- "next monday or wednesday" -> "2026-01-12"
- "morning before 11am" -> "9:30 AM"
- "this week or next week on tuesday" -> "2026-01-14"
"""

from datetime import datetime

from loguru import logger

from ymir.pipeline.llm.ollama import OllamaLLM
from ymir.pipeline.template_parser import ParsedTag


class RuleResolver:
    """Resolves natural language rules using LLM."""

    SYSTEM_PROMPT = """You are a data generation assistant that converts natural language rules into specific concrete values.

Given a rule describing a constraint (like a date preference, time slot, or other requirement), generate ONE specific value that satisfies the constraint.

Rules:
1. Return ONLY the generated value, nothing else
2. No explanations, no quotes, no formatting
3. Be creative but realistic
4. For dates, use format MM/DD/YYYY
5. For times, use format like "9:30 AM" or "2:00 PM"
6. For text choices, just return the chosen text

Current date context: {current_date}
Day of week: {day_of_week}

Examples:
- Rule: "next monday or wednesday" -> 01/13/2026
- Rule: "morning before 11am" -> 9:30 AM
- Rule: "afternoon after 2pm but before 5pm" -> 3:15 PM
- Rule: "any weekday this week" -> 01/08/2026
- Rule: "prefers email contact" -> email"""

    def __init__(
        self,
        model: str = "qwen3:4b",
        temperature: float = 0.5,  # Lower temp for more consistent outputs
    ):
        """Initialize the rule resolver.

        Args:
            model: Ollama model name to use
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.model = model
        self.temperature = temperature
        self._llm: OllamaLLM | None = None

    def _get_llm(self) -> OllamaLLM:
        """Lazily initialize the LLM client."""
        if self._llm is None:
            self._llm = OllamaLLM(
                model=self.model,
                temperature=self.temperature,
                num_predict=64,  # Short responses only
                num_ctx=1024,  # Small context needed
            )
        return self._llm

    def _get_context(self) -> dict[str, str]:
        """Get current date context for the LLM."""
        now = datetime.now()
        return {
            "current_date": now.strftime("%B %d, %Y"),
            "day_of_week": now.strftime("%A"),
        }

    def resolve(self, rule: str) -> str:
        """Resolve a single natural language rule to a concrete value.

        Args:
            rule: Natural language rule string (e.g., "next monday or wednesday")

        Returns:
            Concrete value that satisfies the rule
        """
        context = self._get_context()
        system = self.SYSTEM_PROMPT.format(**context)

        messages = [{"role": "user", "content": f"Rule: {rule}"}]

        try:
            llm = self._get_llm()
            response = llm.generate(messages, system=system)
            # Clean up response - remove any quotes, whitespace, thinking tags
            result = response.strip().strip('"\'')
            # Remove any <think>...</think> tags if present
            if "<think>" in result.lower():
                import re

                result = re.sub(r"<think>.*?</think>", "", result, flags=re.IGNORECASE | re.DOTALL)
            result = result.strip()
            logger.debug(f"Resolved rule '{rule}' -> '{result}'")
            return result
        except Exception as e:
            logger.error(f"Error resolving rule '{rule}': {e}")
            # Return a placeholder on error
            return f"[rule: {rule}]"

    async def aresolve(self, rule: str) -> str:
        """Resolve a single rule asynchronously.

        Args:
            rule: Natural language rule string

        Returns:
            Concrete value that satisfies the rule
        """
        context = self._get_context()
        system = self.SYSTEM_PROMPT.format(**context)

        messages = [{"role": "user", "content": f"Rule: {rule}"}]

        try:
            llm = self._get_llm()
            response = await llm.agenerate(messages, system=system)
            result = response.strip().strip('"\'')
            # Remove any <think>...</think> tags if present
            if "<think>" in result.lower():
                import re

                result = re.sub(r"<think>.*?</think>", "", result, flags=re.IGNORECASE | re.DOTALL)
            result = result.strip()
            logger.debug(f"Resolved rule '{rule}' -> '{result}'")
            return result
        except Exception as e:
            logger.error(f"Error resolving rule '{rule}': {e}")
            return f"[rule: {rule}]"

    def resolve_batch(self, rules: list[str]) -> dict[str, str]:
        """Resolve multiple rules synchronously.

        Args:
            rules: List of natural language rule strings

        Returns:
            Dictionary mapping rule strings to resolved values
        """
        results = {}
        for rule in rules:
            results[rule] = self.resolve(rule)
        return results

    async def aresolve_batch(self, rules: list[str]) -> dict[str, str]:
        """Resolve multiple rules asynchronously.

        Args:
            rules: List of natural language rule strings

        Returns:
            Dictionary mapping rule strings to resolved values
        """
        results = {}
        for rule in rules:
            results[rule] = await self.aresolve(rule)
        return results

    def resolve_tags(self, tags: list[ParsedTag]) -> dict[str, str]:
        """Resolve ParsedTag objects (from template parser).

        Args:
            tags: List of ParsedTag objects (should be rule tags)

        Returns:
            Dictionary mapping tag.full_match to resolved values
        """
        results = {}
        for tag in tags:
            if tag.tag_type == "rule" and tag.content:
                resolved = self.resolve(tag.content)
                results[tag.full_match] = resolved
        return results

    async def aresolve_tags(self, tags: list[ParsedTag]) -> dict[str, str]:
        """Resolve ParsedTag objects asynchronously.

        Args:
            tags: List of ParsedTag objects (should be rule tags)

        Returns:
            Dictionary mapping tag.full_match to resolved values
        """
        results = {}
        for tag in tags:
            if tag.tag_type == "rule" and tag.content:
                resolved = await self.aresolve(tag.content)
                results[tag.full_match] = resolved
        return results
