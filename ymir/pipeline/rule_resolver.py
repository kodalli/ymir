"""LLM-based rule resolver for generating varied natural language.

Resolves <rule>...</rule> tags in actor templates using an LLM to generate
natural language phrases that actors would say.

Example rules:
- "a date preference for scheduling" -> "next Monday or Wednesday"
- "a time preference" -> "sometime in the morning, before 11 if possible"
- "a brief note about patient situation" -> "I've had headaches for about a week"
"""

from loguru import logger

from ymir.pipeline.llm.ollama import OllamaLLM
from ymir.pipeline.template_parser import ParsedTag


class RuleResolver:
    """Resolves natural language rules using LLM."""

    SYSTEM_PROMPT = """You are a data generation assistant creating varied natural language for actor personas.

Given a rule describing what type of text to generate, create ONE natural-sounding phrase that an actor would say.

Rules:
1. Return ONLY the generated phrase, nothing else
2. No explanations, no quotes, no formatting
3. Be creative and varied - each generation should be different
4. Sound natural, like something a real person would say
5. Match the tone and context described in the rule

Examples:
- Rule: "a date preference for scheduling" -> next Monday or Wednesday
- Rule: "a time preference for appointments" -> sometime in the morning, before 11 if possible
- Rule: "a reason for calling about health" -> I've been having persistent headaches
- Rule: "how urgent the request is" -> it's not super urgent but I'd like to be seen soon
- Rule: "a brief note about special requirements" -> I'll need wheelchair access"""

    def __init__(
        self,
        model: str = "mistral-small:latest",
        temperature: float = 0.7,
        num_predict: int = 128,
    ):
        """Initialize the rule resolver.

        Args:
            model: Ollama model name to use
            temperature: Temperature for generation (higher = more varied)
            num_predict: Max tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self._llm: OllamaLLM | None = None

    def _get_llm(self) -> OllamaLLM:
        """Lazily initialize the LLM client."""
        if self._llm is None:
            self._llm = OllamaLLM(
                model=self.model,
                temperature=self.temperature,
                num_predict=self.num_predict,
                num_ctx=1024,
            )
        return self._llm

    def resolve(self, rule: str) -> str:
        """Resolve a single rule to a natural language phrase.

        Args:
            rule: Rule describing what type of text to generate

        Returns:
            Natural language phrase that an actor would say
        """
        system = self.SYSTEM_PROMPT

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

            # If result is empty after cleaning, return a fallback
            if not result:
                logger.warning(f"Empty result for rule '{rule}', using fallback")
                return f"[{rule}]"

            logger.debug(f"Resolved rule '{rule}' -> '{result}'")
            return result
        except Exception as e:
            logger.error(f"Error resolving rule '{rule}': {e}")
            # Return a placeholder on error
            return f"[{rule}]"

    async def aresolve(self, rule: str) -> str:
        """Resolve a single rule asynchronously.

        Args:
            rule: Rule describing what type of text to generate

        Returns:
            Natural language phrase that an actor would say
        """
        system = self.SYSTEM_PROMPT

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

            # If result is empty after cleaning, return a fallback
            if not result:
                logger.warning(f"Empty result for rule '{rule}', using fallback")
                return f"[{rule}]"

            logger.debug(f"Resolved rule '{rule}' -> '{result}'")
            return result
        except Exception as e:
            logger.error(f"Error resolving rule '{rule}': {e}")
            return f"[{rule}]"

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
