"""Batch actor generation from templates.

Generates unique actors from ActorTemplate definitions by:
1. Parsing XML-style placeholders in the template
2. Generating random values for <random:*> tags
3. Resolving natural language rules via LLM for <rule>*</rule> tags
4. Combining results into complete actor data
"""

from typing import AsyncIterator

from loguru import logger

from ymir.core.scenario_schemas import ActorTemplate, GeneratedActorData
from ymir.pipeline.rule_resolver import RuleResolver
from ymir.pipeline.template_parser import TemplateParser


class ActorGenerator:
    """Generate actors from templates with XML-style placeholders."""

    def __init__(
        self,
        model: str = "qwen3:4b",
        temperature: float = 0.5,
    ):
        """Initialize the actor generator.

        Args:
            model: Ollama model name for rule resolution
            temperature: Temperature for LLM generation
        """
        self.parser = TemplateParser()
        self.resolver = RuleResolver(model=model, temperature=temperature)

    def generate_one(
        self,
        template: ActorTemplate,
        background_override: str | None = None,
        goal_override: str | None = None,
    ) -> GeneratedActorData:
        """Generate a single actor from a template (synchronous).

        Args:
            template: The ActorTemplate to generate from
            background_override: Optional override for actor background
            goal_override: Optional override for actor goal

        Returns:
            GeneratedActorData with filled situation, background, and goal
        """
        # Process the main template text (situation details)
        situation = self._process_template(template.template_text)

        # Process background (use template, override, or template description)
        if background_override:
            background = background_override
        elif template.background_template:
            background = self._process_template(template.background_template)
        else:
            background = template.description

        # Process goal (use template, override, or default)
        if goal_override:
            goal = goal_override
        elif template.goal_template:
            goal = self._process_template(template.goal_template)
        else:
            goal = "Complete the interaction successfully"

        return GeneratedActorData(
            situation=situation,
            background=background,
            goal=goal,
            template_id=template.id,
        )

    async def agenerate_one(
        self,
        template: ActorTemplate,
        background_override: str | None = None,
        goal_override: str | None = None,
    ) -> GeneratedActorData:
        """Generate a single actor from a template (asynchronous).

        Args:
            template: The ActorTemplate to generate from
            background_override: Optional override for actor background
            goal_override: Optional override for actor goal

        Returns:
            GeneratedActorData with filled situation, background, and goal
        """
        # Process the main template text (situation details)
        situation = await self._aprocess_template(template.template_text)

        # Process background
        if background_override:
            background = background_override
        elif template.background_template:
            background = await self._aprocess_template(template.background_template)
        else:
            background = template.description

        # Process goal
        if goal_override:
            goal = goal_override
        elif template.goal_template:
            goal = await self._aprocess_template(template.goal_template)
        else:
            goal = "Complete the interaction successfully"

        return GeneratedActorData(
            situation=situation,
            background=background,
            goal=goal,
            template_id=template.id,
        )

    def _process_template(self, template_text: str) -> str:
        """Process a template string synchronously.

        Args:
            template_text: Template with XML placeholders

        Returns:
            Template with all placeholders replaced
        """
        # Parse all tags
        tags = self.parser.parse(template_text)

        # Generate random values
        random_values = self.parser.generate_random_values(tags)

        # Get rule tags and resolve them
        rule_tags = [t for t in tags if t.tag_type == "rule"]
        rule_values = self.resolver.resolve_tags(rule_tags)

        # Merge all values
        all_values = {**random_values, **rule_values}

        # Apply to template
        return self.parser.apply_values(template_text, all_values)

    async def _aprocess_template(self, template_text: str) -> str:
        """Process a template string asynchronously.

        Args:
            template_text: Template with XML placeholders

        Returns:
            Template with all placeholders replaced
        """
        # Parse all tags
        tags = self.parser.parse(template_text)

        # Generate random values (sync is fine, these are fast)
        random_values = self.parser.generate_random_values(tags)

        # Get rule tags and resolve them asynchronously
        rule_tags = [t for t in tags if t.tag_type == "rule"]
        rule_values = await self.resolver.aresolve_tags(rule_tags)

        # Merge all values
        all_values = {**random_values, **rule_values}

        # Apply to template
        return self.parser.apply_values(template_text, all_values)

    def generate_batch(
        self,
        template: ActorTemplate,
        count: int,
        background_override: str | None = None,
        goal_override: str | None = None,
    ) -> list[GeneratedActorData]:
        """Generate multiple actors from a template (synchronous).

        Args:
            template: The ActorTemplate to generate from
            count: Number of actors to generate
            background_override: Optional override for actor background
            goal_override: Optional override for actor goal

        Returns:
            List of GeneratedActorData objects
        """
        results = []
        for i in range(count):
            try:
                actor_data = self.generate_one(
                    template, background_override, goal_override
                )
                results.append(actor_data)
                logger.debug(f"Generated actor {i+1}/{count} from template {template.id}")
            except Exception as e:
                logger.error(f"Error generating actor {i+1}/{count}: {e}")
        return results

    async def agenerate_batch(
        self,
        template: ActorTemplate,
        count: int,
        background_override: str | None = None,
        goal_override: str | None = None,
    ) -> list[GeneratedActorData]:
        """Generate multiple actors from a template (asynchronous).

        Args:
            template: The ActorTemplate to generate from
            count: Number of actors to generate
            background_override: Optional override for actor background
            goal_override: Optional override for actor goal

        Returns:
            List of GeneratedActorData objects
        """
        results = []
        for i in range(count):
            try:
                actor_data = await self.agenerate_one(
                    template, background_override, goal_override
                )
                results.append(actor_data)
                logger.debug(f"Generated actor {i+1}/{count} from template {template.id}")
            except Exception as e:
                logger.error(f"Error generating actor {i+1}/{count}: {e}")
        return results

    async def agenerate_stream(
        self,
        template: ActorTemplate,
        count: int,
        background_override: str | None = None,
        goal_override: str | None = None,
    ) -> AsyncIterator[GeneratedActorData]:
        """Generate actors as an async stream.

        Args:
            template: The ActorTemplate to generate from
            count: Number of actors to generate
            background_override: Optional override for actor background
            goal_override: Optional override for actor goal

        Yields:
            GeneratedActorData objects as they are generated
        """
        for i in range(count):
            try:
                actor_data = await self.agenerate_one(
                    template, background_override, goal_override
                )
                yield actor_data
                logger.debug(f"Streamed actor {i+1}/{count} from template {template.id}")
            except Exception as e:
                logger.error(f"Error generating actor {i+1}/{count}: {e}")

    def preview(
        self,
        template_text: str,
        count: int = 3,
    ) -> list[str]:
        """Preview template output without saving.

        Useful for testing templates before saving them.

        Args:
            template_text: Raw template text with placeholders
            count: Number of previews to generate

        Returns:
            List of processed template strings
        """
        results = []
        for _ in range(count):
            try:
                processed = self._process_template(template_text)
                results.append(processed)
            except Exception as e:
                logger.error(f"Error generating preview: {e}")
                results.append(f"[Error: {e}]")
        return results

    async def apreview(
        self,
        template_text: str,
        count: int = 3,
    ) -> list[str]:
        """Preview template output asynchronously.

        Args:
            template_text: Raw template text with placeholders
            count: Number of previews to generate

        Returns:
            List of processed template strings
        """
        results = []
        for _ in range(count):
            try:
                processed = await self._aprocess_template(template_text)
                results.append(processed)
            except Exception as e:
                logger.error(f"Error generating preview: {e}")
                results.append(f"[Error: {e}]")
        return results

    def validate_template(self, template_text: str) -> tuple[bool, list[str]]:
        """Validate template syntax.

        Args:
            template_text: Template text to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.parser.validate(template_text)
