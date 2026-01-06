"""Actor template parser for XML-style placeholders.

Supports placeholders like:
- <random:name/> - Random person name
- <random:phone/> - Random phone number
- <random:email/> - Random email address
- <random:date min="1950-01-01" max="2000-12-31"/> - Random date in range
- <random:choice>opt1|opt2|opt3</random:choice> - Random from options
- <random:sentence topic="..."/> - Generated contextual sentence
- <random:int min="1" max="100"/> - Random integer in range
- <rule>natural language constraint</rule> - LLM-interpreted rule
"""

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta

from loguru import logger


@dataclass
class ParsedTag:
    """Represents a parsed XML-style placeholder tag."""

    tag_type: str  # 'random' or 'rule'
    generator: str  # 'name', 'phone', 'date', 'choice', 'email', 'sentence', 'int'
    attributes: dict[str, str]
    content: str | None  # For <random:choice> and <rule> tags
    full_match: str  # Original matched string for replacement


class TagGenerator(ABC):
    """Abstract base class for tag value generators."""

    @abstractmethod
    def generate(self, tag: ParsedTag) -> str:
        """Generate a value for the given tag."""
        pass


class NameGenerator(TagGenerator):
    """Generate random person names."""

    FIRST_NAMES = [
        "James",
        "Mary",
        "John",
        "Patricia",
        "Robert",
        "Jennifer",
        "Michael",
        "Linda",
        "William",
        "Elizabeth",
        "David",
        "Barbara",
        "Richard",
        "Susan",
        "Joseph",
        "Jessica",
        "Thomas",
        "Sarah",
        "Charles",
        "Karen",
        "Christopher",
        "Lisa",
        "Daniel",
        "Nancy",
        "Matthew",
        "Betty",
        "Anthony",
        "Margaret",
        "Mark",
        "Sandra",
        "Donald",
        "Ashley",
        "Steven",
        "Kimberly",
        "Paul",
        "Emily",
        "Andrew",
        "Donna",
        "Joshua",
        "Michelle",
    ]

    LAST_NAMES = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Garcia",
        "Miller",
        "Davis",
        "Rodriguez",
        "Martinez",
        "Hernandez",
        "Lopez",
        "Gonzalez",
        "Wilson",
        "Anderson",
        "Thomas",
        "Taylor",
        "Moore",
        "Jackson",
        "Martin",
        "Lee",
        "Perez",
        "Thompson",
        "White",
        "Harris",
        "Sanchez",
        "Clark",
        "Ramirez",
        "Lewis",
        "Robinson",
        "Walker",
        "Young",
        "Allen",
        "King",
        "Wright",
        "Scott",
        "Torres",
        "Nguyen",
        "Hill",
        "Flores",
    ]

    def generate(self, tag: ParsedTag) -> str:
        first = random.choice(self.FIRST_NAMES)
        last = random.choice(self.LAST_NAMES)
        return f"{first} {last}"


class PhoneGenerator(TagGenerator):
    """Generate random US phone numbers."""

    def generate(self, tag: ParsedTag) -> str:
        area = random.randint(200, 999)
        exchange = random.randint(200, 999)
        subscriber = random.randint(1000, 9999)
        return f"({area}) {exchange}-{subscriber}"


class EmailGenerator(TagGenerator):
    """Generate random email addresses."""

    DOMAINS = [
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
        "email.com",
        "mail.com",
        "protonmail.com",
    ]

    def generate(self, tag: ParsedTag) -> str:
        # If a name attribute is provided, use it as base
        name_base = tag.attributes.get("name", "user")
        # Clean and normalize
        name_base = name_base.lower().replace(" ", ".").replace("'", "")
        # Add random suffix
        suffix = random.randint(1, 999)
        domain = random.choice(self.DOMAINS)
        return f"{name_base}{suffix}@{domain}"


class DateGenerator(TagGenerator):
    """Generate random dates within a range."""

    def generate(self, tag: ParsedTag) -> str:
        # Parse min/max dates from attributes
        min_str = tag.attributes.get("min", "1950-01-01")
        max_str = tag.attributes.get("max", "2005-12-31")
        fmt = tag.attributes.get("format", "%m/%d/%Y")

        try:
            min_date = date.fromisoformat(min_str)
            max_date = date.fromisoformat(max_str)
        except ValueError:
            # Fallback to default range
            min_date = date(1950, 1, 1)
            max_date = date(2005, 12, 31)

        # Generate random date in range
        delta = (max_date - min_date).days
        random_days = random.randint(0, max(0, delta))
        result_date = min_date + timedelta(days=random_days)

        return result_date.strftime(fmt)


class ChoiceGenerator(TagGenerator):
    """Generate random choice from pipe-delimited options."""

    def generate(self, tag: ParsedTag) -> str:
        if tag.content:
            choices = [c.strip() for c in tag.content.split("|")]
            if choices:
                return random.choice(choices)
        return ""


class IntGenerator(TagGenerator):
    """Generate random integer within a range."""

    def generate(self, tag: ParsedTag) -> str:
        min_val = int(tag.attributes.get("min", "1"))
        max_val = int(tag.attributes.get("max", "100"))
        return str(random.randint(min_val, max_val))


class SentenceGenerator(TagGenerator):
    """Generate contextual sentences based on topic."""

    # Topic-specific sentence templates
    TEMPLATES = {
        "medical_complaint": [
            "Experiencing {symptom} for the past {duration}",
            "Started feeling {symptom} {duration} ago",
            "Has been dealing with {symptom} since {time_ref}",
            "{symptom} that started {time_ref}, getting {progression}",
        ],
        "general": [
            "Additional information about the situation",
            "No special requirements",
            "Standard case",
        ],
    }

    SYMPTOMS = [
        "persistent pain",
        "mild discomfort",
        "recurring headaches",
        "fatigue",
        "dizziness",
        "nausea",
        "shortness of breath",
        "joint stiffness",
        "muscle soreness",
        "back pain",
    ]

    DURATIONS = [
        "a few days",
        "about a week",
        "two weeks",
        "several days",
        "the past month",
    ]

    TIME_REFS = [
        "last week",
        "Monday",
        "a few days ago",
        "yesterday",
        "last weekend",
    ]

    PROGRESSIONS = ["worse", "better", "about the same", "gradually improving"]

    def generate(self, tag: ParsedTag) -> str:
        topic = tag.attributes.get("topic", "general")
        templates = self.TEMPLATES.get(topic, self.TEMPLATES["general"])
        template = random.choice(templates)

        # Replace placeholders in template
        result = template.format(
            symptom=random.choice(self.SYMPTOMS),
            duration=random.choice(self.DURATIONS),
            time_ref=random.choice(self.TIME_REFS),
            progression=random.choice(self.PROGRESSIONS),
        )

        return result


class TemplateParser:
    """Parse and process actor templates with XML-style placeholders."""

    # Pattern to match both self-closing and content tags
    # Examples:
    #   <random:name/>
    #   <random:date min="1950-01-01" max="2000-12-31"/>
    #   <random:choice>opt1|opt2|opt3</random:choice>
    #   <rule>natural language here</rule>
    TAG_PATTERN = re.compile(
        r"<(random|rule)(?::(\w+))?([^>]*)(?:/>|>(.*?)</\1(?::\2)?>)",
        re.DOTALL,
    )

    # Pattern to extract attributes like key="value"
    ATTR_PATTERN = re.compile(r'(\w+)="([^"]*)"')

    def __init__(self):
        self.generators: dict[str, TagGenerator] = {
            "name": NameGenerator(),
            "phone": PhoneGenerator(),
            "email": EmailGenerator(),
            "date": DateGenerator(),
            "choice": ChoiceGenerator(),
            "int": IntGenerator(),
            "sentence": SentenceGenerator(),
        }

    def parse(self, template: str) -> list[ParsedTag]:
        """Extract all tags from a template string.

        Args:
            template: The template string containing XML-style placeholders

        Returns:
            List of ParsedTag objects representing each placeholder
        """
        tags = []

        for match in self.TAG_PATTERN.finditer(template):
            tag_type = match.group(1)  # 'random' or 'rule'
            generator = match.group(2) or ""  # 'name', 'phone', etc. (empty for rule)
            attr_str = match.group(3) or ""  # Attributes string
            content = match.group(4)  # Content for non-self-closing tags

            # Parse attributes
            attributes = {}
            for attr_match in self.ATTR_PATTERN.finditer(attr_str):
                attributes[attr_match.group(1)] = attr_match.group(2)

            tags.append(
                ParsedTag(
                    tag_type=tag_type,
                    generator=generator,
                    attributes=attributes,
                    content=content.strip() if content else None,
                    full_match=match.group(0),
                )
            )

        return tags

    def generate_random_values(self, tags: list[ParsedTag]) -> dict[str, str]:
        """Generate values for all random tags.

        Args:
            tags: List of ParsedTag objects

        Returns:
            Dictionary mapping full_match strings to generated values
        """
        values = {}

        for tag in tags:
            if tag.tag_type == "random" and tag.generator in self.generators:
                generator = self.generators[tag.generator]
                values[tag.full_match] = generator.generate(tag)
            elif tag.tag_type == "random":
                logger.warning(f"Unknown generator type: {tag.generator}")
                values[tag.full_match] = f"[unknown:{tag.generator}]"

        return values

    def apply_values(self, template: str, values: dict[str, str]) -> str:
        """Replace placeholders in template with generated values.

        Args:
            template: The original template string
            values: Dictionary mapping placeholder strings to replacement values

        Returns:
            Template with all placeholders replaced
        """
        result = template
        for placeholder, value in values.items():
            result = result.replace(placeholder, value)
        return result

    def generate(
        self, template: str, rule_values: dict[str, str] | None = None
    ) -> str:
        """Parse template and replace all random placeholders with generated values.

        Note: Rule tags (<rule>...</rule>) are NOT replaced by this method.
        They should be resolved separately using RuleResolver and passed via rule_values.

        Args:
            template: The template string to process
            rule_values: Optional dict mapping rule tag strings to resolved values

        Returns:
            Template with random placeholders replaced
        """
        # Parse all tags
        tags = self.parse(template)

        # Generate random values
        values = self.generate_random_values(tags)

        # Add rule values if provided
        if rule_values:
            values.update(rule_values)

        # Apply all values
        return self.apply_values(template, values)

    def get_rule_tags(self, template: str) -> list[ParsedTag]:
        """Extract only the rule tags from a template.

        Args:
            template: The template string

        Returns:
            List of ParsedTag objects for rule tags only
        """
        return [tag for tag in self.parse(template) if tag.tag_type == "rule"]

    def validate(self, template: str) -> tuple[bool, list[str]]:
        """Validate a template's syntax.

        Args:
            template: The template string to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        try:
            tags = self.parse(template)

            for tag in tags:
                if tag.tag_type == "random":
                    if not tag.generator:
                        errors.append(
                            f"Random tag missing generator type: {tag.full_match}"
                        )
                    elif tag.generator not in self.generators:
                        errors.append(
                            f"Unknown generator type '{tag.generator}' in: {tag.full_match}"
                        )

                elif tag.tag_type == "rule":
                    if not tag.content:
                        errors.append(f"Rule tag has no content: {tag.full_match}")

        except Exception as e:
            errors.append(f"Parse error: {str(e)}")

        return (len(errors) == 0, errors)
