import yaml
from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any
from pathlib import Path


class PromptConfig(BaseModel):
    """
    Configuration model for storing prompt templates and related settings.
    This model handles serialization to/from YAML with special handling for
    preserving curly braces and whitespace in prompt templates.
    """

    model: str
    max_tokens: int
    temperature: float
    reasoning_effort: Optional[str] = None
    system_prompt: str
    user_prompt: str

    @field_validator("system_prompt", "user_prompt")
    @classmethod
    def validate_prompts(cls, v):
        """Validate that prompt templates have balanced curly braces"""
        # Count open and close braces
        open_count = v.count("{")
        close_count = v.count("}")

        if open_count != close_count:
            raise ValueError(
                f"Unbalanced curly braces in prompt: {open_count} opening and {close_count} closing braces"
            )

        return v

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "PromptConfig":
        """Create a PromptConfig from YAML content string"""
        try:
            config_dict = yaml.safe_load(yaml_content)
            return cls(**config_dict)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML configuration: {str(e)}")

    @classmethod
    def from_yaml_file(cls, file_path: str) -> "PromptConfig":
        """Load a PromptConfig from a YAML file"""
        try:
            with open(file_path, "r") as f:
                yaml_content = f.read()
            return cls.from_yaml(yaml_content)
        except Exception as e:
            raise ValueError(f"Failed to load YAML from file {file_path}: {str(e)}")

    def to_yaml(self) -> str:
        """Convert the config to a YAML string with proper handling of special characters"""
        # Use sort_keys=False to preserve field order
        # Use default_flow_style=False for block style YAML output
        return yaml.dump(self.model_dump(), sort_keys=False, default_flow_style=False)

    def save_to_file(self, file_path: str = None) -> str:
        """
        Save the config to a YAML file
        If file_path is not provided, will use a default location with timestamp
        Returns the path where the file was saved
        """
        import time

        if not file_path:
            timestamp = int(time.time())
            filename = f"prompt_config_{timestamp}.yaml"

            # Create data directory if it doesn't exist
            data_dir = Path("ymir/data/prompt_configs")
            data_dir.mkdir(parents=True, exist_ok=True)

            file_path = str(data_dir / filename)

        with open(file_path, "w") as f:
            f.write(self.to_yaml())

        return file_path

    def format_prompts(self, variables: Dict[str, Any]) -> tuple:
        """
        Format the system and user prompts with the given variables
        Returns a tuple of (formatted_system_prompt, formatted_user_prompt)
        """
        formatted_system = self.system_prompt
        formatted_user = self.user_prompt

        # Replace placeholders with values
        for key, value in variables.items():
            if value is not None:
                value_str = str(value)
            else:
                value_str = ""

            formatted_system = formatted_system.replace(f"{{{key}}}", value_str)
            formatted_user = formatted_user.replace(f"{{{key}}}", value_str)

        return formatted_system, formatted_user


if __name__ == "__main__":
    pass
