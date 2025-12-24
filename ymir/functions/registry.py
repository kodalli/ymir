"""Function registry for managing tool/function definitions."""

import json
from pathlib import Path
from typing import Iterator

import yaml
from loguru import logger

from .schemas import FunctionDefinition, ScenarioTemplate


class FunctionRegistry:
    """Registry for managing function definitions and scenario templates."""

    def __init__(self, data_dir: str = "ymir/data/runtime/functions"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._functions: dict[str, FunctionDefinition] = {}
        self._scenarios: dict[str, ScenarioTemplate] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """Load built-in function templates."""
        from .templates.scheduling import SCHEDULING_FUNCTIONS, SCHEDULING_SCENARIO

        for func in SCHEDULING_FUNCTIONS:
            self._functions[func.name] = func

        self._scenarios[SCHEDULING_SCENARIO.id] = SCHEDULING_SCENARIO

    # Function CRUD
    def add_function(self, func: FunctionDefinition) -> None:
        """Add a function definition."""
        self._functions[func.name] = func
        self._save_functions()

    def get_function(self, name: str) -> FunctionDefinition | None:
        """Get a function by name."""
        return self._functions.get(name)

    def delete_function(self, name: str) -> bool:
        """Delete a function by name."""
        if name in self._functions:
            del self._functions[name]
            self._save_functions()
            return True
        return False

    def list_functions(self, category: str | None = None) -> list[FunctionDefinition]:
        """List all functions, optionally filtered by category."""
        funcs = list(self._functions.values())
        if category:
            funcs = [f for f in funcs if f.category == category]
        return funcs

    def get_categories(self) -> list[str]:
        """Get all unique function categories."""
        return list(set(f.category for f in self._functions.values()))

    # Scenario CRUD
    def add_scenario(self, scenario: ScenarioTemplate) -> None:
        """Add a scenario template."""
        self._scenarios[scenario.id] = scenario
        self._save_scenarios()

    def get_scenario(self, scenario_id: str) -> ScenarioTemplate | None:
        """Get a scenario by ID."""
        return self._scenarios.get(scenario_id)

    def delete_scenario(self, scenario_id: str) -> bool:
        """Delete a scenario by ID."""
        if scenario_id in self._scenarios:
            del self._scenarios[scenario_id]
            self._save_scenarios()
            return True
        return False

    def list_scenarios(self, category: str | None = None) -> list[ScenarioTemplate]:
        """List all scenarios, optionally filtered by category."""
        scenarios = list(self._scenarios.values())
        if category:
            scenarios = [s for s in scenarios if s.category == category]
        return scenarios

    # Persistence
    def _save_functions(self) -> None:
        """Save user-defined functions to disk."""
        user_funcs = [f for f in self._functions.values() if f.category != "scheduling"]
        if not user_funcs:
            return

        path = self.data_dir / "functions.yaml"
        data = [f.model_dump() for f in user_funcs]
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _save_scenarios(self) -> None:
        """Save user-defined scenarios to disk."""
        user_scenarios = [s for s in self._scenarios.values() if s.category != "scheduling"]
        if not user_scenarios:
            return

        path = self.data_dir / "scenarios.yaml"
        data = [s.model_dump() for s in user_scenarios]
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def load_from_disk(self) -> None:
        """Load user-defined functions and scenarios from disk."""
        # Load functions
        func_path = self.data_dir / "functions.yaml"
        if func_path.exists():
            try:
                with open(func_path) as f:
                    data = yaml.safe_load(f)
                    if data:
                        for item in data:
                            func = FunctionDefinition(**item)
                            self._functions[func.name] = func
            except Exception as e:
                logger.error(f"Error loading functions: {e}")

        # Load scenarios
        scenario_path = self.data_dir / "scenarios.yaml"
        if scenario_path.exists():
            try:
                with open(scenario_path) as f:
                    data = yaml.safe_load(f)
                    if data:
                        for item in data:
                            scenario = ScenarioTemplate(**item)
                            self._scenarios[scenario.id] = scenario
            except Exception as e:
                logger.error(f"Error loading scenarios: {e}")

    def export_to_json(self) -> str:
        """Export all functions and scenarios as JSON."""
        return json.dumps(
            {
                "functions": [f.model_dump() for f in self._functions.values()],
                "scenarios": [s.model_dump() for s in self._scenarios.values()],
            },
            indent=2,
        )


# Global registry instance
_registry: FunctionRegistry | None = None


def get_registry() -> FunctionRegistry:
    """Get the global function registry instance."""
    global _registry
    if _registry is None:
        _registry = FunctionRegistry()
        _registry.load_from_disk()
    return _registry
