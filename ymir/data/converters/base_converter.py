"""Base converter interface for dataset conversion."""

from abc import ABC, abstractmethod
from typing import Iterator

from ymir.core import Trajectory


class BaseConverter(ABC):
    """Base class for dataset converters."""

    @abstractmethod
    def convert_file(self, input_path: str) -> Iterator[Trajectory]:
        """Convert a dataset file to trajectories."""
        pass

    @abstractmethod
    def convert_record(self, record: dict) -> Trajectory | None:
        """Convert a single record to a trajectory."""
        pass

    @abstractmethod
    def supports_format(self, format_name: str) -> bool:
        """Check if this converter supports the given format."""
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of the format this converter handles."""
        pass
