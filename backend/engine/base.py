"""Abstract base class for ML engines."""
import logging
from abc import ABC, abstractmethod
from typing import Any


class BaseEngine(ABC):
    """Base class for all ML engines (ASR, diarization, summarization, etc.)."""

    def __init__(self, config: Any):
        """
        Initialize engine with configuration.

        Args:
            config: Engine-specific configuration object
        """
        self._config = config
        self._model = None
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """
        Process input and return results.

        Args:
            *args: Positional arguments (engine-specific)
            **kwargs: Keyword arguments (engine-specific)

        Returns:
            Engine-specific result
        """
        ...

    @abstractmethod
    def load(self) -> None:
        """Load model into memory/VRAM."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free VRAM."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    @property
    @abstractmethod
    def required_vram_gb(self) -> float:
        """Estimated VRAM required in gigabytes."""
        ...

    @property
    def name(self) -> str:
        """Engine name (class name)."""
        return self.__class__.__name__

    async def __aenter__(self):
        """Async context manager entry."""
        self.load()
        return self

    async def __aexit__(self, *args):
        """Async context manager exit."""
        self.unload()
