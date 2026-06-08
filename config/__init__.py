"""
Configuration loader for the VSL Recognition project.

Provides a unified interface to load and access YAML configuration
with dot-notation access support via a recursive DotDict wrapper.

Usage:
    from config import load_config, get_config

    # Load from default path
    cfg = load_config()

    # Access nested values with dot notation
    batch_size = cfg.training.batch_size
    lr = cfg.training.learning_rate

    # Or dict-style access
    num_classes = cfg["data"]["num_classes"]

    # Get singleton config (loads once, reuses thereafter)
    cfg = get_config()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml


# Default config path relative to project root
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Module-level singleton
_config_instance: Optional["DotDict"] = None


class DotDict(dict):
    """A dictionary subclass that supports dot-notation access to nested keys.

    Recursively wraps nested dictionaries so that deeply nested values
    can be accessed like attributes:

        cfg = DotDict({"data": {"num_classes": 161}})
        cfg.data.num_classes  # 161

    Also supports standard dict operations ([], .get(), .keys(), etc.).

    Raises:
        AttributeError: When accessing a key that doesn't exist.
    """

    def __init__(self, data: dict | None = None, **kwargs: Any) -> None:
        """Initialize DotDict from a dictionary or keyword arguments.

        Args:
            data: Source dictionary to wrap. Nested dicts are recursively
                  converted to DotDict instances.
            **kwargs: Additional key-value pairs.
        """
        super().__init__()
        source = data or {}
        source.update(kwargs)
        for key, value in source.items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any) -> Any:
        """Recursively wrap dicts and lists containing dicts.

        Args:
            value: The value to potentially wrap.

        Returns:
            Wrapped value (DotDict for dicts, list with wrapped elements
            for lists, original value otherwise).
        """
        if isinstance(value, dict) and not isinstance(value, DotDict):
            return DotDict(value)
        if isinstance(value, (list, tuple)):
            wrapped = [DotDict._wrap(item) for item in value]
            return type(value)(wrapped)
        return value

    def __getattr__(self, key: str) -> Any:
        """Access dictionary keys as attributes.

        Args:
            key: The attribute/key name.

        Returns:
            The value associated with the key.

        Raises:
            AttributeError: If the key is not found.
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"Config has no attribute '{key}'. "
                f"Available keys: {list(self.keys())}"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        """Set dictionary keys as attributes.

        Args:
            key: The attribute/key name.
            value: The value to set.
        """
        self[key] = self._wrap(value)

    def __delattr__(self, key: str) -> None:
        """Delete dictionary keys as attributes.

        Args:
            key: The attribute/key name.

        Raises:
            AttributeError: If the key is not found.
        """
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"Config has no attribute '{key}'"
            ) from None

    def to_dict(self) -> dict:
        """Convert back to a plain nested dictionary.

        Returns:
            A standard Python dict with all nested DotDicts unwrapped.
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, DotDict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def merge(self, overrides: dict) -> "DotDict":
        """Deep-merge another dictionary into this config.

        Values from `overrides` take precedence. Nested dicts are merged
        recursively rather than replaced wholesale.

        Args:
            overrides: Dictionary of values to merge in.

        Returns:
            Self, for method chaining.
        """
        for key, value in overrides.items():
            if (
                key in self
                and isinstance(self[key], DotDict)
                and isinstance(value, dict)
            ):
                self[key].merge(value)
            else:
                self[key] = self._wrap(value)
        return self

    def __repr__(self) -> str:
        return f"DotDict({dict.__repr__(self)})"


def load_config(
    config_path: Union[str, Path, None] = None,
    overrides: dict | None = None,
) -> DotDict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.
            If None, uses the default ``config/config.yaml`` relative
            to the project root.
        overrides: Optional dictionary of values to override after loading.
            Supports nested keys that will be deep-merged.

    Returns:
        A DotDict instance containing all configuration parameters.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file is malformed.

    Example:
        >>> cfg = load_config()
        >>> cfg.data.num_classes
        161
        >>> cfg = load_config(overrides={"training": {"batch_size": 64}})
        >>> cfg.training.batch_size
        64
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}\n"
            f"Expected location: {_DEFAULT_CONFIG_PATH.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raw_config = {}

    config = DotDict(raw_config)

    if overrides:
        config.merge(overrides)

    return config


def get_config(
    config_path: Union[str, Path, None] = None,
    overrides: dict | None = None,
    force_reload: bool = False,
) -> DotDict:
    """Get the singleton configuration instance.

    Loads the config on first call and caches it for subsequent calls.
    Use ``force_reload=True`` to re-read from disk.

    Args:
        config_path: Path to the YAML config file.
        overrides: Optional overrides to deep-merge.
        force_reload: If True, reload from disk even if already cached.

    Returns:
        The cached (or freshly loaded) DotDict configuration.
    """
    global _config_instance

    if _config_instance is None or force_reload:
        _config_instance = load_config(config_path, overrides)

    return _config_instance


def reset_config() -> None:
    """Reset the singleton config, forcing a reload on next access."""
    global _config_instance
    _config_instance = None
