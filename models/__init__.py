"""
Model factory and registry for Vietnamese Sign Language (VSL) recognition.

Usage::

    from models import build_model

    # Using a config dictionary
    config = {
        "model_type": "hybrid",      # "lstm" | "transformer" | "hybrid"
        "input_shape": (60, 1605),
        "num_classes": 161,
        # ... any model-specific kwargs
    }
    model = build_model(config)
    model.summary()

    # Or import individual builders directly
    from models.hybrid_model import build_hybrid_model
    model = build_hybrid_model(dropout_dense=0.5)
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from keras import Model

from .lstm_model import build_lstm_model
from .transformer_model import build_transformer_model
from .hybrid_model import build_hybrid_model

__all__ = [
    "build_model",
    "build_lstm_model",
    "build_transformer_model",
    "build_hybrid_model",
    "MODEL_REGISTRY",
]

# ---------------------------------------------------------------------------
# Registry — maps human-friendly names to builder functions
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Callable[..., Model]] = {
    "lstm": build_lstm_model,
    "transformer": build_transformer_model,
    "hybrid": build_hybrid_model,
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(config: Dict[str, Any]) -> Model:
    """Build and return a compiled Keras model based on *config*.

    The ``config`` dictionary **must** contain a ``"model_type"`` key whose
    value is one of the registered model names (``"lstm"``,
    ``"transformer"``, or ``"hybrid"``).  All remaining key/value pairs are
    forwarded as keyword arguments to the chosen builder function.

    Args:
        config: Dictionary with at least ``"model_type"`` and optionally any
                keyword arguments accepted by the corresponding builder
                (e.g. ``input_shape``, ``num_classes``, ``dropout_dense``,
                etc.).

    Returns:
        A compiled ``tf.keras.Model``.

    Raises:
        ValueError: If ``model_type`` is missing or not in the registry.

    Example::

        model = build_model({
            "model_type": "hybrid",
            "input_shape": (60, 1605),
            "num_classes": 161,
            "dropout_dense": 0.5,
        })
    """
    config = dict(config)  # Shallow copy so we don't mutate the caller's dict

    model_type: str | None = config.pop("model_type", None)

    if model_type is None:
        raise ValueError(
            "config must contain a 'model_type' key. "
            f"Available types: {list(MODEL_REGISTRY.keys())}"
        )

    model_type = model_type.lower().strip()

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available types: {list(MODEL_REGISTRY.keys())}"
        )

    builder = MODEL_REGISTRY[model_type]
    model = builder(**config)

    # Print a compact summary for quick sanity-checking
    print(f"\n{'=' * 60}")
    print(f"  Model: {model.name}  (type={model_type})")
    print(f"  Total params: {model.count_params():,}")
    print(f"{'=' * 60}\n")
    model.summary(line_length=100, show_trainable=True)

    return model
