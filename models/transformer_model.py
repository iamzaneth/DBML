"""
Transformer Encoder model for Vietnamese Sign Language (VSL) recognition.

Architecture:
    Input (batch, 60, 1605)
    → Linear Projection (1605 → 256)
    → Positional Encoding (learned, 60 positions)
    → Transformer Encoder × 4 layers (8 heads, d_model=256, ff=1024)
    → Global Average Pooling
    → Dense (256, GELU) + Dropout (0.3)
    → Dense (num_classes, Softmax)

Custom Keras layers:
    - ``PositionalEncoding``: Learned positional embeddings.
    - ``TransformerEncoderLayer``: Pre-LN Transformer block with Multi-Head
      Attention and position-wise Feed-Forward Network.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model


# ===========================================================================
# Custom Layers
# ===========================================================================


class PositionalEncoding(layers.Layer):
    """Learned positional encoding added to projected input embeddings.

    Creates a trainable embedding table of shape ``(max_len, d_model)`` and
    adds it element-wise to the input tensor.

    Args:
        max_len: Maximum sequence length (number of positions). Default: 60.
        d_model: Embedding / model dimensionality. Default: 256.
    """

    def __init__(self, max_len: int = 60, d_model: int = 256, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape: tf.TensorShape) -> None:
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(self.max_len, self.d_model),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor with positional information added, same shape as input.
        """
        seq_len = tf.shape(x)[1]
        return x + self.pos_embedding[:seq_len, :]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


class TransformerEncoderLayer(layers.Layer):
    """Single Transformer Encoder block (Pre-LayerNorm variant).

    Consists of Multi-Head Self-Attention followed by a position-wise
    Feed-Forward Network, each wrapped with residual connections and
    Layer Normalization.

    Args:
        d_model: Model / embedding dimensionality. Default: 256.
        num_heads: Number of attention heads. Default: 8.
        ff_dim: Inner dimensionality of the feed-forward network. Default: 1024.
        dropout_rate: Dropout rate for attention and FFN. Default: 0.1.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout_rate: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # --- Multi-Head Attention ---
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name="mha",
        )
        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6, name="ln_1")
        self.dropout_1 = layers.Dropout(dropout_rate, name="dropout_attn")

        # --- Feed-Forward Network ---
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu", name="ffn_expand"),
                layers.Dropout(dropout_rate, name="ffn_dropout"),
                layers.Dense(d_model, name="ffn_project"),
            ],
            name="ffn",
        )
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6, name="ln_2")
        self.dropout_2 = layers.Dropout(dropout_rate, name="dropout_ffn")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Forward pass through the Transformer encoder block.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            training: Whether in training mode (enables dropout).

        Returns:
            Output tensor, same shape as input.
        """
        # --- Self-Attention with Pre-LN ---
        x_norm = self.layernorm_1(x)
        attn_output = self.mha(x_norm, x_norm, training=training)
        attn_output = self.dropout_1(attn_output, training=training)
        x = x + attn_output  # Residual

        # --- Feed-Forward with Pre-LN ---
        x_norm = self.layernorm_2(x)
        ffn_output = self.ffn(x_norm, training=training)
        ffn_output = self.dropout_2(ffn_output, training=training)
        x = x + ffn_output  # Residual

        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


# ===========================================================================
# Model Builder
# ===========================================================================


def build_transformer_model(
    input_shape: Tuple[int, int] = (60, 1605),
    num_classes: int = 161,
    d_model: int = 256,
    num_heads: int = 8,
    ff_dim: int = 1024,
    num_layers: int = 4,
    dropout_encoder: float = 0.1,
    dropout_classifier: float = 0.3,
    learning_rate: float = 1e-4,
) -> Model:
    """Build a Transformer Encoder model for sequence classification.

    The architecture linearly projects the raw landmark features into
    ``d_model`` dimensions, adds learned positional encodings, and passes
    the result through a stack of Transformer encoder layers.  The output
    sequence is globally averaged and fed to a classification head.

    Args:
        input_shape: Tuple of (sequence_length, num_features).
                     Default: (60, 1605).
        num_classes: Number of output classes. Default: 161.
        d_model: Model dimensionality. Default: 256.
        num_heads: Number of attention heads per encoder layer. Default: 8.
        ff_dim: Feed-forward hidden size inside each encoder layer. Default: 1024.
        num_layers: Number of stacked Transformer encoder layers. Default: 4.
        dropout_encoder: Dropout rate inside encoder layers. Default: 0.1.
        dropout_classifier: Dropout rate in the classification head. Default: 0.3.
        learning_rate: Adam optimizer learning rate. Default: 1e-4.

    Returns:
        A compiled ``tf.keras.Model`` ready for training.
    """
    seq_len = input_shape[0]

    # --- Input ---
    inputs = layers.Input(shape=input_shape, name="input_landmarks")

    # --- Linear Projection ---
    x = layers.Dense(d_model, name="linear_projection")(inputs)

    # --- Positional Encoding ---
    x = PositionalEncoding(max_len=seq_len, d_model=d_model, name="pos_encoding")(x)

    # --- Transformer Encoder Stack ---
    for i in range(num_layers):
        x = TransformerEncoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_encoder,
            name=f"transformer_block_{i}",
        )(x)

    # --- Final Layer Norm (Pre-LN convention) ---
    x = layers.LayerNormalization(epsilon=1e-6, name="final_layernorm")(x)

    # --- Pooling ---
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # --- Classification Head ---
    x = layers.Dense(d_model, activation="gelu", name="dense_hidden")(x)
    x = layers.Dropout(dropout_classifier, name="dropout_classifier")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_softmax")(x)

    # --- Compile ---
    model = Model(inputs=inputs, outputs=outputs, name="Transformer_VSL")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Quick smoke-test when running the file directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_transformer_model()
    model.summary()
