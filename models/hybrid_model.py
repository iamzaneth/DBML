"""
CNN + LSTM + Attention hybrid model for Vietnamese Sign Language (VSL) recognition.

This is the **recommended** architecture, combining three complementary
inductive biases:

1. **Conv1D** layers extract local temporal patterns and reduce feature
   dimensionality.
2. **Bidirectional LSTM** captures long-range temporal dependencies.
3. **Multi-Head Self-Attention** allows the model to attend to the most
   informative frames across the entire sequence.

Architecture:
    Input (batch, 60, 1605)
    → Conv1D (128, kernel=3, padding='same') + ReLU + BatchNorm
    → Conv1D (256, kernel=3, padding='same') + ReLU + BatchNorm
    → MaxPooling1D (pool=2) → (batch, 30, 256)
    → Bidirectional LSTM (128, return_sequences=True)
    → Multi-Head Self-Attention (4 heads)
    → Global Average Pooling
    → Dense (256, ReLU) + Dropout (0.4)
    → Dense (num_classes, Softmax)
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model


class MultiHeadSelfAttention(layers.Layer):
    """Multi-Head Self-Attention layer.

    Wraps ``tf.keras.layers.MultiHeadAttention`` in a residual block with
    Layer Normalization (Pre-LN style) for use as a standalone attention
    module inside the hybrid model.

    Args:
        num_heads: Number of attention heads. Default: 4.
        key_dim: Dimensionality of each attention head.  When ``None``,
                 it is inferred as ``d_model // num_heads`` during build.
        dropout: Dropout rate applied to attention weights. Default: 0.1.
    """

    def __init__(
        self,
        num_heads: int = 4,
        key_dim: int | None = None,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self._key_dim = key_dim
        self.dropout = dropout

    def build(self, input_shape: tf.TensorShape) -> None:
        d_model = int(input_shape[-1])
        key_dim = self._key_dim or (d_model // self.num_heads)

        self.layernorm = layers.LayerNormalization(epsilon=1e-6, name="ln_attn")
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            name="mha",
        )
        self.dropout_layer = layers.Dropout(self.dropout, name="dropout_attn")
        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Apply self-attention with residual connection.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            training: Whether in training mode (enables dropout).

        Returns:
            Output tensor, same shape as input.
        """
        x_norm = self.layernorm(x)
        attn_output = self.mha(x_norm, x_norm, training=training)
        attn_output = self.dropout_layer(attn_output, training=training)
        return x + attn_output  # Residual

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self._key_dim,
                "dropout": self.dropout,
            }
        )
        return config


# ===========================================================================
# Model Builder
# ===========================================================================


def build_hybrid_model(
    input_shape: Tuple[int, int] = (60, 1605),
    num_classes: int = 161,
    conv_filters: Tuple[int, int] = (128, 256),
    conv_kernel_size: int = 3,
    pool_size: int = 2,
    lstm_units: int = 128,
    attention_heads: int = 4,
    attention_dropout: float = 0.1,
    dense_units: int = 256,
    dropout_dense: float = 0.4,
    learning_rate: float = 1e-3,
) -> Model:
    """Build a CNN + BiLSTM + Attention hybrid model for sequence classification.

    This is the **recommended** model for the VSL recognition task.  The
    Conv1D front-end compresses the 1605-dim landmark features into a
    compact representation while capturing local temporal patterns.  The
    Bidirectional LSTM models sequential dependencies, and Multi-Head
    Self-Attention surfaces the most discriminative frames.

    Args:
        input_shape: Tuple of (sequence_length, num_features).
                     Default: (60, 1605).
        num_classes: Number of output classes. Default: 161.
        conv_filters: Number of filters for each Conv1D layer.
                      Default: (128, 256).
        conv_kernel_size: Kernel width for Conv1D layers. Default: 3.
        pool_size: MaxPooling1D pool size (halves temporal length).
                   Default: 2.
        lstm_units: Units in the Bidirectional LSTM. Default: 128.
        attention_heads: Number of heads in the self-attention layer.
                         Default: 4.
        attention_dropout: Dropout rate inside the attention layer.
                           Default: 0.1.
        dense_units: Units in the dense classification head. Default: 256.
        dropout_dense: Dropout rate after the dense layer. Default: 0.4.
        learning_rate: Adam optimizer learning rate. Default: 1e-3.

    Returns:
        A compiled ``tf.keras.Model`` ready for training.
    """
    # --- Input ---
    inputs = layers.Input(shape=input_shape, name="input_landmarks")

    # --- Conv1D Feature Extractor ---
    x = layers.Conv1D(
        conv_filters[0],
        kernel_size=conv_kernel_size,
        padding="same",
        activation="relu",
        name="conv1d_1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_conv_1")(x)

    x = layers.Conv1D(
        conv_filters[1],
        kernel_size=conv_kernel_size,
        padding="same",
        activation="relu",
        name="conv1d_2",
    )(x)
    x = layers.BatchNormalization(name="bn_conv_2")(x)

    # --- Temporal Downsampling ---
    x = layers.MaxPooling1D(pool_size=pool_size, name="maxpool")(x)
    # Shape: (batch, 30, 256)

    # --- Bidirectional LSTM ---
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, name="lstm"),
        name="bilstm",
    )(x)
    # Shape: (batch, 30, 256)  — BiLSTM doubles the units

    # --- Multi-Head Self-Attention ---
    x = MultiHeadSelfAttention(
        num_heads=attention_heads,
        dropout=attention_dropout,
        name="self_attention",
    )(x)

    # --- Pooling ---
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # --- Classification Head ---
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = layers.Dropout(dropout_dense, name="dropout_dense")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_softmax")(x)

    # --- Compile ---
    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_LSTM_Attn_VSL")

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
    model = build_hybrid_model()
    model.summary()
