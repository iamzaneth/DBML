"""
Bidirectional LSTM baseline model for Vietnamese Sign Language (VSL) recognition.

Architecture:
    Input (batch, 60, 1605)
    → BatchNormalization
    → Bidirectional LSTM (256 units, return_sequences=True)
    → Dropout (0.3)
    → Bidirectional LSTM (128 units, return_sequences=False)
    → Dropout (0.3)
    → Dense (256, ReLU) + BatchNorm
    → Dropout (0.4)
    → Dense (num_classes, Softmax)
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from keras import layers, Model


def build_lstm_model(
    input_shape: Tuple[int, int] = (60, 1605),
    num_classes: int = 161,
    lstm_units_1: int = 256,
    lstm_units_2: int = 128,
    dense_units: int = 256,
    dropout_lstm: float = 0.3,
    dropout_dense: float = 0.4,
    learning_rate: float = 1e-3,
) -> Model:
    """Build a Bidirectional LSTM model for sequence classification.

    This is the baseline model using two stacked Bidirectional LSTM layers
    with batch normalization and dropout regularization.

    Args:
        input_shape: Tuple of (sequence_length, num_features).
                     Default: (60, 1605) — 60 frames × 1605 landmark features.
        num_classes: Number of output classes. Default: 161 VSL signs.
        lstm_units_1: Units in the first BiLSTM layer. Default: 256.
        lstm_units_2: Units in the second BiLSTM layer. Default: 128.
        dense_units: Units in the fully-connected classification head. Default: 256.
        dropout_lstm: Dropout rate after each LSTM layer. Default: 0.3.
        dropout_dense: Dropout rate after the dense layer. Default: 0.4.
        learning_rate: Adam optimizer learning rate. Default: 1e-3.

    Returns:
        A compiled ``tf.keras.Model`` ready for training.
    """
    # --- Input ---
    inputs = layers.Input(shape=input_shape, name="input_landmarks")

    # --- Normalization ---
    x = layers.BatchNormalization(name="input_batchnorm")(inputs)

    # --- Stacked Bidirectional LSTMs ---
    x = layers.Bidirectional(
        layers.LSTM(lstm_units_1, return_sequences=True, name="lstm_1"),
        name="bilstm_1",
    )(x)
    x = layers.Dropout(dropout_lstm, name="dropout_lstm_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units_2, return_sequences=False, name="lstm_2"),
        name="bilstm_2",
    )(x)
    x = layers.Dropout(dropout_lstm, name="dropout_lstm_2")(x)

    # --- Classification Head ---
    x = layers.Dense(dense_units, activation="relu", name="dense_hidden")(x)
    x = layers.BatchNormalization(name="dense_batchnorm")(x)
    x = layers.Dropout(dropout_dense, name="dropout_dense")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output_softmax")(x)

    # --- Compile ---
    model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_VSL")

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
    model = build_lstm_model()
    model.summary()
