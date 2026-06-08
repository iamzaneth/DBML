"""
VSL Training Pipeline - Trainer Module.

Provides the VSLTrainer class for training sign language recognition models
with mixed precision, cosine annealing with warmup, and comprehensive callbacks.

Designed for RTX 4050 Laptop GPU (6GB VRAM) with float16 mixed precision.
"""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

logger = logging.getLogger(__name__)


class CosineAnnealingWarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Cosine annealing learning rate schedule with linear warmup.

    The schedule linearly increases the learning rate from 0 to `max_lr`
    during the warmup phase, then decays it following a cosine curve
    to `min_lr`.

    Args:
        max_lr: Peak learning rate after warmup.
        warmup_steps: Number of steps for linear warmup.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate at the end of cosine decay.
    """

    def __init__(
        self,
        max_lr: float = 1e-3,
        warmup_steps: int = 1000,
        total_steps: int = 50000,
        min_lr: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Linear warmup
        warmup_lr = self.max_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine annealing after warmup
        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.minimum(progress, 1.0)
        cosine_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


class VSLTrainer:
    """Trainer for Vietnamese Sign Language recognition models.

    Handles the full training lifecycle including mixed precision setup,
    optimizer configuration, callback creation, and model fitting.

    Args:
        model: A compiled or uncompiled Keras model.
        config: Training configuration dictionary. Expected keys:
            - output_dir (str): Directory for saving checkpoints and logs.
            - epochs (int): Maximum number of training epochs.
            - batch_size (int): Training batch size.
            - learning_rate (float): Peak learning rate.
            - weight_decay (float): AdamW weight decay.
            - warmup_epochs (int): Number of warmup epochs.
            - label_smoothing (float): Label smoothing factor.
            - patience (int): EarlyStopping patience.
            - use_mixed_precision (bool): Whether to use float16 mixed precision.
            - num_classes (int): Number of output classes.
        labels: Optional list of label names for logging.

    Example:
        >>> trainer = VSLTrainer(model, config)
        >>> history = trainer.train(train_dataset, val_dataset)
    """

    def __init__(
        self,
        model: keras.Model,
        config: Dict[str, Any],
        labels: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.config = self._apply_defaults(config)
        self.labels = labels
        self.history: Optional[keras.callbacks.History] = None

        # Create output directory
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup mixed precision
        if self.config["use_mixed_precision"]:
            self._setup_mixed_precision()

        logger.info("VSLTrainer initialized with config: %s", self.config)

    @staticmethod
    def _apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for missing config keys."""
        defaults = {
            "output_dir": "outputs",
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "warmup_epochs": 5,
            "label_smoothing": 0.1,
            "patience": 20,
            "use_mixed_precision": True,
            "num_classes": 161,
            "min_lr": 1e-6,
        }
        merged = {**defaults, **config}
        return merged

    def _setup_mixed_precision(self) -> None:
        """Enable mixed precision training for faster GPU computation."""
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)
        logger.info(
            "Mixed precision enabled — compute: %s, variable: %s",
            policy.compute_dtype,
            policy.variable_dtype,
        )

    def _build_lr_schedule(
        self, steps_per_epoch: int
    ) -> CosineAnnealingWarmupSchedule:
        """Build cosine annealing LR schedule with warmup.

        Args:
            steps_per_epoch: Number of training steps per epoch.

        Returns:
            Configured learning rate schedule.
        """
        total_steps = steps_per_epoch * self.config["epochs"]
        warmup_steps = steps_per_epoch * self.config["warmup_epochs"]

        schedule = CosineAnnealingWarmupSchedule(
            max_lr=self.config["learning_rate"],
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config["min_lr"],
        )
        logger.info(
            "LR schedule: cosine annealing with %d warmup steps, %d total steps",
            warmup_steps,
            total_steps,
        )
        return schedule

    def _compile_model(self, steps_per_epoch: int) -> None:
        """Compile the model with AdamW optimizer and label-smoothed loss.

        Args:
            steps_per_epoch: Number of training steps per epoch.
        """
        lr_schedule = self._build_lr_schedule(steps_per_epoch)

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.config["weight_decay"],
        )

        # Wrap optimizer for mixed precision (loss scaling)
        if self.config["use_mixed_precision"]:
            # In TF 2.x, AdamW handles mixed precision natively when
            # global policy is set. No manual LossScaleOptimizer needed
            # for tf >= 2.11 with new optimizer API.
            pass

        loss_fn = keras.losses.CategoricalCrossentropy(
            label_smoothing=self.config["label_smoothing"],
            from_logits=False,
        )

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
            ],
        )
        logger.info("Model compiled with AdamW optimizer and label smoothing=%.2f", self.config["label_smoothing"])

    def _build_callbacks(self) -> List[keras.callbacks.Callback]:
        """Build the set of training callbacks.

        Returns:
            List of configured Keras callbacks.
        """
        callbacks = []

        # 1. ModelCheckpoint — save best model by val_loss
        checkpoint_path = str(self.output_dir / "best_model.h5")
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            )
        )
        logger.info("ModelCheckpoint: saving best model to %s", checkpoint_path)

        # 2. EarlyStopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config["patience"],
                restore_best_weights=True,
                verbose=1,
            )
        )

        # 3. ReduceLROnPlateau (as a fallback alongside cosine schedule)
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.config["patience"] // 2,
                min_lr=self.config["min_lr"],
                verbose=1,
            )
        )

        # 4. TensorBoard
        tb_log_dir = str(self.output_dir / "tensorboard_logs")
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=tb_log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq="epoch",
            )
        )
        logger.info("TensorBoard logs: %s", tb_log_dir)

        # 5. CSVLogger
        csv_path = str(self.output_dir / "training_log.csv")
        callbacks.append(
            keras.callbacks.CSVLogger(csv_path, append=True)
        )
        logger.info("CSVLogger: %s", csv_path)

        return callbacks

    def _print_training_summary(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset],
    ) -> None:
        """Print a formatted training summary before training begins.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset (may be None).
        """
        divider = "=" * 60
        print(f"\n{divider}")
        print("  Vietnamese Sign Language — Training Summary")
        print(divider)
        print(f"  Model:              {self.model.name}")
        print(f"  Parameters:         {self.model.count_params():,}")
        print(f"  Epochs:             {self.config['epochs']}")
        print(f"  Batch size:         {self.config['batch_size']}")
        print(f"  Learning rate:      {self.config['learning_rate']}")
        print(f"  Weight decay:       {self.config['weight_decay']}")
        print(f"  Label smoothing:    {self.config['label_smoothing']}")
        print(f"  Mixed precision:    {self.config['use_mixed_precision']}")
        print(f"  Num classes:        {self.config['num_classes']}")
        print(f"  Output dir:         {self.output_dir}")
        print(f"  Early stop patience:{self.config['patience']}")

        if val_dataset is not None:
            print(f"  Validation:         Yes")
        else:
            print(f"  Validation:         No")

        # GPU info
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                print(f"  GPU:                {gpu.name}")
        else:
            print("  GPU:                None (training on CPU)")

        print(divider + "\n")

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
    ) -> keras.callbacks.History:
        """Run the full training loop.

        Args:
            train_dataset: Training tf.data.Dataset (batched).
            val_dataset: Validation tf.data.Dataset (batched). Optional.
            steps_per_epoch: Override for steps per epoch. If None, auto-detect.
            validation_steps: Override for validation steps. If None, auto-detect.

        Returns:
            Keras History object containing training metrics.

        Raises:
            RuntimeError: If training fails unexpectedly.
        """
        try:
            # Estimate steps per epoch if not provided
            if steps_per_epoch is None:
                steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
                if steps_per_epoch <= 0:
                    # Fallback: count manually (expensive for large datasets)
                    logger.warning(
                        "Cannot determine dataset cardinality; counting batches..."
                    )
                    steps_per_epoch = sum(1 for _ in train_dataset)
                logger.info("Steps per epoch: %d", steps_per_epoch)

            # Compile the model
            self._compile_model(steps_per_epoch)

            # Print summary
            self._print_training_summary(train_dataset, val_dataset)

            # Build callbacks
            callbacks = self._build_callbacks()

            # Train
            start_time = time.time()
            self.history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=self.config["epochs"],
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1,
            )
            elapsed = time.time() - start_time
            logger.info("Training completed in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

            # Save final model
            final_path = str(self.output_dir / "final_model.h5")
            self.model.save(final_path)
            logger.info("Final model saved to %s", final_path)

            # Save training config
            config_path = str(self.output_dir / "training_config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info("Training config saved to %s", config_path)

            return self.history

        except Exception as e:
            logger.error("Training failed: %s", str(e))
            raise RuntimeError(f"Training failed: {e}") from e

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the model to disk.

        Args:
            path: Optional custom save path. Defaults to output_dir/model.h5.

        Returns:
            The path where the model was saved.
        """
        if path is None:
            path = str(self.output_dir / "model.h5")
        self.model.save(path)
        logger.info("Model saved to %s", path)
        return path

    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """Return the training history as a dictionary.

        Returns:
            Dictionary mapping metric names to lists of values per epoch,
            or None if training has not been run yet.
        """
        if self.history is None:
            return None
        return self.history.history
