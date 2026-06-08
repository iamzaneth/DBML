#!/usr/bin/env python3
"""
train.py — Main entry point for VSL sign language model training.

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --epochs 50 --batch_size 16
    python train.py --config config/config.yaml --model_type hybrid --output_dir outputs/exp1

This script:
    1. Parses command-line arguments
    2. Loads training configuration via `config` module
    3. Builds the dataset pipeline via `VSLDatasetBuilder`
    4. Builds or loads the model via `models.build_model`
    5. Trains with mixed precision, callbacks, and LR scheduling via `VSLTrainer`
    6. Evaluates on the test set via `VSLEvaluator`
    7. Saves the best model, training logs, and evaluation report
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import load_config
from data.dataset import VSLDatasetBuilder
from models import build_model as build_vsl_model
from training.trainer import VSLTrainer
from training.evaluate import VSLEvaluator
from utils.visualization import VSLVisualizer

# ────────────────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ────────────────────────────────────────────────────────────
# CLI argument parsing
# ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a Vietnamese Sign Language (VSL) recognition model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML training configuration file.",
    )

    # Overrides
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Override model type from config (e.g., 'transformer', 'lstm', 'hybrid').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory for checkpoints and logs.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the dataset directory (raw .npz files).",
    )

    # Training options
    parser.add_argument(
        "--no_mixed_precision",
        action="store_true",
        help="Disable mixed precision (float16) training.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Skip training and only evaluate the model.",
    )

    # GPU
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use.",
    )

    return parser.parse_args()


# ────────────────────────────────────────────────────────────
# GPU configuration
# ────────────────────────────────────────────────────────────
def configure_gpu(gpu_id: str = "0") -> None:
    """Configure GPU memory growth and device visibility."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU configured: %s", [g.name for g in gpus])
        except RuntimeError as e:
            logger.warning("GPU configuration error: %s", e)
    else:
        logger.warning("No GPU found. Training will run on CPU.")


def load_labels_from_json(data_dir: str) -> list[str]:
    """Helper to load labels from labels.json if it exists."""
    labels_path = Path(data_dir) / ".." / "labels.json"
    if not labels_path.exists():
        labels_path = Path("labels.json")
    
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, list):
                return raw
            elif isinstance(raw, dict):
                max_idx = max(int(k) for k in raw.keys())
                labels = [""] * (max_idx + 1)
                for k, v in raw.items():
                    labels[int(k)] = v
                return labels
    return []


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Configure GPU
    configure_gpu(args.gpu)

    # Prepare config overrides
    overrides = {}
    if args.model_type is not None:
        overrides["model"] = {"model_type": args.model_type}
    if args.epochs is not None:
        overrides["training"] = {"epochs": args.epochs}
    if args.batch_size is not None:
        if "training" not in overrides: overrides["training"] = {}
        overrides["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        if "training" not in overrides: overrides["training"] = {}
        overrides["training"]["learning_rate"] = args.learning_rate
    if args.output_dir is not None:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["output_dir"] = args.output_dir
    if args.data_dir is not None:
        overrides["data"] = {"raw_dir": args.data_dir}
    if args.no_mixed_precision:
        overrides["training"] = overrides.get("training", {})
        overrides["training"]["use_mixed_precision"] = False

    # Load config
    cfg = load_config(args.config, overrides=overrides if overrides else None)
    logger.info("Loaded configuration.")

    # Build datasets
    builder = VSLDatasetBuilder(cfg)
    datasets, info = builder.build_with_info(cache=True)
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    # Try to load labels from dataset mapping or labels.json
    labels = []
    mapping = builder.get_label_mapping()
    if mapping:
        max_idx = max(mapping.keys())
        labels = [""] * (max_idx + 1)
        for k, v in mapping.items():
            labels[k] = v
    else:
        labels = load_labels_from_json(getattr(cfg.data, "raw_dir", "data/raw"))
    
    if not labels:
        logger.warning("No labels mapping found. Using dummy labels.")
        labels = [f"Class_{i}" for i in range(info["num_classes"])]

    # Prepare model config
    model_cfg = cfg.model.to_dict()
    model_cfg["input_shape"] = (info["seq_length"], info["num_features"])
    model_cfg["num_classes"] = info["num_classes"]

    # Build or load model
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        model = keras.models.load_model(args.resume, compile=False)
    else:
        model = build_vsl_model(model_cfg)

    # ── Training ──
    if not args.evaluate_only:
        trainer_cfg = cfg.training.to_dict()
        trainer_cfg["num_classes"] = info["num_classes"]
        
        trainer = VSLTrainer(model=model, config=trainer_cfg, labels=labels)
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            steps_per_epoch=info["train_steps"],
            validation_steps=info["val_steps"],
        )

        # Plot training curves
        viz = VSLVisualizer(output_dir=str(Path(trainer.output_dir) / "plots"))
        if history and history.history:
            viz.plot_training_curves(history.history)

        # Use the trained model (with best weights restored by EarlyStopping)
        model = trainer.model
    else:
        trainer_cfg = cfg.training.to_dict()
        trainer_cfg["num_classes"] = info["num_classes"]
        trainer = VSLTrainer(model=model, config=trainer_cfg, labels=labels)

    # ── Evaluation ──
    eval_dir = str(Path(trainer.output_dir) / "eval")
    evaluator = VSLEvaluator(model=model, labels=labels, output_dir=eval_dir)
    results = evaluator.evaluate(test_dataset)
    evaluator.save_report(results)

    # Plot confusion matrix
    viz = VSLVisualizer(output_dir=eval_dir)
    cm = np.array(results["confusion_matrix"])
    viz.plot_confusion_matrix(cm, labels)

    logger.info("Training and evaluation complete.")
    logger.info("Results saved to: %s", trainer.output_dir)


if __name__ == "__main__":
    main()
