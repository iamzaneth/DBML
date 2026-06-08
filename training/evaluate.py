"""
VSL Evaluation Module.

Provides comprehensive evaluation of trained VSL models including:
- Top-1 and Top-5 accuracy
- Per-class precision, recall, F1 (sklearn classification_report)
- Confusion matrix (saved as image)
- Inference latency benchmark
- Evaluation report saved to file
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class VSLEvaluator:
    """Evaluator for Vietnamese Sign Language recognition models.

    Computes comprehensive metrics and generates evaluation reports.

    Args:
        model: Trained Keras model for evaluation.
        labels: List of class label names (length must match num_classes).
        output_dir: Directory to save evaluation artifacts.

    Example:
        >>> evaluator = VSLEvaluator(model, labels, output_dir="outputs/eval")
        >>> results = evaluator.evaluate(test_dataset)
        >>> evaluator.save_report(results)
    """

    def __init__(
        self,
        model: keras.Model,
        labels: List[str],
        output_dir: str = "outputs/eval",
    ) -> None:
        self.model = model
        self.labels = labels
        self.num_classes = len(labels)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "VSLEvaluator initialized with %d classes, output: %s",
            self.num_classes,
            self.output_dir,
        )

    def evaluate(
        self,
        test_dataset: tf.data.Dataset,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Run full evaluation on the test dataset.

        Args:
            test_dataset: Test tf.data.Dataset (batched, yielding (x, y) pairs).
            batch_size: Batch size for evaluation (used in latency benchmark).

        Returns:
            Dictionary containing all evaluation metrics.
        """
        logger.info("Starting evaluation...")

        # Collect predictions and ground truth
        y_true_list: List[np.ndarray] = []
        y_pred_probs_list: List[np.ndarray] = []

        for x_batch, y_batch in test_dataset:
            preds = self.model.predict(x_batch, verbose=0)
            y_pred_probs_list.append(preds)
            y_true_list.append(y_batch.numpy() if isinstance(y_batch, tf.Tensor) else y_batch)

        y_true_onehot = np.concatenate(y_true_list, axis=0)
        y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)

        # Convert from one-hot to class indices
        y_true = np.argmax(y_true_onehot, axis=1)
        y_pred = np.argmax(y_pred_probs, axis=1)

        total_samples = len(y_true)
        logger.info("Evaluation on %d samples", total_samples)

        # Top-1 accuracy
        top1_accuracy = float(np.mean(y_true == y_pred))

        # Top-5 accuracy
        top5_accuracy = self._compute_topk_accuracy(y_true, y_pred_probs, k=5)

        # Per-class metrics via sklearn
        classification_rep, per_class_metrics = self._compute_classification_report(
            y_true, y_pred
        )

        # Confusion matrix
        confusion_mat = self._compute_confusion_matrix(y_true, y_pred)

        # Save confusion matrix image
        self._save_confusion_matrix_image(confusion_mat)

        # Latency benchmark
        latency_stats = self._benchmark_latency(test_dataset, num_runs=50)

        results: Dict[str, Any] = {
            "total_samples": total_samples,
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "classification_report": classification_rep,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": confusion_mat.tolist(),
            "latency": latency_stats,
        }

        logger.info("Top-1 Accuracy: %.4f", top1_accuracy)
        logger.info("Top-5 Accuracy: %.4f", top5_accuracy)
        logger.info(
            "Avg Latency: %.2f ms (±%.2f ms)",
            latency_stats["mean_ms"],
            latency_stats["std_ms"],
        )

        return results

    @staticmethod
    def _compute_topk_accuracy(
        y_true: np.ndarray, y_pred_probs: np.ndarray, k: int = 5
    ) -> float:
        """Compute top-k accuracy.

        Args:
            y_true: Ground truth class indices, shape (N,).
            y_pred_probs: Predicted probabilities, shape (N, C).
            k: Number of top predictions to consider.

        Returns:
            Top-k accuracy as a float.
        """
        top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        return float(np.mean(correct))

    def _compute_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[str, Dict[str, Dict[str, float]]]:
        """Compute per-class precision, recall, F1 using sklearn.

        Args:
            y_true: Ground truth class indices.
            y_pred: Predicted class indices.

        Returns:
            Tuple of (formatted report string, per-class metrics dict).
        """
        try:
            from sklearn.metrics import classification_report

            # Determine which labels actually appear
            present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
            target_names = [
                self.labels[i] if i < len(self.labels) else f"class_{i}"
                for i in present_labels
            ]

            report_str = classification_report(
                y_true,
                y_pred,
                labels=present_labels,
                target_names=target_names,
                zero_division=0,
                digits=4,
            )

            report_dict = classification_report(
                y_true,
                y_pred,
                labels=present_labels,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
            )

            logger.info("\n%s", report_str)
            return report_str, report_dict

        except ImportError:
            logger.warning(
                "scikit-learn not installed; skipping classification report."
            )
            return "sklearn not available", {}

    def _compute_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute the confusion matrix.

        Args:
            y_true: Ground truth class indices.
            y_pred: Predicted class indices.

        Returns:
            Confusion matrix as a numpy array of shape (C, C).
        """
        try:
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(
                y_true, y_pred, labels=list(range(self.num_classes))
            )
            return cm
        except ImportError:
            logger.warning("scikit-learn not installed; computing confusion matrix manually.")
            cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
            for t, p in zip(y_true, y_pred):
                cm[t, p] += 1
            return cm

    def _save_confusion_matrix_image(self, confusion_mat: np.ndarray) -> str:
        """Save the confusion matrix as a heatmap image.

        Args:
            confusion_mat: Confusion matrix array.

        Returns:
            Path to the saved image.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning(
                "matplotlib/seaborn not installed; skipping confusion matrix plot."
            )
            return ""

        fig_path = str(self.output_dir / "confusion_matrix.png")

        # For many classes, use a smaller figure and skip annotations
        n = confusion_mat.shape[0]
        annotate = n <= 30
        fig_size = max(10, n * 0.15)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(
            confusion_mat,
            annot=annotate,
            fmt="d" if annotate else "",
            cmap="Blues",
            xticklabels=self.labels if n <= 50 else False,
            yticklabels=self.labels if n <= 50 else False,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)
        plt.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        logger.info("Confusion matrix saved to %s", fig_path)
        return fig_path

    def _benchmark_latency(
        self,
        test_dataset: tf.data.Dataset,
        num_runs: int = 50,
    ) -> Dict[str, float]:
        """Benchmark single-sample inference latency.

        Args:
            test_dataset: Test dataset to sample from.
            num_runs: Number of inference runs for benchmarking.

        Returns:
            Dictionary with mean_ms, std_ms, min_ms, max_ms latency stats.
        """
        # Get a single sample for benchmarking
        sample_batch = None
        for x_batch, _ in test_dataset.take(1):
            sample_batch = x_batch[:1]  # Single sample
            break

        if sample_batch is None:
            logger.warning("Cannot benchmark latency: empty dataset.")
            return {"mean_ms": 0.0, "std_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

        # Warmup
        for _ in range(5):
            _ = self.model.predict(sample_batch, verbose=0)

        # Benchmark
        latencies: List[float] = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.model.predict(sample_batch, verbose=0)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        stats = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
        }
        logger.info(
            "Latency: %.2f ± %.2f ms (min=%.2f, max=%.2f)",
            stats["mean_ms"],
            stats["std_ms"],
            stats["min_ms"],
            stats["max_ms"],
        )
        return stats

    def save_report(self, results: Dict[str, Any], filename: str = "evaluation_report.json") -> str:
        """Save evaluation report to a JSON file.

        Args:
            results: Evaluation results dictionary from evaluate().
            filename: Name of the output report file.

        Returns:
            Path to the saved report file.
        """
        report_path = str(self.output_dir / filename)

        # Remove non-serializable items, keep text report separate
        serializable = {k: v for k, v in results.items()}
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)

        # Also save the text classification report
        text_report_path = str(self.output_dir / "classification_report.txt")
        with open(text_report_path, "w", encoding="utf-8") as f:
            f.write(str(results.get("classification_report", "")))

        logger.info("Evaluation report saved to %s", report_path)
        logger.info("Classification report saved to %s", text_report_path)

        # Print summary to console
        print("\n" + "=" * 50)
        print("  Evaluation Summary")
        print("=" * 50)
        print(f"  Total samples:  {results['total_samples']}")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.4f}")
        print(f"  Avg Latency:    {results['latency']['mean_ms']:.2f} ms")
        print("=" * 50 + "\n")

        return report_path
