"""
VSL Visualization Utilities.

Provides plotting functions for:
- Training/validation loss and accuracy curves
- Confusion matrix heatmaps
- Landmark sequence visualization
- Class distribution histograms
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class VSLVisualizer:
    """Visualization utilities for VSL sign language recognition.

    Provides static and instance methods for generating training plots,
    confusion matrices, landmark visualizations, and data distribution charts.

    Args:
        output_dir: Directory to save generated plots.

    Example:
        >>> viz = VSLVisualizer(output_dir="outputs/plots")
        >>> viz.plot_training_curves(history)
        >>> viz.plot_confusion_matrix(cm, labels)
    """

    def __init__(self, output_dir: str = "outputs/plots") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("VSLVisualizer output dir: %s", self.output_dir)

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """Plot training and validation loss/accuracy curves.

        Generates a 2×1 subplot figure with loss curves on top and
        accuracy curves on the bottom.

        Args:
            history: Dictionary from Keras History.history containing
                'loss', 'val_loss', 'accuracy', 'val_accuracy', and
                optionally 'top5_accuracy', 'val_top5_accuracy'.
            save_path: Custom path to save the plot. If None, uses default.
            show: Whether to display the plot interactively.

        Returns:
            Path to the saved plot image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = str(self.output_dir / "training_curves.png")

        epochs = range(1, len(history.get("loss", [])) + 1)

        # Determine number of subplots
        has_top5 = "top5_accuracy" in history
        n_plots = 3 if has_top5 else 2

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        fig.suptitle("VSL Training Curves", fontsize=16, fontweight="bold")

        # --- Loss ---
        ax = axes[0]
        ax.plot(epochs, history["loss"], "b-", label="Train Loss", linewidth=2)
        if "val_loss" in history:
            ax.plot(
                epochs, history["val_loss"], "r-",
                label="Val Loss", linewidth=2
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Top-1 Accuracy ---
        ax = axes[1]
        if "accuracy" in history:
            ax.plot(
                epochs, history["accuracy"], "b-",
                label="Train Accuracy", linewidth=2
            )
        if "val_accuracy" in history:
            ax.plot(
                epochs, history["val_accuracy"], "r-",
                label="Val Accuracy", linewidth=2
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Top-1 Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Top-5 Accuracy (optional) ---
        if has_top5:
            ax = axes[2]
            ax.plot(
                epochs, history["top5_accuracy"], "b-",
                label="Train Top-5", linewidth=2
            )
            if "val_top5_accuracy" in history:
                ax.plot(
                    epochs, history["val_top5_accuracy"], "r-",
                    label="Val Top-5", linewidth=2
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Top-5 Accuracy")
            ax.set_title("Top-5 Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        logger.info("Training curves saved to %s", save_path)
        return save_path

    def plot_confusion_matrix(
        self,
        confusion_mat: np.ndarray,
        labels: List[str],
        save_path: Optional[str] = None,
        show: bool = False,
        normalize: bool = False,
        title: str = "Confusion Matrix",
    ) -> str:
        """Plot a confusion matrix heatmap using seaborn.

        Args:
            confusion_mat: Confusion matrix array of shape (C, C).
            labels: List of class label names.
            save_path: Custom path to save the plot.
            show: Whether to display the plot interactively.
            normalize: Whether to normalize rows to percentages.
            title: Plot title.

        Returns:
            Path to the saved plot image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        if save_path is None:
            save_path = str(self.output_dir / "confusion_matrix.png")

        if normalize:
            row_sums = confusion_mat.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
            confusion_mat = confusion_mat.astype(np.float64) / row_sums
            fmt = ".2f"
        else:
            fmt = "d"

        n = confusion_mat.shape[0]
        annotate = n <= 30
        fig_size = max(10, n * 0.15)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(
            confusion_mat,
            annot=annotate,
            fmt=fmt if annotate else "",
            cmap="Blues",
            xticklabels=labels if n <= 50 else False,
            yticklabels=labels if n <= 50 else False,
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        if n <= 50:
            plt.xticks(rotation=45, ha="right", fontsize=max(4, 10 - n // 10))
            plt.yticks(rotation=0, fontsize=max(4, 10 - n // 10))

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        logger.info("Confusion matrix saved to %s", save_path)
        return save_path

    def plot_landmark_sequence(
        self,
        sequence: np.ndarray,
        frame_indices: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        title: str = "Landmark Sequence",
    ) -> str:
        """Visualize landmark positions as 2D scatter plots.

        Plots hand and pose landmark (x, y) coordinates for selected
        frames in the sequence.

        Args:
            sequence: Landmark sequence of shape (T, F) where T is the
                number of frames and F is the feature dimension.
            frame_indices: List of frame indices to plot. If None,
                plots 6 evenly spaced frames.
            save_path: Custom save path.
            show: Whether to display interactively.
            title: Plot title.

        Returns:
            Path to the saved plot image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = str(self.output_dir / "landmark_sequence.png")

        num_frames = sequence.shape[0]
        if frame_indices is None:
            frame_indices = np.linspace(0, num_frames - 1, 6, dtype=int).tolist()

        n_plots = len(frame_indices)
        fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        if n_plots == 1:
            axes = [axes]

        for ax, idx in zip(axes, frame_indices):
            frame_data = sequence[idx]

            # Extract pose landmarks (first 33 × 4 = 132 features)
            # x, y pairs from pose (skip z, visibility)
            pose_x = frame_data[0:132:4]  # every 4th starting at 0
            pose_y = frame_data[1:132:4]  # every 4th starting at 1

            # Left hand: starts at 132 + 1404 = 1536 (after pose and face)
            # But total feature layout may vary. Use approximate offsets.
            # We'll plot pose and hand landmarks using estimated offsets.
            lh_start = 132 + 468 * 3  # after pose and face
            rh_start = lh_start + 21 * 3

            lh_x = frame_data[lh_start:lh_start + 63:3]
            lh_y = frame_data[lh_start + 1:lh_start + 63:3]

            rh_x = frame_data[rh_start:rh_start + 63:3]
            rh_y = frame_data[rh_start + 1:rh_start + 63:3]

            ax.scatter(pose_x, -pose_y, c="blue", s=10, alpha=0.7, label="Pose")
            ax.scatter(lh_x, -lh_y, c="red", s=15, alpha=0.8, label="Left Hand")
            ax.scatter(rh_x, -rh_y, c="green", s=15, alpha=0.8, label="Right Hand")

            ax.set_title(f"Frame {idx}", fontsize=10)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-1.1, 0.1)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

        # Add legend to first plot
        axes[0].legend(fontsize=7, loc="lower right")

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        logger.info("Landmark sequence plot saved to %s", save_path)
        return save_path

    def plot_class_distribution(
        self,
        labels: List[str],
        counts: List[int],
        save_path: Optional[str] = None,
        show: bool = False,
        top_n: Optional[int] = None,
        title: str = "Class Distribution",
    ) -> str:
        """Plot a histogram of class distribution.

        Args:
            labels: List of class label names.
            counts: List of sample counts per class.
            save_path: Custom save path.
            show: Whether to display interactively.
            top_n: If set, show only the top N most frequent classes.
            title: Plot title.

        Returns:
            Path to the saved plot image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = str(self.output_dir / "class_distribution.png")

        # Sort by count descending
        sorted_pairs = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)

        if top_n is not None:
            sorted_pairs = sorted_pairs[:top_n]

        sorted_labels, sorted_counts = zip(*sorted_pairs) if sorted_pairs else ([], [])

        n = len(sorted_labels)
        fig_width = max(10, n * 0.25)
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        bars = ax.bar(
            range(n),
            sorted_counts,
            color=plt.cm.viridis(np.linspace(0.2, 0.8, n)),
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xlabel("Class", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")

        if n <= 50:
            ax.set_xticks(range(n))
            ax.set_xticklabels(
                sorted_labels,
                rotation=45,
                ha="right",
                fontsize=max(4, 8 - n // 20),
            )
        else:
            ax.set_xticks([])

        # Add mean line
        mean_count = np.mean(sorted_counts) if sorted_counts else 0
        ax.axhline(
            y=mean_count, color="red", linestyle="--",
            alpha=0.7, label=f"Mean: {mean_count:.0f}"
        )
        ax.legend()

        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        logger.info("Class distribution plot saved to %s", save_path)
        return save_path

    def plot_learning_rate_schedule(
        self,
        lr_values: List[float],
        save_path: Optional[str] = None,
        show: bool = False,
    ) -> str:
        """Plot the learning rate schedule over training steps.

        Args:
            lr_values: List of learning rate values per step or epoch.
            save_path: Custom save path.
            show: Whether to display interactively.

        Returns:
            Path to the saved plot.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if save_path is None:
            save_path = str(self.output_dir / "lr_schedule.png")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(lr_values, color="blue", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        logger.info("LR schedule plot saved to %s", save_path)
        return save_path
