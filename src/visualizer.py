import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loguru import logger


class NNVisualizer:
    @staticmethod
    def plot_decision_boundary(
        model, X: np.ndarray, y: np.ndarray,
        title: str = "Decision Boundary",
        save_path: Optional[Path] = None,
    ) -> None:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm", levels=[0, 0.5, 1])
        plt.scatter(
            X[:, 0], X[:, 1], c=y, cmap="coolwarm",
            edgecolor="k", s=40, alpha=0.8,
        )
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_loss_curve(
        losses: list, save_path: Optional[Path] = None,
    ) -> None:
        plt.figure(figsize=(8, 4))
        plt.plot(losses, linewidth=1.5)
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Loss curve saved to {save_path}")
        plt.close()
