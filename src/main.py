import typer
import sys
from loguru import logger

from .config import settings
from .data import generate_xor_data, generate_circle_data, generate_spiral_data
from .model import NeuralNetwork
from .visualizer import NNVisualizer

app = typer.Typer(help="Neural Network from Scratch CLI")

logger.remove()
logger.add(sys.stderr, level=settings.log_level)

DATASETS = {"xor": generate_xor_data, "circle": generate_circle_data, "spiral": generate_spiral_data}


@app.command()
def train(
    dataset: str = typer.Option("xor", help="Dataset: xor, circle, spiral"),
    hidden: int = typer.Option(4, help="Hidden layer size"),
    activation: str = typer.Option("relu", help="Activation: relu, sigmoid, tanh"),
    lr: float = typer.Option(0.1, help="Learning rate"),
    epochs: int = typer.Option(2000, help="Training epochs"),
):
    logger.info(f"Training NN on {dataset} | hidden={hidden} | act={activation} | lr={lr}")
    X, y = DATASETS[dataset]()
    nn = NeuralNetwork(
        layer_sizes=[2, hidden, 1],
        activation=activation,
        learning_rate=lr,
    )
    losses = nn.fit(X, y, epochs=epochs)
    acc = nn.accuracy(X, y)
    logger.success(f"Final accuracy: {acc:.2%} | Final loss: {losses[-1]:.6f}")
    vis = NNVisualizer()
    vis.plot_decision_boundary(
        nn, X, y, title=f"NN on {dataset} — {acc:.1%}",
        save_path=settings.plots_dir / f"{dataset}_boundary.png",
    )
    vis.plot_loss_curve(losses, save_path=settings.plots_dir / f"{dataset}_loss.png")


if __name__ == "__main__":
    app()
