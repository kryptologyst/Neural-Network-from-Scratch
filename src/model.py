import numpy as np
from loguru import logger
from typing import Optional, Tuple, Literal


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: list = None,
        activation: Literal["relu", "sigmoid", "tanh"] = "relu",
        learning_rate: float = 0.01,
        random_state: int = 42,
    ):
        self.layer_sizes = layer_sizes or [2, 4, 1]
        self.activation_name = activation
        self.lr = learning_rate
        self.rng = np.random.default_rng(random_state)
        self.weights: list = []
        self.biases: list = []
        self._init_params()

    def _init_params(self) -> None:
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            self.weights.append(
                self.rng.uniform(-limit, limit, (fan_in, fan_out))
            )
            self.biases.append(np.zeros((1, fan_out)))

    def _activation(self, Z: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return np.maximum(0, Z)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        elif self.activation_name == "tanh":
            return np.tanh(Z)

    def _activation_derivative(self, A: np.ndarray) -> np.ndarray:
        if self.activation_name == "relu":
            return (A > 0).astype(float)
        elif self.activation_name == "sigmoid":
            return A * (1 - A)
        elif self.activation_name == "tanh":
            return 1 - A**2

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, list, list]:
        A = X
        activations = [A]
        z_list = []
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            Z = A @ W + b
            z_list.append(Z)
            A = self._activation(Z)
            activations.append(A)
        W_out, b_out = self.weights[-1], self.biases[-1]
        Z_out = A @ W_out + b_out
        z_list.append(Z_out)
        A_out = 1 / (1 + np.exp(-np.clip(Z_out, -500, 500)))
        activations.append(A_out)
        return A_out, activations, z_list

    def _binary_cross_entropy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def backward(
        self, X: np.ndarray, y: np.ndarray,
        activations: list, z_list: list,
    ) -> None:
        m = X.shape[0]
        y = y.reshape(-1, 1)
        A_out = activations[-1]
        dA = -(y / np.clip(A_out, 1e-15, 1 - 1e-15) - (1 - y) / np.clip(1 - A_out, 1e-15, 1 - 1e-15))
        dZ = dA * A_out * (1 - A_out)

        for l in range(len(self.weights) - 1, -1, -1):
            A_prev = activations[l]
            dW = (A_prev.T @ dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            self.weights[l] -= self.lr * dW
            self.biases[l] -= self.lr * db
            if l > 0:
                dA = dZ @ self.weights[l].T
                dZ = dA * self._activation_derivative(activations[l])

    def fit(
        self, X: np.ndarray, y: np.ndarray,
        epochs: int = 1000, verbose: bool = True,
    ) -> list:
        losses = []
        for epoch in range(epochs):
            A_out, activations, z_list = self.forward(X)
            loss = self._binary_cross_entropy(A_out, y)
            losses.append(loss)
            self.backward(X, y, activations, z_list)
            if verbose and epoch % 200 == 0:
                logger.info(f"Epoch {epoch:4d} — Loss: {loss:.6f}")
        return losses

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        A_out, _, _ = self.forward(X)
        return (A_out >= threshold).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        A_out, _, _ = self.forward(X)
        return A_out.flatten()

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y.flatten()))
