import numpy as np
from loguru import logger
from typing import Tuple


def generate_xor_data(n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, (n_samples, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    logger.info(f"XOR data: {n_samples} samples, {sum(y)} positive")
    return X, y


def generate_circle_data(n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    radius = rng.uniform(0, 1, n_samples)
    angle = rng.uniform(0, 2 * np.pi, n_samples)
    X = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    y = (radius < 0.5).astype(int)
    logger.info(f"Circle data: {n_samples} samples, {sum(y)} inner")
    return X, y


def generate_spiral_data(n_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    n = n_samples // 2
    theta = np.sqrt(rng.uniform(0, 1, n)) * 2 * np.pi * 2
    r = theta
    X0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    theta = np.sqrt(rng.uniform(0, 1, n)) * 2 * np.pi * 2
    r = theta
    X1 = np.column_stack([-r * np.cos(theta), -r * np.sin(theta)])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    logger.info(f"Spiral data: {len(X)} samples")
    return X, y
