import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import NeuralNetwork
from src.data import generate_xor_data


class TestNeuralNetwork:
    @pytest.fixture
    def xor_data(self):
        return generate_xor_data(200)

    @pytest.fixture
    def trained_nn(self, xor_data):
        X, y = xor_data
        nn = NeuralNetwork(
            layer_sizes=[2, 8, 1],
            activation="relu",
            learning_rate=0.1,
        )
        nn.fit(X, y, epochs=1000, verbose=False)
        return nn

    def test_forward_output_shape(self, xor_data):
        X, _ = xor_data
        nn = NeuralNetwork([2, 4, 1])
        out, acts, zs = nn.forward(X)
        assert out.shape == (len(X), 1)

    def test_predict_binary(self, trained_nn, xor_data):
        X, _ = xor_data
        preds = trained_nn.predict(X)
        assert set(preds).issubset({0, 1})

    def test_xor_accuracy(self, trained_nn, xor_data):
        X, y = xor_data
        acc = trained_nn.accuracy(X, y)
        assert acc > 0.95, f"Expected >95%, got {acc:.2%}"

    def test_loss_decreases(self, xor_data):
        X, y = xor_data
        nn = NeuralNetwork([2, 4, 1], learning_rate=0.1)
        losses = nn.fit(X, y, epochs=500, verbose=False)
        assert losses[-1] < losses[0]

    def test_proba_range(self, trained_nn, xor_data):
        X, _ = xor_data
        proba = trained_nn.predict_proba(X)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_different_activations(self, xor_data):
        X, y = xor_data
        for act in ["relu", "sigmoid", "tanh"]:
            nn = NeuralNetwork([2, 4, 1], activation=act, learning_rate=0.1)
            nn.fit(X, y, epochs=500, verbose=False)
            acc = nn.accuracy(X, y)
            assert acc > 0.80, f"{act} failed: {acc:.2%}"
