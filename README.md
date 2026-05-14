# Neural Network from Scratch

A **pure NumPy** feedforward neural network — no TensorFlow, no PyTorch. Implements forward pass, backpropagation, and gradient descent by hand.

## Overview

- Configurable architecture: `[input, hidden, output]`
- Three activation functions: **ReLU**, **Sigmoid**, **Tanh**
- Binary cross-entropy loss with manual gradient computation
- Three synthetic datasets: **XOR**, **Circle**, **Spiral**
- Interactive **Streamlit dashboard** with Plotly decision boundary visualization

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main train --dataset xor --hidden 8 --epochs 2000
pytest tests/ -v
```

## How It Works

1. **Forward pass**: `A = activation(X @ W + b)` through each layer, sigmoid on output
2. **Loss**: Binary cross-entropy `-mean(y*log(ŷ) + (1-y)*log(1-ŷ))`
3. **Backward pass**: Chain rule from output → hidden → input
4. **Update**: `W -= lr * dW`, `b -= lr * db`

## Docker

```bash
docker compose up --build
```

## License

MIT
# Neural-Network-from-Scratch
