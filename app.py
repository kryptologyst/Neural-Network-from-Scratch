import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data import generate_xor_data, generate_circle_data, generate_spiral_data
from src.model import NeuralNetwork

st.set_page_config(page_title="NN from Scratch", page_icon="", layout="wide")
st.title("Neural Network from Scratch")
st.markdown("Pure NumPy implementation — no TensorFlow, no PyTorch. Forward pass, backprop, gradient descent.")

DATASETS = {"XOR": generate_xor_data, "Circle": generate_circle_data, "Spiral": generate_spiral_data}

col1, col2, col3 = st.columns(3)
with col1:
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
with col2:
    hidden = st.slider("Hidden Neurons", 2, 32, 8)
    activation = st.selectbox("Activation", ["relu", "sigmoid", "tanh"])
with col3:
    lr = st.select_slider(
        "Learning Rate",
        options=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        value=0.1,
    )
    epochs = st.slider("Epochs", 100, 5000, 1000, 100)

if st.button("Train Network", type="primary"):
    X, y = DATASETS[dataset_name]()
    nn = NeuralNetwork(
        layer_sizes=[2, hidden, 1],
        activation=activation,
        learning_rate=lr,
    )
    with st.spinner(f"Training for {epochs} epochs..."):
        losses = nn.fit(X, y, epochs=epochs, verbose=False)
    acc = nn.accuracy(X, y)
    st.success(f"Accuracy: **{acc:.2%}** | Final Loss: **{losses[-1]:.6f}**")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Decision Boundary")
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 150),
            np.linspace(y_min, y_max, 150),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = nn.predict(grid).reshape(xx.shape)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Contour(
            z=Z, x=np.linspace(x_min, x_max, 150),
            y=np.linspace(y_min, y_max, 150),
            colorscale="RdBu", opacity=0.4,
            showscale=False,
        ))
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1],
            mode="markers",
            marker=dict(color=y, colorscale="RdBu", size=8, line=dict(width=1, color="black")),
            showlegend=False,
        ))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Loss Curve")
        st.line_chart({"Loss": losses}, height=400)
