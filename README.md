# Optimizer Visualizer — From SGD to AdamW

An interactive Streamlit application demonstrating seven optimizers (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW) implemented from scratch in NumPy.

- **Part A** — Live playground on the elongated bowl `L(x,y) = x² + c·y²` with animation, contour view, and per-optimizer explanations.
- **Part B** — Live MLP training dashboard on the Breast Cancer Wisconsin dataset with an auto-computed comparison table.

## Highlights

- All optimizers built from scratch — no `torch.optim` or `tf.keras`
- Manual backpropagation with finite-difference gradient check (~1e-9 accuracy)
- Interactive controls: Play/Pause/Step/Reset animation
- Live training visualization with real-time loss curves

## Tech Stack

- **Python** + **NumPy** (optimizers & neural network)
- **Streamlit** (UI)
- **Matplotlib** (visualization)

## Live Demo

Deployed on Streamlit Cloud: [https://optimizers-player.streamlit.app](https://optimizers-player.streamlit.app)
