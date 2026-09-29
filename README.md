# Optimizer Visualizer — From SGD to AdamW

An interactive Streamlit application built for the *Deep Learning Laboratory: From SGD to AdamW*
assignment. Seven optimizers (SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW) are implemented
from scratch in NumPy and used identically in both panels:

- **Part A** — a live playground on the elongated bowl `L(x,y) = x² + c·y²`, with Play/Pause/Step/Reset
  animation, a synchronized contour view + loss curve, and per-optimizer plain-language explanations.
- **Part B** — a live training dashboard for a from-scratch MLP (`16→8→1`, ReLU/Sigmoid, manual
  backprop) on the Breast Cancer Wisconsin dataset, with an auto-computed comparison table.

## Project structure

```
optimizer_viz/
├── app.py             # Streamlit UI — Part A and Part B (entry point)
├── optimizers.py       # 7 from-scratch optimizer classes (pure NumPy, no plotting)
├── loss_surfaces.py    # The 4 elongated-bowl loss surfaces (L1–L4) for Part A
├── neural_net.py        # From-scratch MLP: forward, BCE loss, manual backprop
├── requirements.txt
└── README.md
```

No `torch.optim`, no `tf.keras.optimizers` / `keras.optimizers`, and no autograd are used anywhere —
every update rule and every gradient in `neural_net.py::backward()` is derived and coded by hand.
A finite-difference gradient check is included at the bottom of `neural_net.py`
(`python neural_net.py`) and matches the analytic backprop to ~1e-9 relative error on every weight.

## Running locally

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) — open it in your browser.

## Deploying it (Streamlit Community Cloud — free, easiest option)

1. **Push this folder to a GitHub repository.**
   ```bash
   git init
   git add .
   git commit -m "Optimizer visualizer: SGD to AdamW"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   Make sure `app.py` and `requirements.txt` are both at the repo root (or note their sub-path).

2. **Go to** [share.streamlit.io](https://share.streamlit.io) **and sign in with GitHub.**

3. Click **"New app"**, then:
   - Repository: `<your-username>/<repo-name>`
   - Branch: `main`
   - Main file path: `app.py`

4. Click **"Deploy"**. The first build takes a few minutes while it installs `requirements.txt`.

5. You'll get a public URL like `https://<repo-name>-<random>.streamlit.app` — this is what you
   submit/share with your professor. Any push to `main` auto-redeploys the app.

### Alternative: Hugging Face Spaces
If GitHub + Streamlit Cloud isn't available: create a new Space at
[huggingface.co/new-space](https://huggingface.co/new-space), choose the **Streamlit** SDK, and
push this same folder (with `app.py` at the root) to the Space's git repo the same way as above.

### Alternative: run it locally and record a screen capture
If you don't need a public deployment, `streamlit run app.py` on your own machine is sufficient —
the assignment only requires a short screen-recording or annotated screenshots demonstrating the
app (submission item #4), not necessarily a public URL.

## Design notes (mapped to the rubric)

| Requirement | Where it's satisfied |
|---|---|
| 7 optimizers, correct math | `optimizers.py`, verified against analytic convergence in Part A |
| NN forward/backprop, no autograd | `neural_net.py`, verified with a numeric gradient check |
| Interactivity (sliders, Play/Pause/Step, live training) | `app.py`, session-state-driven animation loop (Part A) and epoch-by-epoch live chart updates (Part B) |
| Comparison table (B3), auto-computed | `app.py`, `pandas.DataFrame` built from recorded per-epoch histories, no manual entry |
| Separation of optimizer logic vs. UI | optimizer/NN classes contain zero plotting or Streamlit calls |
| Input validation | learning-rate fields reject ≤0 values and fall back to a safe default with a warning instead of crashing |
| Consistent colour-per-optimizer | `optimizers.py::OPTIMIZER_COLORS`, reused in every chart in both panels |

## A note on the NAG implementation

The lab brief states the look-ahead point as `θₜ − βvₜ₋₁`. Implemented literally with the
`vₜ = βvₜ₋₁ + (1−β)gₜ` moving-average form of momentum (where `v` is on the same numerical scale
as the raw gradient, not the parameter-update step), that formula overshoots and diverges on a
curved surface, because the look-ahead offset isn't scaled by the learning rate the way the actual
parameter step `θₜ₊₁ = θₜ − ηvₜ` is. This app instead uses the learning-rate-consistent look-ahead
`θₜ − η·β·vₜ₋₁` — i.e., it extrapolates to where the momentum term alone would carry the parameters
on the *next actual step*, which is the standard interpretation of "look ahead" in Nesterov's method
and is what keeps NAG numerically stable while still evaluating the gradient at a genuine look-ahead
point rather than at `θₜ`. This is worth mentioning to your professor if asked about the exact
formula used.
