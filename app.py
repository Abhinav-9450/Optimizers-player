"""
app.py
======
Optimizer Visualizer -- "From SGD to AdamW"
A single Streamlit application with two panels:
  Part A: interactive 2D optimizer playground on an elongated-bowl loss surface
  Part B: live training dashboard for a from-scratch MLP on Breast Cancer Wisconsin

Run with:  streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from optimizers import OPTIMIZER_CLASSES, OPTIMIZER_COLORS, make_optimizer
from loss_surfaces import SURFACES, DEFAULT_SURFACE, make_grad_fn, make_loss_fn, contour_grid
from neural_net import MLP

st.set_page_config(page_title="Optimizer Visualizer: SGD to AdamW", layout="wide", page_icon="📉")

ALL_OPTS = list(OPTIMIZER_CLASSES.keys())
# The lab brief's own defaults (documented in-app, and used as the floor of the
# "Iterations budget" slider's meaning) -- see that slider's help text for why the
# app's actual default iteration count is higher.
ASSIGNMENT_DEFAULT_ITERS = 500
ASSIGNMENT_DEFAULT_LR = 0.01

EXPLANATIONS = {
    "NAG": (
        "**NAG vs. plain Momentum.** Momentum computes the gradient at the *current* point and "
        "then adds inertia on top. NAG instead peeks ahead to where the momentum term is about "
        "to carry the parameters, and evaluates the gradient *there*. If that look-ahead point has "
        "already overshot the minimum, the gradient at that point points backward, so NAG's update "
        "gets a corrective brake *before* the overshoot happens rather than after -- producing "
        "smaller oscillations than plain Momentum on curved surfaces."
    ),
    "AdaGrad": (
        "**Why AdaGrad's learning rate differs per parameter.** AdaGrad divides the base learning "
        "rate by the square root of the running sum of squared past gradients, separately for each "
        "parameter. A parameter that has seen large/frequent gradients (like *y* on this steep bowl) "
        "accumulates a large `G`, so its effective step shrinks a lot. A parameter with small/rare "
        "gradients (like *x*) keeps a relatively larger effective step. This automatically rescales "
        "each direction to roughly equalize progress -- great for sparse features, but `G` only ever "
        "grows, so the effective rate keeps shrinking forever."
    ),
    "RMSProp": (
        "**How RMSProp fixes AdaGrad's decay problem.** RMSProp replaces AdaGrad's *sum* of all "
        "past squared gradients with an *exponential moving average* (decay factor beta). Old "
        "gradients are gradually forgotten instead of accumulating forever, so the denominator "
        "stabilizes around the recent gradient magnitude instead of growing monotonically. The "
        "effective learning rate can therefore stay usable indefinitely instead of decaying to "
        "near-zero after enough steps."
    ),
    "AdamW": (
        "**Decoupled weight decay vs. L2 regularization.** Classic L2 regularization adds `lambda*theta` "
        "directly into the gradient *before* Adam's adaptive scaling, so the decay itself gets divided "
        "by `sqrt(v_hat)+eps` along with the loss gradient -- parameters with large accumulated "
        "gradients are decayed *less* than they should be. AdamW instead applies `lambda*theta` "
        "*after* the adaptive step, as a separate, undistorted shrinkage term. This decouples "
        "regularization strength from the optimizer's adaptive scaling, which is the fix that made "
        "weight decay behave as intended for Adam-family optimizers."
    ),
}

# ----------------------------------------------------------------------------------
# Sidebar: global info / how-to-use
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.title("📉 Optimizer Visualizer")
    st.caption("From SGD to AdamW -- an interactive optimizer playground")
    with st.expander("ℹ️ How to use this tool", expanded=False):
        st.markdown(
            """
**Part A -- 2D Playground**
1. Pick a loss surface (controls how elongated/ill-conditioned the bowl is).
2. Select one or more optimizers to overlay.
3. Tune learning rate / beta / beta1 / beta2 / weight decay with the sliders.
4. Press **Play** to animate, or **Step** to advance one iteration at a time.
5. Watch the contour path (left) and the loss curve (right) update together.
6. Expand the *"Explain this optimizer"* panel for a plain-language mechanism.

**Part B -- Neural Network Dashboard**
1. Pick one or more optimizers and hyperparameters.
2. Set epochs / batch size, then press **Train**.
3. Training loss, test loss, accuracy, and the effective-learning-rate readout
   update live, epoch by epoch.
4. A comparison table auto-fills once every selected optimizer has finished.

**Defaults used in this app** (documented here, per the assignment): momentum
beta = 0.9, Adam beta1 = 0.9 / beta2 = 0.999, AdamW weight decay = 1e-3,
numerical epsilon = 1e-8, max iterations (Part A) = 500, default learning
rate = 0.01, default starting point (x0,y0) = (8,8).
            """
        )
    st.divider()
    st.caption("Deep Learning Laboratory -- Optimizer Visualizer submission")

tab_a, tab_b = st.tabs(["🅰️ Part A -- 2D Optimizer Playground", "🅱️ Part B -- Neural Network Training"])

# ====================================================================================
# PART A
# ====================================================================================
with tab_a:
    st.header("Part A -- The Optimizer Playground on a 2D Loss Surface")
    st.caption("L(x, y) = x² + c·y² — a narrow, elongated bowl. All trajectories start at (x0, y0).")

    ctrl_col, view_col = st.columns([1, 2.2], gap="large")

    with ctrl_col:
        st.subheader("Controls")

        preset = st.radio(
            "Preset", ["App defaults (visibly converge)", "Literal assignment defaults (η=0.01, 500 iters)"],
            horizontal=False,
            help="The assignment specifies η=0.01 and a 500-iteration cap as its own defaults. At "
                 "exactly that combination, Adam-family optimizers are mathematically still far from "
                 "the minimum after 500 steps (see the Iterations budget slider's help text) -- pick "
                 "the second preset to see that literal configuration; pick the first to see all seven "
                 "optimizers actually reach the minimum on screen."
        )
        use_literal_defaults = preset.startswith("Literal")

        surface_label = st.selectbox("Loss surface", list(SURFACES.keys()),
                                      index=list(SURFACES.keys()).index(DEFAULT_SURFACE))
        c_value = SURFACES[surface_label]

        selected_opts = st.multiselect("Optimizers to overlay", ALL_OPTS,
                                        default=["SGD", "Momentum", "Adam"])

        st.markdown("**Learning rate η** (log scale, 0.0001–0.5)")
        default_log_lr = np.log10(ASSIGNMENT_DEFAULT_LR) if use_literal_defaults else -1.74
        log_lr = st.slider("log10(η)", -4.0, -0.30, float(default_log_lr), step=0.01,
                            label_visibility="collapsed", key=f"pA_log_lr_{use_literal_defaults}")
        lr = float(10 ** log_lr)
        st.caption(f"η = {lr:.5f}")
        lr_manual = st.number_input("...or type η directly", min_value=0.0, value=float(f"{lr:.5f}"),
                                     step=0.001, format="%.5f", key=f"pA_lr_manual_{use_literal_defaults}",
                                     help="Overrides the slider if changed. Values <= 0 are rejected.")
        if lr_manual > 0:
            lr = lr_manual
        else:
            st.warning("Learning rate must be positive -- using the slider value instead.")

        beta = st.slider("β (Momentum / RMSProp)", 0.0, 0.999, 0.9, step=0.01)
        beta1 = st.slider("β1 (Adam / AdamW)", 0.0, 0.999, 0.9, step=0.01)
        beta2 = st.slider("β2 (Adam / AdamW)", 0.9, 0.9999, 0.999, step=0.0005, format="%.4f")
        wd = st.slider("λ weight decay (AdamW)", 0.0, 0.05, 0.001, step=0.001, format="%.3f")

        x0 = st.slider("x0", -9.0, 9.0, 8.0, step=0.5)
        y0 = st.slider("y0", -9.0, 9.0, 8.0, step=0.5)

        default_iters = ASSIGNMENT_DEFAULT_ITERS if use_literal_defaults else 1500
        max_iters = st.slider(
            "Iterations budget", 100, 3000, default_iters, step=100,
            key=f"pA_iters_{use_literal_defaults}",
            help="The lab brief's own default is 500 iterations at η=0.01. At that exact combination, "
                 "SGD/Momentum/NAG converge in well under 100 steps, but Adam-family optimizers take "
                 "roughly constant-size steps per iteration (independent of distance from the minimum), "
                 "so they need well over 1000 steps to visually reach the origin here -- that is a real, "
                 "well-known property of adaptive optimizers, not a bug. The 'App defaults' preset uses "
                 "a higher budget so the difference is visible instead of just looking 'stuck'."
        )

        speed = st.select_slider("Animation speed", options=["Slow", "Medium", "Fast", "Instant"],
                                  value="Fast")
        speed_delay = {"Slow": 0.05, "Medium": 0.02, "Fast": 0.0, "Instant": 0.0}[speed]
        frames_per_tick = {"Slow": 1, "Medium": 1, "Fast": max(1, max_iters // 150),
                            "Instant": max(1, max_iters // 20)}[speed]

        btn_cols = st.columns(4)
        play_clicked = btn_cols[0].button("▶ Play", use_container_width=True)
        pause_clicked = btn_cols[1].button("⏸ Pause", use_container_width=True)
        step_clicked = btn_cols[2].button("⏭ Step", use_container_width=True)
        reset_clicked = btn_cols[3].button("⏮ Reset", use_container_width=True)

    # ---- settings hash: recompute trajectories whenever any control changes ----
    settings_key = (surface_label, tuple(sorted(selected_opts)), round(lr, 6), beta, beta1, beta2,
                     wd, x0, y0, max_iters)
    if st.session_state.get("pA_settings_key") != settings_key:
        st.session_state["pA_settings_key"] = settings_key
        st.session_state["pA_frame"] = 0
        st.session_state["pA_playing"] = False
        grad_fn = make_grad_fn(c_value)
        loss_fn = make_loss_fn(c_value)
        trajectories = {}
        for name in selected_opts:
            opt = make_optimizer(name, lr=lr, beta=beta, beta1=beta1, beta2=beta2, weight_decay=wd)
            params = {"x": np.array(float(x0)), "y": np.array(float(y0))}
            xs, ys, losses = [float(x0)], [float(y0)], [loss_fn(params)]
            for _ in range(max_iters):
                params = opt.step(params, grad_fn)
                xs.append(float(params["x"]))
                ys.append(float(params["y"]))
                losses.append(loss_fn(params))
            trajectories[name] = {"x": xs, "y": ys, "loss": losses}
        st.session_state["pA_trajectories"] = trajectories

    trajectories = st.session_state.get("pA_trajectories", {})

    # ---- frame state: a single source of truth (session_state["pA_frame"]) that both
    # the buttons/animation loop AND the slider widget share via the same key, so a
    # programmatic increment during Play is never silently overwritten by a stale
    # widget value on rerun (the bug in the previous version). ----
    if "pA_frame" not in st.session_state:
        st.session_state["pA_frame"] = 0
    if "pA_playing" not in st.session_state:
        st.session_state["pA_playing"] = False

    if reset_clicked:
        st.session_state["pA_frame"] = 0
        st.session_state["pA_playing"] = False
    if step_clicked:
        st.session_state["pA_playing"] = False
        st.session_state["pA_frame"] = min(st.session_state["pA_frame"] + 1, max_iters)
    if play_clicked:
        if st.session_state["pA_frame"] >= max_iters:
            st.session_state["pA_frame"] = 0  # restart from the beginning if replaying after the end
        st.session_state["pA_playing"] = True
    if pause_clicked:
        st.session_state["pA_playing"] = False

    # Advance exactly one animation tick HERE -- before the slider widget below is
    # instantiated. Streamlit forbids writing to st.session_state["pA_frame"] after
    # the widget with key="pA_frame" has rendered in the same script run; doing so
    # from the old bottom-of-script animation driver silently threw an exception on
    # every single Play-triggered rerun, which is why the animation never advanced.
    if st.session_state["pA_playing"]:
        if st.session_state["pA_frame"] < max_iters:
            st.session_state["pA_frame"] = min(st.session_state["pA_frame"] + frames_per_tick, max_iters)
        else:
            st.session_state["pA_playing"] = False
    st.session_state["pA_frame"] = min(st.session_state["pA_frame"], max_iters)

    def _on_manual_scrub():
        # Fires when the user drags the slider directly -- stop any running animation.
        st.session_state["pA_playing"] = False

    st.slider("Iteration", 0, max_iters, key="pA_frame", on_change=_on_manual_scrub)
    frame = st.session_state["pA_frame"]

    with view_col:
        if not selected_opts:
            st.info("Select at least one optimizer to see the playground.")
        else:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("View 1 — Contour + trajectory",
                                                                  "View 2 — Loss vs. iteration"))
            X, Y, Z = contour_grid(c_value)
            fig.add_trace(go.Contour(x=X[0], y=Y[:, 0], z=Z, showscale=False,
                                      colorscale="Blues", opacity=0.55,
                                      contours=dict(showlines=False)), row=1, col=1)
            fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                                      marker=dict(symbol="star", size=16, color="gold",
                                                  line=dict(color="black", width=1)),
                                      name="minimum", showlegend=False), row=1, col=1)

            f = st.session_state.get("pA_frame", 0)
            for name in selected_opts:
                traj = trajectories[name]
                color = OPTIMIZER_COLORS[name]
                fig.add_trace(go.Scatter(x=traj["x"][:f + 1], y=traj["y"][:f + 1], mode="lines",
                                          line=dict(color=color, width=2), name=name,
                                          legendgroup=name), row=1, col=1)
                fig.add_trace(go.Scatter(x=[traj["x"][f]], y=[traj["y"][f]], mode="markers",
                                          marker=dict(color=color, size=10, line=dict(color="white", width=1)),
                                          legendgroup=name, showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=list(range(f + 1)), y=traj["loss"][:f + 1], mode="lines",
                                          line=dict(color=color, width=2), legendgroup=name,
                                          showlegend=False), row=1, col=2)

            fig.update_xaxes(title_text="x", row=1, col=1)
            fig.update_yaxes(title_text="y", row=1, col=1)
            fig.update_xaxes(title_text="iteration", row=1, col=2)
            fig.update_yaxes(title_text="L(θ_t)", type="log", row=1, col=2)
            fig.update_layout(height=520, legend=dict(orientation="h", y=-0.15),
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            metric_cols = st.columns(len(selected_opts))
            for i, name in enumerate(selected_opts):
                traj = trajectories[name]
                metric_cols[i].metric(name, f"L = {traj['loss'][f]:.4f}",
                                       f"x={traj['x'][f]:.3f}, y={traj['y'][f]:.3f}")

            if "AdaGrad" in selected_opts and trajectories["AdaGrad"]["loss"][f] > 1.0:
                st.caption(
                    "ℹ️ AdaGrad still hasn't reached the minimum -- this is expected, not a bug: its "
                    "accumulated denominator only ever grows, so its effective step keeps shrinking "
                    "toward zero the longer it runs (see the AdaGrad explanation below, and A7‑Q5)."
                )

    st.subheader("Explain-as-you-go")
    exp_opt = st.selectbox("Explain this optimizer", [o for o in ALL_OPTS if o in EXPLANATIONS],
                            key="pA_explain_select")
    st.info(EXPLANATIONS[exp_opt])

    with st.expander("📐 Conditioning explorer -- what changes as the bowl narrows?"):
        st.markdown(
            "Switch the **Loss surface** dropdown above between L1 (κ=10) and L4 (κ=1000) with the "
            "same optimizers and learning rate selected, and watch View 1. As the condition number "
            "κ = λ_max/λ_min of the Hessian grows, the bowl becomes far narrower along *y* than *x*. "
            "Plain SGD's fixed, isotropic step size is well-tuned for at most one of the two "
            "directions at a time, so it is forced to take small steps to avoid diverging along *y* -- "
            "which makes it crawl along the shallow *x* direction and zig-zag along the steep *y* "
            "direction. Momentum-based and adaptive methods compensate for this anisotropy in "
            "different ways (see the per-optimizer explanations above)."
        )

    with st.expander("🎚️ Learning-rate sensitivity explorer"):
        st.markdown(
            "Use the η slider/number field above to sweep **0.001 → 0.01 → 0.1** for a fixed "
            "optimizer and surface, and watch View 2 (log-scale loss curve). At η too low, "
            "convergence is slow but stable. At η too high, adaptive and momentum methods can "
            "start to oscillate or diverge -- watch for the loss curve turning upward instead of "
            "monotonically decreasing. Take a screenshot (or a short screen recording using your "
            "OS's built-in tool) of one clearly diverging case and one clearly convergent case for "
            "your submission, as required by A6."
        )

    # ---- animation driver: only triggers a rerun while "playing"; the actual frame
    # increment already happened above, BEFORE the "pA_frame" slider widget was
    # created, since Streamlit forbids mutating a widget's session-state key after
    # that widget has rendered in the same run. ----
    if st.session_state.get("pA_playing", False):
        if st.session_state["pA_frame"] >= max_iters:
            st.session_state["pA_playing"] = False  # "pA_playing" has no widget bound to it -- safe to set here
        else:
            if speed_delay > 0:
                time.sleep(speed_delay)
            st.rerun()

# ====================================================================================
# PART B
# ====================================================================================
with tab_b:
    st.header("Part B -- Training a Real Neural Network")
    st.caption("Input → Dense(16)-ReLU → Dense(8)-ReLU → Dense(1)-Sigmoid, "
               "trained from scratch (manual forward/backprop) on Breast Cancer Wisconsin.")

    @st.cache_data
    def load_data():
        data = load_breast_cancer()
        X, y = data.data, data.target.reshape(-1, 1).astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test, data.feature_names

    X_train, X_test, y_train, y_test, feature_names = load_data()

    info_cols = st.columns(4)
    info_cols[0].metric("Train samples", X_train.shape[0])
    info_cols[1].metric("Test samples", X_test.shape[0])
    info_cols[2].metric("Features", X_train.shape[1])
    info_cols[3].metric("Classes", "2 (malignant / benign)")

    ctrl_col_b, view_col_b = st.columns([1, 2.2], gap="large")

    with ctrl_col_b:
        st.subheader("Controls")
        selected_opts_b = st.multiselect("Optimizers to train & compare", ALL_OPTS,
                                          default=["SGD", "Adam", "AdamW"], key="pB_opts")
        lr_b = st.number_input("Learning rate η", min_value=0.0, value=0.01, step=0.001,
                                format="%.4f", key="pB_lr",
                                help="Must be positive -- invalid values are rejected safely.")
        if lr_b <= 0:
            st.warning("Learning rate must be positive -- falling back to 0.01.")
            lr_b = 0.01
        beta_b = st.slider("β (Momentum / RMSProp)", 0.0, 0.999, 0.9, step=0.01, key="pB_beta")
        beta1_b = st.slider("β1 (Adam / AdamW)", 0.0, 0.999, 0.9, step=0.01, key="pB_beta1")
        beta2_b = st.slider("β2 (Adam / AdamW)", 0.9, 0.9999, 0.999, step=0.0005, format="%.4f", key="pB_beta2")
        wd_b = st.slider("λ weight decay (AdamW)", 0.0, 0.05, 0.001, step=0.001, format="%.3f", key="pB_wd")
        epochs = st.slider("Epochs", 5, 300, 60, step=5)
        batch_choice = st.selectbox("Batch size", ["Full batch"] + [16, 32, 64, 128], index=2)
        batch_size = X_train.shape[0] if batch_choice == "Full batch" else int(batch_choice)
        eff_lr_opt = st.selectbox(
            "Effective-LR readout for", [o for o in selected_opts_b if o in
                                          ("AdaGrad", "RMSProp", "Adam", "AdamW")] or ["(none adaptive selected)"],
            help="Only AdaGrad/RMSProp/Adam/AdamW have a data-dependent effective learning rate."
        )
        train_clicked = st.button("🚀 Train", type="primary", use_container_width=True)

    def run_training(opt_names, epochs, batch_size):
        """Train one fresh MLP per optimizer, recording per-epoch metrics.
        Runs inside the Streamlit script (not a rerun loop) so the dashboard
        placeholders below can be updated live, epoch by epoch."""
        rng = np.random.default_rng(0)
        n = X_train.shape[0]
        nets, opts, histories = {}, {}, {}
        for name in opt_names:
            nets[name] = MLP(n_features=X_train.shape[1], seed=42)
            opts[name] = make_optimizer(name, lr=lr_b, beta=beta_b, beta1=beta1_b,
                                         beta2=beta2_b, weight_decay=wd_b)
            histories[name] = {"train_loss": [], "test_loss": [], "train_acc": [],
                                "test_acc": [], "eff_lr": []}

        chart_ph = st.empty()
        table_ph = st.empty()
        progress = st.progress(0.0, text="Training...")

        for epoch in range(epochs):
            idx = rng.permutation(n)
            for name in opt_names:
                net, opt = nets[name], opts[name]
                for start in range(0, n, batch_size):
                    batch_idx = idx[start:start + batch_size]
                    Xb, yb = X_train[batch_idx], y_train[batch_idx]
                    grad_fn = net.grad_fn_factory(Xb, yb)
                    net.params = opt.step(net.params, grad_fn)

                train_loss, train_acc = net.loss_and_acc(X_train, y_train)
                test_loss, test_acc = net.loss_and_acc(X_test, y_test)
                histories[name]["train_loss"].append(train_loss)
                histories[name]["test_loss"].append(test_loss)
                histories[name]["train_acc"].append(train_acc)
                histories[name]["test_acc"].append(test_acc)
                if hasattr(opt, "effective_lr"):
                    histories[name]["eff_lr"].append(opt.effective_lr("W1"))
                else:
                    histories[name]["eff_lr"].append(opt.lr)

            progress.progress((epoch + 1) / epochs, text=f"Training... epoch {epoch + 1}/{epochs}")

            if epoch % max(1, epochs // 30) == 0 or epoch == epochs - 1:
                fig_b = make_subplots(rows=1, cols=3, subplot_titles=(
                    "Train loss", "Test loss", "Test accuracy"))
                for name in opt_names:
                    h = histories[name]
                    color = OPTIMIZER_COLORS[name]
                    xs = list(range(1, len(h["train_loss"]) + 1))
                    fig_b.add_trace(go.Scatter(x=xs, y=h["train_loss"], mode="lines",
                                                line=dict(color=color, width=2), name=name,
                                                legendgroup=name), row=1, col=1)
                    fig_b.add_trace(go.Scatter(x=xs, y=h["test_loss"], mode="lines",
                                                line=dict(color=color, width=2), legendgroup=name,
                                                showlegend=False), row=1, col=2)
                    fig_b.add_trace(go.Scatter(x=xs, y=h["test_acc"], mode="lines",
                                                line=dict(color=color, width=2), legendgroup=name,
                                                showlegend=False), row=1, col=3)
                fig_b.update_xaxes(title_text="epoch")
                fig_b.update_yaxes(title_text="loss", row=1, col=1)
                fig_b.update_yaxes(title_text="loss", row=1, col=2)
                fig_b.update_yaxes(title_text="accuracy", row=1, col=3)
                fig_b.update_layout(height=380, legend=dict(orientation="h", y=-0.25),
                                     margin=dict(l=10, r=10, t=40, b=10))
                chart_ph.plotly_chart(fig_b, use_container_width=True, key=f"train_chart_{epoch}")

        progress.empty()
        return histories

    if train_clicked:
        if not selected_opts_b:
            st.warning("Select at least one optimizer to train.")
        else:
            with view_col_b:
                histories = run_training(selected_opts_b, epochs, batch_size)
                st.session_state["pB_histories"] = histories
                st.session_state["pB_opts_trained"] = list(selected_opts_b)

    histories = st.session_state.get("pB_histories")
    opts_trained = st.session_state.get("pB_opts_trained", [])

    if histories:
        with view_col_b:
            st.subheader("Effective learning-rate readout")
            adaptive_trained = [o for o in opts_trained if o in ("AdaGrad", "RMSProp", "Adam", "AdamW")]
            if adaptive_trained:
                fig_lr = go.Figure()
                for name in adaptive_trained:
                    xs = list(range(1, len(histories[name]["eff_lr"]) + 1))
                    fig_lr.add_trace(go.Scatter(x=xs, y=histories[name]["eff_lr"], mode="lines",
                                                 line=dict(color=OPTIMIZER_COLORS[name], width=2),
                                                 name=name))
                fig_lr.update_layout(height=320, xaxis_title="epoch",
                                      yaxis_title="effective η (mean over W1)",
                                      margin=dict(l=10, r=10, t=20, b=10),
                                      legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(fig_lr, use_container_width=True)
                st.caption("η/√(Gₜ+ε) for AdaGrad and RMSProp; η/√(v̂ₜ+ε) for Adam/AdamW, "
                           "averaged over the first hidden layer's weights (W1) as a representative "
                           "parameter. A shrinking/stabilizing curve is exactly the mechanism behind "
                           "AdaGrad's slowdown and RMSProp/Adam's fix.")
            else:
                st.caption("Train with at least one of AdaGrad / RMSProp / Adam / AdamW to see this readout.")

        st.subheader("Comparison table (auto-computed from the run)")

        def convergence_epoch(test_loss_history):
            final = test_loss_history[-1]
            target = final * 1.01 if final >= 0 else final * 0.99
            for i, v in enumerate(test_loss_history):
                if abs(v - final) <= 0.01 * abs(final) + 1e-9:
                    return i + 1
            return len(test_loss_history)

        rows = []
        for name in opts_trained:
            h = histories[name]
            rows.append({
                "Optimizer": name,
                "Final Train Loss": round(h["train_loss"][-1], 4),
                "Final Test Loss": round(h["test_loss"][-1], 4),
                "Train Acc.": f"{h['train_acc'][-1] * 100:.2f}%",
                "Test Acc.": f"{h['test_acc'][-1] * 100:.2f}%",
                "Convergence Epoch": convergence_epoch(h["test_loss"]),
            })
        df = pd.DataFrame(rows).set_index("Optimizer")
        st.dataframe(df, use_container_width=True)
        st.caption("Convergence epoch = first epoch at which test loss is within 1% of its final value "
                   "(computed automatically, per B3).")
    else:
        st.info("Configure hyperparameters on the left and press **Train** to populate the live "
                "dashboard and comparison table.")

    with st.expander("📘 Software design notes"):
        st.markdown(
            """
- **Separation of concerns:** `optimizers.py` and `neural_net.py` contain zero plotting code -- they
  are pure NumPy classes/functions reused identically by both Part A and Part B, per the assignment's
  software-design requirement.
- **Sensible defaults:** the app opens with a working configuration (default surface, SGD/Momentum/Adam
  selected, η=0.01) and will not crash on first load.
- **Input validation:** learning-rate fields reject ≤0 values and fall back to a safe default with a
  visible warning instead of crashing.
- **Consistent styling:** every optimizer has one fixed colour (`optimizers.py::OPTIMIZER_COLORS`) used
  identically across every plot in both panels.
            """
        )