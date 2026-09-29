"""
optimizers.py
=============
From-scratch implementations of seven optimization algorithms:
SGD, SGD+Momentum, NAG, AdaGrad, RMSProp, Adam, AdamW.

Design notes
------------
Every optimizer works on a *dict* of NumPy arrays, e.g.
    params = {"x": np.array(8.0), "y": np.array(8.0)}
or, for the neural network in Part B,
    params = {"W1": ..., "b1": ..., "W2": ..., "b2": ..., "W3": ..., "b3": ...}

Each optimizer exposes:

    step(params, grad_fn) -> new_params

`grad_fn` is a callable  grad_fn(params) -> grads (a dict with the same keys
as params). Passing a *function* rather than a pre-computed gradient is what
lets NAG evaluate the gradient at the look-ahead point
(theta - beta * v_prev) instead of at theta itself, while every other
optimizer simply calls grad_fn(params) at the current point. This keeps a
single, uniform, stateful interface across every optimizer for both the 2D
toy surface (Part A) and the real backprop-driven MLP (Part B), while still
satisfying the "self-contained class with .step() that retains internal
state between calls" requirement from the lab brief.

No autograd, no torch.optim, no keras.optimizers -- pure NumPy arithmetic.
"""

import numpy as np


class BaseOptimizer:
    """Common bookkeeping shared by every optimizer."""

    name = "Base"
    color = "#000000"

    def __init__(self, lr=0.01):
        self.lr = lr
        self.t = 0  # timestep, used by Adam/AdamW for bias correction

    def reset(self):
        """Clear all internal state (velocities, moving averages, timestep)."""
        self.t = 0

    def step(self, params: dict, grad_fn):
        raise NotImplementedError

    @staticmethod
    def _zeros_like(params):
        return {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}


class SGD(BaseOptimizer):
    """theta_{t+1} = theta_t - eta * g_t"""

    name = "SGD"
    color = "#e74c3c"

    def step(self, params, grad_fn):
        grads = grad_fn(params)
        self.t += 1
        new_params = {}
        for k in params:
            new_params[k] = params[k] - self.lr * grads[k]
        return new_params


class Momentum(BaseOptimizer):
    """v_t = beta*v_{t-1} + (1-beta)*g_t ; theta_{t+1} = theta_t - eta*v_t"""

    name = "Momentum"
    color = "#f39c12"

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def reset(self):
        super().reset()
        self.v = None

    def step(self, params, grad_fn):
        grads = grad_fn(params)
        if self.v is None:
            self.v = self._zeros_like(params)
        self.t += 1
        new_params = {}
        for k in params:
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * grads[k]
            new_params[k] = params[k] - self.lr * self.v[k]
        return new_params


class NAG(BaseOptimizer):
    """Nesterov Accelerated Gradient: gradient evaluated at the look-ahead
    point theta_t - beta * v_{t-1}, then a standard momentum update."""

    name = "NAG"
    color = "#9b59b6"

    def __init__(self, lr=0.01, beta=0.9):
        super().__init__(lr)
        self.beta = beta
        self.v = None

    def reset(self):
        super().reset()
        self.v = None

    def step(self, params, grad_fn):
        if self.v is None:
            self.v = self._zeros_like(params)
        # Look-ahead point: extrapolate to where the momentum term alone
        # would carry the parameters next (theta - eta*beta*v_{t-1}).
        # v is an EMA of the raw gradient (same units as g_t, per the
        # momentum rule v_t = beta*v_{t-1}+(1-beta)*g_t used in this lab),
        # and the actual parameter step is eta*v -- so the look-ahead
        # offset must also carry the eta scale factor, or it overshoots by
        # 1/eta and diverges on any curved surface.
        lookahead = {k: params[k] - self.lr * self.beta * self.v[k] for k in params}
        grads = grad_fn(lookahead)
        self.t += 1
        new_params = {}
        for k in params:
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * grads[k]
            new_params[k] = params[k] - self.lr * self.v[k]
        return new_params


class AdaGrad(BaseOptimizer):
    """G_t = G_{t-1} + g_t^2 ; theta_{t+1} = theta_t - eta*g_t/sqrt(G_t+eps)"""

    name = "AdaGrad"
    color = "#1abc9c"

    def __init__(self, lr=0.01, eps=1e-8):
        super().__init__(lr)
        self.eps = eps
        self.G = None

    def reset(self):
        super().reset()
        self.G = None

    def step(self, params, grad_fn):
        grads = grad_fn(params)
        if self.G is None:
            self.G = self._zeros_like(params)
        self.t += 1
        new_params = {}
        for k in params:
            self.G[k] = self.G[k] + grads[k] ** 2
            new_params[k] = params[k] - self.lr * grads[k] / (np.sqrt(self.G[k]) + self.eps)
        return new_params

    def effective_lr(self, key):
        """eta / sqrt(G_t + eps) for a representative parameter (for B2)."""
        if self.G is None:
            return self.lr
        return float(np.mean(self.lr / (np.sqrt(self.G[key]) + self.eps)))


class RMSProp(BaseOptimizer):
    """v_t = beta*v_{t-1} + (1-beta)*g_t^2 ; theta -= eta*g_t/sqrt(v_t+eps)"""

    name = "RMSProp"
    color = "#3498db"

    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.v = None

    def reset(self):
        super().reset()
        self.v = None

    def step(self, params, grad_fn):
        grads = grad_fn(params)
        if self.v is None:
            self.v = self._zeros_like(params)
        self.t += 1
        new_params = {}
        for k in params:
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * grads[k] ** 2
            new_params[k] = params[k] - self.lr * grads[k] / (np.sqrt(self.v[k]) + self.eps)
        return new_params

    def effective_lr(self, key):
        if self.v is None:
            return self.lr
        return float(np.mean(self.lr / (np.sqrt(self.v[key]) + self.eps)))


class Adam(BaseOptimizer):
    """Adam: bias-corrected first and second moment estimates."""

    name = "Adam"
    color = "#2ecc71"

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def reset(self):
        super().reset()
        self.m = None
        self.v = None

    def step(self, params, grad_fn):
        grads = grad_fn(params)
        if self.m is None:
            self.m = self._zeros_like(params)
            self.v = self._zeros_like(params)
        self.t += 1
        new_params = {}
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            new_params[k] = params[k] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return new_params

    def effective_lr(self, key):
        if self.v is None or self.t == 0:
            return self.lr
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        return float(np.mean(self.lr / (np.sqrt(v_hat) + self.eps)))


class AdamW(BaseOptimizer):
    """AdamW: Adam update plus DECOUPLED weight decay (not folded into g_t)."""

    name = "AdamW"
    color = "#34495e"

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-3):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = None
        self.v = None

    def reset(self):
        super().reset()
        self.m = None
        self.v = None

    def step(self, params, grad_fn):
        grads = grad_fn(params)
        if self.m is None:
            self.m = self._zeros_like(params)
            self.v = self._zeros_like(params)
        self.t += 1
        new_params = {}
        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            new_params[k] = params[k] - self.lr * (
                m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * params[k]
            )
        return new_params

    def effective_lr(self, key):
        if self.v is None or self.t == 0:
            return self.lr
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        return float(np.mean(self.lr / (np.sqrt(v_hat) + self.eps)))


# Registry used by the app to build fresh optimizer instances from UI state.
OPTIMIZER_CLASSES = {
    "SGD": SGD,
    "Momentum": Momentum,
    "NAG": NAG,
    "AdaGrad": AdaGrad,
    "RMSProp": RMSProp,
    "Adam": Adam,
    "AdamW": AdamW,
}

OPTIMIZER_COLORS = {name: cls.color for name, cls in OPTIMIZER_CLASSES.items()}


def make_optimizer(name, lr, beta=0.9, beta1=0.9, beta2=0.999, weight_decay=1e-3):
    """Factory: build a fresh, zero-state optimizer instance from UI params."""
    if name == "SGD":
        return SGD(lr=lr)
    if name == "Momentum":
        return Momentum(lr=lr, beta=beta)
    if name == "NAG":
        return NAG(lr=lr, beta=beta)
    if name == "AdaGrad":
        return AdaGrad(lr=lr)
    if name == "RMSProp":
        return RMSProp(lr=lr, beta=beta)
    if name == "Adam":
        return Adam(lr=lr, beta1=beta1, beta2=beta2)
    if name == "AdamW":
        return AdamW(lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")
