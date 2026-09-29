"""
loss_surfaces.py
=================
The elongated-bowl loss surfaces used in Part A:

    L(x, y) = x^2 + c * y^2 ,   grad = [2x, 2c*y]

c controls the condition number of the Hessian: kappa = max(2, 2c)/min(2, 2c) = c
(for c >= 1). Larger c => a narrower, more elongated bowl => worse zig-zagging
for plain SGD.
"""

import numpy as np

SURFACES = {
    "L1: x^2 + 10y^2   (kappa=10)": 10.0,
    "L2: x^2 + 50y^2   (kappa=50, default)": 50.0,
    "L3: x^2 + 100y^2  (kappa=100)": 100.0,
    "L4: x^2 + 1000y^2 (kappa=1000)": 1000.0,
}

DEFAULT_SURFACE = "L2: x^2 + 50y^2   (kappa=50, default)"


def loss_value(x, y, c):
    return x ** 2 + c * y ** 2


def make_grad_fn(c):
    """Return grad_fn(params) -> grads for the bowl with curvature c."""

    def grad_fn(params):
        x, y = params["x"], params["y"]
        return {"x": 2.0 * x, "y": 2.0 * c * y}

    return grad_fn


def make_loss_fn(c):
    def loss_fn(params):
        return float(loss_value(params["x"], params["y"], c))

    return loss_fn


def contour_grid(c, half_range=9.0, n=120):
    xs = np.linspace(-half_range, half_range, n)
    ys = np.linspace(-half_range, half_range, n)
    X, Y = np.meshgrid(xs, ys)
    Z = loss_value(X, Y, c)
    return X, Y, Z
