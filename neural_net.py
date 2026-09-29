"""
neural_net.py
=============
A small Multi-Layer Perceptron implemented entirely with NumPy:
Input -> Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid

Forward pass, binary cross-entropy loss, and full backpropagation are all
hand-derived and hand-coded here -- no autograd, no framework layers.
Parameters are exposed as a flat dict of NumPy arrays so the same
from-scratch optimizer classes in optimizers.py can update them.
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(z.dtype)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class MLP:
    """Input -> Dense(16)-ReLU -> Dense(8)-ReLU -> Dense(1)-Sigmoid."""

    def __init__(self, n_features, hidden1=16, hidden2=8, seed=42):
        rng = np.random.default_rng(seed)
        # He-ish initialization for ReLU layers, Xavier-ish for output.
        self.params = {
            "W1": rng.normal(0, np.sqrt(2.0 / n_features), size=(n_features, hidden1)),
            "b1": np.zeros(hidden1),
            "W2": rng.normal(0, np.sqrt(2.0 / hidden1), size=(hidden1, hidden2)),
            "b2": np.zeros(hidden2),
            "W3": rng.normal(0, np.sqrt(1.0 / hidden2), size=(hidden2, 1)),
            "b3": np.zeros(1),
        }
        self.n_features = n_features
        self.hidden1 = hidden1
        self.hidden2 = hidden2

    # ---- forward -------------------------------------------------
    def forward(self, X, params=None):
        """Returns (y_hat, cache) where cache holds intermediate values
        needed by backward(). X: (n_samples, n_features)."""
        p = params if params is not None else self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = relu(z1)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = relu(z2)
        z3 = a2 @ p["W3"] + p["b3"]
        y_hat = sigmoid(z3)
        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "y_hat": y_hat}
        return y_hat, cache

    # ---- loss ------------------------------------------------------
    @staticmethod
    def bce_loss(y_hat, y_true):
        """Binary cross-entropy, y_true/y_hat shape (n, 1)."""
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat)))

    @staticmethod
    def accuracy(y_hat, y_true):
        preds = (y_hat >= 0.5).astype(int)
        return float(np.mean(preds == y_true))

    # ---- backward ----------------------------------------------------
    def backward(self, y_true, cache, params=None):
        """Full manual backprop. Returns grads dict matching self.params keys."""
        p = params if params is not None else self.params
        n = y_true.shape[0]
        X, z1, a1, z2, a2, z3, y_hat = (
            cache["X"], cache["z1"], cache["a1"], cache["z2"], cache["a2"], cache["z3"], cache["y_hat"]
        )

        # dL/dz3 for BCE-with-sigmoid simplifies to (y_hat - y_true)/n
        dz3 = (y_hat - y_true) / n                      # (n,1)
        dW3 = a2.T @ dz3                                  # (h2,1)
        db3 = dz3.sum(axis=0)                             # (1,)

        da2 = dz3 @ p["W3"].T                             # (n,h2)
        dz2 = da2 * relu_grad(z2)                          # (n,h2)
        dW2 = a1.T @ dz2                                   # (h1,h2)
        db2 = dz2.sum(axis=0)                              # (h2,)

        da1 = dz2 @ p["W2"].T                              # (n,h1)
        dz1 = da1 * relu_grad(z1)                           # (n,h1)
        dW1 = X.T @ dz1                                     # (nf,h1)
        db1 = dz1.sum(axis=0)                                # (h1,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def grad_fn_factory(self, X, y_true):
        """Return a grad_fn(params) -> grads closure for use with the
        optimizer .step(params, grad_fn) interface (needed so NAG can
        evaluate the gradient at a look-ahead set of weights)."""

        def grad_fn(params):
            y_hat, cache = self.forward(X, params=params)
            return self.backward(y_true, cache, params=params)

        return grad_fn

    def loss_and_acc(self, X, y_true, params=None):
        y_hat, _ = self.forward(X, params=params)
        return self.bce_loss(y_hat, y_true), self.accuracy(y_hat, y_true)


def gradient_check(n_features=5, n_samples=8, eps=1e-5, seed=0):
    """Numerical gradient check -- used only for internal verification,
    not part of the app UI. Compares backward() against finite differences."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, 2, size=(n_samples, 1)).astype(float)
    net = MLP(n_features, hidden1=4, hidden2=3, seed=seed)

    y_hat, cache = net.forward(X)
    analytic = net.backward(y, cache)

    max_rel_err = 0.0
    for key in net.params:
        p = net.params[key]
        it = np.nditer(p, flags=["multi_index"])
        # Sample a handful of entries per param for speed.
        idxs = list(np.ndindex(p.shape))
        rng.shuffle(idxs)
        for idx in idxs[: min(5, len(idxs))]:
            orig = p[idx]
            p[idx] = orig + eps
            y_hat_p, _ = net.forward(X)
            loss_p = net.bce_loss(y_hat_p, y)
            p[idx] = orig - eps
            y_hat_m, _ = net.forward(X)
            loss_m = net.bce_loss(y_hat_m, y)
            p[idx] = orig
            numeric = (loss_p - loss_m) / (2 * eps)
            ana = analytic[key][idx]
            rel_err = abs(numeric - ana) / (abs(numeric) + abs(ana) + 1e-8)
            max_rel_err = max(max_rel_err, rel_err)
    return max_rel_err


if __name__ == "__main__":
    err = gradient_check()
    print(f"Max relative gradient-check error: {err:.2e} (should be well under 1e-4)")
