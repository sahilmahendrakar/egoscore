"""The offline proxy metric: Avg-MSE of a small behaviour-cloning model.

We adopt the EgoVerse paper's own evaluation choice:

    "we report primarily the offline Avg-MSE metric in the human-based evaluation
     setting. While this metric does not directly measure downstream robot
     performance, it provides a stable signal for comparing generalization"

It is a proxy. We report it as one, never as success rate. Its job is to *rank
training subsets* against each other under otherwise identical conditions.

The model predicts a 30-step action chunk of bimanual end-effector poses from a short
window of proprioceptive history. No vision: pulling images for this slice costs 84.6 GB
against 2.3 GB for the pose arrays, and a curation engine that is more expensive than
the training it saves is not worth running. Every condition sees the identical
architecture, optimiser, and step count — only the training subset changes.
"""

from __future__ import annotations

import numpy as np

HISTORY = 10        # frames of proprioceptive context
HORIZON = 30        # action chunk length, matching Human.ACTION_HORIZON in egomimic
STRIDE = 15         # frames between sampled windows
POSE_KEYS = ["left__obs_ee_pose", "right__obs_ee_pose", "obs_head_pose"]


def episode_windows(ep: dict[str, np.ndarray], stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Turn one episode into (X, Y) windows.

    X: (N, HISTORY * D) flattened proprioceptive history
    Y: (N, HORIZON * 14) future bimanual EE poses, relative to the current pose
    """
    parts = [ep[k] for k in POSE_KEYS if k in ep]
    if len(parts) < 3:
        return np.zeros((0, 0)), np.zeros((0, 0))
    obs = np.nan_to_num(np.concatenate(parts, axis=1).astype(np.float32))

    act = np.nan_to_num(
        np.concatenate([ep["left__obs_ee_pose"], ep["right__obs_ee_pose"]], axis=1).astype(np.float32)
    )

    T = len(obs)
    if T < HISTORY + HORIZON + 1:
        return np.zeros((0, 0)), np.zeros((0, 0))

    starts = np.arange(HISTORY, T - HORIZON, stride)
    X = np.stack([obs[s - HISTORY:s].ravel() for s in starts])
    # Predict the future *relative to the current pose*: absolute world position is
    # scene-specific and would let the model win by memorising which room it is in.
    Y = np.stack([(act[s:s + HORIZON] - act[s - 1]).ravel() for s in starts])
    return X, Y


class RidgeChunkPolicy:
    """Closed-form ridge regression from proprioceptive history to action chunk.

    Deliberately not a deep net. With a fixed feature map the solution is exact and
    seed-independent, so any difference between conditions comes from the *data*, not
    from optimiser noise — which is the entire point of the comparison. It also trains
    in under a second, which is what lets us afford every condition x seed cell.
    """

    def __init__(self, alpha: float = 1.0, n_features: int = 512, seed: int = 0):
        self.alpha = alpha
        self.n_features = n_features
        self.seed = seed
        self.W = None
        self.mu = None
        self.sd = None
        self.proj = None
        self.b = None

    def _phi(self, X: np.ndarray) -> np.ndarray:
        """Random Fourier features — a fixed, seed-determined nonlinearity."""
        Z = (X - self.mu) / self.sd
        if self.proj is None:
            rng = np.random.default_rng(self.seed)
            self.proj = rng.normal(0, 1.0 / np.sqrt(Z.shape[1]), (Z.shape[1], self.n_features)).astype(np.float32)
            self.b = rng.uniform(0, 2 * np.pi, self.n_features).astype(np.float32)
        H = np.cos(Z @ self.proj + self.b)
        return np.hstack([H, np.ones((len(H), 1), dtype=np.float32)])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeChunkPolicy":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-6
        H = self._phi(X)
        A = H.T @ H + self.alpha * np.eye(H.shape[1], dtype=np.float32)
        self.W = np.linalg.solve(A, H.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._phi(X) @ self.W


def avg_mse(model: RidgeChunkPolicy, X: np.ndarray, Y: np.ndarray) -> float:
    if len(X) == 0:
        return float("nan")
    return float(np.mean((model.predict(X) - Y) ** 2))
