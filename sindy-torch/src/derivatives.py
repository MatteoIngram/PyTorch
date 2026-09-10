"""Time-derivative estimation from trajectory data.

SINDy needs Xdot, and in any real application you only observe X sampled at
discrete times -- not the analytic derivative. Estimating Xdot from data is
the single most numerically delicate step in the whole method: differentiation
is an ill-posed operation on noisy signals. A finite-difference estimator
divides a noise-sized numerator by a small dt, so it amplifies measurement
noise by a factor that grows as dt shrinks -- exactly backwards from what you
want (smaller dt should mean a *better* derivative estimate, and does, but
only until noise starts to dominate the numerator). In practice this means:
which estimator you use, and how much smoothing you apply, often matters more
to the final discovered model than any part of the sparse regression itself.
A poorly-smoothed derivative doesn't fail loudly -- it produces plausible-
looking "discovered" terms that are really just curve-fitting derivative
noise, which STLSQ has no way to distinguish from real dynamics.
"""

from __future__ import annotations

import torch
from scipy.signal import savgol_filter


def finite_difference(X: torch.Tensor, dt: float) -> torch.Tensor:
    """Second-order central differences. No smoothing.

    Interior points use a central difference, accurate to O(dt^2) for smooth
    noiseless data; endpoints fall back to a one-sided difference, O(dt).
    Assumes X is sampled on a uniform time grid of spacing dt.

    This is the right choice for clean (or near-noiseless) data, where its
    truncation error is tiny and it introduces no smoothing bias. It is the
    wrong choice for noisy data -- see module docstring.

    Args:
        X: [n_time, n_dims] state trajectory.
        dt: uniform sample spacing.

    Returns:
        Xdot: [n_time, n_dims] estimated derivatives.
    """
    if X.dim() != 2:
        raise ValueError(f"expected X of shape [n_time, n_dims], got {tuple(X.shape)}")
    if X.shape[0] < 3:
        raise ValueError("need at least 3 time points for central differences")

    Xdot = torch.empty_like(X)
    Xdot[1:-1] = (X[2:] - X[:-2]) / (2 * dt)
    Xdot[0] = (X[1] - X[0]) / dt
    Xdot[-1] = (X[-1] - X[-2]) / dt
    return Xdot


def savgol_derivative(
    X: torch.Tensor,
    dt: float,
    window_length: int = 21,
    polyorder: int = 3,
) -> torch.Tensor:
    """Savitzky-Golay derivative: fit a local polynomial in a sliding window
    and differentiate that polynomial analytically at the window center.

    This smooths and differentiates in a single step, which is the standard
    choice for noisy SINDy data -- smoothing X first and then taking finite
    differences still amplifies whatever high-frequency noise survives the
    smoothing pass, since finite-differencing is itself noise-amplifying.

    Args:
        X: [n_time, n_dims] state trajectory, uniform time grid.
        dt: sample spacing.
        window_length: number of samples per local fit window (must be odd
            and > polyorder). This is the main knob to tune against the
            actual noise level: larger windows smooth more (lower variance,
            higher bias -- can flatten real fast dynamics); smaller windows
            track fast dynamics better but let more noise through. There is
            no default that is correct for all noise levels; it should be
            chosen relative to the noise amplitude and the trajectory's
            fastest timescale.
        polyorder: degree of the local polynomial fit (must be < window_length).

    Returns:
        Xdot: [n_time, n_dims] estimated derivatives.
    """
    if X.dim() != 2:
        raise ValueError(f"expected X of shape [n_time, n_dims], got {tuple(X.shape)}")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if window_length <= polyorder:
        raise ValueError("window_length must be > polyorder")
    if window_length > X.shape[0]:
        raise ValueError(
            f"window_length ({window_length}) exceeds number of time points ({X.shape[0]})"
        )

    X_np = X.detach().cpu().numpy()
    Xdot_np = savgol_filter(
        X_np, window_length=window_length, polyorder=polyorder, deriv=1, delta=dt, axis=0
    )
    return torch.from_numpy(Xdot_np).to(dtype=X.dtype, device=X.device)
