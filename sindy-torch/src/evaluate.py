"""Pretty-printing, comparison, and trajectory validation for discovered
SINDy models.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.integrate import solve_ivp

from .sindy import FeatureLibrary

ZERO_TOL = 1e-12


def format_equations(
    Xi: torch.Tensor,
    library: FeatureLibrary,
    state_names: Sequence[str],
    precision: int = 2,
) -> list[str]:
    """Render Xi as human-readable 'dx_i/dt = ...' strings, dropping zeroed terms.

    e.g. dx/dt = -10.02 x + 10.01 y
    """
    names = library.names()
    equations = []
    for j, state_name in enumerate(state_names):
        parts: list[str] = []
        for i, name in enumerate(names):
            coef = Xi[i, j].item()
            if abs(coef) < ZERO_TOL:
                continue
            magnitude = f"{abs(coef):.{precision}f}" + ("" if name == "1" else f" {name}")
            if not parts:
                parts.append(f"{'-' if coef < 0 else ''}{magnitude}")
            else:
                parts.append(f"{'-' if coef < 0 else '+'} {magnitude}")
        rhs = " ".join(parts) if parts else "0"
        equations.append(f"d{state_name}/dt = {rhs}")
    return equations


def print_comparison(
    Xi_discovered: torch.Tensor,
    Xi_true: torch.Tensor,
    library: FeatureLibrary,
    state_names: Sequence[str],
    precision: int = 4,
) -> None:
    """Print a term-by-term table of true vs. discovered coefficients.

    Only rows where either the true or the discovered coefficient is
    nonzero are shown, so the table stays readable even for large libraries.
    """
    names = library.names()
    col_width = 22
    header = f"{'term':<10}" + "".join(f"{s + ' true/disc':>{col_width}}" for s in state_names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        cells = []
        any_nonzero = False
        for j in range(len(state_names)):
            true_val = Xi_true[i, j].item()
            disc_val = Xi_discovered[i, j].item()
            if abs(true_val) > ZERO_TOL or abs(disc_val) > ZERO_TOL:
                any_nonzero = True
            cells.append(f"{true_val:>8.{precision}f} /{disc_val:>8.{precision}f}")
        if any_nonzero:
            print(f"{name:<10}" + "".join(f"{c:>{col_width}}" for c in cells))


def coefficient_error(Xi_discovered: torch.Tensor, Xi_true: torch.Tensor) -> dict[str, float]:
    """Error metrics between discovered and true coefficient matrices.

    frobenius_rel is the headline number for the noise-sweep experiment:
    the relative Frobenius-norm error ||Xi_disc - Xi_true|| / ||Xi_true||.
    """
    diff = Xi_discovered - Xi_true
    true_norm = torch.linalg.norm(Xi_true).item()
    return {
        "frobenius_abs": torch.linalg.norm(diff).item(),
        "frobenius_rel": torch.linalg.norm(diff).item() / true_norm if true_norm > 0 else float("nan"),
        "max_abs_error": diff.abs().max().item(),
        "n_correct_support": int(
            ((Xi_discovered.abs() > ZERO_TOL) == (Xi_true.abs() > ZERO_TOL)).all(dim=1).sum().item()
        ),
    }


def simulate_discovered(
    Xi: torch.Tensor,
    library: FeatureLibrary,
    x0: Sequence[float],
    t_span: tuple[float, float],
    dt: float,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate the discovered model dx/dt = Theta(x) @ Xi forward in time.

    Used to sanity-check a discovered model beyond just comparing
    coefficients: even small coefficient errors can compound in a chaotic
    system, so the discovered and true trajectories are expected to diverge
    after a Lyapunov-time or so even when the equations were recovered
    essentially exactly -- that divergence is expected chaotic behavior, not
    evidence the model is wrong. Compare coefficients for correctness,
    compare trajectories only qualitatively (does it stay on the attractor).
    """
    dtype = Xi.dtype

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        x_t = torch.as_tensor(x, dtype=dtype).unsqueeze(0)
        xdot = (library.transform(x_t) @ Xi).squeeze(0)
        return xdot.numpy()

    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        rhs, t_span, np.asarray(x0, dtype=np.float64), t_eval=t_eval, rtol=rtol, atol=atol
    )
    if not sol.success:
        raise RuntimeError(f"integration of discovered model failed: {sol.message}")
    t = torch.from_numpy(sol.t.copy()).to(torch.float64)
    X = torch.from_numpy(sol.y.T.copy()).to(torch.float64)
    return t, X
