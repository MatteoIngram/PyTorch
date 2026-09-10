"""Known dynamical systems used to generate SINDy training data.

Each system is a `DynamicalSystem`: a right-hand side `f(t, x)` for
generating trajectories, plus its true governing equations as strings so
`evaluate.py` can compare discovered coefficients against ground truth.

Trajectories are integrated with `scipy.integrate.solve_ivp` (adaptive-step
RK45 by default, tight `rtol`/`atol`) rather than a fixed-step scheme in
torch. The point of this project is to test whether SINDy can recover the
equations from data -- that test is only meaningful if the "clean" reference
trajectory itself is accurate to near machine precision, so integration
error doesn't get mistaken for regression error later on. Everything
downstream of simulation (feature library, regression) is torch; only the
one-off trajectory generation step borrows scipy's solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from scipy.integrate import solve_ivp


@dataclass
class DynamicalSystem:
    name: str
    state_dim: int
    state_names: tuple[str, ...]
    params: dict[str, float]
    rhs: Callable[[float, np.ndarray, dict], np.ndarray]
    true_equations_fn: Callable[[dict], list[str]]

    def f(self, t: float, x: np.ndarray) -> np.ndarray:
        return self.rhs(t, x, self.params)

    def true_equations(self) -> list[str]:
        return self.true_equations_fn(self.params)

    def simulate(
        self,
        x0: Sequence[float],
        t_span: tuple[float, float],
        dt: float,
        method: str = "RK45",
        rtol: float = 1e-10,
        atol: float = 1e-10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_eval = np.arange(t_span[0], t_span[1], dt)
        sol = solve_ivp(
            self.f,
            t_span,
            np.asarray(x0, dtype=np.float64),
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"integration of '{self.name}' failed: {sol.message}")
        t = torch.from_numpy(sol.t.copy()).to(torch.float64)
        X = torch.from_numpy(sol.y.T.copy()).to(torch.float64)
        return t, X


def lorenz(sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> DynamicalSystem:
    """The Lorenz (1963) system. Chaotic for the classic (sigma, rho, beta) above."""
    params = {"sigma": sigma, "rho": rho, "beta": beta}

    def rhs(t: float, x: np.ndarray, p: dict) -> np.ndarray:
        x1, x2, x3 = x
        return np.array(
            [
                p["sigma"] * (x2 - x1),
                x1 * (p["rho"] - x3) - x2,
                x1 * x2 - p["beta"] * x3,
            ]
        )

    def true_equations(p: dict) -> list[str]:
        return [
            f"dx/dt = {-p['sigma']:.3f} x + {p['sigma']:.3f} y",
            f"dy/dt = {p['rho']:.3f} x - 1.000 y - 1.000 x z",
            f"dz/dt = {-p['beta']:.3f} z + 1.000 x y",
        ]

    return DynamicalSystem(
        name="lorenz",
        state_dim=3,
        state_names=("x", "y", "z"),
        params=params,
        rhs=rhs,
        true_equations_fn=true_equations,
    )


def van_der_pol(mu: float = 2.0) -> DynamicalSystem:

    params = {"mu": mu}

    def rhs(t: float, x: np.ndarray, p: dict) -> np.ndarray:
        x1, x2 = x
        return np.array([x2, p["mu"] * (1 - x1**2) * x2 - x1])

    def true_equations(p: dict) -> list[str]:
        mu = p["mu"]
        return [
            "dx/dt = 1.000 y",
            f"dy/dt = -1.000 x + {mu:.3f} y - {mu:.3f} x^2 y",
        ]

    return DynamicalSystem(
        name="van_der_pol",
        state_dim=2,
        state_names=("x", "y"),
        params=params,
        rhs=rhs,
        true_equations_fn=true_equations,
    )


def rossler(a: float = 0.2, b: float = 0.2, c: float = 5.7) -> DynamicalSystem:
    params = {"a": a, "b": b, "c": c}

    def rhs(t: float, x: np.ndarray, p: dict) -> np.ndarray:
        x1, x2, x3 = x
        return np.array(
            [
                -x2 - x3,
                x1 + p["a"] * x2,
                p["b"] + x3 * (x1 - p["c"]),
            ]
        )

    def true_equations(p: dict) -> list[str]:
        return [
            "dx/dt = -1.000 y - 1.000 z",
            f"dy/dt = 1.000 x + {p['a']:.3f} y",
            f"dz/dt = {p['b']:.3f} + 1.000 x z - {p['c']:.3f} z",
        ]

    return DynamicalSystem(
        name="rossler",
        state_dim=3,
        state_names=("x", "y", "z"),
        params=params,
        rhs=rhs,
        true_equations_fn=true_equations,
    )


SYSTEMS: dict[str, Callable[..., DynamicalSystem]] = {
    "lorenz": lorenz,
    "van_der_pol": van_der_pol,
    "rossler": rossler,
}
