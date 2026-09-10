"""Smoke test: recover a trivial linear system (dx/dt = -x) end to end.

Fast (milliseconds) sanity check that simulation, feature library, and
STLSQ are wired together correctly, without needing the full Lorenz
pipeline to pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.derivatives import finite_difference
from src.sindy import FeatureLibrary, stlsq
from src.systems import DynamicalSystem


def _decay_system() -> DynamicalSystem:
    def rhs(t, x, p):
        return -x

    def true_equations(p):
        return ["dx/dt = -1.000 x"]

    def true_coefficients(p):
        return [{"x": -1.0}]

    return DynamicalSystem(
        name="decay",
        state_dim=1,
        state_names=("x",),
        params={},
        rhs=rhs,
        true_equations_fn=true_equations,
        true_coefficients_fn=true_coefficients,
    )


def test_recovers_linear_decay() -> None:
    torch.manual_seed(0)
    system = _decay_system()
    dt = 0.001
    t, X = system.simulate(x0=[1.0], t_span=(0.0, 5.0), dt=dt)
    Xdot = finite_difference(X, dt)

    library = FeatureLibrary(state_names=system.state_names, poly_degree=3)
    Theta = library.transform(X)

    result = stlsq(Theta, Xdot, threshold=0.1)
    Xi = result.Xi
    Xi_true = system.true_Xi(library)

    assert result.converged
    assert torch.allclose(Xi, Xi_true, atol=1e-2)


if __name__ == "__main__":
    test_recovers_linear_decay()
    print("smoke test passed")
