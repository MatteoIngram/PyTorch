"""experiment_clean.py: recover the Lorenz equations from clean trajectory data.

The full clean-data SINDy pipeline:
  simulate -> estimate derivatives -> build feature library -> STLSQ
  -> compare discovered vs. true equations -> integrate the discovered
     model and overlay it against the true trajectory.

On clean (noiseless) data this should nail the exact Lorenz equations, which
is the headline demo for this project. See experiment_noisy.py for what
happens as measurement noise is added -- that's where derivative-estimation
choices actually start to matter.

Usage:
    python scripts/experiment_clean.py [--system lorenz|van_der_pol|rossler]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch

from src.derivatives import finite_difference
from src.evaluate import (
    coefficient_error,
    format_equations,
    print_comparison,
    simulate_discovered,
)
from src.sindy import FeatureLibrary, stlsq
from src.systems import SYSTEMS

SEED = 0

DEFAULT_X0 = {
    "lorenz": [1.0, 1.0, 1.0],
    "van_der_pol": [0.5, 0.0],
    "rossler": [1.0, 1.0, 1.0],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEMS), default="lorenz")
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    system = SYSTEMS[args.system]()
    x0 = DEFAULT_X0[args.system]
    t_span = (0.0, args.t_end)

    # Fine dt keeps the central-difference truncation error (O(dt^2)) far
    # below the STLSQ threshold, so clean data should recover exact coefficients.
    t, X = system.simulate(x0=x0, t_span=t_span, dt=args.dt)
    Xdot = finite_difference(X, args.dt)

    library = FeatureLibrary(state_names=system.state_names, poly_degree=args.poly_degree)
    Theta = library.transform(X)

    result = stlsq(Theta, Xdot, threshold=args.threshold, max_iter=20)
    Xi = result.Xi
    Xi_true = system.true_Xi(library)

    print(f"=== {system.name}: clean-data recovery ===")
    print(
        f"n_time={X.shape[0]}, dt={args.dt}, poly_degree={args.poly_degree}, "
        f"STLSQ threshold={args.threshold}"
    )
    print(f"STLSQ converged={result.converged} after {result.n_iters} sweep(s)\n")

    print("true equations:")
    for eq in system.true_equations():
        print(f"  {eq}")

    print("\ndiscovered equations:")
    for eq in format_equations(Xi, library, system.state_names):
        print(f"  {eq}")

    print("\ncoefficient comparison (true / discovered):")
    print_comparison(Xi, Xi_true, library, system.state_names)

    err = coefficient_error(Xi, Xi_true)
    n_true_terms = int((Xi_true.abs() > 1e-12).sum().item())
    print(
        f"\ncoefficient error: frobenius_rel={err['frobenius_rel']:.2e}, "
        f"max_abs={err['max_abs_error']:.2e}, "
        f"support recovered exactly for {err['n_correct_support']}/{Xi.shape[0]} terms "
        f"({n_true_terms} true nonzero terms)"
    )

    if args.no_plot:
        return

    t_disc, X_disc = simulate_discovered(Xi, library, x0=x0, t_span=t_span, dt=args.dt)

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.system}_clean_trajectories.png"

    if X.shape[1] == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(projection="3d")
        ax.plot(X[:, 0], X[:, 1], X[:, 2], color="tab:blue", linewidth=0.7, label="true")
        ax.plot(
            X_disc[:, 0], X_disc[:, 1], X_disc[:, 2],
            color="tab:orange", linewidth=0.7, linestyle="--", label="discovered",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(X[:, 0], X[:, 1], color="tab:blue", linewidth=0.7, label="true")
        ax.plot(X_disc[:, 0], X_disc[:, 1], color="tab:orange", linewidth=0.7, linestyle="--", label="discovered")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    ax.set_title(f"{system.name}: true vs. discovered trajectory (clean data)")
    ax.legend()
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved trajectory plot to {out_path}")


if __name__ == "__main__":
    main()
