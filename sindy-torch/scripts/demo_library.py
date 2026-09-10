"""Demo: simulate a system and build its SINDy feature library.

This does NOT discover equations yet -- the STLSQ solver isn't implemented.
It just exercises the two pieces that exist so far: the trajectory
simulator (src/systems.py) and the candidate feature library (src/sindy.py).

Usage:
    python scripts/demo_library.py [--system lorenz|van_der_pol|rossler]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sindy import FeatureLibrary
from src.systems import SYSTEMS

DEFAULT_X0 = {
    "lorenz": [1.0, 1.0, 1.0],
    "van_der_pol": [0.5, 0.0],
    "rossler": [1.0, 1.0, 1.0],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEMS), default="lorenz")
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--poly-degree", type=int, default=3)
    args = parser.parse_args()

    system = SYSTEMS[args.system]()

    print(f"=== {system.name} ===")
    print("true equations:")
    for eq in system.true_equations():
        print(f"  {eq}")

    t, X = system.simulate(
        x0=DEFAULT_X0[args.system], t_span=(0.0, args.t_end), dt=args.dt
    )
    print(f"\nsimulated trajectory: t={tuple(t.shape)}, X={tuple(X.shape)}")

    library = FeatureLibrary(state_names=system.state_names, poly_degree=args.poly_degree)
    Theta = library.transform(X)
    print(f"\nfeature library (poly_degree={args.poly_degree}): {library.n_terms} terms")
    print(library.names())
    print(f"Theta(X) shape: {tuple(Theta.shape)}")


if __name__ == "__main__":
    main()
