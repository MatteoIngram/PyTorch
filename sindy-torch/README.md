# SINDy in PyTorch

An implementation of **SINDy** (Sparse Identification of Nonlinear Dynamics,
Brunton/Proctor/Kutz 2016) — rediscovering the governing equations of a
dynamical system from trajectory data alone, no `pysindy` dependency.

The idea: given a state trajectory and its time derivatives, build a library
of candidate nonlinear terms (polynomials, trig functions), then fit a
*sparse* coefficient matrix so that `Xdot = Theta(X) @ Xi`. Sparsity is the
whole trick — most physical laws are governed by only a handful of active
terms, so thresholding out the small coefficients after least squares
recovers the true equations rather than an overfit dense model.

**Status: in progress.** Built so far:

- [`src/systems.py`](src/systems.py) — Lorenz, Van der Pol, and Rössler
  systems, each exposing a right-hand side (integrated via SciPy's adaptive
  RK45 for a near-machine-precision reference trajectory) and its true
  equations as strings for later comparison.
- [`src/sindy.py`](src/sindy.py) — the candidate feature library `Theta(X)`:
  constant, polynomial terms up to a configurable degree, optional sin/cos
  terms, all built with `torch` tensors.

Still to come: derivative estimation from noisy data (finite differences +
Savitzky-Golay smoothing), the sequentially-thresholded-least-squares
(STLSQ) solver that actually fits `Xi`, and the clean/noisy Lorenz-recovery
experiments.
