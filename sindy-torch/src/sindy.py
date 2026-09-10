"""SINDy: Sparse Identification of Nonlinear Dynamics (Brunton, Proctor, Kutz 2016).

The method fits Xdot = Theta(X) @ Xi, where Theta(X) is a library of candidate
nonlinear functions of the state and Xi is a coefficient matrix that is
regularized to be sparse -- the working assumption being that most physical
laws are governed by only a few active terms per equation.

This module currently implements the feature library, `Theta(X)`. The
sequentially thresholded least squares (STLSQ) solver that fits Xi is not
yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Sequence

import torch


@dataclass
class LibraryTerm:
    """One column of Theta(X)."""

    kind: str  
    exponents: tuple[int, ...] | None = None  
    index: int | None = None 
    freq: float | None = None  


@dataclass
class FeatureLibrary:

    state_names: Sequence[str]
    poly_degree: int = 3
    include_trig: bool = False
    trig_freqs: Sequence[float] = (1.0,)
    _terms: list[LibraryTerm] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.poly_degree < 0:
            raise ValueError("poly_degree must be >= 0")
        self._terms = self._build_terms()

    def _build_terms(self) -> list[LibraryTerm]:
        n = len(self.state_names)
        terms: list[LibraryTerm] = [LibraryTerm(kind="poly", exponents=(0,) * n)]
        for degree in range(1, self.poly_degree + 1):
            for combo in combinations_with_replacement(range(n), degree):
                exponents = [0] * n
                for idx in combo:
                    exponents[idx] += 1
                terms.append(LibraryTerm(kind="poly", exponents=tuple(exponents)))
        if self.include_trig:
            for i in range(n):
                for freq in self.trig_freqs:
                    terms.append(LibraryTerm(kind="sin", index=i, freq=freq))
                    terms.append(LibraryTerm(kind="cos", index=i, freq=freq))
        return terms

    @property
    def n_terms(self) -> int:
        return len(self._terms)

    def names(self) -> list[str]:
        
        out = []
        for term in self._terms:
            if term.kind == "poly":
                out.append(self._poly_name(term.exponents))
            else:
                freq_str = "" if term.freq == 1.0 else f"{term.freq:g} "
                out.append(f"{term.kind}({freq_str}{self.state_names[term.index]})")
        return out

    def _poly_name(self, exponents: tuple[int, ...]) -> str:
        if all(e == 0 for e in exponents):
            return "1"
        parts = []
        for name, e in zip(self.state_names, exponents):
            if e == 0:
                continue
            parts.append(name if e == 1 else f"{name}^{e}")
        return " ".join(parts)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
       
        if X.dim() != 2 or X.shape[1] != len(self.state_names):
            raise ValueError(
                f"expected X of shape [n_time, {len(self.state_names)}], got {tuple(X.shape)}"
            )
        n_time = X.shape[0]
        columns = []
        for term in self._terms:
            if term.kind == "poly":
                col = torch.ones(n_time, dtype=X.dtype, device=X.device)
                for dim_idx, e in enumerate(term.exponents):
                    if e:
                        col = col * X[:, dim_idx] ** e
                columns.append(col)
            elif term.kind == "sin":
                columns.append(torch.sin(term.freq * X[:, term.index]))
            elif term.kind == "cos":
                columns.append(torch.cos(term.freq * X[:, term.index]))
            else:
                raise ValueError(f"unknown term kind: {term.kind}")
        return torch.stack(columns, dim=1)
