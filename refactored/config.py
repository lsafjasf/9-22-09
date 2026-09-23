"""Immutable per-run configuration. No module-level mutable knobs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverConfig:
    max_iterations: int = 1000
    preconditioner: object = "jacobi"
    criterion: object = "relative_rhs"
    ssor_omega: float = 1.2
