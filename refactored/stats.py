"""Boundary 5: per-run statistics.

A fresh RunStats is created inside every solve() call and returned to the
caller. Nothing here is class-level or module-level mutable, so concurrent
solves in one process can never see each other's counters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RunStats:
    preconditioner: str
    criterion: str
    max_iterations: int
    iterations: int = 0
    matvecs: int = 0
    initial_residual_norm: float = 0.0
    final_residual_norm: Optional[float] = None
    residual_history: List[float] = field(default_factory=list)
    converged: bool = False
    stop_reason: str = ""

    def record_matvec(self) -> None:
        self.matvecs += 1

    def record_iteration(self, residual_norm: float) -> None:
        self.iterations += 1
        self.residual_history.append(residual_norm)
        self.final_residual_norm = residual_norm
