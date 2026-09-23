"""Modular iterative solver package (PCG).

Boundaries, in call order:
  linalg -> preconditioners/criteria (pluggable) -> kernel -> stats/logging
The solver facade wires them together per solve with fresh, isolated state.
"""

from .config import SolverConfig
from .criteria import (AbsoluteResidual, ConvergenceCriterion,
                       InitialResidualReduction, RelativeToRHS,
                       register_criterion, resolve_criterion)
from .kernel import pcg_solve
from .logging import ListLogger, NullLogger, SolverLogger, StreamLogger
from .preconditioners import (Identity, Jacobi, Preconditioner, SSOR,
                              register_preconditioner, resolve_preconditioner)
from .solver import IterativeSolver
from .stats import RunStats

__all__ = [
    "SolverConfig", "IterativeSolver", "RunStats", "pcg_solve",
    "Preconditioner", "Identity", "Jacobi", "SSOR",
    "register_preconditioner", "resolve_preconditioner",
    "ConvergenceCriterion", "RelativeToRHS", "InitialResidualReduction",
    "AbsoluteResidual", "register_criterion", "resolve_criterion",
    "SolverLogger", "NullLogger", "StreamLogger", "ListLogger",
]
