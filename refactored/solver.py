"""Boundary 7: facade assembling the other boundaries.

``IterativeSolver`` is stateless apart from its (immutable) config and the
matrix. Each ``solve`` builds a fresh preconditioner instance, criterion
instance, stats object and logger, so two solves on the same solver - and
two solvers in the same process/threads - never share mutable state.
"""

from .config import SolverConfig
from .criteria import resolve_criterion
from .kernel import pcg_solve
from .logging import NullLogger
from .preconditioners import SSOR, resolve_preconditioner
from .stats import RunStats


class IterativeSolver:
    def __init__(self, A, b, config: SolverConfig = None):
        self.A = A
        self.b = b
        self.config = config if config is not None else SolverConfig()

    def solve(self, x0, logger=None):
        cfg = self.config
        if logger is None:
            logger = NullLogger()

        pre_spec = cfg.preconditioner
        if pre_spec == "ssor":
            preconditioner = SSOR(self.A, omega=cfg.ssor_omega)
        else:
            preconditioner = resolve_preconditioner(pre_spec, self.A)
        criterion = resolve_criterion(cfg.criterion)

        stats = RunStats(
            preconditioner=self._preconditioner_name(pre_spec),
            criterion=self._criterion_name(cfg.criterion),
            max_iterations=cfg.max_iterations,
        )
        x = pcg_solve(self.A, self.b, x0, preconditioner, criterion,
                      cfg, stats, logger)
        return x, stats

    @staticmethod
    def _preconditioner_name(spec):
        if hasattr(spec, "__class__") and not isinstance(spec, (str, type)):
            return type(spec).__name__
        if isinstance(spec, type):
            return spec.__name__
        return str(spec)

    @staticmethod
    def _criterion_name(spec):
        if hasattr(spec, "__class__") and not isinstance(spec, (str, type)):
            return type(spec).__name__
        if isinstance(spec, type):
            return spec.__name__
        return str(spec)
