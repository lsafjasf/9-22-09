"""Demo: plug in a custom preconditioner / criterion without touching any
caller, and run two independent solver tasks concurrently in one process.

Run: python3 examples/demo.py
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refactored import (IterativeSolver, ListLogger, Preconditioner,
                        ConvergenceCriterion, SolverConfig,
                        register_preconditioner, register_criterion)
from tests.problems import dense_spd, rhs, tridiagonal_laplacian, zero


class ScaledJacobi(Preconditioner):
    """User-defined preconditioner: 1.5 * diag(A)^-1 (contrived but valid)."""

    def __init__(self, A):
        from refactored import linalg
        n = len(A) if isinstance(A, list) else len(A[0])
        self._diag = linalg.matrix_diag(A, n)

    def apply(self, r):
        return [1.5 * ri / di for ri, di in zip(r, self._diag)]


class FixedIterationLimit(ConvergenceCriterion):
    """User-defined criterion: stop exactly after ``limit`` iterations."""

    def __init__(self, limit=5):
        self._limit = limit

    def met(self, iteration, residual_norm):
        return iteration >= self._limit


register_preconditioner("scaled_jacobi", ScaledJacobi)
register_criterion("fixed_iterations", FixedIterationLimit)


def task(label, A, b, config):
    logger = ListLogger()
    _, stats = IterativeSolver(A, b, config).solve(zero(len(b)), logger=logger)
    return label, stats, len(logger.lines)


def main():
    A1, b1 = tridiagonal_laplacian(100), rhs(100, seed=2)
    A2, b2 = dense_spd(20, seed=3), rhs(20, seed=4)

    cfg1 = SolverConfig(preconditioner="scaled_jacobi")
    cfg2 = SolverConfig(preconditioner="ssor",
                        criterion=FixedIterationLimit(5))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda args: task(*args),
                                [("tridiagonal+scaled_jacobi", A1, b1, cfg1),
                                 ("dense+ssor+5iters", A2, b2, cfg2)]))

    for label, stats, log_lines in results:
        print("%-32s iters=%3d matvecs=%3d final||r||=%.3e converged=%s "
              "loglines=%d"
              % (label, stats.iterations, stats.matvecs,
                 stats.final_residual_norm, stats.converged, log_lines))


if __name__ == "__main__":
    main()
