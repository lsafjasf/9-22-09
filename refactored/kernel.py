"""Boundary 4: the PCG iteration kernel.

The kernel is a pure orchestrator; it owns no matrices, no configuration
globals and no counters. Call order per solve:

1. preconditioner and criterion are resolved/bound once,
2. a fresh RunStats + (per-run) logger are passed in,
3. each iteration: matvec -> update x, r -> record residual/stat/log ->
   test criterion -> precondition -> direction update.

The arithmetic ordering below matches the legacy class operation for
operation (same accumulation order in dot, same in-place axpy), which keeps
the refactor bit-for-bit close instead of merely algorithmically equivalent.
"""

from . import linalg
from .criteria import CriterionContext
from .stats import RunStats


def pcg_solve(A, b, x0, preconditioner, criterion, config, stats, logger):
    n = len(b)
    bnorm = linalg.dot(b, b) ** 0.5

    x = [float(v) for v in x0]
    Ax = linalg.matvec(A, x)
    r = [b[i] - Ax[i] for i in range(n)]
    stats.record_matvec()

    initial_rnorm = linalg.dot(r, r) ** 0.5
    stats.initial_residual_norm = initial_rnorm

    criterion.bind(CriterionContext(
        b_norm=bnorm,
        initial_residual_norm=initial_rnorm,
        dimension=n,
    ))
    logger.start(n, bnorm)

    z = preconditioner.apply(r)
    p = z[:]
    rz = linalg.dot(r, z)

    for _ in range(config.max_iterations):
        Ap = linalg.matvec(A, p)
        stats.record_matvec()

        alpha = rz / linalg.dot(p, Ap)
        linalg.axpy(x, alpha, p)
        linalg.axpy(r, -alpha, Ap)

        rnorm = linalg.dot(r, r) ** 0.5
        stats.record_iteration(rnorm)
        logger.iteration(stats.iterations, rnorm)

        if criterion.met(stats.iterations, rnorm):
            stats.converged = True
            stats.stop_reason = "converged"
            logger.finish(stats)
            return x

        z = preconditioner.apply(r)
        rz_new = linalg.dot(r, z)
        beta = rz_new / rz
        p = [z[i] + beta * p[i] for i in range(n)]
        rz = rz_new

    stats.converged = False
    stats.stop_reason = "max_iterations"
    logger.finish(stats)
    return x
