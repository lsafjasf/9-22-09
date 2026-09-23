"""Per-task isolation: concurrent solves in one process must not share
counters, residual histories or configuration.
"""

import dataclasses
import os
import sys
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refactored import (IterativeSolver, ListLogger, SolverConfig)  # noqa: E402
from tests.problems import dense_spd, rhs, tridiagonal_laplacian, zero  # noqa: E402


def _run(A, b, preconditioner):
    logger = ListLogger()
    solver = IterativeSolver(A, b, SolverConfig(
        preconditioner=preconditioner, max_iterations=1000))
    x, stats = solver.solve(zero(len(b)), logger=logger)
    return x, stats, logger


class IsolationTests(unittest.TestCase):
    def test_concurrent_solves_keep_independent_stats(self):
        A1, b1 = tridiagonal_laplacian(120), rhs(120, seed=11)
        A2, b2 = dense_spd(30, seed=9), rhs(30, seed=5)
        barrier = threading.Barrier(2)

        def worker(fn):
            barrier.wait()
            return fn()

        with ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(worker,
                             lambda: _run(A1, b1, "jacobi"))
            f2 = pool.submit(worker,
                             lambda: _run(A2, b2, "ssor"))
            (x1, s1, log1), (x2, s2, log2) = f1.result(), f2.result()

        self.assertTrue(s1.converged and s2.converged)
        self.assertNotEqual(s1.iterations, 0)
        self.assertNotEqual(s2.iterations, 0)
        self.assertEqual(len(s1.residual_history), s1.iterations)
        self.assertEqual(len(s2.residual_history), s2.iterations)
        self.assertEqual(s1.preconditioner, "jacobi")
        self.assertEqual(s2.preconditioner, "ssor")
        # Per-solution loggers received only their own iterations.
        self.assertEqual(len(log1.lines), s1.iterations + 2)
        self.assertEqual(len(log2.lines), s2.iterations + 2)
        # Solutions are not accidentally identical.
        self.assertNotEqual(len(x1), len(x2))

    def test_sequential_solves_do_not_accumulate_state(self):
        A, b = tridiagonal_laplacian(60), rhs(60, seed=1)
        solver = IterativeSolver(A, b)
        _, first = solver.solve(zero(60))
        _, second = solver.solve(zero(60))
        self.assertEqual(first.iterations, second.iterations)
        self.assertEqual(first.residual_history, second.residual_history)
        self.assertEqual(first.matvecs, second.matvecs)

    def test_config_is_immutable(self):
        cfg = SolverConfig(max_iterations=5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.max_iterations = 9
        # Two solvers may hold different configs without interference.
        A, b = tridiagonal_laplacian(40), rhs(40)
        x_short, s_short = IterativeSolver(
            A, b, SolverConfig(max_iterations=2)).solve(zero(40))
        x_long, s_long = IterativeSolver(
            A, b, SolverConfig(max_iterations=1000)).solve(zero(40))
        self.assertFalse(s_short.converged)
        self.assertTrue(s_long.converged)
        self.assertEqual(s_short.iterations, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
