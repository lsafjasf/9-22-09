"""Old-vs-new equivalence: same problem + preconditioner -> same iteration
count, residual history (within FP tolerance) and solution.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refactored import IterativeSolver, SolverConfig  # noqa: E402
from tests.problems import PROBLEMS, rhs, zero  # noqa: E402
import legacy.pcg_jacobi as legacy_jacobi  # noqa: E402
import legacy.pcg_ssor as legacy_ssor  # noqa: E402

RESIDUAL_TOL = 1e-10
SOLUTION_TOL = 1e-9


def reset_legacy(*modules):
    for m in modules:
        m.TOL = 1e-8
        m.MAX_ITER = 1000
        m.VERBOSE = False
        m.HISTORY.clear()
        m.PCGSolver.iter_count = 0
        m.PCGSolver.matvec_count = 0


class EquivalenceTests(unittest.TestCase):
    def setUp(self):
        reset_legacy(legacy_jacobi, legacy_ssor)

    def tearDown(self):
        reset_legacy(legacy_jacobi, legacy_ssor)

    def _assert_close_vectors(self, got, want, tol, label):
        self.assertEqual(len(got), len(want))
        for i, (g, w) in enumerate(zip(got, want)):
            scale = max(1.0, abs(w))
            self.assertLessEqual(
                abs(g - w) / scale, tol,
                "%s differs at index %d: %.15g vs %.15g" % (label, i, g, w))

    def test_jacobi_matches_legacy(self):
        for name, A, b in PROBLEMS:
            with self.subTest(problem=name):
                reset_legacy(legacy_jacobi)
                x_old = legacy_jacobi.PCGSolver(A, b).solve(zero(len(b)))
                old_iters = legacy_jacobi.PCGSolver.iter_count
                old_hist = list(legacy_jacobi.HISTORY)

                solver = IterativeSolver(
                    A, b, SolverConfig(preconditioner="jacobi"))
                x_new, stats = solver.solve(zero(len(b)))

                self.assertEqual(stats.iterations, old_iters, name)
                self.assertTrue(stats.converged, name)
                self.assertEqual(len(stats.residual_history), len(old_hist))
                for k, (new_r, old_r) in enumerate(
                        zip(stats.residual_history, old_hist)):
                    self.assertLessEqual(
                        abs(new_r - old_r) / max(1e-30, old_r),
                        RESIDUAL_TOL, "residual[%d] %s" % (k, name))
                self._assert_close_vectors(
                    x_new, x_old, SOLUTION_TOL, "x " + name)

    def test_ssor_matches_legacy(self):
        for name, A, b in PROBLEMS:
            with self.subTest(problem=name):
                reset_legacy(legacy_ssor)
                x_old = legacy_ssor.PCGSolver(A, b).solve(zero(len(b)))
                old_iters = legacy_ssor.PCGSolver.iter_count
                old_hist = list(legacy_ssor.HISTORY)

                solver = IterativeSolver(
                    A, b, SolverConfig(preconditioner="ssor",
                                       ssor_omega=1.2))
                x_new, stats = solver.solve(zero(len(b)))

                self.assertEqual(stats.iterations, old_iters, name)
                self.assertTrue(stats.converged, name)
                self.assertEqual(len(stats.residual_history), len(old_hist))
                for k, (new_r, old_r) in enumerate(
                        zip(stats.residual_history, old_hist)):
                    self.assertLessEqual(
                        abs(new_r - old_r) / max(1e-30, old_r),
                        RESIDUAL_TOL, "residual[%d] %s" % (k, name))
                self._assert_close_vectors(
                    x_new, x_old, SOLUTION_TOL, "x " + name)

    def test_identity_runs_without_preconditioning(self):
        from tests.problems import tridiagonal_laplacian
        A, b = tridiagonal_laplacian(50), rhs(50, seed=3)
        x, stats = IterativeSolver(
            A, b, SolverConfig(preconditioner="identity")).solve(zero(50))
        self.assertTrue(stats.converged)
        self.assertGreater(stats.iterations, 0)
        self.assertLessEqual(stats.final_residual_norm, 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
