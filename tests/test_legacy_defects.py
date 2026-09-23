"""Characterization tests proving the LEGACY defects still exist, so the
defect -> new-structure mapping in README.md is backed by executable
evidence. The refactored package is asserted to behave correctly for the
same scenarios.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refactored import IterativeSolver, SolverConfig  # noqa: E402
from tests.problems import dense_spd, rhs, tridiagonal_laplacian, zero  # noqa: E402
import legacy.pcg_jacobi as legacy_jacobi  # noqa: E402


class LegacyDefectTests(unittest.TestCase):
    def tearDown(self):
        legacy_jacobi.TOL = 1e-8
        legacy_jacobi.MAX_ITER = 1000
        legacy_jacobi.VERBOSE = False
        legacy_jacobi.HISTORY.clear()
        legacy_jacobi.PCGSolver.iter_count = 0
        legacy_jacobi.PCGSolver.matvec_count = 0

    def test_legacy_history_is_shared_across_instances(self):
        A1, b1 = tridiagonal_laplacian(40), rhs(40, seed=7)
        A2, b2 = tridiagonal_laplacian(20), rhs(20, seed=8)

        legacy_jacobi.PCGSolver(A1, b1).solve(zero(40))
        n_after_first = len(legacy_jacobi.HISTORY)
        solver2 = legacy_jacobi.PCGSolver(A2, b2)
        solver2.solve(zero(20))

        # The second solver's instance attribute points at the SAME global
        # list, which now contains both runs' residuals concatenated.
        self.assertIs(solver2.history, legacy_jacobi.HISTORY)
        self.assertGreater(len(legacy_jacobi.HISTORY), n_after_first)
        self.assertGreater(n_after_first, 0)

        # New API: each solve hands back its own, correctly sized history.
        _, s1 = IterativeSolver(A1, b1).solve(zero(40))
        _, s2 = IterativeSolver(A2, b2).solve(zero(20))
        self.assertEqual(len(s1.residual_history), s1.iterations)
        self.assertEqual(len(s2.residual_history), s2.iterations)
        self.assertIsNot(s1.residual_history, s2.residual_history)

    def test_legacy_class_counters_are_shared(self):
        A1, b1 = tridiagonal_laplacian(40), rhs(40, seed=7)
        A2, b2 = dense_spd(15, seed=42), rhs(15, seed=99)
        legacy_jacobi.PCGSolver(A1, b1).solve(zero(40))
        legacy_jacobi.PCGSolver(A2, b2).solve(zero(15))
        # After the second solve the class counter reflects only the last
        # run; per-instance counters were never available.
        last_iters = legacy_jacobi.PCGSolver.iter_count

        _, s1 = IterativeSolver(A1, b1).solve(zero(40))
        _, s2 = IterativeSolver(A2, b2).solve(zero(15))
        self.assertNotEqual(s1.iterations, last_iters)
        self.assertEqual(s2.iterations, last_iters)  # same math, isolated stats

    def test_legacy_global_tol_change_leaks_between_tasks(self):
        A, b = tridiagonal_laplacian(80), rhs(80, seed=3)
        legacy_jacobi.PCGSolver(A, b).solve(zero(80))
        baseline = legacy_jacobi.PCGSolver.iter_count

        legacy_jacobi.TOL = 1e-2  # "task B" loosens tolerance globally
        legacy_jacobi.HISTORY.clear()
        legacy_jacobi.PCGSolver(A, b).solve(zero(80))
        loosened = legacy_jacobi.PCGSolver.iter_count
        self.assertLess(loosened, baseline)

        # New API: tolerance lives in a per-solver criterion instance.
        from refactored import RelativeToRHS
        _, s_strict = IterativeSolver(
            A, b, SolverConfig(criterion=RelativeToRHS(1e-8))
        ).solve(zero(80))
        _, s_loose = IterativeSolver(
            A, b, SolverConfig(criterion=RelativeToRHS(1e-2))
        ).solve(zero(80))
        self.assertEqual(s_strict.iterations, baseline)
        self.assertEqual(s_loose.iterations, loosened)
        self.assertLess(s_loose.iterations, s_strict.iterations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
