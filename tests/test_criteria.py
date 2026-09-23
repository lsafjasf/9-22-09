"""Swapping the convergence criterion without touching caller code."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refactored import (InitialResidualReduction, IterativeSolver,
                        RelativeToRHS, SolverConfig)  # noqa: E402
from tests.problems import PROBLEMS, zero  # noqa: E402


def reference_pcg(A, b, x0, apply_pre, stop):
    """Independent compact reference PCG used only for criterion checks."""
    from refactored import linalg

    bnorm = linalg.dot(b, b) ** 0.5
    x = list(map(float, x0))
    r = [b[i] - v for i, v in enumerate(linalg.matvec(A, x))]
    r0 = linalg.dot(r, r) ** 0.5
    z = apply_pre(r)
    p = z[:]
    rz = linalg.dot(r, z)
    history = [r0]
    for _ in range(1000):
        Ap = linalg.matvec(A, p)
        alpha = rz / linalg.dot(p, Ap)
        linalg.axpy(x, alpha, p)
        linalg.axpy(r, -alpha, Ap)
        rn = linalg.dot(r, r) ** 0.5
        history.append(rn)
        if stop(rn, r0, bnorm):
            return x, len(history) - 1, history
        z = apply_pre(r)
        rzn = linalg.dot(r, z)
        p = [z[i] + (rzn / rz) * p[i] for i in range(len(b))]
        rz = rzn
    return x, len(history) - 1, history


class CriterionTests(unittest.TestCase):
    def test_initial_reduction_is_really_r0_reduction(self):
        for name, A, b in PROBLEMS:
            with self.subTest(problem=name):
                solver = IterativeSolver(A, b, SolverConfig(
                    preconditioner="jacobi",
                    criterion=InitialResidualReduction(1e-6)))
                _, stats = solver.solve(zero(len(b)))
                self.assertTrue(stats.converged)
                # Final residual <= factor * r0, and one step earlier was not.
                self.assertLessEqual(
                    stats.final_residual_norm,
                    1e-6 * stats.initial_residual_norm)
                self.assertGreater(
                    stats.residual_history[-2],
                    1e-6 * stats.initial_residual_norm)

    def test_criterion_matches_independent_reference(self):
        from refactored import Jacobi
        for name, A, b in PROBLEMS:
            with self.subTest(problem=name):
                pre = Jacobi(A)
                x_ref, k_ref, hist_ref = reference_pcg(
                    A, b, zero(len(b)), pre.apply,
                    lambda rn, r0, bn: rn <= 1e-6 * r0)
                _, stats = IterativeSolver(
                    A, b, SolverConfig(
                        criterion="initial_reduction")).solve(zero(len(b)))
                self.assertEqual(stats.iterations, k_ref, name)
                self.assertAlmostEqual(
                    stats.final_residual_norm, hist_ref[-1],
                    delta=1e-12 * max(1.0, abs(hist_ref[-1])))

    def test_two_criteria_give_different_iteration_counts(self):
        name, A, b = PROBLEMS[2]  # dense: looser r0 test stops earlier
        _, s_rhs = IterativeSolver(
            A, b, SolverConfig(criterion=RelativeToRHS(1e-8))
        ).solve(zero(len(b)))
        _, s_red = IterativeSolver(
            A, b, SolverConfig(criterion=InitialResidualReduction(1e-6))
        ).solve(zero(len(b)))
        self.assertTrue(s_rhs.converged and s_red.converged)
        self.assertNotEqual(s_rhs.iterations, s_red.iterations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
