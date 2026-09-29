"""预条件 × 判据全组合测试：同一方程组上切换组合。

验证点：
  1. 所有组合都收敛且满足最终残差断言；
  2. 切换组合不改变接口语义：调用点永远是 solver.solve(A, b, x0)，
     返回的 SolveResult 字段集合一致；
  3. 残差下降率判据在 x0=0 时退化为相对残差判据（逐位一致）；
  4. IterationState 新增字段不破坏既有判据的三字段构造方式。

运行：python3 -m unittest discover -s tests -v   （仓库根目录下）
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver import (AbsoluteResidualCriterion, IterativeSolver, IterationState,
                    JacobiPreconditioner, RelativeResidualCriterion,
                    ResidualDecreaseRateCriterion, SSORPreconditioner,
                    SolverConfig, SolveResult)
from solver.linalg import norm2, residual
from tests.test_equivalence import make_rhs, make_spd

N, SEED = 30, 7
TOL = 1e-8
A = make_spd(N, SEED)
b = make_rhs(N, SEED + 1000)
x0 = make_rhs(N, SEED + 2000)
B_NORM = norm2(b)
R0_NORM = norm2(residual(A, b, x0))

PRECONDITIONERS = [
    ("Jacobi", JacobiPreconditioner),
    ("SSOR(1.0)", lambda M: SSORPreconditioner(M, 1.0)),
    ("SSOR(1.2)", lambda M: SSORPreconditioner(M, 1.2)),
]
CRITERIA = [
    ("relative", RelativeResidualCriterion(TOL)),
    ("absolute", AbsoluteResidualCriterion(TOL * B_NORM)),
    ("decrease-rate", ResidualDecreaseRateCriterion(TOL)),
]

# 所有组合统一满足的最终残差上限（取各判据基准的最大者）
RESIDUAL_LIMIT = TOL * max(B_NORM, R0_NORM)
RESULT_FIELDS = {"x", "iterations", "final_residual", "converged", "stats"}


def solve_with(preconditioner_factory, criterion):
    solver = IterativeSolver(
        config=SolverConfig(max_iterations=5000, omega=1.0),
        preconditioner_factory=preconditioner_factory,
        criterion=criterion,
    )
    return solver.solve(A, b, x0)  # 调用点不随组合改变


class TestAllCombinations(unittest.TestCase):
    """3 预条件 × 3 判据 = 9 组合，全部满足最终残差断言。"""

    def test_every_combo_converges_below_residual_limit(self):
        for pn, pf in PRECONDITIONERS:
            for cn, criterion in CRITERIA:
                with self.subTest(preconditioner=pn, criterion=cn):
                    result = solve_with(pf, criterion)
                    self.assertTrue(result.converged)
                    self.assertLessEqual(result.final_residual,
                                         RESIDUAL_LIMIT)
                    # 残差历史单调记录且末项即最终残差
                    self.assertEqual(result.stats.residual_history[-1],
                                     result.final_residual)
                    self.assertEqual(result.stats.iterations,
                                      result.iterations)

    def test_interface_semantics_stable_across_combos(self):
        """切换组合只改构造参数；返回类型与字段集合完全一致。"""
        for pn, pf in PRECONDITIONERS:
            for cn, criterion in CRITERIA:
                with self.subTest(preconditioner=pn, criterion=cn):
                    result = solve_with(pf, criterion)
                    self.assertIsInstance(result, SolveResult)
                    self.assertEqual(set(vars(result)), RESULT_FIELDS)
                    self.assertEqual(len(result.x), N)


class TestDecreaseRateCriterion(unittest.TestCase):
    """残差下降率判据的语义与兼容性。"""

    def test_reduces_to_relative_when_x0_is_zero(self):
        """x0=0 时 ||r0||=||b||，两种判据应给出逐位一致的结果。"""
        kw = dict(config=SolverConfig(max_iterations=5000),
                  preconditioner_factory=JacobiPreconditioner)
        rel = IterativeSolver(criterion=RelativeResidualCriterion(TOL),
                              **kw).solve(A, b)
        dec = IterativeSolver(criterion=ResidualDecreaseRateCriterion(TOL),
                              **kw).solve(A, b)
        self.assertEqual(rel.iterations, dec.iterations)
        self.assertEqual(rel.final_residual, dec.final_residual)
        self.assertEqual(rel.stats.residual_history,
                         dec.stats.residual_history)

    def test_differs_from_relative_when_x0_nonzero(self):
        """x0 非零且 ||r0|| != ||b|| 时，基准不同，停机点一般不同。"""
        self.assertNotAlmostEqual(B_NORM, R0_NORM, places=6)
        rel = solve_with(JacobiPreconditioner,
                         RelativeResidualCriterion(TOL))
        dec = solve_with(JacobiPreconditioner,
                         ResidualDecreaseRateCriterion(TOL))
        self.assertNotEqual(rel.iterations, dec.iterations)

    def test_state_without_initial_residual_falls_back_to_rhs(self):
        """旧的三字段快照仍可使用（接口语义向后兼容）。"""
        criterion = ResidualDecreaseRateCriterion(1e-3)
        state = IterationState(iteration=5, residual_norm=1e-4,
                               rhs_norm=1.0)
        self.assertTrue(criterion.is_converged(state))
        state = IterationState(iteration=5, residual_norm=1e-2,
                               rhs_norm=1.0)
        self.assertFalse(criterion.is_converged(state))

    def test_exact_initial_guess_converges_at_iteration_zero(self):
        """x0 即真解时 ||r0||=0，应在第 0 步判定收敛（不除零）。"""
        from solver.linalg import matvec
        x_exact = make_rhs(N, SEED + 3000)
        b_exact = matvec(A, x_exact)
        solver = IterativeSolver(
            criterion=ResidualDecreaseRateCriterion(TOL),
            preconditioner_factory=JacobiPreconditioner,
        )
        result = solver.solve(A, b_exact, x_exact)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)


if __name__ == "__main__":
    unittest.main()
