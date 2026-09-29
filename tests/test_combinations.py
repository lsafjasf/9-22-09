"""预条件 × 判据 组合迭代的功能测试。

覆盖：
  1. 残差下降率判据的单元行为（含参数校验与无状态性）；
  2. 同一方程组上 3 预条件 × 3 判据共 9 种组合全部收敛，
     且满足与各自判据语义一致的最终残差断言；
  3. 切换组合不改变调用接口语义（solve(A, b[, x0]) -> SolveResult）。

运行：python3 -m unittest discover -s tests -v   （仓库根目录下）
"""

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solver import (AbsoluteResidualCriterion, IdentityPreconditioner,
                    IterationState, IterativeSolver, JacobiPreconditioner,
                    RelativeResidualCriterion, ResidualDecreaseRateCriterion,
                    SolverConfig, SSORPreconditioner)
from solver.linalg import norm2
from tests.test_equivalence import make_rhs, make_spd

N, SEED = 12, 2026
MAX_ITER = 5000
TOL = 1e-8
RATE, MIN_ITERS = 0.5, 5


def gershgorin_lambda_max(A):
    return max(A[i][i] + sum(abs(A[i][j]) for j in range(len(A)) if j != i)
               for i in range(len(A)))


def combos(A):
    """返回 [(名称, 构造 kwargs, 最终残差断言函数)]，覆盖全部 9 种组合。"""
    omega_identity = 1.8 / gershgorin_lambda_max(A)
    preconds = [("identity", IdentityPreconditioner, omega_identity),
                ("jacobi", JacobiPreconditioner, 1.0),
                ("ssor", lambda M: SSORPreconditioner(M, 1.0), 1.0)]
    criteria = [
        ("absolute", AbsoluteResidualCriterion(TOL),
         lambda res, r0, k, b_norm: res <= TOL),
        ("relative", RelativeResidualCriterion(TOL),
         lambda res, r0, k, b_norm: res <= TOL * b_norm),
        ("decrease-rate", ResidualDecreaseRateCriterion(RATE, MIN_ITERS),
         lambda res, r0, k, b_norm: (res / r0) ** (1.0 / k) <= RATE),
    ]
    out = []
    for p_name, p_factory, omega in preconds:
        for c_name, criterion, check in criteria:
            out.append((f"{p_name}+{c_name}",
                        dict(config=SolverConfig(max_iterations=MAX_ITER,
                                                 omega=omega),
                             preconditioner_factory=p_factory,
                             criterion=criterion),
                        check))
    return out


class TestDecreaseRateCriterion(unittest.TestCase):
    """残差下降率判据的单元行为。"""

    def test_average_rate_rule(self):
        c = ResidualDecreaseRateCriterion(0.5, min_iterations=2)
        # k=2, (0.2/1.0)^(1/2) ≈ 0.447 <= 0.5 → 收敛
        self.assertTrue(c.is_converged(IterationState(
            iteration=2, residual_norm=0.2, rhs_norm=1.0,
            initial_residual_norm=1.0, previous_residual_norm=0.4)))
        # k=2, (0.3/1.0)^(1/2) ≈ 0.548 > 0.5 → 未收敛
        self.assertFalse(c.is_converged(IterationState(
            iteration=2, residual_norm=0.3, rhs_norm=1.0,
            initial_residual_norm=1.0, previous_residual_norm=0.5)))

    def test_min_iterations_guard(self):
        c = ResidualDecreaseRateCriterion(0.9, min_iterations=3)
        state = IterationState(iteration=1, residual_norm=1e-12,
                               rhs_norm=1.0, initial_residual_norm=1.0,
                               previous_residual_norm=0.5)
        self.assertFalse(c.is_converged(state))  # 步数不足，残差再小也不判收敛

    def test_zero_residual_always_converged(self):
        c = ResidualDecreaseRateCriterion(0.5)
        self.assertTrue(c.is_converged(IterationState(
            iteration=0, residual_norm=0.0, rhs_norm=1.0,
            initial_residual_norm=0.0)))

    def test_invalid_params_rejected(self):
        with self.assertRaises(ValueError):
            ResidualDecreaseRateCriterion(1.5)
        with self.assertRaises(ValueError):
            ResidualDecreaseRateCriterion(0.5, min_iterations=0)

    def test_criterion_is_stateless_and_shareable(self):
        """判据不保存跨调用状态，可被并发任务共享。"""
        c = ResidualDecreaseRateCriterion(RATE, MIN_ITERS)
        A, b = make_spd(8, 11), make_rhs(8, 1011)
        solver = IterativeSolver(
            config=SolverConfig(max_iterations=MAX_ITER),
            preconditioner_factory=JacobiPreconditioner, criterion=c)
        serial = solver.solve(A, b)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: solver.solve(A, b), range(8)))
        for r in results:
            self.assertEqual(serial.iterations, r.iterations)
            self.assertEqual(serial.final_residual, r.final_residual)


class TestAllCombinations(unittest.TestCase):
    """同一方程组上 9 种组合全部收敛，且满足各自判据的最终残差断言。"""

    def test_all_combos_converge_with_residual_assertion(self):
        A, b = make_spd(N, SEED), make_rhs(N, SEED + 1000)
        b_norm = norm2(b)
        for name, kwargs, check in combos(A):
            with self.subTest(combo=name):
                result = IterativeSolver(**kwargs).solve(A, b)
                self.assertTrue(result.converged)
                r0 = result.stats.residual_history[0]
                self.assertTrue(check(result.final_residual, r0,
                                      result.iterations, b_norm),
                                f"{name}: final_residual="
                                f"{result.final_residual:.3e}")

    def test_interface_semantics_unchanged(self):
        """切换组合时调用点与返回结构完全一致。"""
        A, b = make_spd(N, SEED), make_rhs(N, SEED + 1000)
        x0 = make_rhs(N, SEED + 2000)
        for name, kwargs, _ in combos(A):
            with self.subTest(combo=name):
                solver = IterativeSolver(**kwargs)
                r1 = solver.solve(A, b)          # 两种调用形式都可用
                r2 = solver.solve(A, b, x0)
                for r in (r1, r2):
                    self.assertEqual(len(r.x), N)
                    self.assertIsInstance(r.iterations, int)
                    self.assertIsInstance(r.final_residual, float)
                    self.assertIsInstance(r.converged, bool)
                    # 残差历史 = 初始残差 + 每次迭代后的残差
                    self.assertEqual(len(r.stats.residual_history),
                                     r.iterations + 1)
                    self.assertEqual(r.final_residual,
                                     r.stats.residual_history[-1])

    def test_iteration_state_backward_compatible(self):
        """旧的三字段构造方式仍然可用（新字段有默认值）。"""
        state = IterationState(iteration=1, residual_norm=1e-9, rhs_norm=1.0)
        self.assertTrue(RelativeResidualCriterion(1e-8).is_converged(state))
        self.assertTrue(AbsoluteResidualCriterion(1e-8).is_converged(state))


if __name__ == "__main__":
    unittest.main()
