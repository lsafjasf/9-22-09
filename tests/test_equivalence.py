"""新旧实现对拍测试 + 并发隔离测试。

运行：python3 -m unittest discover -s tests -v   （仓库根目录下）
"""

import random
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import legacy_solver
from solver import (AbsoluteResidualCriterion, IterativeSolver,
                    JacobiPreconditioner, RelativeResidualCriterion,
                    SSORPreconditioner, SolverConfig)


def make_spd(n, seed):
    """确定性生成对称严格对角占优矩阵（保证 Richardson+Jacobi 收敛）。"""
    rng = random.Random(seed)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        off = [rng.uniform(-0.5, 0.5) for _ in range(n)]
        for j in range(n):
            if i != j:
                A[i][j] = off[j]
        A[i][i] = sum(abs(v) for v in off) + rng.uniform(0.5, 1.5)
    for i in range(n):  # 对称化
        for j in range(i + 1, n):
            v = 0.5 * (A[i][j] + A[j][i])
            A[i][j] = A[j][i] = v
    return A


def make_rhs(n, seed):
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def run_legacy(A, b, x0, tol, max_iter, omega):
    """按旧实现的方式调用：先改全局配置，再 solve。"""
    legacy_solver.TOLERANCE = tol
    legacy_solver.MAX_ITER = max_iter
    legacy_solver.OMEGA = omega
    legacy_solver.LOG_EVERY = 0
    before = len(legacy_solver.IterativeSolver.stats["residual_history"])
    x, iters, res = legacy_solver.IterativeSolver(A, b).solve(x0)
    history = legacy_solver.IterativeSolver.stats["residual_history"][before:]
    return x, iters, res, history


def run_new(A, b, x0, tol, max_iter, omega):
    solver = IterativeSolver(
        config=SolverConfig(max_iterations=max_iter, omega=omega),
        preconditioner_factory=JacobiPreconditioner,
        criterion=RelativeResidualCriterion(tol),
    )
    return solver.solve(A, b, x0)


CASES = [  # (n, seed, tol, max_iter, omega, use_x0)
    (5, 1, 1e-8, 1000, 1.0, False),
    (20, 7, 1e-10, 2000, 1.0, False),
    (30, 42, 1e-6, 500, 0.7, True),
    (12, 99, 1e-12, 5000, 1.2, True),
    (8, 3, 1e-4, 50, 1.0, False),   # 故意不收敛，走到 max_iter 分支
]


class TestEquivalence(unittest.TestCase):
    """相同输入 + 相同预条件（Jacobi）+ 相同判据（相对残差）下逐位对拍。"""

    def test_iteration_and_residual_match(self):
        for n, seed, tol, max_iter, omega, use_x0 in CASES:
            with self.subTest(n=n, seed=seed, omega=omega):
                A, b = make_spd(n, seed), make_rhs(n, seed + 1000)
                x0 = make_rhs(n, seed + 2000) if use_x0 else None

                lx, li, lres, lhist = run_legacy(A, b, x0, tol, max_iter, omega)
                result = run_new(A, b, x0, tol, max_iter, omega)

                # 迭代次数必须完全一致
                self.assertEqual(li, result.iterations)
                # 最终残差允许浮点容差（实际运行为逐位一致）
                self.assertAlmostEqual(lres, result.final_residual, places=14)
                # 解向量一致
                for a, c in zip(lx, result.x):
                    self.assertAlmostEqual(a, c, places=12)
                # 残差历史逐项一致
                self.assertEqual(len(lhist),
                                 len(result.stats.residual_history))
                for a, c in zip(lhist, result.stats.residual_history):
                    self.assertAlmostEqual(a, c, places=14)


class TestSharedStateEliminated(unittest.TestCase):
    """同一进程内多个任务并行：统计与配置互不干扰。"""

    def test_concurrent_solves_are_isolated(self):
        jobs = [(n, seed, tol, mi, om, x0) for n, seed, tol, mi, om, x0 in CASES]
        A_b = [(make_spd(n, s), make_rhs(n, s + 1000))
               for n, s, *_ in jobs]

        # 串行基准
        expected = [run_new(A, b, None, tol, mi, om)
                    for (A, b), (n, s, tol, mi, om, _) in zip(A_b, jobs)]

        # 并发执行（不同配置混跑）
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(run_new, A, b, None, tol, mi, om)
                       for (A, b), (n, s, tol, mi, om, _) in zip(A_b, jobs)]
            actual = [f.result() for f in futures]

        for exp, act in zip(expected, actual):
            self.assertEqual(exp.iterations, act.iterations)
            self.assertEqual(exp.final_residual, act.final_residual)
            self.assertEqual(exp.stats.residual_history,
                             act.stats.residual_history)

    def test_legacy_shared_stats_defect_is_documented(self):
        """反向验证旧缺陷：旧实现的统计确实跨实例/跨运行污染。"""
        legacy_solver.IterativeSolver.stats = {
            "runs": 0, "total_iterations": 0, "last_iterations": None,
            "last_residual": None, "residual_history": [],
        }
        A, b = make_spd(5, 1), make_rhs(5, 1001)
        run_legacy(A, b, None, 1e-8, 1000, 1.0)
        run_legacy(A, b, None, 1e-8, 1000, 1.0)
        stats = legacy_solver.IterativeSolver.stats
        # 两次运行混在同一份 stats 里，无法区分各自的历史
        self.assertEqual(stats["runs"], 2)
        self.assertGreater(stats["total_iterations"],
                           stats["last_iterations"])


class TestPluggability(unittest.TestCase):
    """替换预条件/判据不需要改调用点：调用点永远是 solver.solve(A, b)。"""

    def test_swap_preconditioner_and_criterion(self):
        A, b = make_spd(15, 5), make_rhs(15, 1005)

        combos = [
            dict(preconditioner_factory=JacobiPreconditioner,
                 criterion=RelativeResidualCriterion(1e-8)),
            dict(preconditioner_factory=lambda M: SSORPreconditioner(M, 1.0),
                 criterion=RelativeResidualCriterion(1e-8)),
            dict(preconditioner_factory=JacobiPreconditioner,
                 criterion=AbsoluteResidualCriterion(1e-6)),
            dict(preconditioner_factory=lambda M: SSORPreconditioner(M, 1.2),
                 criterion=AbsoluteResidualCriterion(1e-6)),
        ]
        for kw in combos:
            with self.subTest(**{k: type(v).__name__ for k, v in kw.items()}):
                solver = IterativeSolver(
                    config=SolverConfig(max_iterations=2000), **kw)
                result = solver.solve(A, b)  # 调用点不变
                self.assertTrue(result.converged)
                self.assertLessEqual(result.stats.iterations, 2000)


if __name__ == "__main__":
    unittest.main()
