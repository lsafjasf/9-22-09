"""预条件 × 收敛判据 组合对比（可复跑验证脚本）。

在同一个方程组（固定随机种子的对称严格对角占优矩阵）上，枚举
{Identity, Jacobi, SSOR} × {绝对, 相对, 残差下降率} 共 9 种组合，
输出每种组合的迭代次数、最终残差、残差曲线与耗时，并对每个组合
做与其判据语义一致的最终残差断言（任一失败则退出码为 1）。

运行：python3 compare_combinations.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver import (AbsoluteResidualCriterion, IdentityPreconditioner,
                    IterativeSolver, JacobiPreconditioner,
                    RelativeResidualCriterion, ResidualDecreaseRateCriterion,
                    SolverConfig, SSORPreconditioner)
from solver.linalg import norm2
from tests.test_equivalence import make_rhs, make_spd

N, SEED = 12, 2026
MAX_ITER = 5000
REPEATS = 7          # 每种组合重复求解次数，取最优/均值耗时
TOL = 1e-8
RATE = 0.5           # 下降率判据阈值：平均几何下降率 <= 0.5
MIN_ITERS = 5        # 下降率判据最少迭代步数

SPARKS = "▁▂▃▄▅▆▇█"


def gershgorin_lambda_max(A):
    """Gershgorin 圆盘上界：lambda_max <= max_i (a_ii + sum_{j!=i} |a_ij|)。"""
    return max(A[i][i] + sum(abs(A[i][j]) for j in range(len(A)) if j != i)
               for i in range(len(A)))


def sparkline(history, width=48):
    """把残差历史压成一行 log 刻度 sparkline（高位=残差大）。"""
    import math
    logs = [math.log10(max(h, 1e-300)) for h in history]
    if len(logs) > width:  # 均匀抽样到 width 个点
        step = (len(logs) - 1) / (width - 1)
        logs = [logs[round(i * step)] for i in range(width)]
    lo, hi = min(logs), max(logs)
    span = (hi - lo) or 1.0
    return "".join(SPARKS[min(7, int((v - lo) / span * 7.999))] for v in logs)


def run_combo(A, b, precond_factory, criterion, omega):
    solver = IterativeSolver(
        config=SolverConfig(max_iterations=MAX_ITER, omega=omega),
        preconditioner_factory=precond_factory,
        criterion=criterion,
    )
    result = solver.solve(A, b)  # 调用点对所有组合完全一致
    times = []
    for _ in range(REPEATS):     # 计时运行（结果与首次一致，仅测耗时）
        t0 = time.perf_counter()
        solver.solve(A, b)
        times.append((time.perf_counter() - t0) * 1e3)
    return result, min(times), sum(times) / len(times)


def main():
    A, b = make_spd(N, SEED), make_rhs(N, SEED + 1000)
    b_norm = norm2(b)
    omega_identity = 1.8 / gershgorin_lambda_max(A)  # Richardson 收敛安全值

    preconds = [
        ("identity", IdentityPreconditioner, omega_identity),
        ("jacobi", JacobiPreconditioner, 1.0),
        ("ssor(w=1.0)", lambda M: SSORPreconditioner(M, 1.0), 1.0),
    ]
    criteria = [
        ("absolute", AbsoluteResidualCriterion(TOL)),
        ("relative", RelativeResidualCriterion(TOL)),
        ("decrease-rate", ResidualDecreaseRateCriterion(RATE, MIN_ITERS)),
    ]

    print(f"方程组: n={N}, seed={SEED}, ||b||={b_norm:.6f}, "
          f"max_iter={MAX_ITER}")
    print(f"omega: identity={omega_identity:.4f} (Gershgorin 安全值), "
          f"jacobi/ssor=1.0")
    print()

    header = (f"{'preconditioner':<15}{'criterion':<15}{'conv':<6}"
              f"{'iters':>6} {'final_res':>11} {'t_min_ms':>9} {'t_avg_ms':>9}")
    print(header + "   residual curve (log10)")
    print("-" * len(header) + "   " + "-" * 48)

    failures = []
    for p_name, p_factory, omega in preconds:
        for c_name, criterion in criteria:
            result, t_min, t_avg = run_combo(A, b, p_factory, criterion, omega)
            hist = result.stats.residual_history
            print(f"{p_name:<15}{c_name:<15}{str(result.converged):<6}"
                  f"{result.iterations:>6} {result.final_residual:>11.3e}"
                  f" {t_min:>9.2f} {t_avg:>9.2f}   {sparkline(hist)}")

            # —— 与判据语义一致的最终残差断言 ——
            ok = result.converged
            if c_name == "absolute":
                ok &= result.final_residual <= TOL
            elif c_name == "relative":
                ok &= result.final_residual <= TOL * b_norm
            else:
                r0, k = hist[0], result.iterations
                ok &= (result.final_residual / r0) ** (1.0 / k) <= RATE
            if not ok:
                failures.append((p_name, c_name))

    print()
    if failures:
        print("FAILED 组合:", failures)
        return 1
    print(f"OK: 全部 {len(preconds) * len(criteria)} 个组合收敛，"
          f"且满足各自判据的最终残差断言。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
