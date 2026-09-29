"""预条件 × 收敛判据 组合对比（同一方程组，可复跑）。

用法：
    python3 compare_combinations.py            # 打印对比报告到 stdout
    python3 compare_combinations.py --save docs/comparison_output.txt

对比内容：
    - 迭代次数 / 最终残差 / 是否收敛
    - 残差下降曲线（log10 纵轴 ASCII 图，所有组合叠加）
    - 耗时（每组合重复 REPEAT 次取最小值，降低调度噪声）

可复跑性：矩阵/右端项/初猜均由固定种子生成；判据与预条件为无共享
状态的组件，重复运行结果逐位一致（耗时除外）。

接口语义：所有组合走同一个调用点 solver.solve(A, b, x0)，
切换组合只改 IterativeSolver 的构造参数。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from solver import (AbsoluteResidualCriterion, IterativeSolver,
                    JacobiPreconditioner, RelativeResidualCriterion,
                    ResidualDecreaseRateCriterion, SSORPreconditioner,
                    SolverConfig)
from solver.linalg import norm2, residual
from tests.test_equivalence import make_rhs, make_spd

# ---- 固定问题：同一方程组 Ax = b，同一初猜 x0 ----
N, SEED = 30, 7
MAX_ITER = 5000
OMEGA = 1.0
TOL = 1e-8
REPEAT = 7  # 每组计时重复次数，取最小值

A = make_spd(N, SEED)
b = make_rhs(N, SEED + 1000)
x0 = make_rhs(N, SEED + 2000)  # 非零初猜：让三种判据的基准差异可见

B_NORM = norm2(b)
R0_NORM = norm2(residual(A, b, x0))

PRECONDITIONERS = [
    ("Jacobi", JacobiPreconditioner),
    ("SSOR(1.0)", lambda M: SSORPreconditioner(M, 1.0)),
    ("SSOR(1.2)", lambda M: SSORPreconditioner(M, 1.2)),
]

# 判据阈值对齐到同一量级：相对/下降率为 TOL，绝对判据用 TOL*||b||
CRITERIA = [
    ("relative", lambda: RelativeResidualCriterion(TOL)),
    ("absolute", lambda: AbsoluteResidualCriterion(TOL * B_NORM)),
    ("decrease-rate", lambda: ResidualDecreaseRateCriterion(TOL)),
]


def run_combo(precond_factory, criterion_factory):
    """同一调用点 solve(A, b, x0)；返回 (result, best_ms)。"""
    solver = IterativeSolver(
        config=SolverConfig(max_iterations=MAX_ITER, omega=OMEGA),
        preconditioner_factory=precond_factory,
        criterion=criterion_factory(),
    )
    result, best = None, float("inf")
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        result = solver.solve(A, b, x0)  # 调用点不随组合改变
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return result, best


def ascii_curves(series, width=64, height=14):
    """多条残差曲线叠加的 ASCII 图。series: [(symbol, label, history)]。"""
    import math

    def logs(hist):
        return [math.log10(max(v, 1e-300)) for v in hist]

    # x 轴为真实迭代步数（0 .. 所有曲线中的最大步数），不按各自长度拉伸
    x_max = max(len(hist) for _, _, hist in series) - 1
    sampled = []
    for symbol, label, hist in series:
        ls = logs(hist)
        cols = {}
        for it, v in enumerate(ls):
            c = round(it / max(x_max, 1) * (width - 1))
            cols[c] = v
        sampled.append((symbol, label, cols))

    lo = min(min(cols.values()) for _, _, cols in sampled)
    hi = max(max(cols.values()) for _, _, cols in sampled)
    span = max(hi - lo, 1e-12)

    lines = []
    for row in range(height):
        y = hi - span * row / (height - 1)
        grid = [" "] * width
        for symbol, _, cols in sampled:
            for c, v in cols.items():
                if abs(v - y) <= span / (2 * (height - 1)):
                    grid[c] = symbol if grid[c] == " " else "*"
        lines.append(f"1e{y:+7.1f} |{''.join(grid)}|")
    lines.append(" " * 10 + "+" + "-" * width + "+")
    lines.append(" " * 12 + "iter 0" + " " * (width - 16) + f"iter {x_max}")
    return lines


def main():
    combos = [(pn, pf, cn, cf)
              for pn, pf in PRECONDITIONERS for cn, cf in CRITERIA]

    print(f"problem: n={N} SPD (seed={SEED}), ||b||={B_NORM:.6e}, "
          f"||r0||={R0_NORM:.6e}, omega={OMEGA}, max_iter={MAX_ITER}")
    print(f"timing: best of {REPEAT} runs per combo\n")

    header = (f"{'preconditioner':<15}{'criterion':<15}{'iters':>7}"
              f"{'final_res':>12}{'converged':>11}{'time_ms':>10}")
    print(header)
    print("-" * len(header))

    series, symbols = [], "123456789"
    worst_ratio = 0.0
    for i, (pn, pf, cn, cf) in enumerate(combos):
        result, ms = run_combo(pf, cf)
        hist = result.stats.residual_history
        series.append((symbols[i], f"{pn}+{cn}", hist))
        # 最终残差断言：所有组合都必须把残差压到 TOL 量级以下
        assert result.converged, f"{pn}+{cn} 未收敛"
        limit = TOL * max(B_NORM, R0_NORM)
        assert result.final_residual <= limit, (
            f"{pn}+{cn} 最终残差 {result.final_residual:.3e} 超过 {limit:.3e}")
        worst_ratio = max(worst_ratio, result.final_residual / limit)
        print(f"{pn:<15}{cn:<15}{result.iterations:>7}"
              f"{result.final_residual:>12.3e}{str(result.converged):>11}"
              f"{ms:>10.2f}")

    print("\nresidual curves (log10 ||r|| vs iteration):")
    for symbol, label, _ in series:
        print(f"  {symbol} = {label}")
    print()
    for line in ascii_curves(series):
        print(line)

    print(f"\nfinal-residual assertion: all {len(combos)} combos converged, "
          f"worst final_res / limit = {worst_ratio:.3f} <= 1.0  [PASS]")


if __name__ == "__main__":
    if "--save" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--save") + 1])
        import io
        buf = io.StringIO()
        real = sys.stdout
        sys.stdout = buf
        try:
            main()
        finally:
            sys.stdout = real
        text = buf.getvalue()
        out.write_text(text, encoding="utf-8")
        print(text, end="")
        print(f"[saved] {out}", file=sys.stderr)
    else:
        main()
