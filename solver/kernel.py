"""迭代内核：预条件 Richardson 迭代。

职责边界：内核只编排“算残差 → 问判据 → 应用预条件 → 更新解”的
循环骨架，并通过观察者回调向外汇报进度；它不知道统计如何汇总、
日志写到哪里、判据的具体规则、预条件的具体算法。

调用顺序（每轮迭代）：
    1. r = b - A x                      （linalg.residual）
    2. 通知所有 observer: on_iteration  （统计/日志在此挂接）
    3. criterion.is_converged(state)    （收敛则退出）
    4. z = preconditioner.apply(r)
    5. x = x + omega * z
"""

from .criteria import IterationState
from .linalg import norm2, residual


def richardson_kernel(A, b, x0, *, omega, max_iterations,
                      preconditioner, criterion, observers=()):
    """执行迭代，返回 (x, iterations, final_residual_norm, converged)。

    所有参数显式传入，不读任何全局状态；多次调用互不干扰。
    """
    x = list(x0) if x0 is not None else [0.0] * len(b)
    rhs_norm = norm2(b)

    iterations = 0
    final_residual = None
    converged = False
    for k in range(max_iterations + 1):
        r = residual(A, b, x)
        r_norm = norm2(r)
        final_residual = r_norm

        for obs in observers:
            obs.on_iteration(k, r_norm)

        state = IterationState(iteration=k, residual_norm=r_norm,
                               rhs_norm=rhs_norm)
        if criterion.is_converged(state):
            iterations = k
            converged = True
            break
        if k == max_iterations:
            iterations = max_iterations
            break

        z = preconditioner.apply(r)
        x = [x[i] + omega * z[i] for i in range(len(x))]
        iterations = k + 1

    return x, iterations, final_residual, converged
