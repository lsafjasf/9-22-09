"""收敛判据：只负责判断“当前迭代状态是否收敛”。

接口约定（隐式协议）：
    class ConvergenceCriterion(Protocol):
        def is_converged(self, state: IterationState) -> bool: ...

判据是无状态纯函数对象：同样的 state 永远给出同样的结论，
因此可以被任意多个求解任务并发共享。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IterationState:
    """一次迭代结束时的快照，作为判据的唯一输入。"""

    iteration: int          # 当前迭代步数（0 = 初始残差）
    residual_norm: float    # ||r||_2
    rhs_norm: float         # ||b||_2


class RelativeResidualCriterion:
    """相对残差判据：||r|| <= tol * ||b||（与旧实现硬编码的判据一致）。"""

    def __init__(self, tolerance):
        self.tolerance = tolerance

    def is_converged(self, state):
        return state.residual_norm <= self.tolerance * state.rhs_norm


class AbsoluteResidualCriterion:
    """绝对残差判据：||r|| <= tol。适用于 b 的量级已知的场景。"""

    def __init__(self, tolerance):
        self.tolerance = tolerance

    def is_converged(self, state):
        return state.residual_norm <= self.tolerance
