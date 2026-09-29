"""收敛判据：只负责判断“当前迭代状态是否收敛”。

接口约定（隐式协议）：
    class ConvergenceCriterion(Protocol):
        def is_converged(self, state: IterationState) -> bool: ...

判据是无状态纯函数对象：同样的 state 永远给出同样的结论，
因此可以被任意多个求解任务并发共享。
判据所需的全部历史信息（初始残差、上一步残差）都由内核
快照进 IterationState，判据自身不保存任何跨调用状态。

三种判据的适用性详见 docs/CRITERIA.md。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IterationState:
    """一次迭代结束时的快照，作为判据的唯一输入。

    initial_residual_norm / previous_residual_norm 由内核在迭代
    过程中填充；iteration=0（初始残差）时 previous 为 None。
    带默认值是为了不破坏既有调用方手工构造 state 的用法。
    """

    iteration: int          # 当前迭代步数（0 = 初始残差）
    residual_norm: float    # ||r_k||_2
    rhs_norm: float         # ||b||_2
    initial_residual_norm: float = None   # ||r_0||_2
    previous_residual_norm: float = None  # ||r_{k-1}||_2，k=0 时为 None


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


class ResidualDecreaseRateCriterion:
    """残差下降率判据：平均几何下降率 (||r_k||/||r_0||)^(1/k) <= rate。

    刻画“迭代是否足够快地把残差压下去”，而不是“残差是否足够小”。
    rate 取值 (0, 1)，越小越严格；min_iterations 保证至少迭代若干步，
    避免初始波动导致过早下结论。
    """

    def __init__(self, rate, min_iterations=1):
        if not 0.0 < rate < 1.0:
            raise ValueError("rate 必须在 (0, 1) 内")
        if min_iterations < 1:
            raise ValueError("min_iterations 必须 >= 1")
        self.rate = rate
        self.min_iterations = min_iterations

    def is_converged(self, state):
        if state.residual_norm == 0.0:
            return True
        if state.iteration < self.min_iterations:
            return False
        initial = state.initial_residual_norm
        if not initial:  # 初始残差为 0（或旧调用方未提供）：已精确收敛
            return True
        avg_rate = (state.residual_norm / initial) ** (1.0 / state.iteration)
        return avg_rate <= self.rate
