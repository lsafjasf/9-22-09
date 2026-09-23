"""重构后的迭代求解器包。

分层（依赖只允许向下）：
    linalg          向量/矩阵纯函数
    config          不可变配置
    preconditioners 预条件（可插拔）
    criteria        收敛判据（可插拔）
    stats           统计/日志观察者（每次求解独立）
    kernel          迭代内核（编排循环骨架）
    solver          组合层（依赖注入 + 稳定调用接口）
"""

from .config import SolverConfig
from .criteria import (AbsoluteResidualCriterion, IterationState,
                       RelativeResidualCriterion)
from .preconditioners import (IdentityPreconditioner, JacobiPreconditioner,
                              SSORPreconditioner)
from .solver import IterativeSolver, SolveResult
from .stats import LogObserver, SolveStats, StatsCollector

__all__ = [
    "SolverConfig",
    "IterationState",
    "RelativeResidualCriterion",
    "AbsoluteResidualCriterion",
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "SSORPreconditioner",
    "IterativeSolver",
    "SolveResult",
    "SolveStats",
    "StatsCollector",
    "LogObserver",
]
