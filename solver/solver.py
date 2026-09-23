"""组合层：把配置、预条件、判据、观察者装配成可调用的求解器。

调用方视角保持稳定：solver.solve(A, b, x0) -> SolveResult。
替换预条件/判据只发生在构造期（依赖注入），调用点一行不用改。
"""

from dataclasses import dataclass

from .config import SolverConfig
from .criteria import RelativeResidualCriterion
from .kernel import richardson_kernel
from .preconditioners import JacobiPreconditioner
from .stats import LogObserver, SolveStats, StatsCollector


@dataclass
class SolveResult:
    x: list
    iterations: int
    final_residual: float
    converged: bool
    stats: SolveStats


class IterativeSolver:
    """每次 solve 都是自包含的：新建统计、新建预条件，不碰共享状态。"""

    def __init__(self, *, config=None, preconditioner_factory=None,
                 criterion=None, log_stream=None):
        self.config = config or SolverConfig()
        # 预条件依赖具体矩阵，因此注入的是“工厂”：A -> Preconditioner
        self.preconditioner_factory = (preconditioner_factory
                                       or JacobiPreconditioner)
        self.criterion = criterion or RelativeResidualCriterion(1e-8)
        self.log_stream = log_stream

    def solve(self, A, b, x0=None):
        cfg = self.config
        collector = StatsCollector()
        observers = [collector]
        if cfg.log_every:
            observers.append(LogObserver(cfg.log_every, stream=self.log_stream))

        preconditioner = self.preconditioner_factory(A)
        x, iterations, final_residual, converged = richardson_kernel(
            A, b, x0,
            omega=cfg.omega,
            max_iterations=cfg.max_iterations,
            preconditioner=preconditioner,
            criterion=self.criterion,
            observers=observers,
        )
        stats = SolveStats(iterations=iterations,
                           final_residual=final_residual,
                           converged=converged,
                           residual_history=collector.residual_history)
        return SolveResult(x=x, iterations=iterations,
                           final_residual=final_residual,
                           converged=converged, stats=stats)
