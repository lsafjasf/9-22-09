"""统计与日志：以观察者身份挂到迭代内核上，每次求解独立实例。

接口约定（隐式协议）：
    class Observer(Protocol):
        def on_iteration(self, iteration: int, residual_norm: float) -> None: ...

关键点：统计对象是“每次求解一份”，由调用方（Solver.solve）创建，
不存在任何跨任务共享的可变状态。
"""

import sys
from dataclasses import dataclass, field


@dataclass
class SolveStats:
    """单次求解的完整统计，随 SolveResult 一起返回。"""

    iterations: int = 0
    final_residual: float = None
    converged: bool = False
    residual_history: list = field(default_factory=list)


class StatsCollector:
    """观察者：收集残差历史，供求解结束后汇总进 SolveStats。"""

    def __init__(self):
        self.residual_history = []

    def on_iteration(self, iteration, residual_norm):
        self.residual_history.append(residual_norm)


class LogObserver:
    """观察者：按固定间隔把进度写到指定流（默认 stdout）。

    流是构造时注入的，因此不同任务可以各写各的文件/缓冲，
    并发时不会互相串行污染（旧实现直接 print 到全局 stdout）。
    """

    def __init__(self, every, stream=None, tag="solver"):
        self.every = every
        self.stream = stream if stream is not None else sys.stdout
        self.tag = tag

    def on_iteration(self, iteration, residual_norm):
        if self.every and iteration % self.every == 0:
            print(f"[{self.tag}] iter={iteration} residual={residual_norm:.6e}",
                  file=self.stream)
