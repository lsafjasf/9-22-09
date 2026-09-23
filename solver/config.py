"""求解配置：不可变、按实例持有，杜绝全局可变配置。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverConfig:
    """一次求解任务的运行参数。

    frozen=True 保证配置在创建后不可被意外改写；
    每个 IterativeSolver 实例持有自己的一份，任务之间互不影响。
    """

    max_iterations: int = 1000
    omega: float = 1.0        # Richardson 松弛因子
    log_every: int = 0        # 0 = 不输出日志；>0 = 每隔多少步输出一次
