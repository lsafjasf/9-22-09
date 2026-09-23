# 结构对照说明

## 1. 模块职责与调用顺序

```
调用方
  │  solver.solve(A, b, x0)                ← 唯一稳定入口，签名不变
  ▼
solver/solver.py      组合层：装配本次求解所需的一切（每次 solve 新建统计与预条件）
  │  ① preconditioner = factory(A)         ← 预条件只依赖矩阵，构造期注入工厂
  │  ② observers = [StatsCollector, LogObserver?]
  ▼
solver/kernel.py      迭代内核：只编排循环骨架，每轮迭代顺序固定为
  │  1. r = b − A·x            （linalg.residual）
  │  2. observer.on_iteration(k, ‖r‖)      ← 统计/日志在此挂接
  │  3. criterion.is_converged(state)      ← 收敛则退出
  │  4. z = preconditioner.apply(r)
  │  5. x ← x + ω·z
  ▼
solver/linalg.py      纯函数数值原语（matvec / residual / norm2 / axpy）
```

| 模块 | 职责 | 不知道（也不许知道） |
|---|---|---|
| `linalg.py` | 向量/矩阵运算 | 迭代、判据、统计的存在 |
| `config.py` | 不可变运行参数（`frozen` dataclass） | 谁在使用它 |
| `preconditioners.py` | 由 `r` 算 `z ≈ A⁻¹r` | 迭代进行到第几步、是否收敛 |
| `criteria.py` | 由迭代快照判断是否收敛 | 残差怎么算出来、解怎么更新 |
| `kernel.py` | 编排迭代循环骨架 | 统计如何汇总、日志写去哪、判据/预条件的具体算法 |
| `stats.py` | 残差历史收集、日志输出（观察者） | 迭代算法本身 |
| `solver.py` | 依赖注入 + 稳定调用接口 | 各组件的内部实现 |

替换预条件或判据 = 构造 `IterativeSolver` 时换注入参数，调用点
`solver.solve(A, b)` 一行不改（见 `tests/test_equivalence.py::TestPluggability`）。

## 2. 隐式耦合 → 缺陷 → 新结构对应表

| # | 旧实现的隐式耦合 | 导致的缺陷 | 新结构对应 |
|---|---|---|---|
| 1 | 配置是模块级全局变量 `TOLERANCE/MAX_ITER/OMEGA/LOG_EVERY`，`solve()` 运行时读取 | 同进程两个任务想要不同配置必须“改全局→求解→再改回”，并发时互相覆盖；配置在任意时刻可被第三方改写，行为不可复现 | `SolverConfig` 为 `frozen` dataclass，每个 `IterativeSolver` 实例持有一份，创建后不可变 |
| 2 | 统计挂在**类属性** `IterativeSolver.stats` 上 | 所有实例、所有线程共享同一份可变 dict；多次 `solve` 的残差历史混在一个列表里，`runs`/`total_iterations` 跨任务累积，无法得到单次运行的干净统计（`tests/...::test_legacy_shared_stats_defect_is_documented` 反向验证该缺陷） | `SolveStats`/`StatsCollector` 每次 `solve` 新建，随 `SolveResult` 返回；进程内无任何共享可变状态 |
| 3 | Jacobi 预条件硬编码在迭代循环内 | 换预条件只能复制整个文件再改，两份代码随之发散；预条件逻辑无法单独测试 | `Preconditioner` 协议（`apply(r)→z`），提供 `Identity`/`Jacobi`/`SSOR` 三种实现，构造期以工厂注入 |
| 4 | 收敛判据（相对残差）硬编码在循环内 | 换判据同样要复制文件；判据无法单独测试 | `ConvergenceCriterion` 协议（`is_converged(state)→bool`），提供相对/绝对残差两种实现；判据是纯函数对象，可并发共享 |
| 5 | 日志直接 `print` 到全局 stdout | 多任务并发时输出交错无法归属；无法静默或重定向到文件 | `LogObserver` 构造时注入 `stream`，每个任务可各写各的流 |
| 6 | 向量/矩阵运算、迭代、统计、日志全在一个方法里 | 任何小改动都要动整个类；运算原语无法复用与单测 | `linalg` 纯函数 + `kernel` 只编排骨架，各层可独立测试 |

## 3. 数值等价性如何保证

- 新内核的每步运算与旧实现**同序同式**：`r = b − A·x` → 记录 → 判据
  `‖r‖ ≤ tol·‖b‖` → `z_i = r_i / A_ii` → `x ← x + ω·z`。
- Jacobi 预条件保存对角元本身并做除法，而非预存倒数做乘法
  （`r/d` 与 `r*(1/d)` 可能差 1 ulp），保证逐位一致。
- 对拍测试（`TestEquivalence`）在 5 组用例（含不收敛走满 `max_iter`
  的分支）上断言：迭代次数完全相等、最终残差与残差历史在 1e-14
  容差内一致、解向量在 1e-12 容差内一致（实际运行为逐位相同）。
