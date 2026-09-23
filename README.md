# 迭代求解器重构（PCG：预条件共轭梯度）

纯 Python 3 标准库实现，无第三方依赖。`legacy/` 是重构前的单体实现（刻意保留缺陷），
`refactored/` 是重构后的分层实现，`tests/` 包含新旧等价对拍、判据替换、并发隔离与
旧版缺陷表征测试。

## 目录

```
legacy/
  pcg_jacobi.py          旧版单体类（Jacobi）：全局配置 + 类级共享统计
  pcg_ssor.py            换预条件时整文件复制出来的副本（SSOR）
refactored/
  linalg.py              边界1：向量/矩阵运算（纯函数，无状态）
  preconditioners.py     边界2：预条件（Identity/Jacobi/SSOR + 注册表）
  criteria.py            边界3：收敛判据（相对b / 相对r0 / 绝对残差 + 注册表）
  kernel.py              边界4：PCG 迭代内核（只编排，不持有状态）
  stats.py               边界5：每次求解独立的统计 RunStats
  logging.py             边界6：日志接口 Null/Stream/ListLogger
  config.py              不可变每任务配置 SolverConfig（frozen dataclass）
  solver.py              边界7：门面，按 solve() 组装上述组件
tests/
  problems.py            确定性 SPD 测试问题（三对角 Laplacian、稠密 SPD）
  test_equivalence.py    新旧对拍：迭代次数、逐次残差、解向量
  test_criteria.py       判据可替换；与独立参考 PCG 对拍
  test_isolation.py      线程池并发两任务统计/日志互不干扰
  test_legacy_defects.py 旧版缺陷的可执行证据
examples/demo.py         自定义预条件+自定义判据、并发两任务示例
run_tests.py             一键测试入口
```

## 接口边界、职责与调用顺序

```
调用方
  │  IterativeSolver(A, b, SolverConfig(...)).solve(x0, logger=...)
  ▼
solver.IterativeSolver（门面，无共享可变状态）
  │  每个 solve() 内部新建：预条件实例、判据实例、RunStats、日志
  ▼
criteria.bind(CriterionContext)          ← 每次求解绑定一次（r0、||b||）
  │
  ▼
kernel.pcg_solve（迭代编排）
  │  每次迭代严格按以下顺序：
  │  1. linalg.matvec(A, p)             （统计 matvecs）
  │  2. linalg.dot / axpy 更新 x、r
  │  3. linalg.dot 计算残差范数
  │  4. stats.record_iteration + logger.iteration
  │  5. criterion.met(k, ||r||) 判定，收敛则 stats/logger.finish 返回
  │  6. preconditioner.apply(r)
  │  7. linalg.dot 更新 beta 与搜索方向 p
  ▼
返回 (x, RunStats)
```

- **边界1 `linalg.py`**：`dot / axpy / scale / add / matvec / matrix_diag`，
  全部为纯函数；矩阵支持可调用对象、稠密行表（list）、对称三对角元组
  `(diag, offdiag)` 三种形态。
- **边界2 预条件**：统一接口 `Preconditioner.apply(r) -> z`。内置
  `Identity`（无预条件）、`Jacobi`（对角）、`SSOR`（对称 SOR，ω 可配）。
  注册表 `register_preconditioner(name, cls)`，`SolverConfig(preconditioner=...)`
  接受名字/类/实例/可调用对象，**调用方代码不变即可替换**。
- **边界3 判据**：`bind(context)` + `met(iteration, residual_norm)`。
  内置两种以上：`RelativeToRHS`（`||r||<=tol*||b||`，与旧版完全一致的判据）、
  `InitialResidualReduction`（`||r||<=factor*||r0||`）、`AbsoluteResidual`。
  同样支持注册表替换。
- **边界4 `kernel.py`**：不含任何矩阵格式判断以外的逻辑，不知道日志写哪、
  计数存哪、容差是多少；这些全部由参数注入。
- **边界5 统计**：`RunStats` 是每次 `solve()` 新建的 dataclass，含迭代数、
  matvec 数、初始/最终残差、逐次残差历史、收敛状态与停止原因。
- **边界6 日志**：`SolverLogger` 协议（`start/iteration/finish`），默认
  `NullLogger`，另有 `StreamLogger`、`ListLogger`；每任务传入各自实例。
- **配置**：`SolverConfig` 为 frozen dataclass，随任务传入，不存在全局旋钮。

## 数值等价性

重构没有改变算法与运算顺序：`dot` 的求和次序、`axpy` 的原地更新、残差记录
位置、判据判定时机、Jacobi 与 SSOR 的逐行公式在新旧两侧逐行对应（SSOR 反向
扫描两侧使用同一正确公式 `y_i = ω/d_i·(t_i − upper_i·y_{i+1})`）。

`tests/test_equivalence.py` 在 4 个确定性 SPD 问题（2 个三对角、2 个稠密）上
分别对 Jacobi、SSOR 断言：

- 迭代次数完全相等；
- 逐次残差相对误差 ≤ 1e-10；
- 解向量相对误差 ≤ 1e-9。

## 旧版隐式耦合 → 缺陷 → 新结构对照

| 旧版耦合点（位置） | 导致的缺陷 | 新结构对应 |
|---|---|---|
| 模块级 `TOL/MAX_ITER/VERBOSE/HISTORY`（`pcg_jacobi.py` 顶部） | 一个任务改容差/上限/日志开关会泄漏到同进程所有任务；测试间必须手动复位 | `config.py` 的 frozen `SolverConfig` 按任务传入；`test_legacy_defects.py::test_legacy_global_tol_change_leaks_between_tasks` 复现旧行为 |
| 类属性 `iter_count/matvec_count` | 所有实例共享计数器，后跑的任务覆盖前跑的结果，无法回答"第一个任务跑了几次" | 每次 `solve()` 新建 `stats.RunStats` 并返回；见 `test_legacy_class_counters_are_shared` |
| 全局 `HISTORY` 列表，`solve()` 直接 `HISTORY.append` | 多次求解残差历史首尾相接，实例属性 `self.history` 也指向同一对象，并发/连续任务互相污染 | `RunStats.residual_history` 每次新建，长度恒等于本任务迭代数；并发隔离见 `test_concurrent_solves_keep_independent_stats` |
| 预条件逻辑是类方法 `_precondition` | 换预条件只能整文件复制（`pcg_ssor.py` 即如此），两份文件的迭代代码此后各自漂移、易不一致 | `preconditioners.py` 统一接口 + 注册表；内核零改动即可按名字/实例切换 |
| 收敛判定硬编码 `rnorm <= TOL*bnorm` | 无法换判据（如相对 `r0`、绝对残差），改判据要动迭代循环 | `criteria.py` 策略对象，每次求解 `bind` 上下文；`test_criteria.py` 验证两种判据给出不同迭代次数 |
| 日志靠全局 `VERBOSE` + 直接 `print` | 无法静默、无法重定向、并发任务输出交错 | `logging.py` 的 `SolverLogger` 接口按任务注入，默认 `NullLogger` |
| 代数运算、矩阵分派、求解流程混在同一个类 | 无法单独测试/复用任一环节，矩阵类型判断散落在多个方法中 | `linalg.py` 纯函数集中所有代数与矩阵分派；`kernel.py` 只负责编排 |
| 预条件对象无生命周期（每次调用临时解析 A） | 每次迭代重复提取对角元，且无法持有预计算状态 | 预条件在 `solve()` 开始时构造一次、迭代中复用（SSOR 只扫一次结构） |

## 扩展方式（不改调用方）

```python
from refactored import (IterativeSolver, SolverConfig,
                        Preconditioner, register_preconditioner)

class MyPre(Preconditioner):
    def __init__(self, A): ...
    def apply(self, r): ...

register_preconditioner("my_pre", MyPre)
x, stats = IterativeSolver(A, b,
    SolverConfig(preconditioner="my_pre", criterion="initial_reduction")
).solve(x0)
```

判据同理：实现 `ConvergenceCriterion.bind/met` 后
`register_criterion("my_rule", MyCriterion)` 或直接传实例。

## 运行命令

```bash
# 全部测试（等价对拍 + 判据 + 隔离 + 旧版缺陷表征）
python3 run_tests.py
# 等价于
python3 -m unittest discover -s tests -v

# 只跑新旧对拍
python3 -m unittest tests.test_equivalence -v

# 扩展示例（自定义预条件/判据 + 同进程并发两任务）
python3 examples/demo.py
```

要求 Python 3.10+（开发验证环境：Python 3.12），仅使用标准库。
