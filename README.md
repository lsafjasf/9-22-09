# 迭代求解器重构（Python 3，仅标准库）

## 目录

- `legacy_solver.py` — 重构前基线（单类 + 全局配置 + 共享统计，缺陷见文件头注释）
- `solver/` — 重构后包（linalg / config / preconditioners / criteria / kernel / stats / solver）
- `compare_combinations.py` — 预条件 × 判据组合对比（同一方程组上枚举 3×3 组合，
  输出迭代次数 / 残差曲线 / 耗时，并对每个组合做最终残差断言）
- `tests/test_equivalence.py` — 新旧对拍 + 并发隔离 + 可插拔性测试
- `tests/test_combinations.py` — 组合扫描（9 种组合全收敛断言）+ 判据单元测试 + 接口语义测试
- `docs/ARCHITECTURE.md` — 职责边界、调用顺序、隐式耦合→缺陷→新结构对照表
- `docs/CRITERIA.md` — 收敛判据适用性说明（绝对 / 相对 / 残差下降率）

## 运行

```bash
# 全部测试（对拍 + 并发隔离 + 可插拔 + 组合扫描）
python3 -m unittest discover -s tests -v

# 组合对比（可复跑验证：任一组合不满足最终残差断言则退出码为 1）
python3 compare_combinations.py

# 简单使用
python3 - << 'PY'
from solver import IterativeSolver, SolverConfig, SSORPreconditioner, RelativeResidualCriterion
A = [[4.0, 1.0], [1.0, 3.0]]
b = [1.0, 2.0]
solver = IterativeSolver(
    config=SolverConfig(max_iterations=100, omega=1.0),
    preconditioner_factory=lambda M: SSORPreconditioner(M, 1.0),
    criterion=RelativeResidualCriterion(1e-10),
)
r = solver.solve(A, b)
print(r.iterations, f"{r.final_residual:.3e}", r.converged)
PY
```
