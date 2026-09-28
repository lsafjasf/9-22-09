# 新旧实现对拍报告（冻结基线）

本文件由 `python3 tools/generate_report.py` 自动生成，请勿手改。

- 基线：`baseline/expected.json`（由 `tools/freeze_baseline.py` 从 `legacy/` 实测冻结），输入完整性见 `baseline/SHA256SUMS`。
- 对照三方：冻结基线 / 当前 `legacy/` 复算 / 当前 `refactored/`。
- 判据：迭代次数相等；逐次残差历史与解向量对基线位级一致（强于重构前文档声称的 1e-10 / 1e-9）。
- 运行环境：Python 3.12.3 (Linux)；纯标准库、固定种子、固定求和顺序，重跑结果可复现。

## 结论

- 对拍用例数：8（2 种预条件 x 4 个确定性 SPD 问题）
- 迭代次数与基线完全一致：8/8
- 逐次残差历史与基线位级一致（bit-identical）：8/8
- 新版 vs 冻结基线 逐次残差最大相对误差：0 (bit-identical)
- 新版 vs 冻结基线 解向量最大相对误差：0 (bit-identical)
- 当前 legacy 代码复算与冻结基线位级一致：是
- 结论：**行为等价成立（位级一致）**

## 逐用例对照

| 用例 | 基线迭代 | 旧版迭代 | 新版迭代 | 新旧逐次残差最大相对差 | 新旧解向量最大相对差 |
|---|---|---|---|---|---|
| jacobi/laplacian_n40 | 40 | 40 | 40 | 0 (bit-identical) | 0 (bit-identical) |
| jacobi/laplacian_n120 | 120 | 120 | 120 | 0 (bit-identical) | 0 (bit-identical) |
| jacobi/dense_spd_n15 | 14 | 14 | 14 | 0 (bit-identical) | 0 (bit-identical) |
| jacobi/dense_spd_n30 | 23 | 23 | 23 | 0 (bit-identical) | 0 (bit-identical) |
| ssor/laplacian_n40 | 19 | 19 | 19 | 0 (bit-identical) | 0 (bit-identical) |
| ssor/laplacian_n120 | 42 | 42 | 42 | 0 (bit-identical) | 0 (bit-identical) |
| ssor/dense_spd_n15 | 15 | 15 | 15 | 0 (bit-identical) | 0 (bit-identical) |
| ssor/dense_spd_n30 | 24 | 24 | 24 | 0 (bit-identical) | 0 (bit-identical) |

## 复现

```bash
python3 tools/freeze_baseline.py --verify   # 基线无漂移
python3 tools/generate_report.py --check    # 报告/README 与实测一致
python3 run_tests.py                        # 含 tests/test_baseline.py
```
