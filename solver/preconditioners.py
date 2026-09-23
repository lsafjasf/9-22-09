"""预条件子：只负责给定残差 r 计算 z ≈ A^{-1} r。

接口约定（隐式协议）：
    class Preconditioner(Protocol):
        def apply(self, r: list[float]) -> list[float]: ...

预条件子在构造时从矩阵 A 提取所需数据（如对角元），
之后对 apply 的多次调用只读不写，可安全地被并发任务共享使用
（本实现中每个求解任务各自构造，连共享都不需要）。
"""


class IdentityPreconditioner:
    """不做事的预条件：z = r。用于对照实验。"""

    def __init__(self, A):
        pass

    def apply(self, r):
        return list(r)


class JacobiPreconditioner:
    """Jacobi（对角）预条件：z_i = r_i / A_ii。"""

    def __init__(self, A):
        # 注意：保存对角元本身而非倒数，apply 里做除法，
        # 与旧实现逐位一致（r/d 与 r*(1/d) 可能差 1 ulp）。
        self._diag = [A[i][i] for i in range(len(A))]

    def apply(self, r):
        d = self._diag
        return [r[i] / d[i] for i in range(len(r))]


class SSORPreconditioner:
    """SSOR 预条件（对称 Gauss-Seidel 前向+后向各一次）。

    M = (D + wL) D^{-1} (D + wU) / (w(2-w))，apply 通过前代/回代完成，
    不需要显式求逆。要求 A 对称且对角非零。
    """

    def __init__(self, A, omega=1.0):
        n = len(A)
        self._A = A
        self._n = n
        self._omega = omega
        self._diag = [A[i][i] for i in range(n)]

    def apply(self, r):
        A, n, w, diag = self._A, self._n, self._omega, self._diag
        # 前向：(D + wL) y = r
        y = [0.0] * n
        for i in range(n):
            s = r[i]
            row = A[i]
            for j in range(i):
                s -= w * row[j] * y[j]
            y[i] = s / diag[i]
        # 后向：(D + wU) z = D y
        z = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = diag[i] * y[i]
            row = A[i]
            for j in range(i + 1, n):
                s -= w * row[j] * z[j]
            z[i] = s / diag[i]
        # 归一化因子 w(2-w)（w=1 时为 1）
        scale = w * (2.0 - w)
        if scale != 1.0:
            z = [zi / scale for zi in z]
        return z
