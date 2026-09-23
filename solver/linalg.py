"""稠密向量/矩阵基础运算（纯函数，无状态）。

职责：只提供数值原语，不知道迭代、判据、统计的存在。
矩阵用 list[list[float]]（行主序）表示，向量用 list[float]。
"""


def matvec(A, x):
    """返回 A @ x。"""
    n = len(x)
    return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]


def residual(A, b, x):
    """返回 b - A @ x。"""
    Ax = matvec(A, x)
    return [b[i] - Ax[i] for i in range(len(b))]


def norm2(v):
    """欧氏范数。"""
    return sum(c * c for c in v) ** 0.5


def axpy(alpha, x, y):
    """返回 alpha*x + y（不修改入参）。"""
    return [alpha * x[i] + y[i] for i in range(len(y))]
