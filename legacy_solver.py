"""旧版迭代求解器（重构前基线）。

已知缺陷（重构动机，详见 docs/ARCHITECTURE.md）：
  1. 配置是模块级全局变量，调用方靠“先改全局再调用”传参；
  2. 统计信息挂在类属性上，所有实例、所有线程共享且跨运行累积；
  3. Jacobi 预条件硬编码在迭代循环里，换预条件只能复制整个文件；
  4. 收敛判据硬编码，无法替换；
  5. 日志直接 print 到 stdout，无法按任务隔离或关闭到指定流；
  6. 矩阵/向量运算与迭代内核、统计、日志全部揉在一个方法里。
"""

# ---- 全局配置（调用方必须在 solve 前手动设置，且影响进程内所有求解任务）----
TOLERANCE = 1e-8
MAX_ITER = 1000
OMEGA = 1.0
LOG_EVERY = 0  # 0 表示不打印；>0 表示每隔多少步打印一次


class IterativeSolver:
    """单类包揽：向量/矩阵运算 + 预条件 + 判据 + 迭代内核 + 统计 + 日志。"""

    # 缺陷 2：可变统计挂在“类”上 —— 所有实例共享，且多次 solve 之间互相污染。
    stats = {
        "runs": 0,
        "total_iterations": 0,
        "last_iterations": None,
        "last_residual": None,
        "residual_history": [],
    }

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.n = len(b)

    # ---- 向量/矩阵运算也塞在类里，无法复用 ----
    def _matvec(self, x):
        A, n = self.A, self.n
        return [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]

    def _norm(self, v):
        return sum(c * c for c in v) ** 0.5

    def solve(self, x0=None):
        # 缺陷 1：运行时读全局变量，两个任务想要不同配置就会打架。
        tol = TOLERANCE
        max_iter = MAX_ITER
        omega = OMEGA
        log_every = LOG_EVERY

        x = list(x0) if x0 is not None else [0.0] * self.n
        b = self.b
        n = self.n

        b_norm = self._norm(b)
        threshold = tol * b_norm

        iterations = 0
        final_residual = None
        for k in range(max_iter + 1):
            Ax = self._matvec(x)
            r = [b[i] - Ax[i] for i in range(n)]
            r_norm = self._norm(r)

            # 缺陷 2：统计直接写共享的类属性。
            IterativeSolver.stats["residual_history"].append(r_norm)
            final_residual = r_norm

            # 缺陷 5：日志直接 print。
            if log_every and k % log_every == 0:
                print(f"[legacy] iter={k} residual={r_norm:.6e}")

            # 缺陷 4：判据硬编码（相对残差）。
            if r_norm <= threshold:
                iterations = k
                break

            if k == max_iter:
                iterations = max_iter
                break

            # 缺陷 3：Jacobi 预条件硬编码在循环里。
            z = [r[i] / self.A[i][i] for i in range(n)]
            x = [x[i] + omega * z[i] for i in range(n)]
            iterations = k + 1

        IterativeSolver.stats["runs"] += 1
        IterativeSolver.stats["total_iterations"] += iterations
        IterativeSolver.stats["last_iterations"] = iterations
        IterativeSolver.stats["last_residual"] = final_residual
        return x, iterations, final_residual
