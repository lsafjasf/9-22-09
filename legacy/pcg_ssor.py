"""Legacy monolithic preconditioned conjugate-gradient solver (SSOR).

This is a full copy of legacy/pcg_jacobi.py with only _precondition and the
omega constant changed - exactly the "copy the whole file to switch
preconditioners" situation the refactor removes.
"""


TOL = 1e-8
MAX_ITER = 1000
OMEGA = 1.2
VERBOSE = False
HISTORY = []


class PCGSolver:
    iter_count = 0
    matvec_count = 0

    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.history = HISTORY

    def _dot(self, x, y):
        return sum(xi * yi for xi, yi in zip(x, y))

    def _axpy(self, y, a, x):
        for i in range(len(y)):
            y[i] = y[i] + a * x[i]
        return y

    def _matvec(self, A, x):
        if callable(A):
            return A(x)
        if isinstance(A, list):
            return [sum(A[i][j] * x[j] for j in range(len(x)))
                    for i in range(len(A))]
        n = len(x)
        d, lo = A
        y = [0.0] * n
        for i in range(n):
            y[i] = d[i] * x[i]
            if i > 0:
                y[i] += lo[i - 1] * x[i - 1]
            if i < n - 1:
                y[i] += lo[i] * x[i + 1]
        return y

    def _diag_offdiag(self):
        n = len(self.b)
        if isinstance(self.A, list):
            diag = [self.A[i][i] for i in range(n)]
            lower = [self.A[i][i - 1] if i > 0 else 0.0 for i in range(1, n)]
            upper = [self.A[i][i + 1] if i < n - 1 else 0.0
                     for i in range(n - 1)]
        else:
            d, lo = self.A
            diag = d[:]
            lower = lo[:]
            upper = lo[:]
        return diag, lower, upper

    def _precondition(self, r):
        n = len(self.b)
        diag, lower, upper = self._diag_offdiag()
        w = OMEGA
        t = [0.0] * n
        for i in range(n):
            s = r[i]
            if i > 0:
                s -= lower[i - 1] * t[i - 1]
            t[i] = w * s / diag[i]
        y = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = t[i]
            if i < n - 1:
                s -= upper[i] * y[i + 1]
            y[i] = w * s / diag[i]
        return y

    def solve(self, x0):
        n = len(self.b)
        bnorm = self._dot(self.b, self.b) ** 0.5
        x = [float(v) for v in x0]
        r = [self.b[i] - self._matvec(self.A, x)[i] for i in range(n)]
        PCGSolver.iter_count = 0
        PCGSolver.matvec_count = 1
        self.history = HISTORY
        z = self._precondition(r)
        p = z[:]
        rz = self._dot(r, z)
        for _ in range(MAX_ITER):
            Ap = self._matvec(self.A, p)
            PCGSolver.matvec_count += 1
            alpha = rz / self._dot(p, Ap)
            self._axpy(x, alpha, p)
            self._axpy(r, -alpha, Ap)
            rnorm = self._dot(r, r) ** 0.5
            HISTORY.append(rnorm)
            PCGSolver.iter_count += 1
            if VERBOSE:
                print("PCG iter %d residual %.12e" % (PCGSolver.iter_count, rnorm))
            if rnorm <= TOL * bnorm:
                return x
            z = self._precondition(r)
            rz_new = self._dot(r, z)
            beta = rz_new / rz
            p = [z[i] + beta * p[i] for i in range(n)]
            rz = rz_new
        return x
