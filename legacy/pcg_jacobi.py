"""Legacy monolithic preconditioned conjugate-gradient solver (Jacobi).

NOTE (refactoring baseline, intentionally NOT cleaned up):
- Configuration lives in module-level globals (TOL, MAX_ITER, VERBOSE, HISTORY).
- Counters are CLASS attributes shared by every instance in the process.
- Solving, preconditioning, convergence checks, stats and logging are all
  methods of one class.
- To switch preconditioners you copy this whole file -> legacy/pcg_ssor.py.
"""


TOL = 1e-8
MAX_ITER = 1000
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

    def _precondition(self, r):
        if callable(self.A):
            raise ValueError("Jacobi needs an explicit matrix")
        if isinstance(self.A, list):
            diag = [self.A[i][i] for i in range(len(self.b))]
        else:
            diag = self.A[0]
        return [ri / di for ri, di in zip(r, diag)]

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
