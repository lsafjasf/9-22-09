"""Deterministic symmetric positive-definite test problems."""


def lcg(seed=123456789):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state

    return nxt


def tridiagonal_laplacian(n):
    """1-D Laplacian: diag 2, off-diagonal -1 (tridiagonal tuple)."""
    return [2.0] * n, [-1.0] * (n - 1)


def dense_spd(n, seed=42, scale=4.0):
    """A = scale*I + B^T B with dense entries from a deterministic LCG."""
    rnd = lcg(seed)
    B = [[(rnd() % 2001) / 1000.0 - 1.0 for _ in range(n)] for _ in range(n)]
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = sum(B[k][i] * B[k][j] for k in range(n))
        A[i][i] += scale
    return A


def rhs(n, seed=7):
    rnd = lcg(seed)
    return [(rnd() % 1000) / 500.0 - 1.0 for _ in range(n)]


def zero(n):
    return [0.0] * n


PROBLEMS = [
    ("laplacian_n40", tridiagonal_laplacian(40), rhs(40, seed=7)),
    ("laplacian_n120", tridiagonal_laplacian(120), rhs(120, seed=11)),
    ("dense_spd_n15", dense_spd(15, seed=42), rhs(15, seed=99)),
    ("dense_spd_n30", dense_spd(30, seed=9), rhs(30, seed=5)),
]
