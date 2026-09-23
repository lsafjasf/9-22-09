"""Boundary 1: vector / matrix primitives.

Pure functions only, no state. Two sparse matrix shapes are accepted so the
matvec arithmetic is identical to the legacy implementation:

* callable ``A(x)`` returning A*x
* dense: list of rows (symmetric positive definite), or
* tridiagonal: a tuple ``(diag, offdiag)`` with one shared sub/super-diagonal.
"""

from typing import Callable, List, Sequence, Tuple, Union

MatrixLike = Union[Callable[[Sequence[float]], Sequence[float]],
                   List[List[float]],
                   Tuple[List[float], List[float]]]
Vector = List[float]


def dot(x: Sequence[float], y: Sequence[float]) -> float:
    return sum(xi * yi for xi, yi in zip(x, y))


def axpy(y: Vector, a: float, x: Sequence[float]) -> Vector:
    """y <- y + a*x in place; returns y."""
    for i in range(len(y)):
        y[i] = y[i] + a * x[i]
    return y


def scale(a: float, x: Sequence[float]) -> Vector:
    return [a * xi for xi in x]


def add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [ai + bi for ai, bi in zip(a, b)]


def matvec(A: MatrixLike, x: Sequence[float]) -> Vector:
    if callable(A):
        return list(A(x))
    if isinstance(A, list):
        return [sum(A[i][j] * x[j] for j in range(len(x)))
                for i in range(len(A))]
    diag, off = A
    n = len(x)
    y = [0.0] * n
    for i in range(n):
        yi = diag[i] * x[i]
        if i > 0:
            yi += off[i - 1] * x[i - 1]
        if i < n - 1:
            yi += off[i] * x[i + 1]
        y[i] = yi
    return y


def matrix_diag(A: MatrixLike, n: int) -> Vector:
    if callable(A):
        raise TypeError("diagonal of a callable matrix is unavailable")
    if isinstance(A, list):
        return [A[i][i] for i in range(n)]
    return list(A[0])


def is_tridiagonal(A: MatrixLike) -> bool:
    return not callable(A) and isinstance(A, tuple)
