"""Boundary 2: preconditioners (M approx A, applies M^{-1} r).

Every preconditioner exposes ``apply(r) -> z``. Preconditioners are selected
by name (registry) or passed as an object/factory, so the caller of
``IterativeSolver`` never changes when a new one is added.
"""

from typing import Callable, Dict, Sequence, Type

from . import linalg


class Preconditioner:
    def apply(self, r: Sequence[float]) -> list:
        raise NotImplementedError


class Identity(Preconditioner):
    """M = I (plain CG)."""

    def apply(self, r):
        return list(r)


class Jacobi(Preconditioner):
    """M = diag(A)."""

    def __init__(self, A):
        self._A = A

    def apply(self, r):
        diag = linalg.matrix_diag(self._A, len(r))
        return [ri / di for ri, di in zip(r, diag)]


class SSOR(Preconditioner):
    """Symmetric SOR preconditioner, M^{-1} r via two triangular sweeps.

    Forward:  t_i = omega/diag_i * (r_i - lower_{i,i-1} t_{i-1})
    Backward: y_i = diag_i/omega * t_i - upper_{i,i+1} y_{i+1}
    """

    def __init__(self, A, omega: float = 1.2):
        if linalg.is_tridiagonal(A):
            self._diag = list(A[0])
            self._lower = list(A[1])
            self._upper = list(A[1])
        else:
            n = len(A)
            self._diag = [A[i][i] for i in range(n)]
            self._lower = [A[i][i - 1] if i > 0 else 0.0 for i in range(1, n)]
            self._upper = [A[i][i + 1] if i < n - 1 else 0.0
                           for i in range(n - 1)]
        self._omega = float(omega)

    def apply(self, r):
        n = len(self._diag)
        w = self._omega
        diag, lower, upper = self._diag, self._lower, self._upper
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


_REGISTRY: Dict[str, Type[Preconditioner]] = {
    "identity": Identity,
    "jacobi": Jacobi,
    "ssor": SSOR,
}


def register_preconditioner(name: str, cls: Type[Preconditioner]) -> None:
    _REGISTRY[name] = cls


def resolve_preconditioner(spec, A) -> Preconditioner:
    """Build a preconditioner instance from a name, class, or instance."""
    if isinstance(spec, Preconditioner):
        return spec
    if isinstance(spec, str):
        try:
            cls = _REGISTRY[spec]
        except KeyError as exc:
            raise KeyError("unknown preconditioner %r; registered: %s"
                           % (spec, sorted(_REGISTRY))) from exc
        return cls(A) if cls is not Identity else cls()
    if isinstance(spec, type) and issubclass(spec, Preconditioner):
        return spec(A) if spec is not Identity else spec()
    if callable(spec):
        return _CallablePreconditioner(spec)
    raise TypeError("unsupported preconditioner spec: %r" % (spec,))


class _CallablePreconditioner(Preconditioner):
    def __init__(self, fn: Callable):
        self._fn = fn

    def apply(self, r):
        return list(self._fn(r))
