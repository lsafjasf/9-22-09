"""Boundary 3: convergence criteria.

A criterion is a small object created fresh per solve (so it can hold
per-run state such as the initial residual without any globals):

* ``bind(context)`` is called once before iterations start; ``context``
  carries b-norm and initial residual-norm information.
* ``met(iteration, residual_norm)`` is called after every update.

Criteria are selected by name or passed directly, just like
preconditioners, so swapping them needs no caller change.
"""

from dataclasses import dataclass
from typing import Dict, Type


@dataclass(frozen=True)
class CriterionContext:
    b_norm: float
    initial_residual_norm: float
    dimension: int


class ConvergenceCriterion:
    def bind(self, ctx: CriterionContext) -> None:
        pass

    def met(self, iteration: int, residual_norm: float) -> bool:
        raise NotImplementedError


class RelativeToRHS(ConvergenceCriterion):
    """||r|| <= tol * ||b||  -- exactly the legacy test."""

    def __init__(self, tol: float = 1e-8):
        self._tol = float(tol)
        self._threshold = 0.0

    def bind(self, ctx):
        self._threshold = self._tol * ctx.b_norm

    def met(self, iteration, residual_norm):
        return residual_norm <= self._threshold


class InitialResidualReduction(ConvergenceCriterion):
    """||r_k|| <= factor * ||r_0|| (common alternative test)."""

    def __init__(self, factor: float = 1e-6):
        self._factor = float(factor)
        self._threshold = 0.0

    def bind(self, ctx):
        self._threshold = self._factor * ctx.initial_residual_norm

    def met(self, iteration, residual_norm):
        return residual_norm <= self._threshold


class AbsoluteResidual(ConvergenceCriterion):
    """||r_k|| <= tol."""

    def __init__(self, tol: float = 1e-10):
        self._tol = float(tol)

    def met(self, iteration, residual_norm):
        return residual_norm <= self._tol


_REGISTRY: Dict[str, Type[ConvergenceCriterion]] = {
    "relative_rhs": RelativeToRHS,
    "initial_reduction": InitialResidualReduction,
    "absolute": AbsoluteResidual,
}


def register_criterion(name: str, cls: Type[ConvergenceCriterion]) -> None:
    _REGISTRY[name] = cls


def resolve_criterion(spec) -> ConvergenceCriterion:
    if isinstance(spec, ConvergenceCriterion):
        return spec
    if isinstance(spec, str):
        try:
            return _REGISTRY[spec]()
        except KeyError as exc:
            raise KeyError("unknown criterion %r; registered: %s"
                           % (spec, sorted(_REGISTRY))) from exc
    if isinstance(spec, type) and issubclass(spec, ConvergenceCriterion):
        return spec()
    raise TypeError("unsupported criterion spec: %r" % (spec,))
