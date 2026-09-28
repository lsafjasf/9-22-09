"""Frozen-baseline anchor tests.

The equivalence claim references baseline/expected.json, frozen from the
pre-refactor legacy implementation by tools/freeze_baseline.py. These tests
fail if:
  1. any pinned baseline input (legacy sources, problem generator) changes
     without re-freezing (sha256 drift), or
  2. the live legacy solvers no longer reproduce the frozen baseline, or
  3. the refactored solver stops being bit-identical to the frozen baseline.
"""

import hashlib
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from freeze_baseline import reset_legacy  # noqa: E402
from refactored import IterativeSolver, SolverConfig  # noqa: E402
from tests.problems import PROBLEMS, zero  # noqa: E402
import legacy.pcg_jacobi as legacy_jacobi  # noqa: E402
import legacy.pcg_ssor as legacy_ssor  # noqa: E402

BASELINE_DIR = os.path.join(ROOT, "baseline")

CONFIGS = {
    "jacobi": (legacy_jacobi, SolverConfig(preconditioner="jacobi")),
    "ssor": (legacy_ssor, SolverConfig(preconditioner="ssor", ssor_omega=1.2)),
}


def load_baseline():
    with open(os.path.join(BASELINE_DIR, "expected.json")) as fh:
        return json.load(fh)["cases"]


class BaselineIntegrityTests(unittest.TestCase):
    def test_pinned_inputs_unchanged(self):
        with open(os.path.join(BASELINE_DIR, "SHA256SUMS")) as fh:
            entries = [line.split() for line in fh if line.strip()]
        self.assertTrue(entries, "SHA256SUMS is empty")
        for digest, rel in entries:
            with self.subTest(file=rel):
                with open(os.path.join(ROOT, rel), "rb") as fh:
                    actual = hashlib.sha256(fh.read()).hexdigest()
                self.assertEqual(
                    digest, actual,
                    "%s changed after the baseline was frozen; if this is "
                    "intentional, re-run tools/freeze_baseline.py and "
                    "regenerate the report" % rel)

    def test_baseline_file_present_and_complete(self):
        cases = load_baseline()
        for pre in CONFIGS:
            for name, _, _ in PROBLEMS:
                key = "%s/%s" % (pre, name)
                with self.subTest(case=key):
                    self.assertIn(key, cases)
                    case = cases[key]
                    self.assertTrue(case["converged"], key)
                    self.assertEqual(
                        len(case["residual_history"]), case["iterations"])


class FrozenBaselineEquivalenceTests(unittest.TestCase):
    """Refactored solver must match the frozen baseline bit for bit."""

    def _check_against_baseline(self, runner):
        cases = load_baseline()
        for pre, (module, cfg) in CONFIGS.items():
            for name, A, b in PROBLEMS:
                key = "%s/%s" % (pre, name)
                with self.subTest(case=key):
                    base = cases[key]
                    result = runner(pre, module, cfg, A, b)
                    self.assertEqual(result["iterations"],
                                     base["iterations"], key)
                    self.assertEqual(result["residual_history"],
                                     base["residual_history"], key)
                    self.assertEqual(result["solution"],
                                     base["solution"], key)

    def test_refactored_matches_frozen_baseline(self):
        def run(pre, module, cfg, A, b):
            x, stats = IterativeSolver(A, b, cfg).solve(zero(len(b)))
            self.assertTrue(stats.converged)
            return {"iterations": stats.iterations,
                    "residual_history": list(stats.residual_history),
                    "solution": list(x)}
        self._check_against_baseline(run)

    def test_legacy_still_reproduces_frozen_baseline(self):
        def run(pre, module, cfg, A, b):
            reset_legacy(module)
            x = module.PCGSolver(A, b).solve(zero(len(b)))
            result = {"iterations": module.PCGSolver.iter_count,
                      "residual_history": list(module.HISTORY),
                      "solution": list(x)}
            reset_legacy(module)
            return result
        self._check_against_baseline(run)


if __name__ == "__main__":
    unittest.main(verbosity=2)
