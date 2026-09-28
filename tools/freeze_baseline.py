"""Freeze (or verify) the pre-refactor equivalence baseline.

Runs the LEGACY solvers on the deterministic test problems and records their
exact outputs (iteration counts, matvec counts, full residual histories and
solution vectors, floats round-tripped via repr) into baseline/expected.json.
Also pins sha256 of the baseline inputs (legacy sources + problem generator)
into baseline/SHA256SUMS.

Usage:
    python3 tools/freeze_baseline.py            # (re)generate baseline/
    python3 tools/freeze_baseline.py --verify   # check only, exit 1 on drift

The baseline is READ-ONLY after freezing: tests/test_baseline.py fails if any
pinned input changes or if the refactored solver stops matching expected.json.
"""

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASELINE_DIR = os.path.join(ROOT, "baseline")
EXPECTED_PATH = os.path.join(BASELINE_DIR, "expected.json")
SUMS_PATH = os.path.join(BASELINE_DIR, "SHA256SUMS")

# Files that define the benchmark: the pre-refactor implementation and the
# deterministic problem generator. Any change here invalidates the baseline.
PINNED_INPUTS = [
    "legacy/pcg_jacobi.py",
    "legacy/pcg_ssor.py",
    "tests/problems.py",
]

TOL = 1e-8
MAX_ITER = 1000
SSOR_OMEGA = 1.2


def reset_legacy(module):
    module.TOL = TOL
    module.MAX_ITER = MAX_ITER
    module.VERBOSE = False
    module.HISTORY.clear()
    module.PCGSolver.iter_count = 0
    module.PCGSolver.matvec_count = 0


def run_legacy_cases():
    import legacy.pcg_jacobi as legacy_jacobi
    import legacy.pcg_ssor as legacy_ssor
    from tests.problems import PROBLEMS, zero

    cases = {}
    for pre_name, module in (("jacobi", legacy_jacobi),
                             ("ssor", legacy_ssor)):
        for prob_name, A, b in PROBLEMS:
            reset_legacy(module)
            x = module.PCGSolver(A, b).solve(zero(len(b)))
            bnorm = module.PCGSolver(A, b)._dot(b, b) ** 0.5
            key = "%s/%s" % (pre_name, prob_name)
            cases[key] = {
                "preconditioner": pre_name,
                "problem": prob_name,
                "iterations": module.PCGSolver.iter_count,
                "matvec_count": module.PCGSolver.matvec_count,
                "converged": bool(
                    module.HISTORY and
                    module.HISTORY[-1] <= TOL * bnorm),
                "residual_history": list(module.HISTORY),
                "solution": list(x),
            }
            reset_legacy(module)
    return cases


def build_expected():
    return {
        "format": 1,
        "generated_by": "tools/freeze_baseline.py",
        "source": "legacy/ (pre-refactor implementation)",
        "solver_params": {
            "tol": TOL,
            "max_iter": MAX_ITER,
            "ssor_omega": SSOR_OMEGA,
            "criterion": "residual_norm <= tol * rhs_norm",
            "x0": "zero vector",
        },
        "cases": run_legacy_cases(),
    }


def compute_sums():
    lines = []
    for rel in PINNED_INPUTS:
        with open(os.path.join(ROOT, rel), "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        lines.append("%s  %s" % (digest, rel))
    return "\n".join(lines) + "\n"


def main():
    verify = "--verify" in sys.argv[1:]
    expected = json.dumps(build_expected(), indent=2, sort_keys=True) + "\n"
    sums = compute_sums()

    if not verify:
        os.makedirs(BASELINE_DIR, exist_ok=True)
        with open(EXPECTED_PATH, "w") as fh:
            fh.write(expected)
        with open(SUMS_PATH, "w") as fh:
            fh.write(sums)
        print("froze %d cases -> %s" % (
            len(json.loads(expected)["cases"]),
            os.path.relpath(EXPECTED_PATH, ROOT)))
        print("pinned %d inputs -> %s" % (
            len(PINNED_INPUTS), os.path.relpath(SUMS_PATH, ROOT)))
        return 0

    ok = True
    for path, content, label in (
            (EXPECTED_PATH, expected, "expected outputs"),
            (SUMS_PATH, sums, "pinned input hashes")):
        if not os.path.exists(path):
            print("MISSING: %s (run tools/freeze_baseline.py)" % path)
            ok = False
        elif open(path).read() != content:
            print("DRIFT: %s no longer matches the frozen %s" % (path, label))
            ok = False
    if ok:
        print("baseline verified: %d cases, %d pinned inputs, no drift" % (
            len(json.loads(expected)["cases"]), len(PINNED_INPUTS)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
