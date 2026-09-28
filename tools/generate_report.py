"""Regenerate the equivalence report and the README numbers from real runs.

Three-way comparison per case:
    frozen baseline (baseline/expected.json)
    vs live legacy solver (legacy/)
    vs refactored solver (refactored/)

Writes EQUIVALENCE_REPORT.md and rewrites the generated block in README.md
between the markers <!-- BEGIN GENERATED EQUIVALENCE --> and
<!-- END GENERATED EQUIVALENCE -->. Doc numbers are never edited by hand.

Usage:
    python3 tools/generate_report.py          # regenerate report + README
    python3 tools/generate_report.py --check  # fail if report/README stale
"""

import json
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REPORT_PATH = os.path.join(ROOT, "EQUIVALENCE_REPORT.md")
README_PATH = os.path.join(ROOT, "README.md")
BEGIN = "<!-- BEGIN GENERATED EQUIVALENCE -->"
END = "<!-- END GENERATED EQUIVALENCE -->"

sys.path.insert(0, os.path.join(ROOT, "tools"))
from freeze_baseline import reset_legacy  # noqa: E402


def run_all():
    import legacy.pcg_jacobi as legacy_jacobi
    import legacy.pcg_ssor as legacy_ssor
    from refactored import IterativeSolver, SolverConfig
    from tests.problems import PROBLEMS, zero

    with open(os.path.join(ROOT, "baseline", "expected.json")) as fh:
        baseline = json.load(fh)["cases"]

    rows = []
    for pre_name, module, cfg in (
            ("jacobi", legacy_jacobi, SolverConfig(preconditioner="jacobi")),
            ("ssor", legacy_ssor,
             SolverConfig(preconditioner="ssor", ssor_omega=1.2))):
        for prob_name, A, b in PROBLEMS:
            key = "%s/%s" % (pre_name, prob_name)
            base = baseline[key]

            reset_legacy(module)
            x_old = module.PCGSolver(A, b).solve(zero(len(b)))
            old = {
                "iterations": module.PCGSolver.iter_count,
                "residual_history": list(module.HISTORY),
                "solution": list(x_old),
            }
            reset_legacy(module)

            x_new, stats = IterativeSolver(A, b, cfg).solve(zero(len(b)))
            new = {
                "iterations": stats.iterations,
                "residual_history": list(stats.residual_history),
                "solution": list(x_new),
            }

            rows.append({
                "case": key,
                "base_iters": base["iterations"],
                "old_iters": old["iterations"],
                "new_iters": new["iterations"],
                "converged": stats.converged,
                "legacy_vs_base": _diff(old, base),
                "new_vs_base": _diff(new, base),
            })
    return rows


def _diff(run, base):
    hist, base_hist = run["residual_history"], base["residual_history"]
    sol, base_sol = run["solution"], base["solution"]
    res = max((abs(a - c) / max(1e-30, c)
               for a, c in zip(hist, base_hist)), default=0.0)
    sol_rel = max((abs(a - c) / max(1.0, abs(c))
                   for a, c in zip(sol, base_sol)), default=0.0)
    return {
        "iters_equal": run["iterations"] == base["iterations"],
        "hist_len_equal": len(hist) == len(base_hist),
        "max_rel_residual": res,
        "max_rel_solution": sol_rel,
        "bit_identical": (run["iterations"] == base["iterations"]
                          and hist == base_hist and sol == base_sol),
    }


def fmt(x):
    return "0 (bit-identical)" if x == 0.0 else "%.3e" % x


def render_table(rows):
    lines = [
        "| 用例 | 基线迭代 | 旧版迭代 | 新版迭代 | 新旧逐次残差最大相对差 | 新旧解向量最大相对差 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = r["new_vs_base"]
        lines.append("| %s | %d | %d | %d | %s | %s |" % (
            r["case"], r["base_iters"], r["old_iters"], r["new_iters"],
            fmt(d["max_rel_residual"]), fmt(d["max_rel_solution"])))
    return "\n".join(lines)


def render_summary(rows):
    n = len(rows)
    bit = sum(1 for r in rows if r["new_vs_base"]["bit_identical"])
    iters_ok = sum(1 for r in rows
                   if r["new_vs_base"]["iters_equal"]
                   and r["new_vs_base"]["hist_len_equal"])
    max_r = max(r["new_vs_base"]["max_rel_residual"] for r in rows)
    max_x = max(r["new_vs_base"]["max_rel_solution"] for r in rows)
    legacy_ok = all(r["legacy_vs_base"]["bit_identical"] for r in rows)
    verdict = (bit == n and iters_ok == n and legacy_ok)
    lines = [
        "- 对拍用例数：%d（2 种预条件 x 4 个确定性 SPD 问题）" % n,
        "- 迭代次数与基线完全一致：%d/%d" % (iters_ok, n),
        "- 逐次残差历史与基线位级一致（bit-identical）：%d/%d" % (bit, n),
        "- 新版 vs 冻结基线 逐次残差最大相对误差：%s" % fmt(max_r),
        "- 新版 vs 冻结基线 解向量最大相对误差：%s" % fmt(max_x),
        "- 当前 legacy 代码复算与冻结基线位级一致：%s"
        % ("是" if legacy_ok else "否（基线输入已被改动！）"),
        "- 结论：%s" % ("**行为等价成立（位级一致）**"
                       if verdict else "**行为等价不成立**"),
    ]
    return "\n".join(lines), verdict


def generated_block(rows):
    summary, _ = render_summary(rows)
    return "\n".join([
        BEGIN,
        "以下数字由 `python3 tools/generate_report.py` 实测生成，请勿手改。",
        "",
        summary,
        "",
        render_table(rows),
        END,
    ])


def render_report(rows):
    summary, verdict = render_summary(rows)
    env = "Python %s (%s)" % (platform.python_version(), platform.system())
    parts = [
        "# 新旧实现对拍报告（冻结基线）",
        "",
        "本文件由 `python3 tools/generate_report.py` 自动生成，请勿手改。",
        "",
        "- 基线：`baseline/expected.json`（由 `tools/freeze_baseline.py` 从"
        " `legacy/` 实测冻结），输入完整性见 `baseline/SHA256SUMS`。",
        "- 对照三方：冻结基线 / 当前 `legacy/` 复算 / 当前 `refactored/`。",
        "- 判据：迭代次数相等；逐次残差历史与解向量对基线位级一致"
        "（强于重构前文档声称的 1e-10 / 1e-9）。",
        "- 运行环境：%s；纯标准库、固定种子、固定求和顺序，"
        "重跑结果可复现。" % env,
        "",
        "## 结论",
        "",
        summary,
        "",
        "## 逐用例对照",
        "",
        render_table(rows),
        "",
        "## 复现",
        "",
        "```bash",
        "python3 tools/freeze_baseline.py --verify   # 基线无漂移",
        "python3 tools/generate_report.py --check    # 报告/README 与实测一致",
        "python3 run_tests.py                        # 含 tests/test_baseline.py",
        "```",
        "",
    ]
    return "\n".join(parts), verdict


def update_readme(block):
    with open(README_PATH) as fh:
        text = fh.read()
    start = text.index(BEGIN)
    end = text.index(END) + len(END)
    return text[:start] + block + text[end:]


def main():
    check = "--check" in sys.argv[1:]
    rows = run_all()
    report, verdict = render_report(rows)
    readme = update_readme(generated_block(rows))

    if check:
        ok = True
        if not os.path.exists(REPORT_PATH) or \
                open(REPORT_PATH).read() != report:
            print("STALE: EQUIVALENCE_REPORT.md "
                  "(run python3 tools/generate_report.py)")
            ok = False
        if open(README_PATH).read() != readme:
            print("STALE: README.md generated block "
                  "(run python3 tools/generate_report.py)")
            ok = False
        if ok:
            print("report and README numbers are up to date; "
                  "verdict: %s" % ("EQUIVALENT" if verdict else "MISMATCH"))
        return 0 if (ok and verdict) else 1

    with open(REPORT_PATH, "w") as fh:
        fh.write(report)
    with open(README_PATH, "w") as fh:
        fh.write(readme)
    print("wrote %s and updated README.md generated block; verdict: %s"
          % (os.path.relpath(REPORT_PATH, ROOT),
             "EQUIVALENT" if verdict else "MISMATCH"))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
