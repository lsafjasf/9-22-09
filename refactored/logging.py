"""Boundary 6: logging sink (interface + stdlib-backed implementations).

The kernel depends only on the ``SolverLogger`` protocol, never on print
redirection or a global VERBOSE flag. Every solve gets its own logger
instance (or a silent default), so concurrent runs do not interleave.
"""

import sys
from typing import TextIO


class SolverLogger:
    def start(self, n: int, b_norm: float) -> None:
        pass

    def iteration(self, iteration: int, residual_norm: float) -> None:
        pass

    def finish(self, stats) -> None:
        pass


class NullLogger(SolverLogger):
    pass


class StreamLogger(SolverLogger):
    def __init__(self, stream: TextIO = None, every: int = 1):
        self._stream = stream if stream is not None else sys.stderr
        self._every = max(1, int(every))

    def start(self, n, b_norm):
        self._stream.write("PCG start: n=%d ||b||=%.12e\n" % (n, b_norm))

    def iteration(self, iteration, residual_norm):
        if iteration % self._every == 0:
            self._stream.write("PCG iter %d residual %.12e\n"
                               % (iteration, residual_norm))

    def finish(self, stats):
        self._stream.write(
            "PCG finish: %s after %d iters, ||r||=%s, matvecs=%d\n"
            % (stats.stop_reason, stats.iterations,
               stats.final_residual_norm, stats.matvecs))


class ListLogger(SolverLogger):
    """Collects lines into a per-instance list; handy for tests."""

    def __init__(self):
        self.lines = []

    def start(self, n, b_norm):
        self.lines.append("start n=%d bnorm=%.12e" % (n, b_norm))

    def iteration(self, iteration, residual_norm):
        self.lines.append("iter %d %.12e" % (iteration, residual_norm))

    def finish(self, stats):
        self.lines.append("finish %s %d"
                          % (stats.stop_reason, stats.iterations))
