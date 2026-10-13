"""A pseudo-oracle: find the best binary assignment by trying every one.

Write the objective and constraints straight from the problem's definition, not from production code, so a bug
would have to occur twice in the same way to slip through. The search is exponential, so keep n at about 20 or
fewer. It lives with the tests because nothing in the program uses it.

Usage:
    solver = EnumerationSolver(objective=lambda x: ..., feasible=lambda x: sum(x) == s)
    best = solver.run(n)          # None when nothing is feasible
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Best:
    """The best assignment found and its objective value (minimisation)."""

    assignment: tuple[int, ...]
    objective_value: float


class EnumerationSolver:
    """Minimises objective(x) over x in {0, 1}^n subject to feasible(x), by enumeration."""

    def __init__(
        self,
        objective: Callable[[Sequence[int]], float],
        feasible: Callable[[Sequence[int]], bool] = lambda x: True,
    ) -> None:
        self.objective = objective
        self.feasible = feasible

    def run(self, n: int) -> Best | None:
        """Returns the best feasible assignment, or None when no assignment is feasible.

        Ties keep the first assignment found, so tests should compare objective values, not assignments,
        unless the optimum is known to be unique.
        """
        best: Best | None = None
        for bits in itertools.product((0, 1), repeat=n):
            if not self.feasible(bits):
                continue
            value = self.objective(bits)
            if best is None or value < best.objective_value:
                best = Best(assignment=bits, objective_value=value)
        return best
