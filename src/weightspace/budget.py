"""Separate worst-case reservations from measured policy work."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    candidate_cap: int
    candidates_used: int
    row_budget: int
    rows_reserved: int
    rows_used: int
    rows_per_candidate: int
    candidates_remaining: int


class RowMeter:
    def __init__(self, reserved: int):
        self.reserved = reserved
        self.used = 0

    def charge(self, count: int) -> None:
        if type(count) is not int or count < 1:
            raise ValueError("row count must be a positive integer")
        if self.used + count > self.reserved:
            raise RuntimeError("policy row reservation exceeded")
        self.used += count


class Budget:
    """Reserve the same worst-case cost for every accepted candidate."""

    def __init__(self, candidate_cap: int, rows_per_candidate: int, row_budget: int | None):
        if type(candidate_cap) is not int or candidate_cap < 1:
            raise ValueError("candidate_cap must be a positive integer")
        if type(rows_per_candidate) is not int or rows_per_candidate < 1:
            raise ValueError("rows_per_candidate must be a positive integer")
        if row_budget is None:
            row_budget = candidate_cap * rows_per_candidate
        if type(row_budget) is not int or row_budget < 0:
            raise ValueError("row_budget must be a nonnegative integer")
        self.candidate_cap = candidate_cap
        self.rows_per_candidate = rows_per_candidate
        self.row_budget = row_budget
        self._meters: list[RowMeter] = []

    def snapshot(self) -> BudgetSnapshot:
        used = len(self._meters)
        reserved = used * self.rows_per_candidate
        remaining = min(self.candidate_cap - used, (self.row_budget - reserved) // self.rows_per_candidate)
        return BudgetSnapshot(
            self.candidate_cap, used, self.row_budget, reserved, sum(meter.used for meter in self._meters),
            self.rows_per_candidate, remaining,
        )

    def reserve(self) -> RowMeter:
        if not self.snapshot().candidates_remaining:
            raise RuntimeError("development budget exhausted")
        meter = RowMeter(self.rows_per_candidate)
        self._meters.append(meter)
        return meter
