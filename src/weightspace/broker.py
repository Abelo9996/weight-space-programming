"""Broker for development feedback and post-selection held-out evaluation."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Callable, Literal

from .artifacts import RunJournal, canonical_json, json_digest
from .budget import Budget, BudgetSnapshot, RowMeter
from .edits import EditError, EditLimits, apply_edits
from .policies import Policy, copy_policy, policy_digest
from .tasks import Score, StartSplit, TaskBundle, TaskSpec, evaluate_split


@dataclass(frozen=True, slots=True)
class CandidateResult:
    candidate_id: int
    status: Literal["scored", "rejected", "failed", "budget_exhausted"]
    score: Score | None
    reason: str | None
    budget: BudgetSnapshot
    changed_entries: int | None = None
    touched_entries: int | None = None


@dataclass(frozen=True, slots=True)
class FinalResult:
    candidate_id: int
    ops: list[dict]
    dev: Score
    heldout: Score
    budget: BudgetSnapshot
    policy_sha256: str
    heldout_rows_reserved: int
    heldout_rows_used: int


@dataclass(frozen=True, slots=True)
class DevSession:
    """The method-facing interface. Keep the trusted broker and fixtures in the runner."""

    task: TaskSpec
    limits: EditLimits
    _probe: Callable[[list[dict] | str], CandidateResult]
    _policy: Callable[[], Policy]
    _budget: Callable[[], BudgetSnapshot]

    def probe(self, ops: list[dict] | str) -> CandidateResult:
        return self._probe(ops)

    def policy(self) -> Policy:
        return self._policy()

    @property
    def budget(self) -> BudgetSnapshot:
        return self._budget()


class BudgetBroker:
    """Run one strategy against a fixed starting checkpoint and development split.

    Strategy code gets a DevSession. Keep untrusted callers outside the evaluator
    process and serve only the declared protocol through the runner.
    """

    def __init__(
        self,
        task: TaskBundle,
        *,
        policy: Policy | None = None,
        candidate_cap: int,
        row_budget: int | None = None,
        limits: EditLimits = EditLimits(),
        journal: RunJournal | None = None,
    ):
        self._task = task.spec
        self._start = copy_policy(task.policy if policy is None else policy, task.spec.sizes)
        self._dev_split = StartSplit(task.dev.failure, task.dev.normal)
        self._heldout_split = StartSplit(task.heldout.failure, task.heldout.normal)
        self._ledger = Budget(candidate_cap, self._dev_split.max_rows(task.spec.horizon), row_budget)
        self._limits = limits
        self._journal = journal
        self._lock = threading.RLock()
        self._next_id = 0
        self._sealed = False
        self._scored: dict[int, tuple[Score, list[dict]]] = {}
        self._selected_policy: Policy | None = None
        self.dev = DevSession(task.spec, limits, self._probe, self._policy_snapshot, self._budget_snapshot)
        self._append(
            "broker_started",
            task_id=task.spec.id,
            evaluator_source=task.spec.evaluator_source,
            starting_policy_sha256=policy_digest(self._start),
            dev_sha256=self._dev_split.digest(),
            limits=asdict(limits),
            budget=asdict(self._ledger.snapshot()),
        )

    def _append(self, event: str, **fields) -> None:
        if self._journal is not None:
            self._journal.append(event, **fields)

    def _policy_snapshot(self) -> Policy:
        return {name: value.copy() for name, value in self._start.items()}

    def _budget_snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._ledger.snapshot()

    def _check_open(self) -> None:
        if self._sealed:
            raise RuntimeError("development is sealed")

    def _probe(self, ops: list[dict] | str) -> CandidateResult:
        with self._lock:
            self._check_open()
            candidate_id = self._next_id
            self._next_id += 1
            try:
                canonical_json(ops)
            except (TypeError, ValueError, RecursionError):
                self._append("candidate_request", candidate_id=candidate_id, ops=None, valid_json=False)
            else:
                self._append("candidate_request", candidate_id=candidate_id, ops=ops)
            before = self._ledger.snapshot()
            if not before.candidates_remaining:
                result = CandidateResult(candidate_id, "budget_exhausted", None, "development budget exhausted", before)
            else:
                result = self._submit(candidate_id, ops)
            self._append("candidate_result", **asdict(result))
            return result

    def _submit(self, candidate_id: int, ops: list[dict] | str) -> CandidateResult:
        try:
            edit = apply_edits(self._start, ops, self._limits)
        except EditError as error:
            return CandidateResult(candidate_id, "rejected", None, str(error), self._ledger.snapshot())
        meter = self._ledger.reserve()
        self._append(
            "candidate_reserved",
            candidate_id=candidate_id,
            ops=edit.ops,
            edit_sha256=json_digest(edit.ops),
            policy_sha256=policy_digest(edit.policy),
            budget=asdict(self._ledger.snapshot()),
        )
        try:
            score = self._evaluate(edit.policy, self._dev_split, meter)
        except Exception as error:
            return CandidateResult(
                candidate_id, "failed", None, type(error).__name__, self._ledger.snapshot(),
                edit.changed_entries, edit.touched_entries,
            )
        self._scored[candidate_id] = score, edit.ops
        return CandidateResult(
            candidate_id, "scored", score, None, self._ledger.snapshot(), edit.changed_entries, edit.touched_entries,
        )

    def _evaluate(self, policy: Policy, split: StartSplit, meter: RowMeter) -> Score:
        score = evaluate_split(self._task, policy, split, meter.charge)
        if score.rows != meter.used:
            raise RuntimeError("evaluator row total does not match the broker meter")
        return score

    def finalize(self, candidate_id: int) -> FinalResult:
        """Seal development and evaluate the chosen successful candidate once."""
        with self._lock:
            self._check_open()
            if type(candidate_id) is not int or candidate_id not in self._scored:
                raise ValueError("final selection must be a scored candidate ID")
            self._sealed = True
            dev_score, ops = self._scored[candidate_id]
            policy = apply_edits(self._start, ops, self._limits).policy
            checkpoint_sha256 = policy_digest(policy)
            meter = RowMeter(self._heldout_split.max_rows(self._task.horizon))
            self._append(
                "final_selected",
                candidate_id=candidate_id,
                ops=ops,
                edit_sha256=json_digest(ops),
                policy_sha256=checkpoint_sha256,
                dev=asdict(dev_score),
                budget=asdict(self._ledger.snapshot()),
                heldout_sha256=self._heldout_split.digest(),
                heldout_rows_reserved=meter.reserved,
            )
            try:
                heldout_score = self._evaluate(policy, self._heldout_split, meter)
            except Exception as error:
                self._append(
                    "heldout_failed",
                    candidate_id=candidate_id,
                    reason=type(error).__name__,
                    heldout_rows_reserved=meter.reserved,
                    heldout_rows_used=meter.used,
                )
                raise
            result = FinalResult(
                candidate_id, ops, dev_score, heldout_score, self._ledger.snapshot(), checkpoint_sha256,
                meter.reserved, meter.used,
            )
            self._append("heldout_result", **asdict(result))
            self._selected_policy = policy
            return result

    def selected_policy(self) -> Policy:
        with self._lock:
            if self._selected_policy is None:
                raise RuntimeError("no finalized checkpoint")
            return {name: value.copy() for name, value in self._selected_policy.items()}

    def abort(self, reason: str) -> None:
        """Close an incomplete run without evaluating held-out starts."""
        with self._lock:
            if not self._sealed:
                self._sealed = True
                self._append("broker_aborted", reason=reason, budget=asdict(self._ledger.snapshot()))
