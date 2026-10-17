"""Offline best-model selection: a pure function of harness scores.

``docs/design_env_ladder_and_structure_20261016.md`` M3 and
``docs/design_campaign_structure_20261016.md`` step L2. The in-loop
evaluator (``callbacks.info_dict_eval.InfoDictEvalCallback``) selects
while training runs; :func:`select` makes the same decision afterwards,
from the rows :func:`courtside_dynamics.experiments.harness.rescore`
writes, so a selection can be re-made under another protocol (an era
bump) without retraining.

The rule (:class:`SelectionRule`, from a protocol's
``selection_rule()``) is the in-loop one: scores compare
lexicographically over the selection keys, a key decides only when it
differs from the best's by more than its delta (:func:`improves`, the
one implementation the in-loop callback calls too), and with
``confirm`` a candidate that beats the best on the standard block must
also beat it on the confirmation block.

**Confirmation is within-block here, and differs from the in-loop
rule.** Each block's score of a candidate is compared only with the
best's score *on the same block*, and the first best is scored on the
confirmation block too, so it has a confirmation score to defend. The
in-loop callback banks its first best without a confirmation score,
and until a later candidate is accepted it compares every candidate's
confirmation-block score with that first best's *standard*-block score
(``repo_review_20261016.md`` section 2.2): a persistent bias whenever
the two blocks differ in difficulty, which a paired, deterministic
policy makes systematic rather than noise. Changing the in-loop rule
waits on the maintainer's decision; :func:`in_loop_select` replays the
in-loop rule over the same rows, so the difference can be measured on
any run (the two agree whenever no candidate's confirmation lands
between the first best's two block scores).

Nothing here imports Stable-Baselines3, torch or MuJoCo: selection
reads rows, never models.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: :class:`Decision` values.
FIRST = "first"
IMPROVED = "improved"
NOT_IMPROVED = "not_improved"
CONFIRM_REJECTED = "confirm_rejected"


def improves(
    score: Sequence[float],
    best: Sequence[float] | None,
    min_deltas: Sequence[float],
) -> bool:
    """Lexicographic improvement with a per-key noise threshold.

    A key only decides the comparison when it differs from the best by
    more than its delta; within the threshold the comparison falls
    through to the next key, and a full tie is not an improvement.
    With every delta 0.0 this is exactly the strict-``>`` tuple
    comparison. ``best=None`` (nothing banked yet) is always improved
    on. ``InfoDictEvalCallback._improves`` calls this function, so the
    in-loop and offline selections share one comparison.
    """
    if best is None:
        return True
    for new, old, delta in zip(score, best, min_deltas, strict=True):
        if new > old + delta:
            return True
        if new < old - delta:
            return False
    return False


@dataclass(frozen=True)
class SelectionRule:
    """The lexicographic per-key-delta rule a protocol selects by.

    ``keys`` are metric names, compared in order; ``min_deltas`` the
    noise threshold of each (aligned with ``keys``); ``confirm`` asks a
    candidate to win on the confirmation block as well.
    """

    keys: tuple[str, ...]
    min_deltas: tuple[float, ...]
    confirm: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.keys, (str, bytes)) or not self.keys:
            raise ValueError(
                f"a selection rule needs at least one key, got {self.keys!r}"
            )
        keys = tuple(self.keys)
        if any(not isinstance(key, str) or not key for key in keys):
            raise TypeError(f"selection keys must be non-empty strings: {keys!r}")
        if len(set(keys)) != len(keys):
            raise ValueError(f"selection keys repeat: {keys!r}")
        deltas = tuple(self.min_deltas)
        if len(deltas) != len(keys):
            raise ValueError(
                f"{len(deltas)} min_deltas for {len(keys)} selection keys"
            )
        for key, delta in zip(keys, deltas, strict=True):
            if (
                isinstance(delta, bool)
                or not isinstance(delta, (int, float))
                or not math.isfinite(delta)
                or delta < 0.0
            ):
                raise ValueError(
                    f"min_delta for {key!r} must be finite and nonnegative, "
                    f"got {delta!r}"
                )
        if not isinstance(self.confirm, bool):
            raise TypeError(f"confirm must be a bool, got {self.confirm!r}")
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "min_deltas", tuple(float(d) for d in deltas))


@dataclass(frozen=True)
class Decision:
    """What the rule decided about one row (in step order)."""

    step: int
    decision: str
    score: tuple[float, ...]
    confirm_score: tuple[float, ...] | None


@dataclass(frozen=True)
class Selection:
    """:func:`select`'s (or :func:`in_loop_select`'s) result.

    ``step`` and ``row`` are the selected row's; ``decisions`` holds one
    :class:`Decision` per row, in step order, so two selections over
    the same rows can be compared decision by decision.
    """

    rule: SelectionRule
    step: int
    row: Mapping[str, Any]
    decisions: tuple[Decision, ...]

    @property
    def accepted_steps(self) -> tuple[int, ...]:
        """Every step that became the best, in order (the last is ``step``)."""
        return tuple(
            decision.step
            for decision in self.decisions
            if decision.decision in (FIRST, IMPROVED)
        )


def _rule_of(rule: Any) -> SelectionRule:
    if isinstance(rule, SelectionRule):
        return rule
    selection_rule = getattr(rule, "selection_rule", None)
    if callable(selection_rule):
        resolved = selection_rule()
        if isinstance(resolved, SelectionRule):
            return resolved
    raise TypeError(
        f"rule must be a SelectionRule or an evaluation protocol, got "
        f"{type(rule).__name__}"
    )


def _ordered_rows(scores: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows in step order; refuses a row without a step, or a repeat.

    Two rows at one step are two candidates for one evaluation (a
    snapshot and a checkpoint saved at the same step hold the same
    policy): which comes first would decide ties, so the caller picks
    one artifact series.
    """
    rows = list(scores)
    if not rows:
        raise ValueError("select() needs at least one score row")
    seen: dict[int, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"score row {index} is not a mapping: {row!r}")
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(
                f"score row {index} has no integer 'step' ({step!r}); a best "
                f"model without a recorded step cannot be ordered"
            )
        if step in seen:
            raise ValueError(
                f"score rows {seen[step]} and {index} are both at step {step}: "
                f"select over one artifact series (the snapshots, or the "
                f"checkpoints), not both"
            )
        seen[step] = index
    return sorted(rows, key=lambda row: int(row["step"]))


def _score(
    row: Mapping[str, Any],
    field: str,
    keys: Sequence[str],
    *,
    missing_as_minus_inf: bool = False,
) -> tuple[float, ...]:
    metrics = row.get(field)
    if not isinstance(metrics, Mapping):
        raise ValueError(
            f"score row at step {row.get('step')} has no {field!r}: "
            + (
                "the rule confirms candidates, so rescore with the "
                "confirmation block"
                if field == "confirm_metrics"
                else "not a harness score row"
            )
        )
    values: list[float] = []
    for key in keys:
        if key not in metrics:
            if missing_as_minus_inf:
                values.append(float("-inf"))
                continue
            raise ValueError(
                f"score row at step {row.get('step')}: {field} has no "
                f"selection key {key!r}"
            )
        values.append(float(metrics[key]))
    return tuple(values)


def select(scores: Iterable[Mapping[str, Any]], rule: Any) -> Selection:
    """The best row under ``rule``, with confirmation compared within block.

    ``scores`` are harness rows (``rescore``'s jsonl): each has an
    integer ``step``, the standard block's ``metrics`` and, where the
    rule confirms, ``confirm_metrics``. ``rule`` is a
    :class:`SelectionRule` or a protocol (its ``selection_rule()``).
    Rows are taken in step order, the order the in-loop evaluator saw
    them:

    - the first row is the first best; under ``confirm`` its
      confirmation-block score is banked with it;
    - a later row is a candidate when its standard score
      :func:`improves` on the best's standard score; under ``confirm``
      it is accepted only when its confirmation score also improves on
      the best's *confirmation* score (each block compared with
      itself; see the module docstring for how the in-loop rule
      differs).

    Pure: no I/O, and the rows are not modified. A selection key
    missing from a row, or a confirmation score the rule needs and a
    row lacks, raises ``ValueError`` (the in-loop callback scores a
    missing key as ``-inf``; offline, a missing key means the rows came
    from another protocol).
    """
    resolved = _rule_of(rule)
    rows = _ordered_rows(scores)
    decisions: list[Decision] = []
    best_row: Mapping[str, Any] | None = None
    best_score: tuple[float, ...] | None = None
    best_confirm: tuple[float, ...] | None = None
    for row in rows:
        step = int(row["step"])
        score = _score(row, "metrics", resolved.keys)
        if best_row is None:
            confirm_score = (
                _score(row, "confirm_metrics", resolved.keys)
                if resolved.confirm
                else None
            )
            best_row, best_score, best_confirm = row, score, confirm_score
            decisions.append(Decision(step, FIRST, score, confirm_score))
            continue
        if not improves(score, best_score, resolved.min_deltas):
            decisions.append(Decision(step, NOT_IMPROVED, score, None))
            continue
        confirm_score = None
        if resolved.confirm:
            confirm_score = _score(row, "confirm_metrics", resolved.keys)
            if not improves(confirm_score, best_confirm, resolved.min_deltas):
                decisions.append(
                    Decision(step, CONFIRM_REJECTED, score, confirm_score)
                )
                continue
        best_row, best_score, best_confirm = row, score, confirm_score
        decisions.append(Decision(step, IMPROVED, score, confirm_score))
    assert best_row is not None
    return Selection(resolved, int(best_row["step"]), best_row, tuple(decisions))


def in_loop_select(scores: Iterable[Mapping[str, Any]], rule: Any) -> Selection:
    """Replay the in-loop callback's paired selection over harness rows.

    For audits, not for choosing a model: this is the rule
    ``InfoDictEvalCallback`` applies today under paired evaluation
    (``eval_seed``), bias included. The first best is banked without a
    confirmation score; until a candidate is accepted on a
    confirmation, a candidate's confirmation-block score is compared
    with the first best's *standard*-block score
    (``repo_review_20261016.md`` section 2.2). A missing selection key
    scores ``-inf``, as in the callback. Where it and :func:`select`
    decide a row differently, the difference is the in-loop rule's
    cross-block comparison.

    It replays the paired, ungated rule only. Unpaired with
    confirmation, the callback banks the weaker of a candidate's two
    samples, and a performance gate resets the selection at every
    advance; neither is replayed. Passed a protocol, it refuses an
    unpaired one that confirms (``ValueError``; gated runs have no
    protocol, ``experiments.protocols``). A bare :class:`SelectionRule`
    says nothing about pairing, so the caller vouches that the rows
    come from a paired, ungated run's evaluations.
    """
    if getattr(rule, "paired", True) is False and getattr(rule, "confirm", False):
        raise ValueError(
            f"protocol {getattr(rule, 'id', rule)!r} is unpaired and confirms: "
            f"the in-loop callback banks the weaker of an unpaired "
            f"candidate's two samples, which in_loop_select does not replay"
        )
    resolved = _rule_of(rule)
    rows = _ordered_rows(scores)
    decisions: list[Decision] = []
    best_row: Mapping[str, Any] | None = None
    best_score: tuple[float, ...] | None = None
    best_confirm: tuple[float, ...] | None = None
    for row in rows:
        step = int(row["step"])
        score = _score(row, "metrics", resolved.keys, missing_as_minus_inf=True)
        if best_row is None:
            best_row, best_score, best_confirm = row, score, None
            decisions.append(Decision(step, FIRST, score, None))
            continue
        if not improves(score, best_score, resolved.min_deltas):
            decisions.append(Decision(step, NOT_IMPROVED, score, None))
            continue
        confirm_score = None
        if resolved.confirm:
            confirm_score = _score(
                row, "confirm_metrics", resolved.keys, missing_as_minus_inf=True
            )
            reference = best_score if best_confirm is None else best_confirm
            if not improves(confirm_score, reference, resolved.min_deltas):
                decisions.append(
                    Decision(step, CONFIRM_REJECTED, score, confirm_score)
                )
                continue
        best_row, best_score, best_confirm = row, score, confirm_score
        decisions.append(Decision(step, IMPROVED, score, confirm_score))
    assert best_row is not None
    return Selection(resolved, int(best_row["step"]), best_row, tuple(decisions))
