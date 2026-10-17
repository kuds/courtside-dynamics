"""Tests for the pure offline selection (experiments.select).

``select()`` implements the in-loop lexicographic per-key-delta rule
with confirmation compared within block; ``in_loop_select()`` replays
the in-loop callback's cross-block confirmation
(``repo_review_20261016.md`` section 2.2) for audits.
"""

from __future__ import annotations

import itertools
import random

import pytest

from courtside_dynamics.experiments.select import (
    CONFIRM_REJECTED,
    FIRST,
    IMPROVED,
    NOT_IMPROVED,
    SelectionRule,
    improves,
    in_loop_select,
    select,
)

KEYS = ("headline_ep_mean", "success_rate", "episode_reward_mean")


def _row(step, headline, success, reward, confirm=None):
    row = {
        "step": step,
        "metrics": {
            "headline_ep_mean": headline,
            "success_rate": success,
            "episode_reward_mean": reward,
        },
        "confirm_metrics": None,
    }
    if confirm is not None:
        row["confirm_metrics"] = dict(zip(KEYS, confirm, strict=True))
    return row


def test_improves_is_the_in_loop_callbacks_comparison():
    """The callback delegates to improves(); pin them on random scores
    (ties and near-ties included) against the callback's own method."""
    from courtside_dynamics.callbacks.info_dict_eval import InfoDictEvalCallback

    class _Stub:
        num_envs = 1

    callback = InfoDictEvalCallback(
        _Stub(),  # type: ignore[arg-type]
        best_metric_keys=KEYS,
        best_metric_min_delta={"headline_ep_mean": 0.05, "episode_reward_mean": 0.25},
    )
    rng = random.Random(7)
    grid = (0.0, 0.03, 0.05, 0.06, 0.25, 0.3, 1.0)
    for _ in range(2000):
        score = tuple(rng.choice(grid) for _ in KEYS)
        best = tuple(rng.choice(grid) for _ in KEYS)
        assert callback._improves(score, best) is improves(
            score, best, (0.05, 0.0, 0.25)
        )
    assert improves((0.0,), None, (0.0,))
    assert not improves((1.0, 2.0), (1.0, 2.0), (0.0, 0.0))  # a full tie


def test_rule_validation():
    with pytest.raises(ValueError, match="at least one key"):
        SelectionRule((), ())
    with pytest.raises(ValueError, match="min_deltas"):
        SelectionRule(("a",), (0.1, 0.2))
    with pytest.raises(ValueError, match="nonnegative"):
        SelectionRule(("a",), (-0.1,))
    with pytest.raises(ValueError, match="repeat"):
        SelectionRule(("a", "a"), (0.0, 0.0))
    with pytest.raises(TypeError):
        SelectionRule(("a",), (0.0,), confirm=1)  # type: ignore[arg-type]


def test_select_is_lexicographic_with_per_key_deltas_and_ordered_by_step():
    rule = SelectionRule(KEYS, (0.5 / 30, 0.5 / 30, 0.25))
    rows = [
        _row(75_000, 0.1, 0.1, 9.0),  # a better headline wins on the first key
        _row(25_000, 0.0, 0.0, 5.0),
        _row(50_000, 0.0, 0.0, 5.2),  # within the reward delta: not a best
        _row(100_000, 0.1, 0.1, 50.0),  # a tied headline: the reward decides
    ]
    chosen = select(rows, rule)
    assert chosen.step == 100_000
    assert [d.step for d in chosen.decisions] == [25_000, 50_000, 75_000, 100_000]
    assert [d.decision for d in chosen.decisions] == [
        FIRST,
        NOT_IMPROVED,
        IMPROVED,
        IMPROVED,
    ]
    assert chosen.accepted_steps == (25_000, 75_000, 100_000)
    # Pure: the rows are untouched.
    assert rows[0]["metrics"]["headline_ep_mean"] == 0.1


def test_confirmation_compares_each_block_with_itself():
    """The 2.2 case: the first best reads 1.0 on the standard block and
    0.6 on the harder confirmation block; a candidate that beats it on
    both blocks (1.2 vs 1.0, 0.8 vs 0.6) is accepted within block. The
    in-loop rule compares the candidate's 0.8 confirmation with the
    first best's 1.0 *standard* score and rejects it."""
    rule = SelectionRule(("headline",), (0.0,), confirm=True)
    rows = [
        {"step": 1, "metrics": {"headline": 1.0}, "confirm_metrics": {"headline": 0.6}},
        {"step": 2, "metrics": {"headline": 1.2}, "confirm_metrics": {"headline": 0.8}},
    ]
    within = select(rows, rule)
    assert within.step == 2
    assert [d.decision for d in within.decisions] == [FIRST, IMPROVED]
    assert within.decisions[0].confirm_score == (0.6,)
    in_loop = in_loop_select(rows, rule)
    assert in_loop.step == 1
    assert [d.decision for d in in_loop.decisions] == [FIRST, CONFIRM_REJECTED]
    # A candidate that fails its own confirmation is rejected by both.
    rows.append(
        {"step": 3, "metrics": {"headline": 2.0}, "confirm_metrics": {"headline": 0.7}}
    )
    assert select(rows, rule).decisions[-1].decision == CONFIRM_REJECTED
    # Once a confirmed best exists, the in-loop rule is within block too.
    after = [
        {"step": 1, "metrics": {"headline": 1.0}, "confirm_metrics": {"headline": 0.6}},
        {"step": 2, "metrics": {"headline": 1.2}, "confirm_metrics": {"headline": 1.1}},
        {"step": 3, "metrics": {"headline": 1.3}, "confirm_metrics": {"headline": 1.0}},
    ]
    assert [d.decision for d in in_loop_select(after, rule).decisions] == [
        FIRST,
        IMPROVED,
        CONFIRM_REJECTED,
    ]
    assert [d.decision for d in select(after, rule).decisions] == [
        FIRST,
        IMPROVED,
        CONFIRM_REJECTED,
    ]


def test_select_agrees_with_the_in_loop_replay_without_confirmation():
    """Without a confirmation block the two rules are one rule."""
    rule = SelectionRule(KEYS, (0.0, 0.0, 0.25))
    rng = random.Random(3)
    for _ in range(200):
        rows = [
            _row(step, rng.choice((0.0, 0.1, 0.2)), rng.choice((0.0, 0.5)), rng.uniform(-1, 1))
            for step in range(1, 9)
        ]
        assert select(rows, rule).decisions == in_loop_select(rows, rule).decisions


def test_select_refuses_rows_it_cannot_order_or_score():
    rule = SelectionRule(KEYS, (0.0, 0.0, 0.0), confirm=True)
    with pytest.raises(ValueError, match="at least one"):
        select([], rule)
    with pytest.raises(ValueError, match="both at step"):
        select(
            [_row(1, 0, 0, 0, (0, 0, 0)), _row(1, 0, 0, 0, (0, 0, 0))], rule
        )
    with pytest.raises(ValueError, match="integer 'step'"):
        select([{**_row(1, 0, 0, 0), "step": None}], rule)
    # The first best needs its confirmation score under a confirming rule.
    with pytest.raises(ValueError, match="confirm_metrics"):
        select([_row(1, 0, 0, 0)], rule)
    missing = _row(1, 0, 0, 0, (0, 0, 0))
    del missing["metrics"]["success_rate"]
    with pytest.raises(ValueError, match="success_rate"):
        select([missing], rule)
    # The in-loop replay scores a missing key as -inf, like the callback.
    assert in_loop_select([missing], rule).decisions[0].score[1] == float("-inf")
    with pytest.raises(TypeError, match="SelectionRule"):
        select([missing], object())


def test_select_accepts_a_protocol_as_the_rule():
    class _Protocol:
        def selection_rule(self):
            return SelectionRule(("a",), (0.0,))

    rows = [
        {"step": step, "metrics": {"a": value}, "confirm_metrics": None}
        for step, value in zip((1, 2, 3), (0.0, 2.0, 1.0), strict=True)
    ]
    assert select(rows, _Protocol()).step == 2


def test_rows_order_does_not_matter():
    rule = SelectionRule(("a",), (0.0,))
    rows = [
        {"step": step, "metrics": {"a": value}}
        for step, value in ((1, 0.0), (2, 1.0), (3, 1.0), (4, 0.5))
    ]
    for permutation in itertools.permutations(rows):
        assert select(permutation, rule).step == 2
