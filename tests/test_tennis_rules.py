"""Deterministic event-trace tests for cooperative tennis rules."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from courtside_dynamics.envs._tennis_physics import (
    COURT_HALF_LENGTH,
    COURT_HALF_WIDTH,
)
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyEvent,
    RallyEventKind,
    RallyPhase,
    RallyRules,
    RallyStateMachine,
    TerminationReason,
)


def _crossing(
    to_side: CourtSide,
    substep: int,
) -> RallyEvent:
    return RallyEvent(
        kind=(
            RallyEventKind.NET_CROSSING_TO_A
            if to_side is CourtSide.A
            else RallyEventKind.NET_CROSSING_TO_B
        ),
        substep=substep,
        from_side=to_side.opponent,
        to_side=to_side,
        position=(0.0, 0.0, 1.2),
    )


def _racket(
    side: CourtSide,
    substep: int,
    *,
    episode: int | None = None,
) -> RallyEvent:
    return RallyEvent(
        kind=(
            RallyEventKind.BALL_RACKET_A
            if side is CourtSide.A
            else RallyEventKind.BALL_RACKET_B
        ),
        substep=substep,
        contact_episode=episode,
        peak_force=12.0,
    )


def _court(
    side: CourtSide,
    substep: int,
    *,
    position: tuple[float, float, float] | None = None,
    in_bounds: bool | None = None,
    episode: int | None = None,
) -> RallyEvent:
    default_x = -2.0 if side is CourtSide.A else 2.0
    return RallyEvent(
        kind=(
            RallyEventKind.BALL_COURT_A
            if side is CourtSide.A
            else RallyEventKind.BALL_COURT_B
        ),
        substep=substep,
        contact_episode=episode,
        position=position or (default_x, 0.0, 0.0),
        in_bounds=in_bounds,
        peak_force=20.0,
    )


def _advance(
    machine: RallyStateMachine,
    events: Iterable[RallyEvent],
):
    transition = machine.advance(events)
    return transition, transition.after


def _deliver_feed(machine: RallyStateMachine, serving_side: CourtSide, step: int = 0):
    return machine.advance([_crossing(serving_side.opponent, step)])


def _launch_return(
    machine: RallyStateMachine,
    serving_side: CourtSide,
) -> CourtSide:
    receiver = serving_side.opponent
    _deliver_feed(machine, serving_side)
    transition = machine.advance([_racket(receiver, 1)])
    assert transition.valid_racket_hits == (receiver,)
    return receiver


def test_rally_event_kind_string_uses_wire_value():
    assert str(RallyEventKind.BALL_NET) == RallyEventKind.BALL_NET.value


def test_reset_exposes_markov_rally_state_and_no_progress():
    snapshot = RallyStateMachine(serving_side=CourtSide.A).snapshot()
    assert snapshot.phase is RallyPhase.INITIAL_FEED
    assert snapshot.serving_side is CourtSide.A
    assert snapshot.expected_returner is CourtSide.B
    assert snapshot.ball_side is CourtSide.A
    assert snapshot.last_valid_racket_hitter is None
    assert snapshot.rally_count == 0
    assert snapshot.bounce_count == 0
    assert not snapshot.terminated
    info = snapshot.to_info()
    assert info["serve_side_name"] == "a"
    assert info["termination_reason_name"] == "none"


@pytest.mark.parametrize("serving_side", list(CourtSide))
def test_initial_feed_and_first_bounce_are_legal_but_not_a_return(serving_side):
    machine = RallyStateMachine(serving_side=serving_side)
    receiver = serving_side.opponent
    feed = _deliver_feed(machine, serving_side)
    assert feed.after.phase is RallyPhase.AWAITING_RETURN
    assert feed.after.net_crossing_count == 1
    assert feed.after.rally_count == 0

    bounce, snapshot = _advance(machine, [_court(receiver, 1)])
    assert bounce.first_bounces == (receiver,)
    assert snapshot.bounce_count == 1
    assert snapshot.bounce_side is receiver
    assert snapshot.rally_count == 0
    assert not snapshot.terminated


@pytest.mark.parametrize("serving_side", list(CourtSide))
def test_legal_first_bounce_then_return_is_confirmed_on_opponent_bounce(
    serving_side,
):
    machine = RallyStateMachine(serving_side=serving_side)
    receiver = serving_side.opponent
    _deliver_feed(machine, serving_side)
    machine.advance([_court(receiver, 1)])
    hit = machine.advance([_racket(receiver, 2)])
    assert hit.valid_racket_hits == (receiver,)
    assert hit.after.bounce_count == 0

    machine.advance([_crossing(serving_side, 3)])
    landing = machine.advance([_court(serving_side, 4)])
    assert landing.confirmed_returns == (receiver,)
    assert landing.after.rally_count == 1
    assert landing.after.bounce_count == 1
    assert not landing.after.terminated


@pytest.mark.parametrize("serving_side", list(CourtSide))
def test_legal_volley_confirms_previous_return_and_starts_next(serving_side):
    machine = RallyStateMachine(serving_side=serving_side)
    receiver = _launch_return(machine, serving_side)
    machine.advance([_crossing(serving_side, 2)])

    volley = machine.advance([_racket(serving_side, 3)])
    assert volley.confirmed_returns == (receiver,)
    assert volley.valid_racket_hits == (serving_side,)
    assert volley.after.rally_count == 1
    assert volley.after.pending_return_hitter is serving_side
    assert volley.after.expected_returner is receiver
    assert volley.after.bounce_count == 0

    machine.advance([_crossing(receiver, 4)])
    landing = machine.advance([_court(receiver, 5)])
    assert landing.confirmed_returns == (serving_side,)
    assert landing.after.rally_count == 2


@pytest.mark.parametrize("serving_side", list(CourtSide))
def test_second_bounce_terminates_without_extra_progress(serving_side):
    machine = RallyStateMachine(serving_side=serving_side)
    receiver = serving_side.opponent
    _deliver_feed(machine, serving_side)
    machine.advance([_court(receiver, 1, episode=0)])
    second = machine.advance([_court(receiver, 2, episode=1)])
    assert second.after.termination_reason is TerminationReason.SECOND_BOUNCE
    assert second.after.rally_count == 0
    assert second.first_bounces == ()


@pytest.mark.parametrize("serving_side", list(CourtSide))
@pytest.mark.parametrize("lateral", [0.0, -COURT_HALF_WIDTH, COURT_HALF_WIDTH])
def test_line_landings_are_in_bounds(serving_side, lateral):
    machine = RallyStateMachine(serving_side=serving_side)
    receiver = _launch_return(machine, serving_side)
    machine.advance([_crossing(serving_side, 2)])
    x = -COURT_HALF_LENGTH if serving_side is CourtSide.A else COURT_HALF_LENGTH
    landing = machine.advance(
        [_court(serving_side, 3, position=(x, lateral, 0.0))]
    )
    assert landing.after.termination_reason is TerminationReason.NONE
    assert landing.after.rally_count == 1
    assert landing.confirmed_returns == (receiver,)


@pytest.mark.parametrize("serving_side", list(CourtSide))
@pytest.mark.parametrize("axis", ["baseline", "sideline"])
def test_out_of_bounds_landing_terminates_and_does_not_confirm_return(
    serving_side,
    axis,
):
    machine = RallyStateMachine(serving_side=serving_side)
    _launch_return(machine, serving_side)
    machine.advance([_crossing(serving_side, 2)])
    if axis == "baseline":
        sign = -1.0 if serving_side is CourtSide.A else 1.0
        position = (sign * (COURT_HALF_LENGTH + 1e-5), 0.0, 0.0)
    else:
        x = -2.0 if serving_side is CourtSide.A else 2.0
        position = (x, COURT_HALF_WIDTH + 1e-5, 0.0)
    landing = machine.advance([_court(serving_side, 3, position=position)])
    assert landing.after.termination_reason is TerminationReason.OUT_OF_BOUNDS
    assert landing.after.rally_count == 0
    assert landing.confirmed_returns == ()


@pytest.mark.parametrize(
    "kind,reason",
    [
        (RallyEventKind.HUMANOID_A_NET, TerminationReason.HUMANOID_A_NET),
        (RallyEventKind.HUMANOID_B_NET, TerminationReason.HUMANOID_B_NET),
        (RallyEventKind.RACKET_A_NET, TerminationReason.RACKET_A_NET),
        (RallyEventKind.RACKET_B_NET, TerminationReason.RACKET_B_NET),
        (RallyEventKind.BALL_HUMANOID_A, TerminationReason.BALL_HUMANOID_A),
        (RallyEventKind.BALL_HUMANOID_B, TerminationReason.BALL_HUMANOID_B),
        (RallyEventKind.BALL_NET, TerminationReason.BALL_NET),
    ],
)
def test_physical_tennis_faults_have_side_specific_terminal_reasons(kind, reason):
    transition = RallyStateMachine().advance([RallyEvent(kind, substep=0)])
    assert transition.after.termination_reason is reason
    flags = [
        value
        for key, value in transition.after.to_info().items()
        if key.startswith("term_")
    ]
    assert sum(bool(value) for value in flags) == 1


def test_ball_net_can_be_recorded_without_progress_or_termination():
    machine = RallyStateMachine(rules=RallyRules(ball_net_is_fault=False))
    transition = machine.advance(
        [RallyEvent(RallyEventKind.BALL_NET, substep=0, contact_episode=0)]
    )
    assert not transition.after.terminated
    assert transition.after.rally_count == 0
    assert transition.after.event_counts[RallyEventKind.BALL_NET] == 1
    assert transition.processed_events == transition.observed_events


def test_hitter_side_bounce_before_crossing_is_a_failed_return():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    receiver = _launch_return(machine, CourtSide.A)
    failed = machine.advance([_court(receiver, 2)])
    assert failed.after.termination_reason is TerminationReason.FAILED_TO_CROSS
    assert failed.after.rally_count == 0


def test_reverse_crossing_does_not_farm_rally_progress():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    _launch_return(machine, CourtSide.A)
    machine.advance([_crossing(CourtSide.A, 2)])
    reverse = machine.advance([_crossing(CourtSide.B, 3)])
    assert reverse.after.termination_reason is TerminationReason.REVERSE_CROSSING
    assert reverse.after.rally_count == 0


def test_forgiving_reverse_crossing_records_but_never_progresses():
    machine = RallyStateMachine(
        serving_side=CourtSide.A,
        rules=RallyRules(strict_reverse_crossing=False),
    )
    _launch_return(machine, CourtSide.A)
    machine.advance([_crossing(CourtSide.A, 2)])
    reverse = machine.advance([_crossing(CourtSide.B, 3)])
    assert not reverse.after.terminated
    assert reverse.after.rally_count == 0
    assert reverse.after.net_crossing_count == 3


def test_expected_player_cannot_hit_before_initial_feed_crosses():
    transition = RallyStateMachine(serving_side=CourtSide.A).advance(
        [_racket(CourtSide.B, 0)]
    )
    assert transition.after.termination_reason is TerminationReason.PREMATURE_HIT


def test_wrong_player_contact_is_a_fault():
    transition = RallyStateMachine(serving_side=CourtSide.A).advance(
        [_racket(CourtSide.A, 0)]
    )
    assert transition.after.termination_reason is TerminationReason.WRONG_HITTER


def test_duplicate_contact_episode_is_suppressed_but_real_double_hit_faults():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    _deliver_feed(machine, CourtSide.A)
    first = machine.advance([_racket(CourtSide.B, 1, episode=0)])
    assert first.after.legal_hit_count == 1

    duplicate = machine.advance([_racket(CourtSide.B, 2, episode=0)])
    assert len(duplicate.suppressed_events) == 1
    assert duplicate.after.legal_hit_count == 1
    assert duplicate.after.duplicate_contact_suppressed_count == 1
    assert not duplicate.after.terminated

    separated = machine.advance([_racket(CourtSide.B, 3, episode=1)])
    assert separated.after.termination_reason is TerminationReason.DOUBLE_HIT
    assert separated.after.legal_hit_count == 1
    assert separated.after.rally_count == 0


def test_forgiving_double_hit_is_ignored_without_progress():
    machine = RallyStateMachine(
        serving_side=CourtSide.A,
        rules=RallyRules(strict_double_hit=False),
    )
    _deliver_feed(machine, CourtSide.A)
    machine.advance([_racket(CourtSide.B, 1, episode=0)])
    recontact = machine.advance([_racket(CourtSide.B, 2, episode=1)])
    assert not recontact.after.terminated
    assert recontact.after.legal_hit_count == 1
    assert recontact.after.rally_count == 0


def test_same_substep_duplicate_without_episode_id_is_idempotent():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    _deliver_feed(machine, CourtSide.A)
    event = _racket(CourtSide.B, 1)
    transition = machine.advance([event, event])
    assert transition.after.legal_hit_count == 1
    assert transition.after.duplicate_contact_suppressed_count == 1
    assert not transition.after.terminated


def test_duplicate_crossing_delivery_is_idempotent_across_calls():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    crossing = _crossing(CourtSide.B, 5)
    first = machine.advance([crossing])
    duplicate = machine.advance([crossing])
    assert first.after.net_crossing_count == 1
    assert not duplicate.after.terminated
    assert duplicate.after.net_crossing_count == 1
    assert duplicate.after.duplicate_contact_suppressed_count == 0
    assert duplicate.after.duplicate_event_suppressed_count == 1
    assert duplicate.duplicate_events == (crossing,)


def test_stale_cross_batch_event_cannot_rewrite_rally_chronology():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    machine.advance([_crossing(CourtSide.B, 10)])
    stale = machine.advance([_racket(CourtSide.B, 5)])
    assert not stale.after.terminated
    assert stale.after.legal_hit_count == 0
    assert stale.after.last_event_substep == 10
    assert stale.after.stale_event_suppressed_count == 1
    assert stale.suppressed_events == (_racket(CourtSide.B, 5),)
    assert stale.stale_events == stale.suppressed_events
    assert stale.duplicate_events == ()


@pytest.mark.parametrize(
    "event",
    [
        RallyEvent(
            RallyEventKind.BALL_COURT_B,
            substep=0,
            position=(-2.0, 0.0, 0.0),
        ),
        RallyEvent(
            RallyEventKind.BALL_COURT_B,
            substep=0,
            position=(COURT_HALF_LENGTH + 1.0, 0.0, 0.0),
            in_bounds=True,
        ),
    ],
)
def test_contradictory_court_payload_is_rejected_before_mutation(event):
    machine = RallyStateMachine()
    before = machine.snapshot()
    with pytest.raises(ValueError, match="contradicts"):
        machine.advance([event])
    assert machine.snapshot() == before


def test_invalid_late_event_cannot_leak_partial_batch_state():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    before = machine.snapshot()
    invalid_court = RallyEvent(RallyEventKind.BALL_COURT_B, substep=1)
    with pytest.raises(ValueError, match="require a position"):
        machine.advance([_crossing(CourtSide.B, 0), invalid_court])
    assert machine.snapshot() == before


def test_simultaneous_two_racket_contacts_are_deterministic_fault():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    _deliver_feed(machine, CourtSide.A)
    transition = machine.advance(
        [_racket(CourtSide.B, 1), _racket(CourtSide.A, 1)]
    )
    assert (
        transition.after.termination_reason
        is TerminationReason.SIMULTANEOUS_RACKET_CONTACT
    )
    assert transition.after.legal_hit_count == 0


def test_out_ball_precedes_simultaneous_racket_fault():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    _deliver_feed(machine, CourtSide.A)
    events = [
        _court(
            CourtSide.B,
            2,
            position=(2.0, COURT_HALF_WIDTH + 1e-5, 0.0),
        ),
        _racket(CourtSide.A, 2),
        _racket(CourtSide.B, 2),
    ]
    transition = machine.advance(events)
    assert transition.after.termination_reason is TerminationReason.OUT_OF_BOUNDS


def test_second_bounce_precedes_simultaneous_racket_fault():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    _deliver_feed(machine, CourtSide.A)
    machine.advance([_court(CourtSide.B, 1, episode=0)])
    events = [
        _court(CourtSide.B, 2, episode=1),
        _racket(CourtSide.A, 2),
        _racket(CourtSide.B, 2),
    ]
    transition = machine.advance(events)
    assert transition.after.termination_reason is TerminationReason.SECOND_BOUNCE


def test_incoming_crossing_precedes_same_substep_volley():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    transition = machine.advance(
        [_racket(CourtSide.B, 0), _crossing(CourtSide.B, 0)]
    )
    assert not transition.after.terminated
    assert transition.after.legal_hit_count_b == 1
    assert [event.kind for event in transition.processed_events] == [
        RallyEventKind.NET_CROSSING_TO_B,
        RallyEventKind.BALL_RACKET_B,
    ]


def test_outgoing_hit_precedes_same_substep_crossing():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    _deliver_feed(machine, CourtSide.A)
    transition = machine.advance(
        [_crossing(CourtSide.A, 1), _racket(CourtSide.B, 1)]
    )
    assert not transition.after.terminated
    assert transition.after.pending_return_hitter is CourtSide.B
    assert transition.after.pending_return_crossed_net
    assert [event.kind for event in transition.processed_events] == [
        RallyEventKind.BALL_RACKET_B,
        RallyEventKind.NET_CROSSING_TO_A,
    ]


def test_fault_priority_is_independent_of_input_order():
    lower_priority = RallyEvent(RallyEventKind.BALL_NET, substep=4)
    higher_priority = RallyEvent(RallyEventKind.NONFINITE_STATE, substep=4)
    for events in ([lower_priority, higher_priority], [higher_priority, lower_priority]):
        transition = RallyStateMachine().advance(events)
        assert (
            transition.after.termination_reason
            is TerminationReason.NONFINITE_STATE
        )
        assert transition.processed_events == (higher_priority,)


def test_safety_fault_precedes_simultaneous_rackets_in_same_substep():
    events = [
        _racket(CourtSide.A, 4),
        _racket(CourtSide.B, 4),
        RallyEvent(RallyEventKind.UNSAFE_PHYSICS, substep=4),
    ]
    transition = RallyStateMachine().advance(events)
    assert transition.after.termination_reason is TerminationReason.UNSAFE_PHYSICS


def test_events_are_processed_by_substep_not_input_order():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    transition = machine.advance(
        [_racket(CourtSide.B, 2), _crossing(CourtSide.B, 1)]
    )
    assert not transition.after.terminated
    assert transition.after.legal_hit_count_b == 1
    assert [event.substep for event in transition.processed_events] == [1, 2]


def test_terminal_state_is_sticky_and_later_events_are_ignored():
    machine = RallyStateMachine()
    terminal = machine.advance([RallyEvent(RallyEventKind.BALL_NET, substep=0)])
    after = machine.advance([_crossing(CourtSide.B, 1)])
    assert after.before == terminal.after
    assert after.after == terminal.after
    assert after.processed_events == ()
    assert after.ignored_events == after.observed_events
    info = after.to_info()
    assert info["observed_event_net_crossing_to_b"] is True
    assert info["event_net_crossing_to_b"] is False


def test_terminal_state_ignores_malformed_later_event_without_validation():
    machine = RallyStateMachine()
    terminal = machine.advance([RallyEvent(RallyEventKind.BALL_NET, substep=0)])
    malformed = RallyEvent(RallyEventKind.BALL_COURT_B, substep=1)
    ignored = machine.advance([malformed])
    assert ignored.after == terminal.after
    assert ignored.ignored_events == (malformed,)


def test_control_step_peaks_update_episode_diagnostics_without_new_edge():
    machine = RallyStateMachine()
    machine.advance([], contact_peaks={RallyEventKind.BALL_RACKET_A: 2.0})
    transition = machine.advance(
        [],
        contact_peaks={RallyEventKind.BALL_RACKET_A: 7.0},
    )
    assert transition.after.contact_peaks[RallyEventKind.BALL_RACKET_A] == 7.0
    info = transition.to_info()
    assert info["step_contact_peak_ball_racket_a"] == 7.0
    assert info["episode_contact_peak_ball_racket_a"] == 7.0


def test_scripted_two_player_oracle_completes_two_controlled_returns():
    machine = RallyStateMachine(serving_side=CourtSide.A)
    trace = [
        _crossing(CourtSide.B, 0),  # initial feed
        _racket(CourtSide.B, 1),
        _crossing(CourtSide.A, 2),
        _racket(CourtSide.A, 3),  # volley confirms B's return
        _crossing(CourtSide.B, 4),
        _court(CourtSide.B, 5),  # landing confirms A's return
    ]
    transition = machine.advance(reversed(trace))
    assert transition.after.rally_count == 2
    assert transition.after.valid_return_count_a == 1
    assert transition.after.valid_return_count_b == 1
    assert transition.after.legal_hit_count == 2
    assert transition.after.bounce_count == 1
    assert not transition.after.terminated


def test_same_substep_feed_crossing_and_landing_is_legal():
    """A feed that crosses the net and lands in the same substep group
    must be a legal first bounce, not FAILED_TO_CROSS: the fault
    pre-scan projects the crossing before classifying the landing.
    Hand-traced correct before this test existed; pinned because a
    regression here silently turns every fast feed into a fault."""
    machine = RallyStateMachine(serving_side=CourtSide.A)
    transition, after = _advance(
        machine, [_crossing(CourtSide.B, 0), _court(CourtSide.B, 0)]
    )
    assert not after.terminated
    assert after.phase is RallyPhase.AWAITING_RETURN
    assert transition.first_bounces == (CourtSide.B,)
    assert after.bounce_count == 1


def test_same_substep_return_crossing_and_landing_confirms_the_return():
    """The RETURN_IN_FLIGHT sibling: a return that crosses and lands in
    one substep group confirms the pending return."""
    machine = RallyStateMachine(serving_side=CourtSide.A)
    _launch_return(machine, CourtSide.A)  # B's racket hit is pending
    transition, after = _advance(
        machine, [_crossing(CourtSide.A, 2), _court(CourtSide.A, 2)]
    )
    assert not after.terminated
    assert transition.confirmed_returns == (CourtSide.B,)
    assert after.rally_count == 1
    assert after.valid_return_count_b == 1


def test_explicit_from_side_a_is_honored_on_the_feed():
    """``CourtSide.A`` is falsy (IntEnum value 0). The reducer used to
    evaluate ``event.from_side or self._ball_side``, silently discarding
    an explicit ``from_side=CourtSide.A``; on every consistent trace the
    fallback coincided with the discarded value, so no end-to-end trace
    can distinguish the two forms today. The source resolution therefore
    lives in one unit-testable helper, ``_event_source``, and this test
    pins its semantics directly: reverting the helper to the ``or`` form
    fails here even though every rally trace stays green."""
    machine = RallyStateMachine(serving_side=CourtSide.A)

    # The unit pin: explicit A wins over a DIFFERENT fallback.
    explicit_a = _crossing(CourtSide.B, 0)
    assert explicit_a.from_side is CourtSide.A  # explicit, and falsy
    assert (
        machine._event_source(explicit_a, ball_side=CourtSide.B)
        is CourtSide.A
    ), "explicit from_side=CourtSide.A was discarded for the fallback"
    # And the fallback engages only when from_side is truly absent.
    implicit = RallyEvent(
        kind=RallyEventKind.NET_CROSSING_TO_B, substep=0
    )
    assert (
        machine._event_source(implicit, ball_side=CourtSide.B)
        is CourtSide.B
    )

    # The end-to-end sanity check: the explicit-A feed still advances.
    transition = machine.advance([explicit_a])
    assert transition.after.phase is RallyPhase.AWAITING_RETURN
    assert not transition.after.terminated
    assert transition.after.feed_crossed_net is True


def test_court_geometry_parameterizes_the_out_of_bounds_rule():
    """The reducer measures OUT_OF_BOUNDS against its ``CourtGeometry``:
    the same landing is in on the regulation court and out on a
    PaddleTennis-scale court (probe-frozen half-length 6.5), so the
    small-court era reuses the rules stack instead of forking it."""
    from courtside_dynamics.envs._tennis_physics import CourtGeometry

    landing = (7.0, 0.0, 0.0)  # inside regulation (11.885), outside 6.5

    regulation = RallyStateMachine(serving_side=CourtSide.A)
    _, after = _advance(
        regulation,
        [
            _crossing(CourtSide.B, 0),
            _court(CourtSide.B, 1, position=landing),
        ],
    )
    assert not after.terminated
    assert after.bounce_count == 1

    small = RallyStateMachine(
        serving_side=CourtSide.A,
        court=CourtGeometry(half_length=6.5, half_width=4.115),
    )
    _, after = _advance(
        small,
        [
            _crossing(CourtSide.B, 0),
            _court(CourtSide.B, 1, position=landing),
        ],
    )
    assert after.terminated
    assert after.termination_reason is TerminationReason.OUT_OF_BOUNDS


def test_court_geometry_validates_dimensions():
    from courtside_dynamics.envs._tennis_physics import CourtGeometry

    with pytest.raises(ValueError, match="half_length"):
        CourtGeometry(half_length=0.0)
    with pytest.raises(ValueError, match="half_width"):
        CourtGeometry(half_width=float("nan"))
    with pytest.raises(ValueError, match="tolerance"):
        CourtGeometry(tolerance=-1e-6)


def test_regulation_geometry_matches_module_predicate():
    """The default CourtGeometry and the module-level predicate are the
    same measurement; drift between them would let the sampler and the
    reducer disagree on the same landing."""
    from courtside_dynamics.envs._tennis_physics import (
        REGULATION_COURT,
        is_in_bounds,
    )

    for x, y in (
        (0.0, 0.0),
        (COURT_HALF_LENGTH, COURT_HALF_WIDTH),
        (COURT_HALF_LENGTH + 0.01, 0.0),
        (0.0, COURT_HALF_WIDTH + 0.01),
        (-COURT_HALF_LENGTH - 0.01, 0.0),
    ):
        assert REGULATION_COURT.is_in_bounds(x, y) == is_in_bounds(x, y)
