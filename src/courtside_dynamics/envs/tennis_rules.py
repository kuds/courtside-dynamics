"""Deterministic cooperative-tennis rally rules.

This module deliberately has no MuJoCo dependency.  Physics code translates
substep contacts and net-plane crossings into :class:`RallyEvent` objects;
the state machine below decides whether those events form a legal rally.  The
separation keeps tennis legality deterministic and directly unit-testable.

The initial ballistic feed is not a scored return.  A racket return becomes
valid only when it has crossed the net and is then either volleyed by the
expected player or lands in bounds on that player's side.  Merely crossing
the net therefore cannot farm rally credit for a shot that later lands out.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from itertools import groupby
from types import MappingProxyType
from typing import Final

from courtside_dynamics.envs._tennis_physics import (
    REGULATION_COURT,
    CourtGeometry,
)


class CourtSide(IntEnum):
    """The two court halves in the fixed court coordinate frame."""

    A = 0  # negative x
    B = 1  # positive x

    @property
    def opponent(self) -> CourtSide:
        """Return the opposite court side."""
        return CourtSide.B if self is CourtSide.A else CourtSide.A

    @property
    def label(self) -> str:
        """Return the stable lower-case label used in diagnostics."""
        return "a" if self is CourtSide.A else "b"

    @classmethod
    def from_x(cls, x: float, *, tie_breaker: CourtSide) -> CourtSide:
        """Classify an x-coordinate, using ``tie_breaker`` at the net plane."""
        if x < 0.0:
            return cls.A
        if x > 0.0:
            return cls.B
        return tie_breaker


class RallyPhase(IntEnum):
    """Small phase code suitable for observations and scalar logging."""

    INITIAL_FEED = 0
    AWAITING_RETURN = 1
    RETURN_IN_FLIGHT = 2
    TERMINAL = 3


class RallyEventKind(StrEnum):
    """Semantic events emitted by the substep physics sampler."""

    BALL_RACKET_A = "ball_racket_a"
    BALL_RACKET_B = "ball_racket_b"
    BALL_COURT_A = "ball_court_a"
    BALL_COURT_B = "ball_court_b"
    BALL_NET = "ball_net"
    HUMANOID_A_NET = "humanoid_a_net"
    HUMANOID_B_NET = "humanoid_b_net"
    RACKET_A_NET = "racket_a_net"
    RACKET_B_NET = "racket_b_net"
    # Tennis-faithful anti-exploit events beyond the required minimum list.
    BALL_HUMANOID_A = "ball_humanoid_a"
    BALL_HUMANOID_B = "ball_humanoid_b"
    NET_CROSSING_TO_A = "net_crossing_to_a"
    NET_CROSSING_TO_B = "net_crossing_to_b"
    NONFINITE_STATE = "nonfinite_state"
    UNSAFE_PHYSICS = "unsafe_physics"


CONTACT_EVENT_KINDS: Final = frozenset(
    {
        RallyEventKind.BALL_RACKET_A,
        RallyEventKind.BALL_RACKET_B,
        RallyEventKind.BALL_COURT_A,
        RallyEventKind.BALL_COURT_B,
        RallyEventKind.BALL_NET,
        RallyEventKind.HUMANOID_A_NET,
        RallyEventKind.HUMANOID_B_NET,
        RallyEventKind.RACKET_A_NET,
        RallyEventKind.RACKET_B_NET,
        RallyEventKind.BALL_HUMANOID_A,
        RallyEventKind.BALL_HUMANOID_B,
    }
)

RACKET_EVENT_KINDS: Final = frozenset(
    {RallyEventKind.BALL_RACKET_A, RallyEventKind.BALL_RACKET_B}
)
COURT_EVENT_KINDS: Final = frozenset(
    {RallyEventKind.BALL_COURT_A, RallyEventKind.BALL_COURT_B}
)
CROSSING_EVENT_KINDS: Final = frozenset(
    {RallyEventKind.NET_CROSSING_TO_A, RallyEventKind.NET_CROSSING_TO_B}
)


class TerminationReason(IntEnum):
    """Mutually exclusive terminal causes, ordered by fault priority."""

    NONE = 0
    NONFINITE_STATE = 1
    UNSAFE_PHYSICS = 2
    HUMANOID_A_NET = 3
    HUMANOID_B_NET = 4
    RACKET_A_NET = 5
    RACKET_B_NET = 6
    BALL_HUMANOID_A = 7
    BALL_HUMANOID_B = 8
    BALL_NET = 9
    OUT_OF_BOUNDS = 10
    SECOND_BOUNCE = 11
    FAILED_TO_CROSS = 12
    REVERSE_CROSSING = 13
    WRONG_HITTER = 14
    DOUBLE_HIT = 15
    PREMATURE_HIT = 16
    SIMULTANEOUS_RACKET_CONTACT = 17


TERMINATION_REASON_NAMES: Final[Mapping[TerminationReason, str]] = MappingProxyType(
    {
        reason: reason.name.lower()
        for reason in TerminationReason
    }
)


@dataclass(frozen=True, slots=True)
class RallyRules:
    """Configurable v1 rally-rule switches."""

    ball_net_is_fault: bool = True
    strict_reverse_crossing: bool = True
    strict_double_hit: bool = True


@dataclass(frozen=True, slots=True)
class RallyEvent:
    """One ordered semantic event from a physics substep.

    ``contact_episode`` is a monotonically increasing identifier scoped to
    the event kind.  Re-delivery of an already-consumed episode is suppressed
    idempotently.  Synthetic tests may omit it; exact same-kind/same-substep
    duplicates are still suppressed.
    """

    kind: RallyEventKind
    substep: int
    event_id: int | None = None
    contact_episode: int | None = None
    position: tuple[float, float, float] | None = None
    in_bounds: bool | None = None
    peak_force: float = 0.0
    geom_names: tuple[str, str] | None = None
    ball_position: tuple[float, float, float] | None = None
    ball_velocity: tuple[float, float, float] | None = None
    from_side: CourtSide | None = None
    to_side: CourtSide | None = None

    def __post_init__(self) -> None:
        if self.substep < 0:
            raise ValueError("substep must be non-negative")
        if self.event_id is not None and self.event_id < 0:
            raise ValueError("event_id must be non-negative")
        if self.contact_episode is not None and self.contact_episode < 0:
            raise ValueError("contact_episode must be non-negative")
        if not math.isfinite(self.peak_force) or self.peak_force < 0.0:
            raise ValueError("peak_force must be finite and non-negative")
        for name, values in (
            ("position", self.position),
            ("ball_position", self.ball_position),
            ("ball_velocity", self.ball_velocity),
        ):
            if values is not None:
                if len(values) != 3:
                    raise ValueError(f"{name} must contain exactly three values")
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class RallySnapshot:
    """Immutable public snapshot of all rule state needed by a policy."""

    phase: RallyPhase
    serving_side: CourtSide
    expected_returner: CourtSide
    ball_side: CourtSide
    last_valid_racket_hitter: CourtSide | None
    pending_return_hitter: CourtSide | None
    pending_return_crossed_net: bool
    feed_crossed_net: bool
    bounce_side: CourtSide | None
    bounce_count: int
    net_crossing_count: int
    shot_crossing_count: int
    legal_hit_count: int
    legal_hit_count_a: int
    legal_hit_count_b: int
    rally_count: int
    valid_return_count_a: int
    valid_return_count_b: int
    duplicate_contact_suppressed_count: int
    duplicate_event_suppressed_count: int
    stale_event_suppressed_count: int
    last_event_substep: int
    termination_reason: TerminationReason
    ball_net_is_fault: bool
    strict_reverse_crossing: bool
    strict_double_hit: bool
    event_counts: Mapping[RallyEventKind, int]
    contact_peaks: Mapping[RallyEventKind, float]

    @property
    def terminated(self) -> bool:
        return self.termination_reason is not TerminationReason.NONE

    def to_info(self) -> dict[str, int | float | bool | str]:
        """Return stable scalar diagnostics for Gymnasium ``info``."""
        info: dict[str, int | float | bool | str] = {
            "rally_phase": int(self.phase),
            "serve_side": int(self.serving_side),
            "serve_side_name": self.serving_side.label,
            "expected_returner": int(self.expected_returner),
            "expected_returner_name": self.expected_returner.label,
            "ball_side": int(self.ball_side),
            "ball_side_name": self.ball_side.label,
            "last_valid_racket_hitter": (
                -1
                if self.last_valid_racket_hitter is None
                else int(self.last_valid_racket_hitter)
            ),
            "pending_return_hitter": (
                -1
                if self.pending_return_hitter is None
                else int(self.pending_return_hitter)
            ),
            "pending_return": self.pending_return_hitter is not None,
            "pending_return_crossed_net": self.pending_return_crossed_net,
            "feed_crossed_net": self.feed_crossed_net,
            "bounce_side": -1 if self.bounce_side is None else int(self.bounce_side),
            "bounce_count": self.bounce_count,
            "net_crossing_count": self.net_crossing_count,
            "shot_crossing_count": self.shot_crossing_count,
            "legal_hit_count": self.legal_hit_count,
            "legal_hit_count_a": self.legal_hit_count_a,
            "legal_hit_count_b": self.legal_hit_count_b,
            "rally_count": self.rally_count,
            "valid_return_count": self.rally_count,
            "valid_return_count_a": self.valid_return_count_a,
            "valid_return_count_b": self.valid_return_count_b,
            "duplicate_contact_suppressed_count": (
                self.duplicate_contact_suppressed_count
            ),
            "duplicate_event_suppressed_count": self.duplicate_event_suppressed_count,
            "stale_event_suppressed_count": self.stale_event_suppressed_count,
            "last_event_substep": self.last_event_substep,
            "ball_net_is_fault": self.ball_net_is_fault,
            "strict_reverse_crossing": self.strict_reverse_crossing,
            "strict_double_hit": self.strict_double_hit,
            "rally_terminal": self.terminated,
            "termination_reason": int(self.termination_reason),
            "termination_reason_name": TERMINATION_REASON_NAMES[
                self.termination_reason
            ],
        }
        for kind in RallyEventKind:
            info[f"event_count_{kind.value}"] = int(self.event_counts.get(kind, 0))
            if kind in CONTACT_EVENT_KINDS:
                info[f"episode_contact_peak_{kind.value}"] = float(
                    self.contact_peaks.get(kind, 0.0)
                )
        for reason in TerminationReason:
            if reason is TerminationReason.NONE:
                continue
            info[f"term_{TERMINATION_REASON_NAMES[reason]}"] = (
                self.termination_reason is reason
            )
        return info


@dataclass(frozen=True, slots=True)
class RallyTransition:
    """Result of consuming one ordered event batch."""

    before: RallySnapshot
    after: RallySnapshot
    observed_events: tuple[RallyEvent, ...]
    processed_events: tuple[RallyEvent, ...]
    suppressed_events: tuple[RallyEvent, ...]
    duplicate_events: tuple[RallyEvent, ...]
    stale_events: tuple[RallyEvent, ...]
    ignored_events: tuple[RallyEvent, ...]
    valid_racket_hits: tuple[CourtSide, ...]
    confirmed_returns: tuple[CourtSide, ...]
    first_bounces: tuple[CourtSide, ...]
    step_contact_peaks: Mapping[RallyEventKind, float]

    @property
    def terminated_now(self) -> bool:
        return not self.before.terminated and self.after.terminated

    def to_info(self) -> dict[str, int | float | bool | str]:
        """Return snapshot diagnostics plus per-batch event flags."""
        info = self.after.to_info()
        observed_kinds = {event.kind for event in self.observed_events}
        processed_kinds = {event.kind for event in self.processed_events}
        for kind in RallyEventKind:
            info[f"observed_event_{kind.value}"] = kind in observed_kinds
            info[f"event_{kind.value}"] = kind in processed_kinds
            if kind in CONTACT_EVENT_KINDS:
                info[f"step_contact_peak_{kind.value}"] = float(
                    self.step_contact_peaks.get(kind, 0.0)
                )
        info["event_valid_racket_hit"] = bool(self.valid_racket_hits)
        info["event_valid_racket_hit_a"] = CourtSide.A in self.valid_racket_hits
        info["event_valid_racket_hit_b"] = CourtSide.B in self.valid_racket_hits
        info["event_valid_return"] = bool(self.confirmed_returns)
        info["event_valid_return_a"] = CourtSide.A in self.confirmed_returns
        info["event_valid_return_b"] = CourtSide.B in self.confirmed_returns
        info["event_first_bounce"] = bool(self.first_bounces)
        info["event_duplicate_suppressed"] = bool(self.duplicate_events)
        info["event_stale_suppressed"] = bool(self.stale_events)
        return info


_UNCONDITIONAL_FAULTS: Final[Mapping[RallyEventKind, TerminationReason]] = (
    MappingProxyType(
        {
            RallyEventKind.NONFINITE_STATE: TerminationReason.NONFINITE_STATE,
            RallyEventKind.UNSAFE_PHYSICS: TerminationReason.UNSAFE_PHYSICS,
            RallyEventKind.HUMANOID_A_NET: TerminationReason.HUMANOID_A_NET,
            RallyEventKind.HUMANOID_B_NET: TerminationReason.HUMANOID_B_NET,
            RallyEventKind.RACKET_A_NET: TerminationReason.RACKET_A_NET,
            RallyEventKind.RACKET_B_NET: TerminationReason.RACKET_B_NET,
            RallyEventKind.BALL_HUMANOID_A: TerminationReason.BALL_HUMANOID_A,
            RallyEventKind.BALL_HUMANOID_B: TerminationReason.BALL_HUMANOID_B,
        }
    )
)

_EVENT_PRIORITY: Final[Mapping[RallyEventKind, int]] = MappingProxyType(
    {
        RallyEventKind.NONFINITE_STATE: 0,
        RallyEventKind.UNSAFE_PHYSICS: 1,
        RallyEventKind.HUMANOID_A_NET: 2,
        RallyEventKind.HUMANOID_B_NET: 3,
        RallyEventKind.RACKET_A_NET: 4,
        RallyEventKind.RACKET_B_NET: 5,
        RallyEventKind.BALL_HUMANOID_A: 6,
        RallyEventKind.BALL_HUMANOID_B: 7,
        RallyEventKind.BALL_NET: 8,
        RallyEventKind.NET_CROSSING_TO_A: 20,
        RallyEventKind.NET_CROSSING_TO_B: 21,
        RallyEventKind.BALL_COURT_A: 30,
        RallyEventKind.BALL_COURT_B: 31,
        RallyEventKind.BALL_RACKET_A: 40,
        RallyEventKind.BALL_RACKET_B: 41,
    }
)


def _event_side(kind: RallyEventKind) -> CourtSide:
    if kind in {
        RallyEventKind.BALL_RACKET_A,
        RallyEventKind.BALL_COURT_A,
        RallyEventKind.NET_CROSSING_TO_A,
    }:
        return CourtSide.A
    if kind in {
        RallyEventKind.BALL_RACKET_B,
        RallyEventKind.BALL_COURT_B,
        RallyEventKind.NET_CROSSING_TO_B,
    }:
        return CourtSide.B
    raise ValueError(f"{kind.value} does not identify one court side")


class RallyStateMachine:
    """Mutable wrapper around the deterministic cooperative rally reducer."""

    def __init__(
        self,
        *,
        serving_side: CourtSide = CourtSide.A,
        rules: RallyRules | None = None,
        court: CourtGeometry | None = None,
    ) -> None:
        self.rules = rules or RallyRules()
        # The surface the OUT_OF_BOUNDS rule measures against. Default
        # is the regulation singles court (byte-identical for every
        # existing consumer); a non-regulation env passes its own
        # geometry instead of forking the reducer.
        self.court = court if court is not None else REGULATION_COURT
        self.reset(serving_side)

    def reset(self, serving_side: CourtSide) -> RallySnapshot:
        """Reset for an initial feed launched from ``serving_side``."""
        self._phase = RallyPhase.INITIAL_FEED
        self._serving_side = CourtSide(serving_side)
        self._expected_returner = self._serving_side.opponent
        self._ball_side = self._serving_side
        self._last_valid_hitter: CourtSide | None = None
        self._pending_hitter: CourtSide | None = None
        self._pending_crossed_net = False
        self._feed_crossed_net = False
        self._bounce_side: CourtSide | None = None
        self._bounce_count = 0
        self._net_crossing_count = 0
        self._shot_crossing_count = 0
        self._legal_hit_counts: Counter[CourtSide] = Counter()
        self._valid_return_counts: Counter[CourtSide] = Counter()
        self._duplicate_suppressed_count = 0
        self._duplicate_contact_suppressed_count = 0
        self._stale_event_suppressed_count = 0
        self._last_event_substep = -1
        self._termination_reason = TerminationReason.NONE
        self._event_counts: Counter[RallyEventKind] = Counter()
        self._contact_peaks: dict[RallyEventKind, float] = {}
        self._last_contact_episode: dict[RallyEventKind, int] = {}
        self._processed_event_ids: set[int] = set()
        self._processed_event_signatures: set[tuple[object, ...]] = set()
        snapshot = self.snapshot()
        self._assert_invariants(snapshot)
        return snapshot

    def snapshot(self) -> RallySnapshot:
        """Return an immutable copy of the current state."""
        legal_a = self._legal_hit_counts[CourtSide.A]
        legal_b = self._legal_hit_counts[CourtSide.B]
        valid_a = self._valid_return_counts[CourtSide.A]
        valid_b = self._valid_return_counts[CourtSide.B]
        return RallySnapshot(
            phase=self._phase,
            serving_side=self._serving_side,
            expected_returner=self._expected_returner,
            ball_side=self._ball_side,
            last_valid_racket_hitter=self._last_valid_hitter,
            pending_return_hitter=self._pending_hitter,
            pending_return_crossed_net=self._pending_crossed_net,
            feed_crossed_net=self._feed_crossed_net,
            bounce_side=self._bounce_side,
            bounce_count=self._bounce_count,
            net_crossing_count=self._net_crossing_count,
            shot_crossing_count=self._shot_crossing_count,
            legal_hit_count=legal_a + legal_b,
            legal_hit_count_a=legal_a,
            legal_hit_count_b=legal_b,
            rally_count=valid_a + valid_b,
            valid_return_count_a=valid_a,
            valid_return_count_b=valid_b,
            duplicate_contact_suppressed_count=(
                self._duplicate_contact_suppressed_count
            ),
            duplicate_event_suppressed_count=self._duplicate_suppressed_count,
            stale_event_suppressed_count=self._stale_event_suppressed_count,
            last_event_substep=self._last_event_substep,
            termination_reason=self._termination_reason,
            ball_net_is_fault=self.rules.ball_net_is_fault,
            strict_reverse_crossing=self.rules.strict_reverse_crossing,
            strict_double_hit=self.rules.strict_double_hit,
            event_counts=MappingProxyType(dict(self._event_counts)),
            contact_peaks=MappingProxyType(dict(self._contact_peaks)),
        )

    def advance(
        self,
        events: Iterable[RallyEvent],
        *,
        contact_peaks: Mapping[RallyEventKind, float] | None = None,
    ) -> RallyTransition:
        """Consume one event batch and optional sampled control-step peaks."""
        observed = tuple(
            sorted(
                events,
                key=lambda event: (
                    event.substep,
                    _EVENT_PRIORITY[event.kind],
                    event.kind.value,
                    -1 if event.contact_episode is None else event.contact_episode,
                ),
            )
        )
        before = self.snapshot()
        if before.terminated:
            return RallyTransition(
                before=before,
                after=before,
                observed_events=observed,
                processed_events=(),
                suppressed_events=(),
                duplicate_events=(),
                stale_events=(),
                ignored_events=observed,
                valid_racket_hits=(),
                confirmed_returns=(),
                first_bounces=(),
                step_contact_peaks={},
            )

        step_peaks = self._validate_batch(observed, contact_peaks or {})

        for kind, peak in step_peaks.items():
            self._contact_peaks[kind] = max(
                self._contact_peaks.get(kind, 0.0),
                peak,
            )

        processed: list[RallyEvent] = []
        suppressed: list[RallyEvent] = []
        duplicate_events: list[RallyEvent] = []
        stale_events: list[RallyEvent] = []
        ignored: list[RallyEvent] = []
        valid_hits: list[CourtSide] = []
        confirmed_returns: list[CourtSide] = []
        first_bounces: list[CourtSide] = []

        for _, grouped_events in groupby(observed, key=lambda event: event.substep):
            group = tuple(grouped_events)
            if self._termination_reason is not TerminationReason.NONE:
                ignored.extend(group)
                continue

            unique: list[RallyEvent] = []
            for event in group:
                if event.substep < self._last_event_substep:
                    self._stale_event_suppressed_count += 1
                    suppressed.append(event)
                    stale_events.append(event)
                    continue
                if self._is_duplicate(event):
                    self._duplicate_suppressed_count += 1
                    if event.kind in CONTACT_EVENT_KINDS:
                        self._duplicate_contact_suppressed_count += 1
                    suppressed.append(event)
                    duplicate_events.append(event)
                else:
                    self._record_observation(event)
                    unique.append(event)
            if group and group[0].substep >= self._last_event_substep:
                self._last_event_substep = max(
                    self._last_event_substep,
                    group[0].substep,
                )

            fault_event = self._first_unconditional_fault(unique)
            if fault_event is not None:
                processed.append(fault_event)
                self._terminate(self._fault_reason(fault_event.kind))
                ignored.extend(event for event in unique if event is not fault_event)
                break

            court_fault = self._court_fault_candidate(unique)
            if court_fault is not None:
                fault_event, reason = court_fault
                processed.append(fault_event)
                self._terminate(reason)
                ignored.extend(event for event in unique if event is not fault_event)
                break

            crossing_fault = self._crossing_fault_candidate(unique)
            if crossing_fault is not None:
                processed.append(crossing_fault)
                self._terminate(TerminationReason.REVERSE_CROSSING)
                ignored.extend(event for event in unique if event is not crossing_fault)
                break

            racket_sides = {
                _event_side(event.kind)
                for event in unique
                if event.kind in RACKET_EVENT_KINDS
            }
            if len(racket_sides) > 1:
                self._terminate(TerminationReason.SIMULTANEOUS_RACKET_CONTACT)
                ignored.extend(unique)
                break

            for event in self._order_normal_events(unique):
                if self._termination_reason is not TerminationReason.NONE:
                    ignored.append(event)
                    continue
                if event.kind is RallyEventKind.BALL_NET:
                    # Non-terminal curriculum forgiveness: record only.
                    processed.append(event)
                elif event.kind in CROSSING_EVENT_KINDS:
                    processed.append(event)
                    self._handle_crossing(event)
                elif event.kind in COURT_EVENT_KINDS:
                    processed.append(event)
                    confirmed = self._handle_court_contact(event)
                    if confirmed is not None:
                        confirmed_returns.append(confirmed)
                    if (
                        self._termination_reason is TerminationReason.NONE
                        and self._bounce_count == 1
                        and self._bounce_side is _event_side(event.kind)
                    ):
                        first_bounces.append(_event_side(event.kind))
                elif event.kind in RACKET_EVENT_KINDS:
                    processed.append(event)
                    hit, confirmed = self._handle_racket_contact(event)
                    if hit is not None:
                        valid_hits.append(hit)
                    if confirmed is not None:
                        confirmed_returns.append(confirmed)

        after = self.snapshot()
        self._assert_invariants(after)
        return RallyTransition(
            before=before,
            after=after,
            observed_events=observed,
            processed_events=tuple(processed),
            suppressed_events=tuple(suppressed),
            duplicate_events=tuple(duplicate_events),
            stale_events=tuple(stale_events),
            ignored_events=tuple(ignored),
            valid_racket_hits=tuple(valid_hits),
            confirmed_returns=tuple(confirmed_returns),
            first_bounces=tuple(first_bounces),
            step_contact_peaks=step_peaks,
        )

    def _record_observation(self, event: RallyEvent) -> None:
        self._event_counts[event.kind] += 1
        if event.kind in CROSSING_EVENT_KINDS:
            self._net_crossing_count += 1
        if event.kind in CONTACT_EVENT_KINDS:
            self._contact_peaks[event.kind] = max(
                self._contact_peaks.get(event.kind, 0.0),
                float(event.peak_force),
            )

    def _validate_batch(
        self,
        events: Iterable[RallyEvent],
        contact_peaks: Mapping[RallyEventKind, float],
    ) -> dict[RallyEventKind, float]:
        """Validate the complete batch before any machine state is mutated.

        An instance method (not the historical staticmethod) because the
        in_bounds cross-check measures against the machine's own
        ``CourtGeometry``.
        """
        validated_peaks: dict[RallyEventKind, float] = {}
        for kind, peak in contact_peaks.items():
            if kind not in CONTACT_EVENT_KINDS:
                raise ValueError(f"{kind.value} is not a contact event")
            value = float(peak)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("contact peaks must be finite and non-negative")
            validated_peaks[kind] = value

        for event in events:
            if event.kind in COURT_EVENT_KINDS:
                if event.position is None:
                    raise ValueError("court-contact events require a position")
                side = _event_side(event.kind)
                x = event.position[0]
                if x < 0.0 and side is not CourtSide.A:
                    raise ValueError("court event side contradicts contact position")
                if x > 0.0 and side is not CourtSide.B:
                    raise ValueError("court event side contradicts contact position")
                derived_in_bounds = self.court.is_in_bounds(
                    event.position[0],
                    event.position[1],
                )
                if (
                    event.in_bounds is not None
                    and event.in_bounds is not derived_in_bounds
                ):
                    raise ValueError("in_bounds contradicts court contact position")
            if event.kind in CROSSING_EVENT_KINDS:
                target = _event_side(event.kind)
                if event.to_side is not None and event.to_side is not target:
                    raise ValueError("crossing target contradicts event kind")
                if event.from_side is target:
                    raise ValueError("crossing source and target must differ")
        return validated_peaks

    def _is_duplicate(self, event: RallyEvent) -> bool:
        if event.event_id is not None:
            if event.event_id in self._processed_event_ids:
                return True
            self._processed_event_ids.add(event.event_id)
        else:
            signature = (
                event.kind,
                event.substep,
                event.contact_episode,
                event.position,
                event.in_bounds,
                event.from_side,
                event.to_side,
            )
            if signature in self._processed_event_signatures:
                return True
            self._processed_event_signatures.add(signature)

        if event.kind not in CONTACT_EVENT_KINDS:
            return False
        if event.contact_episode is not None:
            previous = self._last_contact_episode.get(event.kind, -1)
            if event.contact_episode <= previous:
                return True
            self._last_contact_episode[event.kind] = event.contact_episode
            return False
        return False

    def _court_fault_candidate(
        self,
        events: Iterable[RallyEvent],
    ) -> tuple[RallyEvent, TerminationReason] | None:
        court_events = [event for event in events if event.kind in COURT_EVENT_KINDS]
        if not court_events:
            return None

        incoming_crossing = any(
            event.kind in CROSSING_EVENT_KINDS
            and _event_side(event.kind) is self._expected_returner
            and (
                (
                    self._phase is RallyPhase.INITIAL_FEED
                    and (
                        self._ball_side
                        if event.from_side is None
                        else event.from_side
                    )
                    is self._serving_side
                )
                or (
                    self._phase is RallyPhase.RETURN_IN_FLIGHT
                    and self._pending_hitter is not None
                    and (
                        self._ball_side
                        if event.from_side is None
                        else event.from_side
                    )
                    is self._pending_hitter
                )
            )
            for event in events
        )
        projected_ball_side = (
            self._expected_returner if incoming_crossing else self._ball_side
        )
        projected_ready = (
            self._phase is RallyPhase.AWAITING_RETURN
            or (
                self._phase is RallyPhase.INITIAL_FEED
                and incoming_crossing
            )
            or (
                self._phase is RallyPhase.RETURN_IN_FLIGHT
                and (self._pending_crossed_net or incoming_crossing)
            )
        )
        projected_bounces = self._bounce_count
        candidates: list[tuple[int, RallyEvent, TerminationReason]] = []
        reason_priority = {
            TerminationReason.OUT_OF_BOUNDS: 0,
            TerminationReason.SECOND_BOUNCE: 1,
            TerminationReason.FAILED_TO_CROSS: 2,
        }
        for event in court_events:
            assert event.position is not None  # validated before mutation
            side = _event_side(event.kind)
            if not self.court.is_in_bounds(event.position[0], event.position[1]):
                reason = TerminationReason.OUT_OF_BOUNDS
            elif (
                side is not self._expected_returner
                or not projected_ready
                or projected_ball_side is not side
            ):
                reason = TerminationReason.FAILED_TO_CROSS
            elif projected_bounces >= 1:
                reason = TerminationReason.SECOND_BOUNCE
            else:
                projected_bounces += 1
                continue
            candidates.append((reason_priority[reason], event, reason))
        if not candidates:
            return None
        _, event, reason = min(
            candidates,
            key=lambda candidate: (candidate[0], candidate[1].kind.value),
        )
        return event, reason

    def _crossing_fault_candidate(
        self,
        events: Iterable[RallyEvent],
    ) -> RallyEvent | None:
        event_list = list(events)
        if not self.rules.strict_reverse_crossing:
            return None
        if self._is_outgoing_hit_cross_group(event_list):
            return None

        phase = self._phase
        ball_side = self._ball_side
        feed_crossed = self._feed_crossed_net
        pending_hitter = self._pending_hitter
        pending_crossed = self._pending_crossed_net
        shot_crossings = self._shot_crossing_count
        for event in event_list:
            if event.kind not in CROSSING_EVENT_KINDS:
                continue
            target = _event_side(event.kind)
            source = (
                ball_side if event.from_side is None else event.from_side
            )
            invalid = (
                target is source
                or source is not ball_side
                or target is not self._expected_returner
            )
            if not invalid and phase is RallyPhase.INITIAL_FEED:
                invalid = source is not self._serving_side or feed_crossed
                if not invalid:
                    feed_crossed = True
                    phase = RallyPhase.AWAITING_RETURN
            elif not invalid and phase is RallyPhase.RETURN_IN_FLIGHT:
                invalid = (
                    pending_hitter is None
                    or source is not pending_hitter
                    or pending_crossed
                    or shot_crossings > 0
                )
                if not invalid:
                    pending_crossed = True
                    shot_crossings = 1
            elif not invalid:
                invalid = True
            if invalid:
                return event
            ball_side = target
        return None

    def _is_outgoing_hit_cross_group(self, events: Iterable[RallyEvent]) -> bool:
        rackets = [event for event in events if event.kind in RACKET_EVENT_KINDS]
        crossings = [event for event in events if event.kind in CROSSING_EVENT_KINDS]
        if len(rackets) != 1 or not crossings:
            return False
        hitter = _event_side(rackets[0].kind)
        return (
            self._phase is not RallyPhase.INITIAL_FEED
            and hitter is self._expected_returner
            and self._ball_side is hitter
            and any(
                _event_side(event.kind) is hitter.opponent
                for event in crossings
            )
        )

    def _order_normal_events(
        self,
        events: Iterable[RallyEvent],
    ) -> list[RallyEvent]:
        event_list = list(events)
        outgoing = self._is_outgoing_hit_cross_group(event_list)

        def priority(event: RallyEvent) -> tuple[int, str]:
            if event.kind is RallyEventKind.BALL_NET:
                return 0, event.kind.value
            if outgoing:
                if event.kind in COURT_EVENT_KINDS:
                    return 10, event.kind.value
                if event.kind in RACKET_EVENT_KINDS:
                    return 20, event.kind.value
                if event.kind in CROSSING_EVENT_KINDS:
                    return 30, event.kind.value
            else:
                if event.kind in CROSSING_EVENT_KINDS:
                    return 10, event.kind.value
                if event.kind in COURT_EVENT_KINDS:
                    return 20, event.kind.value
                if event.kind in RACKET_EVENT_KINDS:
                    return 30, event.kind.value
            return 40, event.kind.value

        return sorted(event_list, key=priority)

    def _first_unconditional_fault(
        self,
        events: Iterable[RallyEvent],
    ) -> RallyEvent | None:
        for event in events:
            if event.kind in _UNCONDITIONAL_FAULTS:
                return event
            if event.kind is RallyEventKind.BALL_NET and self.rules.ball_net_is_fault:
                return event
        return None

    def _fault_reason(self, kind: RallyEventKind) -> TerminationReason:
        if kind is RallyEventKind.BALL_NET:
            return TerminationReason.BALL_NET
        return _UNCONDITIONAL_FAULTS[kind]

    def _handle_crossing(self, event: RallyEvent) -> None:
        target = _event_side(event.kind)
        source = (
            self._ball_side if event.from_side is None else event.from_side
        )

        if (
            target is source
            or source is not self._ball_side
            or target is not self._expected_returner
        ):
            if self.rules.strict_reverse_crossing:
                self._terminate(TerminationReason.REVERSE_CROSSING)
            else:
                self._ball_side = target
            return

        if self._phase is RallyPhase.INITIAL_FEED:
            if source is not self._serving_side or self._feed_crossed_net:
                if self.rules.strict_reverse_crossing:
                    self._terminate(TerminationReason.REVERSE_CROSSING)
                else:
                    self._ball_side = target
                return
            self._feed_crossed_net = True
            self._ball_side = target
            self._phase = RallyPhase.AWAITING_RETURN
            return

        if self._phase is not RallyPhase.RETURN_IN_FLIGHT:
            if self.rules.strict_reverse_crossing:
                self._terminate(TerminationReason.REVERSE_CROSSING)
            else:
                self._ball_side = target
            return
        if self._pending_hitter is None or source is not self._pending_hitter:
            if self.rules.strict_reverse_crossing:
                self._terminate(TerminationReason.REVERSE_CROSSING)
            else:
                self._ball_side = target
            return
        if self._pending_crossed_net or self._shot_crossing_count > 0:
            if self.rules.strict_reverse_crossing:
                self._terminate(TerminationReason.REVERSE_CROSSING)
            else:
                self._ball_side = target
            return

        self._pending_crossed_net = True
        self._shot_crossing_count = 1
        self._ball_side = target

    def _handle_court_contact(self, event: RallyEvent) -> CourtSide | None:
        side = _event_side(event.kind)
        assert event.position is not None  # validated before mutation
        if not self.court.is_in_bounds(event.position[0], event.position[1]):
            self._terminate(TerminationReason.OUT_OF_BOUNDS)
            return None

        if side is not self._expected_returner:
            self._terminate(TerminationReason.FAILED_TO_CROSS)
            return None
        if self._phase is RallyPhase.INITIAL_FEED:
            self._terminate(TerminationReason.FAILED_TO_CROSS)
            return None
        if self._phase is RallyPhase.RETURN_IN_FLIGHT and not self._pending_crossed_net:
            self._terminate(TerminationReason.FAILED_TO_CROSS)
            return None
        if self._ball_side is not side:
            self._terminate(TerminationReason.FAILED_TO_CROSS)
            return None
        if self._bounce_count >= 1:
            self._terminate(TerminationReason.SECOND_BOUNCE)
            return None

        confirmed = self._confirm_pending_return()
        self._bounce_count = 1
        self._bounce_side = side
        self._ball_side = side
        if self._phase is RallyPhase.RETURN_IN_FLIGHT:
            self._phase = RallyPhase.AWAITING_RETURN
        return confirmed

    def _handle_racket_contact(
        self,
        event: RallyEvent,
    ) -> tuple[CourtSide | None, CourtSide | None]:
        hitter = _event_side(event.kind)
        if hitter is not self._expected_returner:
            if self._last_valid_hitter is hitter:
                if self.rules.strict_double_hit:
                    self._terminate(TerminationReason.DOUBLE_HIT)
                # Forgiving curricula may ignore a genuine local recontact,
                # but it still produces no hit or rally progress.
            else:
                self._terminate(TerminationReason.WRONG_HITTER)
            return None, None

        if self._phase is RallyPhase.INITIAL_FEED:
            self._terminate(TerminationReason.PREMATURE_HIT)
            return None, None
        if self._ball_side is not hitter:
            self._terminate(TerminationReason.PREMATURE_HIT)
            return None, None
        if self._phase is RallyPhase.RETURN_IN_FLIGHT and not self._pending_crossed_net:
            self._terminate(TerminationReason.PREMATURE_HIT)
            return None, None

        confirmed = self._confirm_pending_return()
        self._legal_hit_counts[hitter] += 1
        self._last_valid_hitter = hitter
        self._pending_hitter = hitter
        self._pending_crossed_net = False
        self._expected_returner = hitter.opponent
        self._bounce_count = 0
        self._bounce_side = None
        self._shot_crossing_count = 0
        self._phase = RallyPhase.RETURN_IN_FLIGHT
        return hitter, confirmed

    def _confirm_pending_return(self) -> CourtSide | None:
        if self._pending_hitter is None:
            return None
        if not self._pending_crossed_net:
            return None
        hitter = self._pending_hitter
        self._valid_return_counts[hitter] += 1
        self._pending_hitter = None
        self._pending_crossed_net = False
        self._shot_crossing_count = 0
        return hitter

    def _terminate(self, reason: TerminationReason) -> None:
        if self._termination_reason is not TerminationReason.NONE:
            return
        self._termination_reason = reason
        self._phase = RallyPhase.TERMINAL

    @staticmethod
    def _assert_invariants(snapshot: RallySnapshot) -> None:
        assert snapshot.terminated == (snapshot.phase is RallyPhase.TERMINAL)
        assert snapshot.rally_count == (
            snapshot.valid_return_count_a + snapshot.valid_return_count_b
        )
        assert snapshot.legal_hit_count == (
            snapshot.legal_hit_count_a + snapshot.legal_hit_count_b
        )
        assert snapshot.valid_return_count_a <= snapshot.legal_hit_count_a
        assert snapshot.valid_return_count_b <= snapshot.legal_hit_count_b
        assert snapshot.last_event_substep >= -1
        assert snapshot.duplicate_contact_suppressed_count >= 0
        assert snapshot.duplicate_event_suppressed_count >= (
            snapshot.duplicate_contact_suppressed_count
        )
        assert snapshot.stale_event_suppressed_count >= 0
        if not snapshot.terminated:
            assert snapshot.bounce_count in (0, 1)
            if snapshot.bounce_count == 1:
                assert snapshot.bounce_side is snapshot.expected_returner
            else:
                assert snapshot.bounce_side is None
            assert snapshot.pending_return_crossed_net == (
                snapshot.shot_crossing_count == 1
            )
            if snapshot.phase is RallyPhase.INITIAL_FEED:
                assert not snapshot.feed_crossed_net
            else:
                assert snapshot.feed_crossed_net
            if snapshot.phase is RallyPhase.RETURN_IN_FLIGHT:
                assert snapshot.pending_return_hitter is not None
                assert snapshot.pending_return_hitter is snapshot.expected_returner.opponent
                assert (
                    snapshot.last_valid_racket_hitter
                    is snapshot.pending_return_hitter
                )
            else:
                assert snapshot.pending_return_hitter is None
            if snapshot.pending_return_crossed_net:
                assert snapshot.phase is RallyPhase.RETURN_IN_FLIGHT
                assert snapshot.shot_crossing_count == 1


__all__ = [
    "CONTACT_EVENT_KINDS",
    "COURT_EVENT_KINDS",
    "CROSSING_EVENT_KINDS",
    "CourtSide",
    "RACKET_EVENT_KINDS",
    "RallyEvent",
    "RallyEventKind",
    "RallyPhase",
    "RallyRules",
    "RallySnapshot",
    "RallyStateMachine",
    "RallyTransition",
    "TERMINATION_REASON_NAMES",
    "TerminationReason",
]
