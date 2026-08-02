"""MuJoCo substep events for the cooperative humanoid-tennis scene.

The sampler is intentionally stateful across Gymnasium control steps.  It
coalesces all geoms belonging to one semantic channel, emits only contact
rising edges, requires a short release hysteresis before re-arming, and keeps
the peak force observed during the full control step.  Calling it after every
``mujoco.mj_step`` prevents fast tennis impacts from disappearing between
control-frame boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Final

import mujoco
import numpy as np

from courtside_dynamics.envs._tennis_physics import (
    COLLISION_BALL,
    COLLISION_COURT,
    COLLISION_NET,
    COLLISION_RACKET,
    COLLISION_ROBOT,
    REGULATION_COURT,
    CourtGeometry,
)
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyEvent,
    RallyEventKind,
)

_SAFETY_EVENT_KINDS: Final = frozenset(
    {RallyEventKind.NONFINITE_STATE, RallyEventKind.UNSAFE_PHYSICS}
)

# Stable observation order for the contact-latch state exposed by
# :meth:`SubstepTennisEventSampler.markov_state`.  Court A/B share one channel
# because they are mutually exclusive classifications of the same physical
# ball/court contact episode.
TENNIS_CONTACT_CHANNELS: Final = (
    "ball_racket_a",
    "ball_racket_b",
    "ball_court",
    "ball_net",
    "humanoid_a_net",
    "humanoid_b_net",
    "racket_a_net",
    "racket_b_net",
    "ball_humanoid_a",
    "ball_humanoid_b",
)

TENNIS_CONTACT_MARKOV_LABELS: Final = tuple(
    f"contact_latched_{channel}" for channel in TENNIS_CONTACT_CHANNELS
) + tuple(
    f"contact_release_progress_{channel}" for channel in TENNIS_CONTACT_CHANNELS
)


def _vec3_tuple(values: np.ndarray) -> tuple[float, float, float]:
    """Convert a three-vector to a precisely typed plain-Python tuple."""
    return (float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True, slots=True)
class TennisSafetyLimits:
    """Conservative finite-state guardrails applied before contact parsing."""

    max_ball_linear_speed: float = 120.0
    max_abs_qvel: float = 2_000.0
    max_abs_qacc: float = 10_000_000.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_ball_linear_speed) or self.max_ball_linear_speed <= 0.0:
            raise ValueError("max_ball_linear_speed must be positive")
        if not math.isfinite(self.max_abs_qvel) or self.max_abs_qvel <= 0.0:
            raise ValueError("max_abs_qvel must be positive")
        if not math.isfinite(self.max_abs_qacc) or self.max_abs_qacc <= 0.0:
            raise ValueError("max_abs_qacc must be positive")


@dataclass(frozen=True, slots=True)
class TennisEventSamplerConfig:
    """Contact threshold, release latch, and crossing detector settings."""

    min_contact_force: float = 0.0
    release_substeps: int = 2
    crossing_deadband: float = 1e-5
    safety: TennisSafetyLimits = TennisSafetyLimits()

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_contact_force) or self.min_contact_force < 0.0:
            raise ValueError("min_contact_force must be finite and non-negative")
        if self.release_substeps < 1:
            raise ValueError("release_substeps must be at least one")
        if not math.isfinite(self.crossing_deadband) or self.crossing_deadband < 0.0:
            raise ValueError("crossing_deadband must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TennisSceneContactIndex:
    """Resolved geom ownership for one compiled humanoid-tennis model."""

    ball_geom: int
    court_geom: int
    court_geoms: frozenset[int]
    net_geoms: frozenset[int]
    racket_a_geoms: frozenset[int]
    racket_b_geoms: frozenset[int]
    humanoid_a_geoms: frozenset[int]
    humanoid_b_geoms: frozenset[int]

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        *,
        ball_geom_name: str = "ball_geom",
        court_geom_name: str = "court_surface",
        racket_prefixes: tuple[str, str] = (
            "player_a_racket_",
            "player_b_racket_",
        ),
        robot_prefixes: tuple[str, str] = ("player_a_", "player_b_"),
        require_robots: bool = True,
    ) -> TennisSceneContactIndex:
        """Resolve semantic sets from stable categories and player prefixes.

        The defaults are the humanoid-tennis scene, unchanged. A
        robot-free scene (two bare paddles across the net) passes
        ``require_robots=False`` and its own name prefixes so the
        substep sampler, contact latching, and crossing detection can
        be reused without a humanoid in the model.
        """
        ball_geom = int(model.geom(ball_geom_name).id)
        court_geom = int(model.geom(court_geom_name).id)
        court_geoms: set[int] = set()
        net_geoms: set[int] = set()
        racket_a: set[int] = set()
        racket_b: set[int] = set()
        humanoid_a: set[int] = set()
        humanoid_b: set[int] = set()

        for geom_id in range(model.ngeom):
            geom = model.geom(geom_id)
            name = geom.name
            category = int(geom.contype[0])
            if category == COLLISION_COURT:
                court_geoms.add(geom_id)
            elif category == COLLISION_NET:
                net_geoms.add(geom_id)
            elif category == COLLISION_RACKET:
                if name.startswith(racket_prefixes[0]):
                    racket_a.add(geom_id)
                elif name.startswith(racket_prefixes[1]):
                    racket_b.add(geom_id)
            elif category == COLLISION_ROBOT:
                if name.startswith(robot_prefixes[0]):
                    humanoid_a.add(geom_id)
                elif name.startswith(robot_prefixes[1]):
                    humanoid_b.add(geom_id)

        index = cls(
            ball_geom=ball_geom,
            court_geom=court_geom,
            court_geoms=frozenset(court_geoms),
            net_geoms=frozenset(net_geoms),
            racket_a_geoms=frozenset(racket_a),
            racket_b_geoms=frozenset(racket_b),
            humanoid_a_geoms=frozenset(humanoid_a),
            humanoid_b_geoms=frozenset(humanoid_b),
        )
        index._validate(model, require_robots=require_robots)
        return index

    def _validate(
        self, model: mujoco.MjModel, *, require_robots: bool = True
    ) -> None:
        groups = {
            "court": self.court_geoms,
            "net": self.net_geoms,
            "racket_a": self.racket_a_geoms,
            "racket_b": self.racket_b_geoms,
            "humanoid_a": self.humanoid_a_geoms,
            "humanoid_b": self.humanoid_b_geoms,
        }
        # Court, net, and both strikers are the minimum rules scene;
        # the robot bodies are optional so a paddle-only court can
        # reuse this index (robot-dependent event kinds simply never
        # fire against empty groups).
        optional = () if require_robots else ("humanoid_a", "humanoid_b")
        missing = [
            name
            for name, values in groups.items()
            if not values and name not in optional
        ]
        if missing:
            raise ValueError(f"missing semantic tennis geom groups: {missing}")
        all_groups = [
            {self.ball_geom},
            *groups.values(),
        ]
        for left_index, left in enumerate(all_groups):
            for right in all_groups[left_index + 1 :]:
                if left & right:
                    raise ValueError("tennis semantic geom groups must be disjoint")
        if int(model.geom(self.ball_geom).contype[0]) != COLLISION_BALL:
            raise ValueError("ball_geom has the wrong collision category")
        if int(model.geom(self.court_geom).contype[0]) != COLLISION_COURT:
            raise ValueError("court_surface has the wrong collision category")
        if self.court_geom not in self.court_geoms:
            raise ValueError("court_surface is missing from the court geom group")
        if any(
            int(model.geom(geom_id).contype[0]) != COLLISION_COURT
            for geom_id in self.court_geoms
        ):
            raise ValueError("court geom group contains the wrong collision category")


@dataclass(frozen=True, slots=True)
class TennisStepEventBatch:
    """Ordered rising edges and force peaks for one control step."""

    control_step: int
    substeps_sampled: int
    events: tuple[RallyEvent, ...]
    contact_peaks: Mapping[RallyEventKind, float]
    active_contact_latches: frozenset[str]


@dataclass(slots=True)
class _ContactAggregate:
    total_force: float
    representative_force: float
    position: tuple[float, float, float]
    geom_ids: tuple[int, int]
    geom_names: tuple[str, str]
    in_bounds: bool | None


class _NonfiniteContactForce(RuntimeError):
    """Internal signal that the constraint solver produced invalid force."""


class SubstepTennisEventSampler:
    """Convert ordered MuJoCo substeps into semantic tennis event batches."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        config: TennisEventSamplerConfig | None = None,
        index: TennisSceneContactIndex | None = None,
        ball_joint_name: str = "ball_free",
        court: CourtGeometry | None = None,
    ) -> None:
        self.model = model
        self.config = config or TennisEventSamplerConfig()
        # A pre-built index lets a non-humanoid scene (or one with
        # different naming) reuse the sampler; the default resolves the
        # humanoid-tennis scene exactly as before.
        self.index = (
            index
            if index is not None
            else TennisSceneContactIndex.from_model(model)
        )
        # The surface the in_bounds annotation measures against;
        # regulation by default, byte-identical for existing scenes.
        self.court = court if court is not None else REGULATION_COURT
        ball_joint = model.joint(ball_joint_name)
        self._ball_qposadr = int(ball_joint.qposadr[0])
        self._ball_dofadr = int(ball_joint.dofadr[0])
        self._latched_contacts: set[str] = set()
        self._inactive_substeps: dict[str, int] = {}
        self._contact_episode: dict[str, int] = {}
        self._stable_ball_side = CourtSide.A
        self._last_stable_position = np.zeros(3, dtype=np.float64)
        self._physics_substep = -1
        self._next_event_id = 0
        self._control_step: int | None = None
        self._last_control_substep = -1
        self._buffer_events: list[RallyEvent] = []
        self._buffer_contact_peaks: dict[RallyEventKind, float] = {}
        self._buffer_episode_peaks: dict[tuple[RallyEventKind, int], float] = {}
        self._substeps_sampled = 0

    def reset(
        self,
        data: mujoco.MjData,
        *,
        ball_side: CourtSide | None = None,
    ) -> None:
        """Prime latches and crossing state after ``mj_forward`` on reset."""
        if self._control_step is not None:
            raise RuntimeError("cannot reset during an open control step")
        self._physics_substep = -1
        self._next_event_id = 0
        self._contact_episode.clear()
        self._inactive_substeps.clear()
        self._latched_contacts.clear()
        position = self._ball_position(data)
        derived_side = self._side_outside_deadband(float(position[0]))
        if ball_side is not None:
            explicit_side = CourtSide(ball_side)
            if derived_side is not None and derived_side is not explicit_side:
                raise ValueError("ball_side contradicts the reset ball position")
            self._stable_ball_side = explicit_side
        else:
            self._stable_ball_side = derived_side or CourtSide.from_x(
                float(position[0]),
                tie_breaker=CourtSide.A,
            )
        self._last_stable_position = position.copy()
        try:
            contacts = self._collect_contacts(data, tie_breaker=self._stable_ball_side)
        except _NonfiniteContactForce as error:
            raise ValueError("cannot prime non-finite contact force") from error
        for kind in contacts:
            channel = self._contact_channel(kind)
            self._latched_contacts.add(channel)
            self._contact_episode[channel] = 0

    def begin_control_step(self, control_step: int) -> None:
        """Open a new buffer while preserving cross-step contact latches."""
        if control_step < 0:
            raise ValueError("control_step must be non-negative")
        if self._control_step is not None:
            raise RuntimeError("previous control step has not been drained")
        self._control_step = control_step
        self._last_control_substep = -1
        self._buffer_events = []
        self._buffer_contact_peaks = {}
        self._buffer_episode_peaks = {}
        self._substeps_sampled = 0

    def markov_state(self) -> tuple[float, ...]:
        """Return contact latches and hysteresis progress in stable order.

        Contact rising-edge suppression persists across control steps.  These
        values therefore belong in the policy observation: two otherwise
        identical MuJoCo states can emit different events depending on whether
        a semantic contact is still latched or is part-way through its release
        hysteresis.  The first block contains latch flags; the second contains
        normalized inactive-substep progress in ``[0, 1]``.
        """
        latched = tuple(
            float(channel in self._latched_contacts)
            for channel in TENNIS_CONTACT_CHANNELS
        )
        release = tuple(
            min(
                float(self._inactive_substeps.get(channel, 0))
                / float(self.config.release_substeps),
                1.0,
            )
            for channel in TENNIS_CONTACT_CHANNELS
        )
        return latched + release

    def sample_substep(
        self,
        data: mujoco.MjData,
        *,
        control_substep: int,
    ) -> RallyEventKind | None:
        """Sample after one ``mj_step``; return only an early-stop safety kind.

        Normal events have a single delivery owner: :meth:`end_control_step`.
        The return value exists solely so a future environment can stop
        propagating non-finite or unsafe physics before draining that batch.
        """
        if self._control_step is None:
            raise RuntimeError("begin_control_step must be called first")
        if control_substep <= self._last_control_substep:
            raise ValueError("control_substep must increase strictly")
        self._last_control_substep = control_substep
        self._physics_substep += 1
        self._substeps_sampled += 1
        safety_kind = self._safety_event_kind(data)
        if safety_kind is not None:
            if not any(event.kind in _SAFETY_EVENT_KINDS for event in self._buffer_events):
                self._buffer_events.append(
                    RallyEvent(
                        kind=safety_kind,
                        substep=self._physics_substep,
                        event_id=self._allocate_event_id(),
                    )
                )
            return safety_kind

        ball_position = self._ball_position(data)
        ball_velocity = self._ball_velocity(data)
        try:
            contacts = self._collect_contacts(data, tie_breaker=self._stable_ball_side)
        except _NonfiniteContactForce:
            if not any(event.kind in _SAFETY_EVENT_KINDS for event in self._buffer_events):
                self._buffer_events.append(
                    RallyEvent(
                        kind=RallyEventKind.NONFINITE_STATE,
                        substep=self._physics_substep,
                        event_id=self._allocate_event_id(),
                    )
                )
            return RallyEventKind.NONFINITE_STATE
        current_channels = {self._contact_channel(kind) for kind in contacts}

        for kind, aggregate in contacts.items():
            channel = self._contact_channel(kind)
            self._inactive_substeps[channel] = 0
            self._buffer_contact_peaks[kind] = max(
                self._buffer_contact_peaks.get(kind, 0.0),
                aggregate.total_force,
            )
            if channel not in self._latched_contacts:
                episode = self._contact_episode.get(channel, -1) + 1
                self._contact_episode[channel] = episode
                self._latched_contacts.add(channel)
                self._buffer_events.append(
                    RallyEvent(
                        kind=kind,
                        substep=self._physics_substep,
                        event_id=self._allocate_event_id(),
                        contact_episode=episode,
                        position=aggregate.position,
                        in_bounds=aggregate.in_bounds,
                        peak_force=aggregate.total_force,
                        geom_names=aggregate.geom_names,
                        ball_position=_vec3_tuple(ball_position),
                        ball_velocity=_vec3_tuple(ball_velocity),
                    )
                )
            episode = self._contact_episode[channel]
            episode_key = (kind, episode)
            self._buffer_episode_peaks[episode_key] = max(
                self._buffer_episode_peaks.get(episode_key, 0.0),
                aggregate.total_force,
            )

        for channel in tuple(self._latched_contacts - current_channels):
            missing = self._inactive_substeps.get(channel, 0) + 1
            self._inactive_substeps[channel] = missing
            if missing >= self.config.release_substeps:
                self._latched_contacts.remove(channel)
                self._inactive_substeps.pop(channel, None)

        new_side = self._side_outside_deadband(float(ball_position[0]))
        if new_side is self._stable_ball_side:
            self._last_stable_position = ball_position.copy()
        elif new_side is not None:
            old_side = self._stable_ball_side
            crossing_position = self._interpolate_net_crossing(
                self._last_stable_position,
                ball_position,
            )
            crossing_kind = (
                RallyEventKind.NET_CROSSING_TO_A
                if new_side is CourtSide.A
                else RallyEventKind.NET_CROSSING_TO_B
            )
            self._buffer_events.append(
                RallyEvent(
                    kind=crossing_kind,
                    substep=self._physics_substep,
                    event_id=self._allocate_event_id(),
                    position=_vec3_tuple(crossing_position),
                    ball_position=_vec3_tuple(ball_position),
                    ball_velocity=_vec3_tuple(ball_velocity),
                    from_side=old_side,
                    to_side=new_side,
                )
            )
            self._stable_ball_side = new_side
            self._last_stable_position = ball_position.copy()

        return None

    def record_safety_event(
        self,
        kind: RallyEventKind,
        *,
        control_substep: int,
    ) -> None:
        """Record a safety stop when MuJoCo cannot complete a substep.

        ``mujoco.FatalError`` occurs before normal post-step sampling can run.
        This narrow escape hatch lets the environment drain a valid batch and
        return a terminal Gymnasium transition instead of leaving the sampler
        permanently open or crashing a vectorized rollout.
        """
        if kind not in _SAFETY_EVENT_KINDS:
            raise ValueError("record_safety_event accepts only safety kinds")
        if self._control_step is None:
            raise RuntimeError("begin_control_step must be called first")
        if control_substep <= self._last_control_substep:
            raise ValueError("control_substep must increase strictly")
        self._last_control_substep = control_substep
        self._physics_substep += 1
        self._substeps_sampled += 1
        if not any(event.kind in _SAFETY_EVENT_KINDS for event in self._buffer_events):
            self._buffer_events.append(
                RallyEvent(
                    kind=kind,
                    substep=self._physics_substep,
                    event_id=self._allocate_event_id(),
                )
            )

    def safety_event_kind(self, data: mujoco.MjData) -> RallyEventKind | None:
        """Return a safety classification without advancing sampler state."""
        return self._safety_event_kind(data)

    def end_control_step(self) -> TennisStepEventBatch:
        """Drain the current buffer exactly once."""
        if self._control_step is None:
            raise RuntimeError("no open control step")
        updated_events: list[RallyEvent] = []
        for event in self._buffer_events:
            if event.contact_episode is None:
                updated_events.append(event)
                continue
            peak = self._buffer_episode_peaks.get(
                (event.kind, event.contact_episode),
                event.peak_force,
            )
            updated_events.append(replace(event, peak_force=peak))
        updated_events.sort(
            key=lambda event: (
                event.substep,
                event.kind.value,
                -1 if event.contact_episode is None else event.contact_episode,
                event.geom_names or ("", ""),
            )
        )
        batch = TennisStepEventBatch(
            control_step=self._control_step,
            substeps_sampled=self._substeps_sampled,
            events=tuple(updated_events),
            contact_peaks=dict(self._buffer_contact_peaks),
            active_contact_latches=frozenset(self._latched_contacts),
        )
        self._control_step = None
        self._buffer_events = []
        self._buffer_contact_peaks = {}
        self._buffer_episode_peaks = {}
        return batch

    def _allocate_event_id(self) -> int:
        event_id = self._next_event_id
        self._next_event_id += 1
        return event_id

    @staticmethod
    def _contact_channel(kind: RallyEventKind) -> str:
        if kind in {
            RallyEventKind.BALL_COURT_A,
            RallyEventKind.BALL_COURT_B,
        }:
            return "ball_court"
        return kind.value

    def _safety_event_kind(self, data: mujoco.MjData) -> RallyEventKind | None:
        arrays = (
            data.qpos,
            data.qvel,
            data.qacc,
            data.qacc_warmstart,
            data.ctrl,
            data.qfrc_applied,
            data.qfrc_constraint,
            data.efc_force,
            data.xfrc_applied,
        )
        if not np.isfinite(data.time) or not all(
            bool(np.isfinite(values).all()) for values in arrays
        ):
            return RallyEventKind.NONFINITE_STATE
        limits = self.config.safety
        ball_speed = float(np.linalg.norm(self._ball_velocity(data)))
        if (
            ball_speed > limits.max_ball_linear_speed
            or float(np.max(np.abs(data.qvel), initial=0.0)) > limits.max_abs_qvel
            or float(np.max(np.abs(data.qacc), initial=0.0)) > limits.max_abs_qacc
        ):
            return RallyEventKind.UNSAFE_PHYSICS
        return None

    def _collect_contacts(
        self,
        data: mujoco.MjData,
        *,
        tie_breaker: CourtSide,
    ) -> dict[RallyEventKind, _ContactAggregate]:
        aggregates: dict[RallyEventKind, _ContactAggregate] = {}
        force = np.zeros(6, dtype=np.float64)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if int(contact.efc_address) < 0:
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            position = _vec3_tuple(contact.pos)
            kind = self._classify_pair(
                geom1,
                geom2,
                contact_x=position[0],
                tie_breaker=tie_breaker,
            )
            if kind is None:
                continue
            force.fill(0.0)
            mujoco.mj_contactForce(self.model, data, contact_index, force)
            if not bool(np.isfinite(force).all()):
                raise _NonfiniteContactForce
            # mj_contactForce uses the local contact frame: component zero is
            # the compressive normal force; components one and two are
            # friction. Thresholds and reported peaks use only the normal
            # component so they do not change with friction tuning.
            force_magnitude = max(0.0, float(force[0]))
            geom_ids = (min(geom1, geom2), max(geom1, geom2))
            geom_names = tuple(self.model.geom(geom_id).name for geom_id in geom_ids)
            contact_in_bounds = (
                self.court.is_in_bounds(position[0], position[1])
                if kind in {
                    RallyEventKind.BALL_COURT_A,
                    RallyEventKind.BALL_COURT_B,
                }
                else None
            )
            aggregate = aggregates.get(kind)
            if aggregate is None:
                aggregates[kind] = _ContactAggregate(
                    total_force=force_magnitude,
                    representative_force=force_magnitude,
                    position=position,
                    geom_ids=geom_ids,
                    geom_names=geom_names,
                    in_bounds=contact_in_bounds,
                )
                continue
            aggregate.total_force += force_magnitude
            if (force_magnitude, tuple(-value for value in geom_ids)) > (
                aggregate.representative_force,
                tuple(-value for value in aggregate.geom_ids),
            ):
                aggregate.representative_force = force_magnitude
                aggregate.position = position
                aggregate.geom_ids = geom_ids
                aggregate.geom_names = geom_names
                aggregate.in_bounds = contact_in_bounds
        return {
            kind: aggregate
            for kind, aggregate in aggregates.items()
            if aggregate.total_force > self.config.min_contact_force
        }

    def _classify_pair(
        self,
        geom1: int,
        geom2: int,
        *,
        contact_x: float,
        tie_breaker: CourtSide,
    ) -> RallyEventKind | None:
        pair = frozenset({geom1, geom2})
        ball = self.index.ball_geom
        if ball in pair and pair & self.index.court_geoms:
            side = CourtSide.from_x(contact_x, tie_breaker=tie_breaker)
            return (
                RallyEventKind.BALL_COURT_A
                if side is CourtSide.A
                else RallyEventKind.BALL_COURT_B
            )
        if ball in pair and pair & self.index.net_geoms:
            return RallyEventKind.BALL_NET
        if ball in pair and pair & self.index.racket_a_geoms:
            return RallyEventKind.BALL_RACKET_A
        if ball in pair and pair & self.index.racket_b_geoms:
            return RallyEventKind.BALL_RACKET_B
        if ball in pair and pair & self.index.humanoid_a_geoms:
            return RallyEventKind.BALL_HUMANOID_A
        if ball in pair and pair & self.index.humanoid_b_geoms:
            return RallyEventKind.BALL_HUMANOID_B
        if pair & self.index.net_geoms and pair & self.index.racket_a_geoms:
            return RallyEventKind.RACKET_A_NET
        if pair & self.index.net_geoms and pair & self.index.racket_b_geoms:
            return RallyEventKind.RACKET_B_NET
        if pair & self.index.net_geoms and pair & self.index.humanoid_a_geoms:
            return RallyEventKind.HUMANOID_A_NET
        if pair & self.index.net_geoms and pair & self.index.humanoid_b_geoms:
            return RallyEventKind.HUMANOID_B_NET
        return None

    def _ball_position(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(
            data.qpos[self._ball_qposadr : self._ball_qposadr + 3],
            dtype=np.float64,
        ).copy()

    def _ball_velocity(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(
            data.qvel[self._ball_dofadr : self._ball_dofadr + 3],
            dtype=np.float64,
        ).copy()

    def _side_outside_deadband(self, x: float) -> CourtSide | None:
        if x < -self.config.crossing_deadband:
            return CourtSide.A
        if x > self.config.crossing_deadband:
            return CourtSide.B
        return None

    @staticmethod
    def _interpolate_net_crossing(
        previous: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        denominator = float(current[0] - previous[0])
        if denominator == 0.0:
            crossing = current.copy()
            crossing[0] = 0.0
            return crossing
        fraction = float(-previous[0] / denominator)
        fraction = min(max(fraction, 0.0), 1.0)
        crossing = previous + fraction * (current - previous)
        crossing[0] = 0.0
        return crossing


__all__ = [
    "SubstepTennisEventSampler",
    "TENNIS_CONTACT_CHANNELS",
    "TENNIS_CONTACT_MARKOV_LABELS",
    "TennisEventSamplerConfig",
    "TennisSafetyLimits",
    "TennisSceneContactIndex",
    "TennisStepEventBatch",
]
