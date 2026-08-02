"""PaddleTennis prototype scene: two calibrated paddles across a net.

**Prototype, not an environment.** Per the design doctrine
(docs/design_paddle_tennis.md §6), no PaddleTennis env class ships
until the probe battery has run and the task definition is frozen and
certified. This module is the probe substrate: the frozen P0-P2
geometry wired through the shared machinery batch 2 unlocked —
:class:`~courtside_dynamics.envs._paddle.PaddleInterface` twice,
``TennisSceneContactIndex`` without robots, the substep event sampler
and rally reducer on the small :class:`CourtGeometry` — plus the
side-relative observation/action candidate that probe P4 certifies.
Nothing here is registered with Gymnasium and nothing here defines
rewards.

Frame convention (the P4 contract): the **side-local frame is side A's
world frame**. Side B sees the world through the exact 180-degree court
rotation — positions/velocities/angular velocities mirrored via
:func:`~courtside_dynamics.envs._serve.mirror_for_side`, actions
mirrored on their x/y components before reaching side B's world-frame
paddle interface. One policy can therefore play either side.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._paddle import PaddleInterface
from courtside_dynamics.envs._serve import (
    mirror_for_side,
    validate_feed_geometry,
)
from courtside_dynamics.envs._tennis_events import (
    SubstepTennisEventSampler,
    TennisSceneContactIndex,
)
from courtside_dynamics.envs._tennis_physics import CourtGeometry
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyStateMachine,
)

#: Probe-frozen court (docs/paddle_tennis_probes_20260802.md §5).
PADDLE_COURT = CourtGeometry(half_length=6.5, half_width=4.115)
NET_HEIGHT = 0.914

#: Probe-frozen paddle placement: bases at x = -/+1.7, damping 8,
#: world-space x workspaces bounded to each half (see paddle_court.xml).
PADDLE_HOME_X = 1.7  # magnitude; side A at -1.7, side B mirrored

_GRAVITY = 9.81

#: Candidate side-relative observation layout, to be frozen by the env
#: definition after the probes (design doc §3 extends the wall-ball
#: template with the opponent paddle's state). All vectors are in the
#: side-local frame; every entry is a plain physical quantity — the
#: rally/reward bookkeeping tail is an env-definition question, not a
#: probe question.
PADDLE_COURT_OBSERVATION_NAMES = (
    *(f"ball_position_{axis}" for axis in "xyz"),
    *(f"ball_linear_velocity_{axis}" for axis in "xyz"),
    *(f"ball_angular_velocity_{axis}" for axis in "xyz"),
    *(f"own_paddle_position_{axis}" for axis in "xyz"),
    *(f"own_paddle_velocity_{axis}" for axis in "xyz"),
    *(f"opponent_paddle_position_{axis}" for axis in "xyz"),
    *(f"opponent_paddle_velocity_{axis}" for axis in "xyz"),
    *(f"ball_minus_own_paddle_{axis}" for axis in "xyz"),
)


@dataclass(frozen=True, slots=True)
class PaddleCourtServe:
    """A side-local ballistic serve draw (probe device, per P3).

    All coordinates are side-A-local; :meth:`PaddleCourtScene.serve`
    mirrors for side B. Validated through the shared feed contract on
    the small court.
    """

    start_distance_from_net: float = 3.25
    lateral_position: float = 0.0
    height: float = 1.3
    speed: float = 11.0
    elevation_degrees: float = 21.0
    lateral_degrees: float = 0.0
    position_noise: tuple[float, float, float] = (0.25, 0.5, 0.05)
    speed_noise: float = 1.0
    elevation_noise_degrees: float = 3.0
    lateral_noise_degrees: float = 4.0

    def __post_init__(self) -> None:
        validate_feed_geometry(
            kind="serve",
            start_distance_from_net=self.start_distance_from_net,
            lateral_position=self.lateral_position,
            height=self.height,
            position_noise=self.position_noise,
            half_length=PADDLE_COURT.half_length,
            half_width=PADDLE_COURT.half_width,
        )
        for name, value, bound in (
            ("speed", self.speed, self.speed_noise),
            ("elevation_degrees", self.elevation_degrees, 0.0),
        ):
            if not np.isfinite(value):
                raise ValueError(f"serve {name} must be finite")
        if self.speed <= 0.0 or not 0.0 <= self.speed_noise < self.speed:
            raise ValueError(
                "serve speed must remain positive across its noise"
            )
        if (
            self.elevation_noise_degrees < 0.0
            or self.lateral_noise_degrees < 0.0
        ):
            raise ValueError("serve angle noise must be non-negative")


class PaddleCourtScene:
    """The compiled paddle court plus its rules/events/paddle wiring."""

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(
            asset_path("paddle_court.xml")
        )
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.index = TennisSceneContactIndex.from_model(
            self.model, require_robots=False
        )
        self.sampler = SubstepTennisEventSampler(
            self.model, index=self.index, court=PADDLE_COURT
        )
        self.rules = RallyStateMachine(
            serving_side=CourtSide.A, court=PADDLE_COURT
        )
        self.paddles = {
            CourtSide.A: PaddleInterface(
                self.model,
                self.data,
                home_x=-PADDLE_HOME_X,
                joint_names=(
                    "player_a_slide_x",
                    "player_a_slide_y",
                    "player_a_slide_z",
                ),
                actuator_names=(
                    "player_a_target_x",
                    "player_a_target_y",
                    "player_a_target_z",
                ),
                head_body_name="player_a_head",
            ),
            CourtSide.B: PaddleInterface(
                self.model,
                self.data,
                home_x=PADDLE_HOME_X,
                joint_names=(
                    "player_b_slide_x",
                    "player_b_slide_y",
                    "player_b_slide_z",
                ),
                actuator_names=(
                    "player_b_target_x",
                    "player_b_target_y",
                    "player_b_target_z",
                ),
                head_body_name="player_b_head",
            ),
        }
        ball_joint = self.model.joint("ball_free")
        self._ball_qposadr = int(ball_joint.qposadr[0])
        self._ball_dofadr = int(ball_joint.dofadr[0])

    # -- state access ------------------------------------------------------

    def ball_position(self) -> np.ndarray:
        return self.data.qpos[
            self._ball_qposadr : self._ball_qposadr + 3
        ].copy()

    def ball_velocity(self) -> np.ndarray:
        return self.data.qvel[
            self._ball_dofadr : self._ball_dofadr + 3
        ].copy()

    def ball_angular_velocity(self) -> np.ndarray:
        return self.data.qvel[
            self._ball_dofadr + 3 : self._ball_dofadr + 6
        ].copy()

    def paddle_position(self, side: CourtSide) -> np.ndarray:
        head = "player_a_head" if side is CourtSide.A else "player_b_head"
        return np.asarray(self.data.body(head).xpos, dtype=np.float64).copy()

    def paddle_velocity(self, side: CourtSide) -> np.ndarray:
        joints = (
            ("player_a_slide_x", "player_a_slide_y", "player_a_slide_z")
            if side is CourtSide.A
            else ("player_b_slide_x", "player_b_slide_y", "player_b_slide_z")
        )
        return np.array(
            [
                float(self.data.qvel[int(self.model.joint(name).dofadr[0])])
                for name in joints
            ],
            dtype=np.float64,
        )

    # -- the P4 contract: side-relative observations and actions ----------

    def observation(self, side: CourtSide) -> np.ndarray:
        """The candidate side-local observation (see module docstring).

        Every vector is mirrored into the side-local frame, so the same
        policy reads either side identically; P4 pins this bit-for-bit.
        """
        side = CourtSide(side)
        own = side
        opponent = side.opponent
        parts = (
            mirror_for_side(self.ball_position(), side),
            mirror_for_side(self.ball_velocity(), side),
            mirror_for_side(self.ball_angular_velocity(), side),
            mirror_for_side(self.paddle_position(own), side),
            mirror_for_side(self.paddle_velocity(own), side),
            mirror_for_side(self.paddle_position(opponent), side),
            mirror_for_side(self.paddle_velocity(opponent), side),
            mirror_for_side(
                self.ball_position() - self.paddle_position(own), side
            ),
        )
        return np.concatenate(parts)

    def apply_action(self, side: CourtSide, local_action: np.ndarray):
        """Apply a side-local normalized action through the side's paddle.

        The x/y components mirror into world frame for side B (the
        paddle interfaces are world-frame; their ctrlranges are exact
        court mirrors by XML construction), so a given local action
        means the same court-relative move on either side.
        """
        side = CourtSide(side)
        world_action = mirror_for_side(
            np.asarray(local_action, dtype=np.float64).copy(), side
        )
        return self.paddles[side].apply(self.data, world_action)

    # -- serve and stepping ------------------------------------------------

    def serve(
        self,
        serve: PaddleCourtServe,
        *,
        serving_side: CourtSide,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place and launch the ball for a side-local ballistic serve.

        Returns the world-frame (position, velocity) actually set.
        Resets the rules machine and primes the sampler.
        """
        serving_side = CourtSide(serving_side)
        position_noise = rng.uniform(
            low=-np.asarray(serve.position_noise),
            high=np.asarray(serve.position_noise),
        )
        speed = float(
            rng.uniform(
                serve.speed - serve.speed_noise,
                serve.speed + serve.speed_noise,
            )
        )
        elevation = np.radians(
            rng.uniform(
                serve.elevation_degrees - serve.elevation_noise_degrees,
                serve.elevation_degrees + serve.elevation_noise_degrees,
            )
        )
        lateral = np.radians(
            rng.uniform(
                serve.lateral_degrees - serve.lateral_noise_degrees,
                serve.lateral_degrees + serve.lateral_noise_degrees,
            )
        )
        position = (
            np.array(
                [
                    -serve.start_distance_from_net,
                    serve.lateral_position,
                    serve.height,
                ]
            )
            + position_noise
        )
        horizontal = speed * np.cos(elevation)
        velocity = np.array(
            [
                horizontal * np.cos(lateral),
                horizontal * np.sin(lateral),
                speed * np.sin(elevation),
            ]
        )
        mirror_for_side(position, serving_side)
        mirror_for_side(velocity, serving_side)

        mujoco.mj_resetData(self.model, self.data)
        qpos = self.data.qpos
        qvel = self.data.qvel
        qpos[self._ball_qposadr : self._ball_qposadr + 3] = position
        qpos[self._ball_qposadr + 3 : self._ball_qposadr + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        qvel[self._ball_dofadr : self._ball_dofadr + 6] = 0.0
        qvel[self._ball_dofadr : self._ball_dofadr + 3] = velocity
        mujoco.mj_forward(self.model, self.data)

        self.rules = RallyStateMachine(
            serving_side=serving_side, court=PADDLE_COURT
        )
        self.sampler.reset(self.data, ball_side=serving_side)
        return position, velocity

    def step_control_frame(
        self, control_step: int, *, frame_skip: int = 5
    ):
        """Advance one control frame, sampling events per substep.

        Returns the sampled :class:`TennisStepEventBatch`; callers feed
        it to ``self.rules.advance(batch.events, ...)``.
        """
        self.sampler.begin_control_step(control_step)
        for substep in range(frame_skip):
            mujoco.mj_step(self.model, self.data)
            self.sampler.sample_substep(self.data, control_substep=substep)
        return self.sampler.end_control_step()
