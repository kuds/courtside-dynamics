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

    # Defaults are the P3-measured primary band (probe doc
    # paddle_tennis_probes_p3_p4_20260802.md): origin 3.25 m behind the
    # net at 9 m/s, 21 degrees. Do not drift them without a probe.
    start_distance_from_net: float = 3.25
    lateral_position: float = 0.0
    height: float = 1.3
    speed: float = 9.0
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
        for name, value in (
            ("speed", self.speed),
            ("elevation_degrees", self.elevation_degrees),
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

#: The P1-frozen scripted controller (probe doc §3): commit gap,
#: swing-through past the ball toward the net, strike-height offset
#: below ball center. Frozen best configuration from the P0-P2 sweep.
LEAD_CHARGE_GAP = 0.8
LEAD_CHARGE_SWING = 0.4
LEAD_CHARGE_STRIKE = -0.12

#: Side-local paddle constants (paddle_court.xml): home pivot, the x
#: mapping span, and the z slide's local range around base height 1.2.
PADDLE_LOCAL_X_RANGE = (-6.4, -0.1)
PADDLE_LOCAL_Y_SPAN = 3.0
PADDLE_LOCAL_Z = (1.2, -0.9, 2.0)  # base height, low, high

_GRAVITY_MPS2 = 9.81


def lead_charge_local_action(
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    paddle_position: np.ndarray,
    *,
    gap: float = LEAD_CHARGE_GAP,
    swing: float = LEAD_CHARGE_SWING,
    strike: float = LEAD_CHARGE_STRIKE,
) -> np.ndarray:
    """The frozen P1 controller, in the side-local frame.

    A world-frame port of the certification ``lead_charge`` oracle
    (training/ladder_certification.py): yield at neutral park while the
    ball is outgoing (serves launch from the controller's own half and
    pass through the home column -- a ball-tracking park would swat the
    server's own feed), pre-position by tracking y/z while the ball is
    incoming, and once it is within ``gap`` of the paddle, charge with
    a ballistic y/z lead aimed ``strike`` below the projected ball
    height, driving the target ``swing`` beyond the ball toward the
    net. Volleys were legal under the original (now superseded)
    ``volley_rule="legal"`` profile, so unlike the wall-ball oracle
    there is no bounce-count precondition -- which is why this frozen
    controller faults every point under the ground-rules default
    (probe: 100% ``volley_return``); the era's reference is
    :func:`ground_lead_charge_local_action`. Kept frozen verbatim for
    the volley-era artifacts and the P3 instruments.
    """
    from courtside_dynamics.envs._base import invert_piecewise_target

    home_x = -PADDLE_HOME_X
    x_low, x_high = PADDLE_LOCAL_X_RANGE
    z_base, z_low, z_high = PADDLE_LOCAL_Z

    ball_x, ball_y, ball_z = ball_position
    vx, vy, vz = ball_velocity
    paddle_x = paddle_position[0]
    incoming = vx < -0.1

    if incoming and (ball_x - paddle_x) <= gap:
        target_x = min(ball_x + swing, x_high)
        t_hit = float(
            np.clip((ball_x - paddle_x) / max(0.5, -vx + 2.5), 0.0, 1.2)
        )
        target_y = ball_y + vy * t_hit
        target_z = (
            max(0.25, ball_z + vz * t_hit - 0.5 * _GRAVITY_MPS2 * t_hit**2)
            + strike
        )
    elif incoming:
        target_x = home_x
        target_y = ball_y
        target_z = max(0.25, ball_z)
    else:
        target_x = home_x
        target_y = 0.0
        target_z = z_base

    action_x = invert_piecewise_target(target_x, x_low, home_x, x_high)
    action_y = float(np.clip(target_y / PADDLE_LOCAL_Y_SPAN, -1.0, 1.0))
    action_z = invert_piecewise_target(
        target_z - z_base, z_low, 0.0, z_high
    )
    return np.array([action_x, action_y, action_z], dtype=np.float64)


def scripted_lead_charge_opponent(observation: np.ndarray) -> np.ndarray:
    """The frozen controller as an opponent over a side-local observation.

    Reads the ball state and own-paddle position from the candidate
    observation layout (indices pinned by
    ``PADDLE_COURT_OBSERVATION_NAMES`` and its test). This is the
    signature opponents share -- a frozen SB3 policy drops in as
    ``lambda obs: policy.predict(obs, deterministic=True)[0]`` once P5
    settles whether the wall-ball champions transfer.
    """
    return lead_charge_local_action(
        observation[0:3],   # ball position
        observation[3:6],   # ball linear velocity
        observation[9:12],  # own paddle position
    )


#: Index of the rules ``bounce_count`` in the full PaddleTennis
#: observation (24 physical values, then the rally block: 4 phase
#: one-hots and 5 flags precede it). Pinned against
#: ``PADDLE_TENNIS_OBSERVATION_NAMES`` by the env tests; this module
#: cannot import the env (the env imports this module).
OBS_BOUNCE_COUNT_INDEX = 33
#: Index of the own-relative ``ball_side_is_own`` flag, same layout.
OBS_BALL_SIDE_INDEX = 30


#: How far behind the predicted landing point the ground oracle waits
#: out an incoming ball's pre-bounce flight, and the bounce height the
#: landing prediction solves for (the ball's radius).
GROUND_WAIT_MARGIN = 0.9
_BALL_BOUNCE_HEIGHT = 0.07


#: The ground stroke's calibrated swing-through: soft, because a
#: 6.5 m court must catch the return (the frozen 0.4 slam lands mean
#: 11.0 m from the net -- every stroke long).
GROUND_SWING = 0.1


def ground_lead_charge_local_action(
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    paddle_position: np.ndarray,
    *,
    bounce_count: float,
    ball_on_own_side: float = 0.0,
    wait_margin: float = GROUND_WAIT_MARGIN,
    gap: float = LEAD_CHARGE_GAP,
    swing: float = GROUND_SWING,
    strike: float = LEAD_CHARGE_STRIKE,
) -> np.ndarray:
    """The ground-rules era's oracle: wait behind the bounce, then charge.

    Restores the wall-ball certification oracle's
    ``floor_bounce_count >= 1`` charge precondition that the P1 port
    deliberately dropped ("volleys are legal on this court") -- and
    fixes the two failure modes the ground-rules bring-up measured:

    - the frozen port's pre-positioning tracks the ball's y/z at the
      home column, INSIDE the incoming serve's descent path, so under
      ``require_bounce_before_return`` the passive touch is an
      instant VOLLEY_RETURN fault (under the old profile it was a
      silent volley -- the P1-era oracle's returns were largely
      volleys);
    - charging a post-bounce ball from a fixed deep park slams a
      high, fast ball flat and long (0/12 confirmed returns across
      the whole swing x strike calibration grid at swing >= 0.3; the
      misses land mean 11.0 m from the net, so the fix is the soft
      ``GROUND_SWING``, not aim);
    - the frozen post-hit recovery races back to the home column,
      chasing straight through the soft outgoing return's path
      (double_hit 9-12/16 in the swing sweep) -- so while the ball is
      outgoing on the hitter's own side (``ball_on_own_side``), the
      controller HOLDS position with the face dropped low, and only
      re-homes once the ball has crossed.

    So the controller ballistically predicts the landing point (the
    wall-ball ``run_up`` instrument), waits ``wait_margin`` behind it
    with the face LOW, and charges the frozen ``lead_charge`` logic
    only after the bounce -- meeting the ball low on the rise, where
    the fixed +10-degree face supplies the loft (the P0-P2 measured
    strike-height control channel).
    """
    from courtside_dynamics.envs._base import invert_piecewise_target

    incoming = float(ball_velocity[0]) < -0.1
    if not incoming and ball_on_own_side >= 0.5:
        # Own soft return still outbound: hold ground, face low --
        # never chase it toward the net.
        x_low, x_high = PADDLE_LOCAL_X_RANGE
        z_base, z_low, z_high = PADDLE_LOCAL_Z
        action_x = invert_piecewise_target(
            float(np.clip(float(paddle_position[0]), x_low, x_high)),
            x_low,
            -PADDLE_HOME_X,
            x_high,
        )
        action_z = invert_piecewise_target(
            0.25 - z_base, z_low, 0.0, z_high
        )
        return np.array([action_x, 0.0, action_z], dtype=np.float64)
    if incoming and bounce_count < 1.0:
        x_low, x_high = PADDLE_LOCAL_X_RANGE
        z_base, z_low, z_high = PADDLE_LOCAL_Z
        ball_x, ball_y, ball_z = (float(v) for v in ball_position[:3])
        vx, vy, vz = (float(v) for v in ball_velocity[:3])
        drop = max(0.0, ball_z - _BALL_BOUNCE_HEIGHT)
        t_land = (
            vz + float(np.sqrt(vz * vz + 2.0 * _GRAVITY_MPS2 * drop))
        ) / _GRAVITY_MPS2
        land_x = ball_x + vx * t_land
        land_y = ball_y + vy * t_land
        wait_x = land_x - wait_margin
        wait_y = land_y
        if land_x < x_low + 0.3:
            # Workspace-margin collapse: the ball lands within 0.3 m
            # of the paddle's deepest reach (or beyond it), so a
            # y-tracking low face pinned at x_low sits in the descent
            # path and self-volleys (the probe's measured 1/100
            # receiver fault, seed 5117: landing 6.32 vs reach 6.4).
            # Dodge laterally for just these; an ordinary clipped
            # wait with >= 0.3 m of margin keeps the y-track (a
            # broader dodge trigger measurably degraded deep-serve
            # returns: ge1 97% -> 82% on the probe block).
            wait_y = land_y + (0.6 if land_y <= 0.0 else -0.6)
        action_x = invert_piecewise_target(
            float(np.clip(wait_x, x_low, x_high)),
            x_low,
            -PADDLE_HOME_X,
            x_high,
        )
        action_y = float(
            np.clip(wait_y / PADDLE_LOCAL_Y_SPAN, -1.0, 1.0)
        )
        action_z = invert_piecewise_target(
            0.25 - z_base, z_low, 0.0, z_high
        )
        return np.array([action_x, action_y, action_z], dtype=np.float64)
    return lead_charge_local_action(
        ball_position,
        ball_velocity,
        paddle_position,
        gap=gap,
        swing=swing,
        strike=strike,
    )


def scripted_ground_opponent(observation: np.ndarray) -> np.ndarray:
    """The bounce-waiting oracle over the full env observation.

    Unlike :func:`scripted_lead_charge_opponent` (which reads only the
    24-value physical block and stays usable on the prototype scene),
    this opponent also needs the rules ``bounce_count`` from the env's
    rally block, so it requires the full 48-value observation.
    """
    return ground_lead_charge_local_action(
        observation[0:3],
        observation[3:6],
        observation[9:12],
        bounce_count=float(observation[OBS_BOUNCE_COUNT_INDEX]),
        ball_on_own_side=float(observation[OBS_BALL_SIDE_INDEX]),
    )


def net_patting_local_action(
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    paddle_position: np.ndarray,
    *,
    net_gap: float = 0.45,
    strike_below: float = 0.15,
) -> np.ndarray:
    """Adversarial probe controller: reproduce the close-net volley loop.

    The first learned GPU run (20260803_004559) maximized cooperative
    return rate by patting the ball across the net at close range (a
    crossing every ~14 control steps, 91.6% of time in
    ``return_in_flight``). This scripted approximation exists so the
    ground-rules probe can measure volley-style net play with a
    reproducible instrument: park ``net_gap`` behind the net, track
    the ball's y, rise ``strike_below`` under the incoming ball (the
    fixed +10-degree pitch lifts each touch back over), and retract
    low while the ball is outgoing so the face never double-touches.
    It volleys every return it makes; it does NOT reproduce the
    learned loop's cadence or persistence (bring-up measured ~1.4
    crossings at a ~97-step cadence under the legal profile before a
    net or crossing fault) -- the loop's full signature remains
    documented by the GPU run itself. Not a frozen reference
    controller: an exploit-style witness.
    """
    from courtside_dynamics.envs._base import invert_piecewise_target

    x_low, x_high = PADDLE_LOCAL_X_RANGE
    z_base, z_low, z_high = PADDLE_LOCAL_Z

    incoming = float(ball_velocity[0]) < -0.05
    target_x = max(x_low, min(-net_gap, x_high))
    target_y = float(ball_position[1])
    target_z = (
        max(0.25, float(ball_position[2]) - strike_below)
        if incoming
        else 0.25
    )

    action_x = invert_piecewise_target(target_x, x_low, -PADDLE_HOME_X, x_high)
    action_y = float(np.clip(target_y / PADDLE_LOCAL_Y_SPAN, -1.0, 1.0))
    action_z = invert_piecewise_target(
        target_z - z_base, z_low, 0.0, z_high
    )
    return np.array([action_x, action_y, action_z], dtype=np.float64)


def scripted_net_patting_opponent(observation: np.ndarray) -> np.ndarray:
    """The exploit witness over a side-local observation."""
    return net_patting_local_action(
        observation[0:3],
        observation[3:6],
        observation[9:12],
    )
