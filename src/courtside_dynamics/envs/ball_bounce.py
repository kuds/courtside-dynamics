"""Ball Bounce environment.

The agent controls a 6-DOF paddle and must deliberately rebound a ball from
its top face.  A bounce is rewarded only when a ball-paddle contact has all of
the following properties:

* the ball approached the paddle's top face;
* the paddle was moving upward into the ball;
* the pair contact exceeded the configured force threshold; and
* the ball left with sufficient upward paddle-relative speed.

Contacts are sampled at every MuJoCo physics substep.  This matters because a
stiff impact can begin and end entirely inside one 10 ms control step.  A
persistent contact is one contact episode and can pay at most once; two
contact-free substeps plus physical clearance are required before the detector
re-arms.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
from gymnasium import utils

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs._base import CourtsideMujocoEnv


class BallBounceEnv(CourtsideMujocoEnv, utils.EzPickle):
    """Deliberately rebound a ball from a 6-DOF paddle's top face."""

    action_names: tuple[str, ...] = (
        "act_x",
        "act_y",
        "act_z",
        "slider_z_motor",
        "slider_y_motor",
        "slider_x_motor",
    )

    def __init__(
        self,
        episode_len: int = 1_000,
        min_force: float = 0.0,
        min_impact_speed: float = 0.25,
        min_rebound_speed: float = 0.5,
        min_paddle_upward_speed: float = 0.05,
        release_substeps: int = 2,
        release_clearance: float = 0.002,
        drop_margin: float = 0.01,
        **kwargs: Any,
    ) -> None:
        utils.EzPickle.__init__(
            self,
            episode_len=episode_len,
            min_force=min_force,
            min_impact_speed=min_impact_speed,
            min_rebound_speed=min_rebound_speed,
            min_paddle_upward_speed=min_paddle_upward_speed,
            release_substeps=release_substeps,
            release_clearance=release_clearance,
            drop_margin=drop_margin,
            **kwargs,
        )

        self.min_force = self._finite_nonnegative("min_force", min_force)
        self.min_impact_speed = self._finite_positive(
            "min_impact_speed", min_impact_speed
        )
        self.min_rebound_speed = self._finite_positive(
            "min_rebound_speed", min_rebound_speed
        )
        self.min_paddle_upward_speed = self._finite_nonnegative(
            "min_paddle_upward_speed", min_paddle_upward_speed
        )
        if isinstance(release_substeps, bool) or not isinstance(release_substeps, int):
            raise TypeError("release_substeps must be an integer")
        if release_substeps < 1:
            raise ValueError("release_substeps must be at least 1")
        self.release_substeps = release_substeps
        self.release_clearance = self._finite_nonnegative(
            "release_clearance", release_clearance
        )
        self.drop_margin = self._finite_nonnegative("drop_margin", drop_margin)

        CourtsideMujocoEnv.__init__(
            self,
            asset_path("ball_bounce.xml"),
            episode_len=episode_len,
            obs_dim=30,
            **kwargs,
        )

        ball_joint = self.model.joint("ball_freejoint")
        self._ball_qposadr = int(ball_joint.qposadr[0])
        self._ball_geom_id = int(self.model.geom("ball_geom").id)
        self._paddle_geom_id = int(self.model.geom("paddle_geom").id)
        compiled_actions = tuple(
            self.model.actuator(index).name for index in range(self.model.nu)
        )
        if compiled_actions != self.action_names:
            raise RuntimeError(
                "BallBounce action_names do not match compiled actuator order"
            )

        self._reset_contact_tracker()
        self._last_finite_observation = np.zeros(30, dtype=np.float64)

    @staticmethod
    def _finite_nonnegative(name: str, value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    @staticmethod
    def _finite_positive(name: str, value: float) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    def _reset_contact_tracker(self) -> None:
        self.bounce_count = 0
        self.contact_episode_count = 0
        self._contact_latched = False
        self._contact_release_progress = 0
        self._contact_candidate = False
        self._contact_peak_force = 0.0
        self._impact_relative_speed = 0.0
        self._impact_paddle_speed = 0.0
        self._rebound_relative_speed = 0.0
        self._contact_local_position = np.zeros(3, dtype=np.float64)
        self._contact_rejection_reason = "none"
        self._substep_touch_peak = 0.0
        self._substep_valid_bounces = 0
        self._substep_contact_started = False

    def _paddle_kinematics(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return normal, rotation, origin, angular and linear velocity."""
        paddle = self.data.geom("paddle_geom")
        rotation = np.asarray(paddle.xmat).reshape(3, 3).copy()
        normal = rotation[:, 2].copy()
        origin = np.asarray(paddle.xpos).copy()
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_GEOM,
            self._paddle_geom_id,
            velocity,
            0,
        )
        return (
            normal,
            rotation,
            origin,
            velocity[:3].copy(),
            velocity[3:].copy(),
        )

    @staticmethod
    def _point_velocity(
        *,
        origin: np.ndarray,
        angular_velocity: np.ndarray,
        linear_velocity: np.ndarray,
        point: np.ndarray,
    ) -> np.ndarray:
        return linear_velocity + np.cross(angular_velocity, point - origin)

    @staticmethod
    def _relative_normal_speed(
        normal: np.ndarray,
        ball_velocity: np.ndarray,
        paddle_point_velocity: np.ndarray,
    ) -> float:
        return float(np.dot(ball_velocity - paddle_point_velocity, normal))

    def _ball_paddle_height(self) -> float:
        """Signed ball-centre height above the paddle centre plane."""
        normal, _, paddle_position, _, _ = self._paddle_kinematics()
        ball_position = np.asarray(self.data.geom("ball_geom").xpos)
        return float(np.dot(ball_position - paddle_position, normal))

    def _geom_clearance(self) -> float:
        """Smallest signed surface distance, including lateral separation."""
        return float(
            mujoco.mj_geomDistance(
                self.model,
                self.data,
                self._ball_geom_id,
                self._paddle_geom_id,
                max(1.0, self.release_clearance + 0.1),
                None,
            )
        )

    def _ball_paddle_contact(
        self,
    ) -> tuple[float, bool, bool, np.ndarray]:
        """Return pair force, presence, top-face flag, and local contact."""
        peak_force = 0.0
        top_peak_force = -1.0
        on_paddle = False
        on_top_face = False
        strongest_local_contact = np.zeros(3, dtype=np.float64)
        top_local_contact = np.zeros(3, dtype=np.float64)
        force6 = np.zeros(6, dtype=np.float64)
        paddle = self.data.geom("paddle_geom")
        rotation = np.asarray(paddle.xmat).reshape(3, 3)
        paddle_normal = rotation[:, 2]
        paddle_position = np.asarray(paddle.xpos)
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if {geom1, geom2} != {self._ball_geom_id, self._paddle_geom_id}:
                continue
            on_paddle = True
            mujoco.mj_contactForce(self.model, self.data, index, force6)
            normal_force = float(force6[0])

            local_contact = rotation.T @ (np.asarray(contact.pos) - paddle_position)
            if normal_force >= peak_force:
                peak_force = normal_force
                strongest_local_contact = local_contact.copy()
            contact_normal = np.asarray(contact.frame[:3])
            face_alignment = abs(float(np.dot(contact_normal, paddle_normal)))
            if local_contact[2] > 0.0 and face_alignment >= 0.75:
                on_top_face = True
                if normal_force >= top_peak_force:
                    top_peak_force = normal_force
                    top_local_contact = local_contact.copy()
        selected_contact = (
            top_local_contact if on_top_face else strongest_local_contact
        )
        return peak_force, on_paddle, on_top_face, selected_contact

    def _begin_contact_episode(
        self,
        *,
        on_top_face: bool,
        relative_speed: float,
        paddle_speed: float,
        local_position: np.ndarray,
    ) -> None:
        self._contact_latched = True
        self._contact_release_progress = 0
        self._contact_peak_force = 0.0
        self._impact_relative_speed = relative_speed
        self._impact_paddle_speed = paddle_speed
        self._rebound_relative_speed = 0.0
        self._contact_local_position = local_position.copy()
        self.contact_episode_count += 1
        self._substep_contact_started = True

        if not on_top_face:
            self._contact_candidate = False
            self._contact_rejection_reason = "underside_or_edge"
        elif relative_speed > -self.min_impact_speed:
            self._contact_candidate = False
            self._contact_rejection_reason = "not_approaching"
        elif paddle_speed < self.min_paddle_upward_speed:
            self._contact_candidate = False
            self._contact_rejection_reason = "passive_paddle"
        else:
            self._contact_candidate = True
            self._contact_rejection_reason = "none"

    def _step_mujoco_simulation(self, ctrl, n_frames):
        """Advance one substep at a time and capture qualified rebound events."""
        self.data.ctrl[:] = ctrl
        self._substep_touch_peak = 0.0
        self._substep_valid_bounces = 0
        self._substep_contact_started = False

        for _ in range(n_frames):
            (
                pre_normal,
                pre_rotation,
                pre_origin,
                pre_angular_velocity,
                pre_linear_velocity,
            ) = self._paddle_kinematics()
            pre_ball_velocity = np.asarray(
                self.data.joint("ball_freejoint").qvel[:3]
            ).copy()

            mujoco.mj_step(self.model, self.data)
            (
                force,
                on_paddle,
                on_top_face,
                local_position,
            ) = self._ball_paddle_contact()
            self._substep_touch_peak = max(self._substep_touch_peak, force)

            if on_paddle:
                self._contact_release_progress = 0
                if not self._contact_latched:
                    pre_contact_point = (
                        pre_origin + pre_rotation @ local_position
                    )
                    pre_paddle_point_velocity = self._point_velocity(
                        origin=pre_origin,
                        angular_velocity=pre_angular_velocity,
                        linear_velocity=pre_linear_velocity,
                        point=pre_contact_point,
                    )
                    pre_relative_speed = self._relative_normal_speed(
                        pre_normal,
                        pre_ball_velocity,
                        pre_paddle_point_velocity,
                    )
                    pre_paddle_speed = float(
                        np.dot(pre_paddle_point_velocity, pre_normal)
                    )
                    self._begin_contact_episode(
                        on_top_face=on_top_face,
                        relative_speed=pre_relative_speed,
                        paddle_speed=pre_paddle_speed,
                        local_position=local_position,
                    )
                else:
                    self._contact_local_position = local_position.copy()
                self._contact_peak_force = max(self._contact_peak_force, force)
            elif self._contact_latched:
                self._contact_release_progress += 1
                if self._contact_release_progress == 1:
                    (
                        normal,
                        rotation,
                        origin,
                        angular_velocity,
                        linear_velocity,
                    ) = self._paddle_kinematics()
                    contact_point = (
                        origin + rotation @ self._contact_local_position
                    )
                    paddle_point_velocity = self._point_velocity(
                        origin=origin,
                        angular_velocity=angular_velocity,
                        linear_velocity=linear_velocity,
                        point=contact_point,
                    )
                    ball_velocity = np.asarray(
                        self.data.joint("ball_freejoint").qvel[:3]
                    )
                    self._rebound_relative_speed = self._relative_normal_speed(
                        normal,
                        ball_velocity,
                        paddle_point_velocity,
                    )
                if (
                    self._contact_release_progress >= self.release_substeps
                    and self._geom_clearance() >= self.release_clearance
                ):
                    if (
                        self._contact_candidate
                        and self._contact_peak_force > self.min_force
                        and self._rebound_relative_speed
                        >= self.min_rebound_speed
                    ):
                        self._substep_valid_bounces += 1
                    self._contact_latched = False
                    self._contact_release_progress = 0
                    self._contact_candidate = False

        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _physics_state_is_finite(self) -> bool:
        arrays = (
            self.data.qpos,
            self.data.qvel,
            self.data.ctrl,
            self.data.act,
            self.data.sensordata,
            self.data.geom("ball_geom").xpos,
            self.data.geom("paddle_geom").xpos,
            self.data.geom("paddle_geom").xmat,
        )
        return all(bool(np.isfinite(values).all()) for values in arrays)

    @staticmethod
    def _finite_sensor(values) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        if not bool(np.isfinite(result).all()):
            result[:] = 0.0
        return result

    def step(self, a):
        action = np.asarray(a, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, "
                f"got {action.shape}"
            )
        action_nonfinite = not bool(np.isfinite(action).all())
        state_nonfinite = not self._physics_state_is_finite()
        if action_nonfinite or state_nonfinite:
            self._substep_touch_peak = 0.0
            self._substep_valid_bounces = 0
            self._substep_contact_started = False
        else:
            self.do_simulation(action, self.frame_skip)
        self.step_number += 1

        detected_bounces = self._substep_valid_bounces

        current_velocity_value = self._finite_sensor(
            self.data.sensor("ball_velocity").data
        )
        current_accelerometer_value = self._finite_sensor(
            self.data.sensor("ball_accelerometer").data
        )
        current_fromto_value = self._finite_sensor(
            self.data.sensor("ball_to_paddle").data
        )

        raw_observation = self._get_obs()
        state_nonfinite = (
            state_nonfinite
            or action_nonfinite
            or not self._physics_state_is_finite()
            or not bool(np.isfinite(raw_observation).all())
        )
        if state_nonfinite:
            obs = self._last_finite_observation.copy()
            ball_paddle_height = 0.0
        else:
            obs = raw_observation
            self._last_finite_observation = obs.copy()
            ball_paddle_height = self._ball_paddle_height()
        ball_below_paddle = (
            not state_nonfinite and ball_paddle_height < -self.drop_margin
        )
        terminated = bool(state_nonfinite or ball_below_paddle)
        truncated = self.step_number >= self.episode_len
        if terminated:
            # A failure step is never itself a successful juggling step.
            valid_bounces = 0
        else:
            valid_bounces = detected_bounces
            self.bounce_count += valid_bounces
        reward = float(valid_bounces)

        term_nonfinite = state_nonfinite
        term_ball_dropped = ball_below_paddle and not term_nonfinite
        term_timeout = truncated and not terminated
        if term_nonfinite:
            termination_reason = "nonfinite_state"
        elif term_ball_dropped:
            termination_reason = "ball_below_paddle"
        elif term_timeout:
            termination_reason = "timeout"
        else:
            termination_reason = "none"

        info = {
            "bounce_count": self.bounce_count,
            # Backward-compatible name, now the ball-paddle pair-force peak
            # sampled over every physics substep rather than one broad sensor
            # value at the control boundary.
            "touch_sensor": self._substep_touch_peak,
            "valid_bounce": valid_bounces,
            "contact_episode_count": self.contact_episode_count,
            "contact_started": self._substep_contact_started,
            "contact_latched": self._contact_latched,
            "contact_release_progress": self._contact_release_progress,
            "contact_peak_force": self._contact_peak_force,
            "impact_relative_speed": self._impact_relative_speed,
            "paddle_normal_speed_at_impact": self._impact_paddle_speed,
            "rebound_relative_speed": self._rebound_relative_speed,
            "contact_rejection_reason": self._contact_rejection_reason,
            "ball_paddle_height": ball_paddle_height,
            "ball_velocity": current_velocity_value,
            "ball_accelerometer": current_accelerometer_value,
            "ball_to_paddle": current_fromto_value,
            "termination_reason": termination_reason,
            "term_ball_dropped": bool(term_ball_dropped),
            "term_nonfinite": bool(term_nonfinite),
            "term_timeout": bool(term_timeout),
        }
        return obs, reward, terminated, truncated, info

    def reset_model(self):
        self.step_number = 0
        self._reset_contact_tracker()

        qpos, qvel = self._noisy_init_state()
        # Keep limited one-DOF joints inside their legal ranges.  In
        # particular, uniform noise used to put slider_z below its zero lower
        # bound in roughly half of resets.
        for name in (
            "rotate_x",
            "rotate_y",
            "rotate_z",
            "slider_x",
            "slider_y",
            "slider_z",
        ):
            joint = self.model.joint(name)
            if bool(joint.limited[0]):
                address = int(joint.qposadr[0])
                qpos[address] = np.clip(
                    qpos[address], float(joint.range[0]), float(joint.range[1])
                )
        quaternion = qpos[self._ball_qposadr + 3 : self._ball_qposadr + 7]
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm <= 0.0 or not math.isfinite(quaternion_norm):
            quaternion[:] = (1.0, 0.0, 0.0, 0.0)
        else:
            quaternion /= quaternion_norm

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        self._last_finite_observation = observation.copy()
        return observation

    observation_names: tuple[str, ...] = (
        "ball_x",
        "ball_y",
        "ball_z",
        "ball_vx",
        "ball_vy",
        "ball_vz",
        "rotate_x_qpos",
        "rotate_x_qvel",
        "rotate_y_qpos",
        "rotate_y_qvel",
        "rotate_z_qpos",
        "rotate_z_qvel",
        "slider_x_qpos",
        "slider_x_qvel",
        "slider_y_qpos",
        "slider_y_qvel",
        "slider_z_qpos",
        "slider_z_qvel",
        "ball_wx",
        "ball_wy",
        "ball_wz",
        "contact_latched",
        "contact_candidate",
        "contact_force_qualified",
        "contact_release_progress",
        "rebound_relative_speed",
        "contact_local_x",
        "contact_local_y",
        "contact_local_z",
        "episode_progress",
    )

    def _get_obs(self) -> np.ndarray:
        ball = self.data.joint("ball_freejoint")
        release_progress = float(self._contact_release_progress) / float(
            self.release_substeps
        )
        episode_progress = min(
            float(self.step_number) / float(self.episode_len), 1.0
        )
        return np.concatenate(
            (
                np.asarray(ball.qpos[:3]),
                np.asarray(ball.qvel[:3]),
                self._joints_obs(
                    "rotate_x",
                    "rotate_y",
                    "rotate_z",
                    "slider_x",
                    "slider_y",
                    "slider_z",
                ),
                np.asarray(ball.qvel[3:6]),
                np.array(
                    [
                        float(self._contact_latched),
                        float(self._contact_candidate),
                        float(self._contact_peak_force > self.min_force),
                        release_progress,
                        self._rebound_relative_speed,
                    ]
                ),
                self._contact_local_position,
                np.array([episode_progress]),
            )
        )
