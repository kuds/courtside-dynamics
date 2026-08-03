"""The P5 transfer shim: wall-ball view of the paddle court.

Pins the observation/action identification in
``tools/paddle_tennis_p5_transfer.py`` (the instrument the real
champion measurements run through on Colab) and smokes the transfer
loop with the wall-ball oracle. Seeds come from the burned calibration
blocks only; the P5 calibration block proper is 5000-5099 and the
reserved held-out block 4100-4199 is never touched.
"""
from __future__ import annotations

import numpy as np
import pytest

from courtside_dynamics.envs.paddle_tennis import (
    PADDLE_TENNIS_OBSERVATION_NAMES,
)
from tools.paddle_tennis_p5_transfer import (
    SHIM_VARIANTS,
    WALL_FACE_X,
    WB_PADDLE_ORIGIN,
    WB_X_MAPPING,
    paddle_court_action,
    run_transfer,
    stub_oracle_policy,
    transfer_policy,
    wall_ball_observation,
)

_IDX = {
    name: index
    for index, name in enumerate(PADDLE_TENNIS_OBSERVATION_NAMES)
}


def _obs(**values: float) -> np.ndarray:
    observation = np.zeros(len(PADDLE_TENNIS_OBSERVATION_NAMES))
    for name, value in values.items():
        observation[_IDX[name]] = value
    return observation


class TestObservationShim:
    def test_net_variant_translates_positions_only(self):
        observation = _obs(
            ball_position_x=-3.25,
            ball_position_y=0.4,
            ball_position_z=1.1,
            ball_linear_velocity_x=7.5,
            ball_linear_velocity_y=-0.2,
            own_paddle_position_x=-1.7,
            own_paddle_position_y=0.5,
            own_paddle_position_z=1.2,
            own_paddle_velocity_x=2.0,
            ball_minus_own_paddle_y=-0.1,
            ball_angular_velocity_y=30.0,
        )
        wall = wall_ball_observation(observation, SHIM_VARIANTS["net"])
        assert wall.shape == (23,)
        assert wall[0] == pytest.approx(WALL_FACE_X - 3.25)
        assert wall[1] == 0.4
        assert wall[3] == 7.5  # velocities unscaled at k=1
        # Paddle qpos are offsets from the wall-ball origin.
        assert wall[6] == pytest.approx(
            (WALL_FACE_X - 1.7) - WB_PADDLE_ORIGIN[0]
        )
        assert wall[8] == 0.5
        assert wall[10] == pytest.approx(1.2 - WB_PADDLE_ORIGIN[2])
        assert wall[7] == 2.0
        # paddle_to_ball_dx is recomputed in the wall frame.
        assert wall[14] == pytest.approx(wall[0] - (wall[6] - 1.7))
        assert wall[15] == -0.1
        assert wall[18] == 30.0
        # No stall clock, curriculum, or recovery on the paddle court.
        assert tuple(wall[20:23]) == (0.0, 0.0, 0.0)

    def test_scaled_variant_scales_x_and_vx(self):
        shim = SHIM_VARIANTS["scaled"]
        observation = _obs(
            ball_position_x=-6.5, ball_linear_velocity_x=-9.0
        )
        wall = wall_ball_observation(observation, shim)
        # The paddle-court baseline lands on the wall-ball baseline.
        assert wall[0] == pytest.approx(-7.985)
        assert wall[3] == pytest.approx(-9.0 * shim.scale)

    def test_opponent_half_ball_held_at_the_wall_face(self):
        """The far half does not exist in the champion's world: the
        rendered ball waits at the face (wall-ball's rebound moment)
        instead of flying up to 6 m beyond it."""
        from tools.paddle_tennis_p5_transfer import WB_BALL_MAX_X

        for variant in ("net", "scaled"):
            shim = SHIM_VARIANTS[variant]
            deep = wall_ball_observation(
                _obs(ball_position_x=5.0), shim
            )
            assert deep[0] == pytest.approx(WB_BALL_MAX_X)
            # paddle_to_ball must be consistent with the held position.
            assert deep[14] == pytest.approx(
                deep[0] - (deep[6] - 1.7)
            )

    def test_gate_flag_is_own_shot_in_flight(self):
        in_flight = _obs(ball_side_is_own=1.0, expected_returner_is_own=0.0)
        ours_to_hit = _obs(
            ball_side_is_own=1.0, expected_returner_is_own=1.0
        )
        crossed = _obs(ball_side_is_own=0.0, expected_returner_is_own=0.0)
        shim = SHIM_VARIANTS["net"]
        assert wall_ball_observation(in_flight, shim)[12] == 1.0
        assert wall_ball_observation(ours_to_hit, shim)[12] == 0.0
        assert wall_ball_observation(crossed, shim)[12] == 0.0


class TestActionShim:
    def test_y_z_pass_through_unchanged(self):
        action = paddle_court_action(
            np.array([0.0, 0.4, -0.7]), SHIM_VARIANTS["scaled"]
        )
        assert action[1] == pytest.approx(0.4)
        assert action[2] == pytest.approx(-0.7)

    def test_scaled_x_endpoints_pinned(self):
        """Hand-computed anchors (scale 11.885/6.5 = 1.828462): the
        decode must land these exactly, catching wrong-home,
        wrong-scale, and swapped-range bugs the earlier monotonicity
        check could not (adversarial-review finding)."""
        shim = SHIM_VARIANTS["scaled"]
        # a=-1 -> wb -8.2 -> local -6.617 -> clamped -6.4 -> action -1.
        assert paddle_court_action(np.array([-1.0, 0, 0]), shim)[
            0
        ] == pytest.approx(-1.0)
        # a=0 -> wb home -5.4 -> local -5.08624 -> (x+1.7)/4.7.
        assert paddle_court_action(np.array([0.0, 0, 0]), shim)[
            0
        ] == pytest.approx(-0.720477, abs=1e-5)
        # a=+1 -> wb 0.3 -> FENCE-clamped -2.6 -> local -3.55490.
        assert paddle_court_action(np.array([1.0, 0, 0]), shim)[
            0
        ] == pytest.approx(-0.394660, abs=1e-5)

    def test_fence_semantics_match_training(self):
        """In training every x action >= +0.491 meant 'paddle to the
        fence front (-2.6)'; the decode must reproduce that plateau,
        not spread the saturated interval over unreachable targets."""
        shim = SHIM_VARIANTS["scaled"]
        at_fence = paddle_court_action(np.array([0.491, 0, 0]), shim)[0]
        for ax in (0.6, 0.8, 1.0):
            assert paddle_court_action(np.array([ax, 0, 0]), shim)[
                0
            ] == pytest.approx(at_fence, abs=1e-3)

    def test_net_variant_pins_the_champion_deep(self):
        """The rigid translation maps the whole champion command range
        behind the paddle-court baseline -- the measured reason the
        ``net`` arm scores zero (P5 snapshot)."""
        shim = SHIM_VARIANTS["net"]
        for ax in (-1.0, 0.0, 1.0):
            target_wb = {
                -1.0: WB_X_MAPPING[0],
                0.0: WB_X_MAPPING[1],
                1.0: WB_X_MAPPING[2],
            }[ax]
            assert shim.to_local_x(target_wb) <= -3.6
        deep = paddle_court_action(np.array([-1.0, 0, 0]), shim)
        assert deep[0] == pytest.approx(-1.0)  # clamped to -6.4


class TestYieldOverlay:
    def test_yields_unless_ball_is_ours(self):
        calls = []

        def recorder(wall_obs: np.ndarray) -> np.ndarray:
            calls.append(wall_obs)
            return np.array([0.5, 0.5, 0.5])

        shim = SHIM_VARIANTS["scaled"]
        player = transfer_policy(recorder, shim, yield_overlay=True)
        parked = player(_obs(expected_returner_is_own=0.0))
        # The park is the CHAMPION'S neutral decoded through the shim
        # (wall-ball home -5.4 -> local -5.086), never the paddle-
        # court home: the release-step self-state must sit inside the
        # champion's lifelong workspace.
        assert np.array_equal(
            parked, paddle_court_action(np.zeros(3), shim)
        )
        assert parked[0] == pytest.approx(-0.720477, abs=1e-5)
        assert not calls
        active = player(_obs(expected_returner_is_own=1.0))
        assert calls and active.shape == (3,)


class TestTransferSmoke:
    def test_stub_oracle_transfer_runs(self):
        result = run_transfer(
            transfer_policy(
                stub_oracle_policy(),
                SHIM_VARIANTS["scaled"],
                yield_overlay=True,
            ),
            label="smoke",
            episodes=2,
            seed_start=1000,
        )
        assert result.episodes == 2
        assert result.mean_crossings >= 0.0
        assert sum(result.terminations.values()) == 2
        assert "smoke" in result.row()
