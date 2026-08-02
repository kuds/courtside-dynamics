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

    def test_x_monotone_and_clamped(self):
        shim = SHIM_VARIANTS["scaled"]
        xs = [
            paddle_court_action(np.array([ax, 0, 0]), shim)[0]
            for ax in np.linspace(-1, 1, 9)
        ]
        assert all(-1.0 <= x <= 1.0 for x in xs)
        assert xs == sorted(xs)

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

        player = transfer_policy(
            recorder, SHIM_VARIANTS["scaled"], yield_overlay=True
        )
        parked = player(_obs(expected_returner_is_own=0.0))
        assert np.array_equal(parked, np.zeros(3))
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
