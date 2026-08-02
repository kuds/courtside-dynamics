"""PaddleInterface extraction and the paddle-court prototype (P4).

Three concerns, in dependency order:

1. ``PaddleInterface`` is the canonical calibrated-paddle
   implementation; a drift test pins its control mapping bit-for-bit
   against the live ``WallBallEnv`` until WallBall's internals are
   mechanically delegated onto it.
2. The prototype scene wires the probe-frozen court through the shared
   rules/events machinery without robots.
3. Probe P4 (design_paddle_tennis.md §6): a mirrored state produces
   bit-for-bit mirrored side-relative observations, mirrored actions
   produce exactly mirrored control targets, and mirrored dynamics stay
   mirrored through real physics.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pytest

from courtside_dynamics.envs import WallBallEnv
from courtside_dynamics.envs._paddle import PaddleInterface
from courtside_dynamics.envs._paddle_court import (
    PADDLE_COURT,
    PaddleCourtScene,
    PaddleCourtServe,
)
from courtside_dynamics.envs.tennis_rules import CourtSide


def _wall_ball_interface(env: WallBallEnv) -> PaddleInterface:
    unwrapped = env.unwrapped
    return PaddleInterface(
        unwrapped.model,
        unwrapped.data,
        home_x=unwrapped.paddle_home_x,
        x_target_range=unwrapped.paddle_x_target_range,
        x_fence=unwrapped.paddle_x_fence,
        joint_damping=unwrapped.paddle_joint_damping,
    )


class TestPaddleInterfaceDrift:
    """Pin the extracted interface to WallBallEnv's live behavior."""

    @pytest.mark.parametrize(
        "env_kwargs",
        [
            {},
            {
                "paddle_home_x": -2.5,
                "paddle_x_target_range": (-4.0, 0.0),
                "paddle_x_fence": (-3.5, -1.0),
                "paddle_joint_damping": 8.0,
            },
        ],
        ids=["default", "fenced"],
    )
    def test_control_mapping_matches_wall_ball(self, env_kwargs):
        env = WallBallEnv(**env_kwargs)
        try:
            interface = _wall_ball_interface(env)
            rng = np.random.default_rng(7)
            actions = [
                np.zeros(3),
                np.ones(3),
                -np.ones(3),
                *(rng.uniform(-1.2, 1.2, size=3) for _ in range(50)),
            ]
            for action in actions:
                expected = env.unwrapped._action_to_controls(
                    action.astype(np.float32)
                )
                produced = interface.targets(action.astype(np.float32))
                assert np.array_equal(expected, produced), action
        finally:
            env.close()

    def test_world_space_views_match_wall_ball(self):
        env = WallBallEnv(paddle_home_x=-2.5)
        try:
            unwrapped = env.unwrapped
            interface = _wall_ball_interface(env)
            assert interface.home_x == unwrapped.paddle_home_x
            assert (
                interface.x_target_range == unwrapped.paddle_x_target_range
            )
            assert (
                interface.physical_world_range
                == unwrapped._paddle_x_world_range
            )
            assert interface.qpos_x_for_world(
                unwrapped.paddle_home_x
            ) == pytest.approx(float(unwrapped._control_home[0]))
        finally:
            env.close()

    def test_validation_matches_wall_ball_contracts(self):
        env = WallBallEnv()
        try:
            interface = _wall_ball_interface(env)
            with pytest.raises(ValueError, match="must stay within"):
                interface.set_x_target_range((-9.0, 0.3))
            with pytest.raises(ValueError, match="must lie inside"):
                interface.set_home_x(2.0)
            with pytest.raises(ValueError, match="strictly increasing"):
                interface.set_fence((-1.0, -2.0))
            with pytest.raises(ValueError, match="finite and positive"):
                interface.set_damping(0.0)
            # Damping override writes and None restores the XML value.
            dofadr = int(env.unwrapped.model.joint("paddle_slide_x").dofadr[0])
            xml_value = float(env.unwrapped.model.dof_damping[dofadr])
            interface.set_damping(9.0)
            assert float(env.unwrapped.model.dof_damping[dofadr]) == 9.0
            interface.set_damping(None)
            assert float(env.unwrapped.model.dof_damping[dofadr]) == xml_value
        finally:
            env.close()


def _mirror_scene_state(
    source: PaddleCourtScene, target: PaddleCourtScene
) -> None:
    """Write the exact 180-degree rotation of ``source`` into ``target``.

    The court rotation swaps the paddle roles: side B's slides in the
    target take the mirror of side A's in the source (their bases and
    ranges are exact mirrors, so the mirrored world pose is
    ``qpos' = -qpos`` for x/y and identical z), and vice versa.
    """
    src, dst = source.data, target.data
    model = source.model
    for a_joint, b_joint in (
        ("player_a_slide_x", "player_b_slide_x"),
        ("player_a_slide_y", "player_b_slide_y"),
        ("player_a_slide_z", "player_b_slide_z"),
    ):
        sign = -1.0 if a_joint.endswith(("_x", "_y")) else 1.0
        for one, other in ((a_joint, b_joint), (b_joint, a_joint)):
            qposadr = int(model.joint(one).qposadr[0])
            dofadr = int(model.joint(one).dofadr[0])
            other_qposadr = int(model.joint(other).qposadr[0])
            other_dofadr = int(model.joint(other).dofadr[0])
            dst.qpos[other_qposadr] = sign * src.qpos[qposadr]
            dst.qvel[other_dofadr] = sign * src.qvel[dofadr]
    ball_q = source._ball_qposadr
    ball_d = source._ball_dofadr
    dst.qpos[ball_q : ball_q + 7] = src.qpos[ball_q : ball_q + 7]
    dst.qpos[ball_q] = -src.qpos[ball_q]
    dst.qpos[ball_q + 1] = -src.qpos[ball_q + 1]
    dst.qvel[ball_d : ball_d + 6] = src.qvel[ball_d : ball_d + 6]
    dst.qvel[ball_d] = -src.qvel[ball_d]
    dst.qvel[ball_d + 1] = -src.qvel[ball_d + 1]
    dst.qvel[ball_d + 3] = -src.qvel[ball_d + 3]
    dst.qvel[ball_d + 4] = -src.qvel[ball_d + 4]
    mujoco.mj_forward(target.model, target.data)


def _randomize_scene(scene: PaddleCourtScene, rng: np.random.Generator):
    data, model = scene.data, scene.model
    for name, low, high in (
        ("player_a_slide_x", -1.0, 1.0),
        ("player_a_slide_y", -1.5, 1.5),
        ("player_a_slide_z", -0.5, 0.5),
        ("player_b_slide_x", -1.0, 1.0),
        ("player_b_slide_y", -1.5, 1.5),
        ("player_b_slide_z", -0.5, 0.5),
    ):
        qposadr = int(model.joint(name).qposadr[0])
        dofadr = int(model.joint(name).dofadr[0])
        data.qpos[qposadr] = rng.uniform(low, high)
        data.qvel[dofadr] = rng.uniform(-1.0, 1.0)
    ball_q, ball_d = scene._ball_qposadr, scene._ball_dofadr
    data.qpos[ball_q : ball_q + 3] = [
        rng.uniform(-5.0, 5.0),
        rng.uniform(-3.0, 3.0),
        rng.uniform(0.3, 2.5),
    ]
    data.qpos[ball_q + 3 : ball_q + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[ball_d : ball_d + 3] = rng.uniform(-8.0, 8.0, size=3)
    data.qvel[ball_d + 3 : ball_d + 6] = rng.uniform(-20.0, 20.0, size=3)
    mujoco.mj_forward(model, data)


class TestPaddleCourtPrototype:
    def test_scene_wires_the_shared_machinery_without_robots(self):
        scene = PaddleCourtScene()
        assert not scene.index.humanoid_a_geoms
        assert not scene.index.humanoid_b_geoms
        assert scene.index.racket_a_geoms and scene.index.racket_b_geoms
        assert scene.rules.court is PADDLE_COURT
        assert scene.sampler.court is PADDLE_COURT
        # Probe-frozen paddle calibration: damping 8 baked into the XML.
        for joint in ("player_a_slide_x", "player_b_slide_x"):
            dofadr = int(scene.model.joint(joint).dofadr[0])
            assert float(scene.model.dof_damping[dofadr]) == 8.0

    def test_serve_launches_a_legal_feed_that_the_rules_score(self):
        scene = PaddleCourtScene()
        rng = np.random.default_rng(0)
        position, velocity = scene.serve(
            PaddleCourtServe(), serving_side=CourtSide.A, rng=rng
        )
        assert position[0] < 0 and velocity[0] > 0
        crossed = False
        for step in range(400):
            batch = scene.step_control_frame(step)
            transition = scene.rules.advance(
                batch.events, contact_peaks=batch.contact_peaks
            )
            crossed = crossed or transition.after.net_crossing_count > 0
            if transition.after.terminated:
                break
        assert crossed, "the seeded serve must cross the net"
        assert transition.after.terminated, (
            "an unreturned serve must terminate through the rules"
        )

    def test_serve_config_validates_through_the_shared_feed_contract(self):
        with pytest.raises(ValueError, match="serve.*one court half"):
            PaddleCourtServe(start_distance_from_net=7.0)  # > 6.5 half
        with pytest.raises(ValueError, match="speed must remain positive"):
            PaddleCourtServe(speed=1.0, speed_noise=2.0)


class TestP4MirroringIdentity:
    """design_paddle_tennis.md §6 P4, on the candidate contracts."""

    def test_mirrored_states_produce_identical_observations(self):
        """obs(A, S) == obs(B, mirror(S)) bit-for-bit, over many states."""
        scene = PaddleCourtScene()
        mirrored = PaddleCourtScene()
        rng = np.random.default_rng(8100)
        for _ in range(25):
            _randomize_scene(scene, rng)
            _mirror_scene_state(scene, mirrored)
            obs_a = scene.observation(CourtSide.A)
            obs_b = mirrored.observation(CourtSide.B)
            assert np.array_equal(obs_a, obs_b)
            # And the other pairing: B's view of S == A's view of mirror(S).
            assert np.array_equal(
                scene.observation(CourtSide.B),
                mirrored.observation(CourtSide.A),
            )

    def test_mirrored_actions_produce_mirrored_targets(self):
        """The same side-local action commands the exact court-mirrored
        world target through either paddle (sign-exact in IEEE)."""
        scene = PaddleCourtScene()
        rng = np.random.default_rng(11)
        for _ in range(50):
            local_action = rng.uniform(-1.0, 1.0, size=3)
            targets_a = scene.apply_action(CourtSide.A, local_action)
            targets_b = scene.apply_action(CourtSide.B, local_action)
            # x targets are in each paddle's qpos space; add the
            # origins back to compare world-space targets, which must
            # mirror exactly.
            a_world_x = (
                targets_a[0] + scene.paddles[CourtSide.A]._x_origin
            )
            b_world_x = (
                targets_b[0] + scene.paddles[CourtSide.B]._x_origin
            )
            assert a_world_x == -b_world_x
            assert targets_a[1] == -targets_b[1]
            assert targets_a[2] == targets_b[2]

    def test_mirrored_dynamics_stay_mirrored_through_physics(self):
        """Playing the same side-local policy from mirrored states keeps
        the trajectories mirrored through real contact-rich physics.

        Not asserted bit-for-bit: MuJoCo's constraint assembly orders
        the two paddles' contacts differently on the two sides, so
        float association can differ at the ulp level. The tolerance is
        far below anything a policy or the rules machinery can resolve;
        the measured drift is recorded in the P3/P4 probe doc.
        """
        scene = PaddleCourtScene()
        mirrored = PaddleCourtScene()
        rng = np.random.default_rng(8150)
        policy_rng = np.random.default_rng(9)
        for _ in range(5):
            _randomize_scene(scene, rng)
            _mirror_scene_state(scene, mirrored)
            actions = policy_rng.uniform(-1.0, 1.0, size=(40, 3))
            for local_action in actions:
                scene.apply_action(CourtSide.A, local_action)
                scene.apply_action(
                    CourtSide.B, np.array([0.1, -0.2, 0.0])
                )
                mirrored.apply_action(CourtSide.B, local_action)
                mirrored.apply_action(
                    CourtSide.A, np.array([0.1, -0.2, 0.0])
                )
                for _ in range(5):
                    mujoco.mj_step(scene.model, scene.data)
                    mujoco.mj_step(mirrored.model, mirrored.data)
            ball = scene.ball_position()
            ball_m = mirrored.ball_position()
            assert ball[0] == pytest.approx(-ball_m[0], abs=1e-6)
            assert ball[1] == pytest.approx(-ball_m[1], abs=1e-6)
            assert ball[2] == pytest.approx(ball_m[2], abs=1e-6)


class TestP3Harness:
    """The serve-rules probe harness (tools/paddle_tennis_probes.py)."""

    def test_quick_sweep_produces_paired_alternation_cells(self):
        import sys

        sys.path.insert(0, "tools")
        try:
            from paddle_tennis_probes import format_report, sweep_serve_rules
        finally:
            sys.path.pop(0)

        cells = sweep_serve_rules(quick=True)
        # One config, both serving sides, sharing a seed block.
        assert len(cells) == 2
        sides = {cell.serving_side for cell in cells}
        assert sides == {CourtSide.A, CourtSide.B}
        # P4's mirror identity, observed end to end: the alternation
        # twin must reproduce the primary cell's statistics.
        primary, twin = cells
        assert primary.legal_serve_rate == twin.legal_serve_rate
        assert primary.mean_crossings == twin.mean_crossings
        report = format_report(cells)
        assert "| depth | speed | elev | side |" in report

    def test_lead_charge_controller_yields_parks_and_charges(self):
        import sys

        sys.path.insert(0, "tools")
        try:
            from paddle_tennis_probes import lead_charge_local_action
        finally:
            sys.path.pop(0)

        # Outgoing ball: neutral yield (exactly zero action).
        outgoing = lead_charge_local_action(
            np.array([-3.0, 0.5, 1.5]),
            np.array([6.0, 0.0, 2.0]),
            np.array([-1.7, 0.0, 1.2]),
        )
        assert np.array_equal(outgoing, np.zeros(3))
        # Incoming far ball: hold the home column, track y.
        waiting = lead_charge_local_action(
            np.array([3.0, 1.0, 2.0]),
            np.array([-8.0, 0.0, 1.0]),
            np.array([-1.7, 0.0, 1.2]),
        )
        assert waiting[0] == pytest.approx(0.0)
        assert waiting[1] > 0.0
        # Incoming ball inside the commit gap: swing toward the net.
        charging = lead_charge_local_action(
            np.array([-1.2, 0.0, 1.4]),
            np.array([-7.0, 0.0, -1.0]),
            np.array([-1.7, 0.0, 1.2]),
        )
        assert charging[0] > 0.0
