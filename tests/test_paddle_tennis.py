"""PaddleTennisEnv: the frozen phase-P1 task contract.

Pins the freeze recorded in ``docs/paddle_tennis_env_20260802.md``:
the 48-value side-relative observation layout, the P4 mirror observed
end-to-end through the registered env, the cooperative reward
accounting, serve alternation, the render-only court styles, opponent
injection, and the nonfinite guards. Cross-env API smoke (check_env,
truncation, registration ids) lives in ``test_envs.py``'s shared
tables.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import courtside_dynamics  # noqa: F401  (triggers registration)
from courtside_dynamics.envs import PaddleCourtServe, PaddleTennisEnv
from courtside_dynamics.envs._paddle_court import (
    OBS_BALL_SIDE_INDEX,
    OBS_BOUNCE_COUNT_INDEX,
    scripted_ground_opponent,
    scripted_lead_charge_opponent,
)
from courtside_dynamics.envs.paddle_tennis import (
    PADDLE_TENNIS_ACTION_NAMES,
    PADDLE_TENNIS_NORMALIZED_SLICE,
    PADDLE_TENNIS_OBSERVATION_NAMES,
)
from courtside_dynamics.envs.tennis_rules import CourtSide

#: Bring-up seeds (calibration block 1000-1119, already burned by the
#: P3 harness diagnostics); the reserved held-out blocks 3100-3199 and
#: 4100-4199 must never appear in tests.
_SMOKE_SEEDS = (1000, 1001, 1002)


def _zero_action() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


class TestObservationContract:
    def test_observation_names_pin(self):
        """The exact frozen layout; any change is a new comparability era."""
        expected = (
            *(f"ball_position_{axis}" for axis in "xyz"),
            *(f"ball_linear_velocity_{axis}" for axis in "xyz"),
            *(f"ball_angular_velocity_{axis}" for axis in "xyz"),
            *(f"own_paddle_position_{axis}" for axis in "xyz"),
            *(f"own_paddle_velocity_{axis}" for axis in "xyz"),
            *(f"opponent_paddle_position_{axis}" for axis in "xyz"),
            *(f"opponent_paddle_velocity_{axis}" for axis in "xyz"),
            *(f"ball_minus_own_paddle_{axis}" for axis in "xyz"),
            "rally_phase_initial_feed",
            "rally_phase_awaiting_return",
            "rally_phase_return_in_flight",
            "rally_phase_terminal",
            "own_is_serving",
            "expected_returner_is_own",
            "ball_side_is_own",
            "feed_crossed_net",
            "pending_return_crossed_net",
            "bounce_count",
            "rally_count",
            "episode_remaining_fraction",
            "contact_latched_own_racket",
            "contact_latched_opponent_racket",
            "contact_latched_court",
            "contact_latched_net",
            "contact_latched_own_racket_net",
            "contact_latched_opponent_racket_net",
            "contact_release_progress_own_racket",
            "contact_release_progress_opponent_racket",
            "contact_release_progress_court",
            "contact_release_progress_net",
            "contact_release_progress_own_racket_net",
            "contact_release_progress_opponent_racket_net",
        )
        assert PADDLE_TENNIS_OBSERVATION_NAMES == expected
        assert PADDLE_TENNIS_ACTION_NAMES == (
            "target_x",
            "target_y",
            "target_z",
        )
        assert PADDLE_TENNIS_NORMALIZED_SLICE == slice(0, 24)
        assert (
            PADDLE_TENNIS_OBSERVATION_NAMES[
                PADDLE_TENNIS_NORMALIZED_SLICE.stop
            ]
            == "rally_phase_initial_feed"
        )

    def test_registration_matches_episode_len_default(self):
        import gymnasium

        spec = gymnasium.spec("CourtsideDynamics/PaddleTennis")
        default = inspect.signature(PaddleTennisEnv.__init__).parameters[
            "episode_len"
        ].default
        assert spec.max_episode_steps == default == 1500

    def test_reset_observation_shape_and_reset_info(self):
        env = PaddleTennisEnv()
        try:
            obs, info = env.reset(seed=_SMOKE_SEEDS[0])
            assert obs.shape == (len(PADDLE_TENNIS_OBSERVATION_NAMES),)
            assert bool(np.isfinite(obs).all())
            assert info["serve_side"] == "a"
            assert info["serve_side_is_policy"] == 1.0
            position = np.asarray(info["serve_ball_position"])
            velocity = np.asarray(info["serve_ball_velocity"])
            # Side A serves from its own (negative-x) half, toward +x.
            assert position[0] < 0.0
            assert velocity[0] > 0.0
            # The serving side's own view of itself must be identical
            # to the policy's view of the reset (side A serves here).
            assert np.array_equal(
                obs, env.observation_for_side(CourtSide.A)
            )
        finally:
            env.close()


class TestMirrorIdentity:
    """P4, observed end-to-end through the registered env.

    Twin envs on the same seed: env_a serves side A; env_b consumes
    one extra reset so the alternation serves side B from the same
    noise draw -- the exact mirrored initial state. With the same
    deterministic controller driving the policy side of both envs (and
    the opponent side, via the default scripted controller), the pair
    must stay mirrored: env_b's policy observation equals env_a's
    side-B observation.
    """

    def test_mirrored_twins_stay_mirrored(self):
        env_a = PaddleTennisEnv()
        env_b = PaddleTennisEnv()
        try:
            seed = _SMOKE_SEEDS[1]
            obs_a, info_a = env_a.reset(seed=seed)
            env_b.reset(seed=seed)
            obs_b, info_b = env_b.reset(seed=seed)
            assert info_a["serve_side"] == "a"
            assert info_b["serve_side"] == "b"
            assert info_b["serve_side_is_policy"] == 0.0

            # Bit-for-bit at the mirrored reset.
            assert np.array_equal(
                obs_b, env_a.observation_for_side(CourtSide.B)
            )
            assert np.array_equal(
                obs_a, env_b.observation_for_side(CourtSide.B)
            )

            # Through contact-rich physics the mirrored trajectories
            # drift only by MuJoCo constraint-ordering ulps (the P4
            # dynamics bound).
            for _ in range(40):
                action_a = scripted_ground_opponent(obs_a)
                action_b = scripted_ground_opponent(obs_b)
                obs_a, _, term_a, trunc_a, _ = env_a.step(action_a)
                obs_b, _, term_b, trunc_b, _ = env_b.step(action_b)
                np.testing.assert_allclose(
                    obs_b,
                    env_a.observation_for_side(CourtSide.B),
                    atol=1e-6,
                )
                assert (term_a, trunc_a) == (term_b, trunc_b)
                if term_a or trunc_a:
                    break
        finally:
            env_a.close()
            env_b.close()


class TestRewardAccounting:
    def test_scripted_pair_rallies_and_reward_decomposes(self):
        """The frozen scripted pair reproduces a P3-band rally, and
        every step's reward equals its ``rew_*`` decomposition."""
        env = PaddleTennisEnv()
        try:
            best_crossings = 0
            for seed in _SMOKE_SEEDS:
                obs, _ = env.reset(seed=seed)
                total = 0.0
                total_return = 0.0
                while True:
                    obs, reward, term, trunc, info = env.step(
                        scripted_ground_opponent(obs)
                    )
                    assert reward == pytest.approx(
                        info["rew_return"]
                        + info["rew_fault"]
                        + info["rew_unsafe"]
                    )
                    total += reward
                    total_return += info["rew_return"]
                    if term or trunc:
                        break
                # Cooperative accounting: the returns component equals
                # the rules' confirmed-return count; a terminated point
                # pays exactly one fault, a cap-truncated rally none
                # (the ground era's scripted pair reaches the 1500-step
                # cap in ~17% of episodes).
                assert total_return == pytest.approx(
                    float(info["valid_return_count"])
                )
                if trunc:
                    assert total == pytest.approx(total_return)
                    assert info["rew_fault"] == 0.0
                else:
                    assert total == pytest.approx(total_return - 1.0)
                    assert info["rew_fault"] == -1.0
                assert info["rew_unsafe"] == 0.0
                best_crossings = max(best_crossings, info["crossings"])
                # Exactly one grouped termination flag fires.
                flags = [
                    info[name]
                    for name in (
                        "term_out_of_bounds",
                        "term_ball_net",
                        "term_second_bounce",
                        "term_failed_to_cross",
                        "term_illegal_hit",
                        "term_net_touch",
                        "term_volley",
                        "term_nonfinite",
                        "term_timeout",
                    )
                ]
                # Exactly one flag fires on every episode end --
                # term_timeout carries the truncation case.
                assert sum(flags) == 1.0
                if trunc:
                    assert info["term_timeout"] == 1.0
            # The ground-era band puts the scripted pair at ~7
            # crossings/point; across three seeds a real rally must
            # appear.
            assert best_crossings >= 2
        finally:
            env.close()

    def test_custom_reward_scales(self):
        env = PaddleTennisEnv(return_reward=2.5, fault_penalty=0.5)
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
            total_return = 0.0
            while True:
                obs, reward, term, trunc, info = env.step(
                    scripted_ground_opponent(obs)
                )
                total_return += info["rew_return"]
                if term or trunc:
                    break
            assert total_return == pytest.approx(
                2.5 * float(info["valid_return_count"])
            )
            if not trunc:
                assert info["rew_fault"] == -0.5
        finally:
            env.close()


class TestVolleyRule:
    """The ground-rules era: volleys fault, the legacy profile survives."""

    def test_kwarg_validation_and_default(self):
        env = PaddleTennisEnv()
        try:
            assert env.volley_rule == "fault"
            assert (
                env.opponent_controller is scripted_ground_opponent
            )
        finally:
            env.close()
        env = PaddleTennisEnv(volley_rule="legal")
        try:
            assert (
                env.opponent_controller
                is scripted_lead_charge_opponent
            )
        finally:
            env.close()
        with pytest.raises(ValueError, match="volley_rule"):
            PaddleTennisEnv(volley_rule="bogus")

    def test_obs_index_pins_for_ground_controller(self):
        """envs/_paddle_court.py cannot import the env, so its obs
        indices are literals -- pin them against the frozen names."""
        assert (
            PADDLE_TENNIS_OBSERVATION_NAMES[OBS_BOUNCE_COUNT_INDEX]
            == "bounce_count"
        )
        assert (
            PADDLE_TENNIS_OBSERVATION_NAMES[OBS_BALL_SIDE_INDEX]
            == "ball_side_is_own"
        )

    def test_volley_capable_player_faults_under_ground_rules(self):
        """The frozen P1 oracle's serve intercept is a volley; under
        the era default it must terminate as term_volley with the
        fault penalty, crediting no crossings."""
        env = PaddleTennisEnv()
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
            while True:
                obs, reward, term, trunc, info = env.step(
                    scripted_lead_charge_opponent(obs)
                )
                if term or trunc:
                    break
            assert term
            assert info["termination_reason_name"] == "volley_return"
            assert info["term_volley"] == 1.0
            # At most the opponent's own legal return precedes the
            # fault (the volleyer contributes no crossing).
            assert info["crossings"] <= 1
        finally:
            env.close()

    def test_legal_profile_reproduces_the_volley_era(self):
        """Under volley_rule='legal' the frozen pair still rallies
        (the superseded era stays reproducible for its artifacts)."""
        env = PaddleTennisEnv(volley_rule="legal")
        try:
            best = 0
            for seed in _SMOKE_SEEDS:
                obs, _ = env.reset(seed=seed)
                while True:
                    obs, _, term, trunc, info = env.step(
                        scripted_lead_charge_opponent(obs)
                    )
                    if term or trunc:
                        break
                assert info["term_volley"] == 0.0
                best = max(best, info["crossings"])
            assert best >= 2
        finally:
            env.close()

    def test_volley_fault_confirms_nothing(self):
        """Rules-level pin: a pre-bounce racket touch under ground
        rules terminates VOLLEY_RETURN and credits NOTHING -- not
        even the incoming shot. Crediting it would make touching a
        doomed out-bound ball strictly better than letting it land
        (+return_reward against the same fault penalty), a learnable
        exploit the adversarial review measured."""
        from courtside_dynamics.envs.tennis_rules import (
            RallyEvent,
            RallyEventKind,
            RallyRules,
            RallyStateMachine,
            TerminationReason,
        )

        machine = RallyStateMachine(
            serving_side=CourtSide.A,
            rules=RallyRules(require_bounce_before_return=True),
        )
        # Feed crosses to B, bounces, B returns legally.
        machine.advance(
            [
                RallyEvent(
                    kind=RallyEventKind.NET_CROSSING_TO_B, substep=0
                )
            ]
        )
        machine.advance(
            [
                RallyEvent(
                    kind=RallyEventKind.BALL_COURT_B,
                    substep=1,
                    position=(3.0, 0.0, 0.0),
                )
            ]
        )
        machine.advance(
            [RallyEvent(kind=RallyEventKind.BALL_RACKET_B, substep=2)]
        )
        machine.advance(
            [
                RallyEvent(
                    kind=RallyEventKind.NET_CROSSING_TO_A, substep=3
                )
            ]
        )
        # A volleys the incoming return before it bounces.
        transition = machine.advance(
            [RallyEvent(kind=RallyEventKind.BALL_RACKET_A, substep=4)]
        )
        assert (
            transition.after.termination_reason
            is TerminationReason.VOLLEY_RETURN
        )
        assert transition.confirmed_returns == ()
        assert transition.valid_racket_hits == ()
        # Legacy profile: the same sequence is a legal volley return.
        machine = RallyStateMachine(serving_side=CourtSide.A)
        for event in (
            RallyEvent(kind=RallyEventKind.NET_CROSSING_TO_B, substep=0),
            RallyEvent(
                kind=RallyEventKind.BALL_COURT_B,
                substep=1,
                position=(3.0, 0.0, 0.0),
            ),
            RallyEvent(kind=RallyEventKind.BALL_RACKET_B, substep=2),
            RallyEvent(kind=RallyEventKind.NET_CROSSING_TO_A, substep=3),
        ):
            machine.advance([event])
        transition = machine.advance(
            [RallyEvent(kind=RallyEventKind.BALL_RACKET_A, substep=4)]
        )
        assert not transition.after.terminated
        assert transition.confirmed_returns == (CourtSide.B,)
        assert transition.valid_racket_hits == (CourtSide.A,)

    def test_reward_kwargs_validated(self):
        with pytest.raises(ValueError, match="return_reward"):
            PaddleTennisEnv(return_reward=float("nan"))
        with pytest.raises(ValueError, match="fault_penalty"):
            PaddleTennisEnv(fault_penalty=-1.0)


class TestServeAlternation:
    def test_sides_alternate_across_resets(self):
        env = PaddleTennisEnv()
        try:
            sides = []
            for index in range(4):
                _, info = env.reset(seed=_SMOKE_SEEDS[0] + index)
                sides.append(info["serve_side"])
                assert info["serve_side_is_policy"] == (
                    1.0 if info["serve_side"] == "a" else 0.0
                )
            assert sides == ["a", "b", "a", "b"]
        finally:
            env.close()

    def test_side_b_serve_mirrors_the_draw(self):
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            _, info = env.reset(seed=_SMOKE_SEEDS[0])
            assert info["serve_side"] == "b"
            position = np.asarray(info["serve_ball_position"])
            velocity = np.asarray(info["serve_ball_velocity"])
            assert position[0] > 0.0
            assert velocity[0] < 0.0
        finally:
            env.close()


class TestCourtStyles:
    """Render-only style sets, derived from the compiled model."""

    @staticmethod
    def _visible(env, name: str) -> bool:
        return float(env.model.site(name).rgba[3]) > 0.0

    def test_style_visibility_matrix(self):
        for style, static_on, tennis_on in (
            ("diagnostic", True, False),
            ("tennis", False, True),
            ("none", False, False),
        ):
            env = PaddleTennisEnv(court_style=style)
            try:
                env.reset(seed=_SMOKE_SEEDS[0])
                assert env._court_static_sites
                assert env._court_tennis_sites
                for name in env._court_static_sites:
                    assert self._visible(env, name) is static_on, (
                        f"{name} wrong in style {style}"
                    )
                for name in env._court_tennis_sites:
                    assert self._visible(env, name) is tennis_on, (
                        f"{name} wrong in style {style}"
                    )
                for name in env._court_preset_sites:
                    assert self._visible(env, name) is static_on, (
                        f"{name} wrong in style {style}"
                    )
            finally:
                env.close()

    def test_serve_markers_follow_the_serve_config(self):
        serve = PaddleCourtServe(start_distance_from_net=4.5)
        env = PaddleTennisEnv(serve_config=serve)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            site_a = int(env.model.site("court_line_serve_a").id)
            site_b = int(env.model.site("court_line_serve_b").id)
            assert env.model.site_pos[site_a][0] == pytest.approx(-4.5)
            assert env.model.site_pos[site_b][0] == pytest.approx(4.5)
        finally:
            env.close()

    def test_none_sentinel_and_validation(self):
        env = PaddleTennisEnv(court_style=None)
        try:
            assert env.court_style == "none"
        finally:
            env.close()
        with pytest.raises(ValueError, match="court_style"):
            PaddleTennisEnv(court_style="bogus")

    def test_style_change_applies_at_next_reset(self):
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            marker = next(iter(env._court_static_sites))
            assert self._visible(env, marker)
            env.court_style = "none"
            env.reset(seed=_SMOKE_SEEDS[0])
            assert not self._visible(env, marker)
        finally:
            env.close()

    def test_styles_do_not_change_physics(self):
        """Sites cannot collide: identical seeds and actions produce
        bit-identical observations across styles."""
        traces = {}
        for style in ("diagnostic", "tennis", "none"):
            env = PaddleTennisEnv(court_style=style)
            try:
                obs, _ = env.reset(seed=_SMOKE_SEEDS[2])
                steps = [obs]
                for _ in range(30):
                    obs, _, term, trunc, _ = env.step(
                        scripted_ground_opponent(obs)
                    )
                    steps.append(obs)
                    if term or trunc:
                        break
                traces[style] = np.concatenate(steps)
            finally:
                env.close()
            assert np.array_equal(
                traces[style], traces["diagnostic"]
            ), f"style {style} altered the physics trace"


class TestOpponentInjection:
    def test_custom_opponent_sees_side_local_views(self):
        seen = []

        def opponent(observation: np.ndarray) -> np.ndarray:
            seen.append(observation.copy())
            return np.zeros(3)

        env = PaddleTennisEnv(opponent_controller=opponent)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            for _ in range(3):
                env.step(_zero_action())
            assert len(seen) == 3
            for observation in seen:
                assert observation.shape == (
                    len(PADDLE_TENNIS_OBSERVATION_NAMES),
                )
                # Side B's own-paddle x sits in its own half (negative
                # in side-local coordinates), like side A's own view.
                assert observation[9] < 0.0
        finally:
            env.close()

    def test_bad_opponent_action_fails_loudly(self):
        env = PaddleTennisEnv(
            opponent_controller=lambda observation: np.zeros(2)
        )
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            with pytest.raises(ValueError, match="opponent_controller"):
                env.step(_zero_action())
        finally:
            env.close()


class TestGuards:
    def test_wrong_action_shape_raises(self):
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            with pytest.raises(ValueError, match="shape"):
                env.step(np.zeros(6))
        finally:
            env.close()

    def test_action_is_clipped_not_rejected(self):
        env_clipped = PaddleTennisEnv()
        env_limit = PaddleTennisEnv()
        try:
            env_clipped.reset(seed=_SMOKE_SEEDS[0])
            env_limit.reset(seed=_SMOKE_SEEDS[0])
            env_clipped.step(np.array([5.0, -5.0, 5.0]))
            env_limit.step(np.array([1.0, -1.0, 1.0]))
            assert np.array_equal(
                env_clipped.data.ctrl, env_limit.data.ctrl
            )
        finally:
            env_clipped.close()
            env_limit.close()

    def test_nonfinite_action_terminates_without_physics(self):
        env = PaddleTennisEnv()
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
            qpos_before = env.data.qpos.copy()
            echoed, reward, term, trunc, info = env.step(
                np.array([np.nan, 0.0, 0.0])
            )
            assert term and not trunc
            assert reward == -2.0
            assert info["rew_unsafe"] == -2.0
            assert info["term_nonfinite"] == 1.0
            assert info["termination_reason_name"] == "nonfinite_state"
            # No physics stepped; the last finite observation echoes.
            assert np.array_equal(env.data.qpos, qpos_before)
            assert np.array_equal(echoed, obs)
        finally:
            env.close()

    def test_nonfinite_physics_state_terminates_before_stepping(self):
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            env.data.qpos[0] = np.nan
            echoed, reward, term, trunc, info = env.step(_zero_action())
            assert term
            assert reward == -2.0
            assert info["term_nonfinite"] == 1.0
            assert bool(np.isfinite(echoed).all())
        finally:
            env.close()

    def test_nonfinite_info_keeps_the_csv_schema(self):
        """The recipe's CSV row reads these keys on every step,
        including forced nonfinite terminations."""
        from courtside_dynamics.recipes import _PADDLE_TENNIS_CSV_KEYS

        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            *_, info = env.step(np.array([np.nan, 0.0, 0.0]))
            missing = [
                key for key in _PADDLE_TENNIS_CSV_KEYS if key not in info
            ]
            assert not missing
        finally:
            env.close()


class TestCertificationHarness:
    """The held-out certification instrument stays runnable and its
    pre-registered contract stays pinned. The real verdict (seeds
    4200-4299) is recorded in the ground-rules snapshot;
    tests must never draw from the reserved blocks."""

    def test_floor_constants_pin(self):
        from tools.paddle_tennis_probes import (
            CERTIFICATION_EPISODES,
            CERTIFICATION_GE1_RATE_FLOOR,
            CERTIFICATION_MEAN_CROSSINGS_FLOOR,
            CERTIFICATION_SEED_START,
        )

        assert CERTIFICATION_SEED_START == 4200
        assert CERTIFICATION_EPISODES == 100
        assert CERTIFICATION_MEAN_CROSSINGS_FLOOR == 5.9
        assert CERTIFICATION_GE1_RATE_FLOOR == 0.90

    def test_certify_smoke_on_calibration_seeds(self):
        from tools.paddle_tennis_probes import certify_frozen_env

        result = certify_frozen_env(
            episodes=2, seed_start=_SMOKE_SEEDS[0]
        )
        assert result.episodes == 2
        assert result.serve_side_a_fraction == 0.5
        assert result.unsafe_terminations == 0
        assert result.mean_crossings >= 0.0
        assert "verdict" in result.report()


class TestDeterminism:
    def test_same_seed_same_trace(self):
        def run(seed: int) -> np.ndarray:
            env = PaddleTennisEnv()
            try:
                obs, _ = env.reset(seed=seed)
                steps = [obs]
                for _ in range(120):
                    obs, _, term, trunc, _ = env.step(
                        scripted_ground_opponent(obs)
                    )
                    steps.append(obs)
                    if term or trunc:
                        break
                return np.concatenate(steps)
            finally:
                env.close()

        first = run(_SMOKE_SEEDS[2])
        second = run(_SMOKE_SEEDS[2])
        assert np.array_equal(first, second)
