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
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import courtside_dynamics  # noqa: F401  (triggers registration)
from courtside_dynamics.envs import PaddleCourtServe, PaddleTennisEnv
from courtside_dynamics.envs._paddle_court import (
    FULL_OBSERVATION_SIZE,
    OBS_BALL_SIDE_INDEX,
    OBS_BOUNCE_COUNT_INDEX,
    scripted_ground_opponent,
    scripted_hard_slam_witness,
    scripted_lead_charge_opponent,
)
from courtside_dynamics.envs.paddle_tennis import (
    _TERM_GROUPS,
    PADDLE_TENNIS_ACTION_NAMES,
    PADDLE_TENNIS_NORMALIZED_SLICE,
    PADDLE_TENNIS_OBSERVATION_NAMES,
    PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES,
)
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyEventKind,
    RallyPhase,
    TerminationReason,
)

#: Bring-up seeds (calibration block 1000-1119, already burned by the
#: P3 harness diagnostics); the reserved held-out blocks 3100-3199 and
#: 4100-4199 must never appear in tests.
_SMOKE_SEEDS = (1000, 1001, 1002)


def _zero_action() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


def _serve_options(index: int) -> dict[str, str]:
    """Reset options alternating the first serve by episode index.

    A seeded reset restarts serve alternation at side A (the
    reproducible-reset contract), so a loop that reuses one env
    across seeds forces the serve side to keep covering both serve
    slots -- the A, B, A, ... sequence those loops relied on when the
    alternation ignored the seed.
    """
    return {"serve_side": "a" if index % 2 == 0 else "b"}


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
            PADDLE_TENNIS_OBSERVATION_NAMES[PADDLE_TENNIS_NORMALIZED_SLICE.stop]
            == "rally_phase_initial_feed"
        )

    def test_registration_matches_episode_len_default(self):
        import gymnasium

        spec = gymnasium.spec("CourtsideDynamics/PaddleTennis")
        default = (
            inspect.signature(PaddleTennisEnv.__init__)
            .parameters["episode_len"]
            .default
        )
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
            assert np.array_equal(obs, env.observation_for_side(CourtSide.A))
        finally:
            env.close()


class TestMirrorIdentity:
    """P4, observed end-to-end through the registered env.

    Twin envs on the same seed: env_a serves side A; env_b forces the
    side-B serve through the reset options, from the same noise draw
    -- the exact mirrored initial state. With the same
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
            obs_b, info_b = env_b.reset(seed=seed, options={"serve_side": "b"})
            assert info_a["serve_side"] == "a"
            assert info_b["serve_side"] == "b"
            assert info_b["serve_side_is_policy"] == 0.0

            # Bit-for-bit at the mirrored reset.
            assert np.array_equal(obs_b, env_a.observation_for_side(CourtSide.B))
            assert np.array_equal(obs_a, env_b.observation_for_side(CourtSide.B))

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
            for index, seed in enumerate(_SMOKE_SEEDS):
                obs, _ = env.reset(seed=seed, options=_serve_options(index))
                total = 0.0
                total_return = 0.0
                while True:
                    obs, reward, term, trunc, info = env.step(
                        scripted_ground_opponent(obs)
                    )
                    assert reward == pytest.approx(
                        info["rew_return"] + info["rew_fault"] + info["rew_unsafe"]
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
                assert total_return == pytest.approx(float(info["valid_return_count"]))
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
                obs, reward, term, trunc, info = env.step(scripted_ground_opponent(obs))
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
            assert env.opponent_controller is scripted_ground_opponent
        finally:
            env.close()
        env = PaddleTennisEnv(volley_rule="legal")
        try:
            assert env.opponent_controller is scripted_lead_charge_opponent
        finally:
            env.close()
        with pytest.raises(ValueError, match="volley_rule"):
            PaddleTennisEnv(volley_rule="bogus")

    def test_obs_index_pins_for_ground_controller(self):
        """envs/_paddle_court.py cannot import the env, so its obs
        indices are literals -- pin them against the frozen names."""
        assert PADDLE_TENNIS_OBSERVATION_NAMES[OBS_BOUNCE_COUNT_INDEX] == "bounce_count"
        assert (
            PADDLE_TENNIS_OBSERVATION_NAMES[OBS_BALL_SIDE_INDEX] == "ball_side_is_own"
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
            for index, seed in enumerate(_SMOKE_SEEDS):
                obs, _ = env.reset(seed=seed, options=_serve_options(index))
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
        machine.advance([RallyEvent(kind=RallyEventKind.NET_CROSSING_TO_B, substep=0)])
        machine.advance(
            [
                RallyEvent(
                    kind=RallyEventKind.BALL_COURT_B,
                    substep=1,
                    position=(3.0, 0.0, 0.0),
                )
            ]
        )
        machine.advance([RallyEvent(kind=RallyEventKind.BALL_RACKET_B, substep=2)])
        machine.advance([RallyEvent(kind=RallyEventKind.NET_CROSSING_TO_A, substep=3)])
        # A volleys the incoming return before it bounces.
        transition = machine.advance(
            [RallyEvent(kind=RallyEventKind.BALL_RACKET_A, substep=4)]
        )
        assert transition.after.termination_reason is TerminationReason.VOLLEY_RETURN
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
    def test_sides_alternate_across_unseeded_resets(self):
        # The contract changed with the reproducible reset: the
        # alternation runs across UNSEEDED resets (the vectorized
        # auto-reset path), and a seeded reset restarts it at side A
        # (previously a seed was ignored, and the same loop over four
        # seeds read a, b, a, b).
        env = PaddleTennisEnv()
        try:
            sides = []
            # An ODD number of resets before the final seeded one, so
            # continued alternation would serve "b" there: only the
            # seed's restart at side A gives "a" (after an even count,
            # both rules agree and the check cannot tell them apart).
            for index in range(5):
                _, info = env.reset(seed=_SMOKE_SEEDS[0] if index == 0 else None)
                sides.append(info["serve_side"])
                assert info["serve_side_is_policy"] == (
                    1.0 if info["serve_side"] == "a" else 0.0
                )
            assert sides == ["a", "b", "a", "b", "a"]
            _, info = env.reset(seed=_SMOKE_SEEDS[1])
            assert info["serve_side"] == "a"
            # ...and the alternation resumes from the restart.
            _, info = env.reset()
            assert info["serve_side"] == "b"
        finally:
            env.close()

    def test_side_b_serve_mirrors_the_draw(self):
        env = PaddleTennisEnv()
        try:
            _, info_a = env.reset(seed=_SMOKE_SEEDS[0])
            _, info = env.reset(seed=_SMOKE_SEEDS[0], options={"serve_side": "b"})
            assert info["serve_side"] == "b"
            position = np.asarray(info["serve_ball_position"])
            velocity = np.asarray(info["serve_ball_velocity"])
            assert position[0] > 0.0
            assert velocity[0] < 0.0
            # The same seed's draw, rotated onto the other half.
            np.testing.assert_allclose(
                position, np.asarray(info_a["serve_ball_position"]) * [-1, -1, 1]
            )
        finally:
            env.close()


class TestReproducibleReset:
    """The reset contract shared with the humanoid env: a seed (plus
    options) fully determines the launch, so evaluations can replay
    identical feeds; ``options={"serve_side": ...}`` picks the slot."""

    @staticmethod
    def _rollout(env, seed, options, actions):
        obs, info = env.reset(seed=seed, options=options)
        trace = [obs]
        rewards = []
        for action in actions:
            obs, reward, term, trunc, _ = env.step(action)
            trace.append(obs)
            rewards.append(reward)
            if term or trunc:
                break
        return info, np.concatenate(trace), rewards

    def test_same_seed_twice_replays_identically(self):
        """Regression: the serve side used to come from the alternation
        ledger alone, so the same seed twice on one env served from
        different sides (different observations)."""
        env = PaddleTennisEnv()
        try:
            obs_first, info_first = env.reset(seed=_SMOKE_SEEDS[0])
            obs_second, info_second = env.reset(seed=_SMOKE_SEEDS[0])
            np.testing.assert_array_equal(obs_first, obs_second)
            assert info_first == info_second
        finally:
            env.close()

        # Whole rollouts replay too, from a dirtied env (paddles moved,
        # n-point relaunches flipping the ledger), with and without a
        # forced serve side.
        actions = np.random.default_rng(7).uniform(-1.0, 1.0, size=(300, 3))
        env = PaddleTennisEnv(points_per_episode=None)
        try:
            for options in (None, {"serve_side": "b"}):
                first = self._rollout(env, _SMOKE_SEEDS[0], options, actions)
                # Dirty the env in between: an unseeded episode moves
                # the paddles and advances the alternation ledger.
                env.reset()
                for action in actions[:200]:
                    env.step(action)
                second = self._rollout(env, _SMOKE_SEEDS[0], options, actions)
                assert first[0] == second[0]
                np.testing.assert_array_equal(first[1], second[1])
                assert first[2] == second[2]
        finally:
            env.close()

    def test_serve_side_option_is_honored_and_alternation_continues(self):
        env = PaddleTennisEnv()
        try:
            for value, label in (
                ("b", "b"),
                ("A", "a"),
                (CourtSide.B, "b"),
                (CourtSide.A, "a"),
            ):
                obs, info = env.reset(seed=_SMOKE_SEEDS[0], options={"serve_side": value})
                assert info["serve_side"] == label
                names = list(PADDLE_TENNIS_OBSERVATION_NAMES)
                assert obs[names.index("own_is_serving")] == (
                    1.0 if label == "a" else 0.0
                )
                # Alternation continues from the forced side.
                _, info = env.reset()
                assert info["serve_side"] == ("a" if label == "b" else "b")
            # An unseeded forced reset leaves the seed stream alone.
            _, info = env.reset(options={"serve_side": "b"})
            assert info["serve_side"] == "b"
        finally:
            env.close()

    def test_bad_options_raise_before_touching_the_ledger(self):
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            for options in (
                {"serve_side": "c"},
                {"serve_side": 1},
                {"ball_position": (0.0, 0.0, 1.0)},
                {"serve_side": "a", "bogus": True},
            ):
                with pytest.raises(ValueError):
                    env.reset(seed=_SMOKE_SEEDS[1], options=options)
            # The failed resets neither restarted nor advanced the
            # alternation: the next unseeded reset still serves B.
            _, info = env.reset()
            assert info["serve_side"] == "b"
            # Empty options are the plain reset.
            _, info = env.reset(seed=_SMOKE_SEEDS[0], options={})
            assert info["serve_side"] == "a"
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
                    obs, _, term, trunc, _ = env.step(scripted_ground_opponent(obs))
                    steps.append(obs)
                    if term or trunc:
                        break
                traces[style] = np.concatenate(steps)
            finally:
                env.close()
            assert np.array_equal(traces[style], traces["diagnostic"]), (
                f"style {style} altered the physics trace"
            )


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
                assert observation.shape == (len(PADDLE_TENNIS_OBSERVATION_NAMES),)
                # Side B's own-paddle x sits in its own half (negative
                # in side-local coordinates), like side A's own view.
                assert observation[9] < 0.0
        finally:
            env.close()

    def test_wrong_shaped_opponent_action_fails_loudly(self):
        """A wrong shape is a controller wiring bug: it still raises."""
        env = PaddleTennisEnv(opponent_controller=lambda observation: np.zeros(2))
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            with pytest.raises(ValueError, match="opponent_controller"):
                env.step(_zero_action())
        finally:
            env.close()

    def test_nonfinite_opponent_action_ends_through_the_unsafe_guard(self):
        """A blown-up opponent (a frozen policy fed an extreme state)
        is a runtime failure, not a bug: the episode ends through the
        nonfinite guard with the unsafe penalty, the pending escrow
        clawed back, and no physics stepped -- it used to raise and
        crash the whole vectorized rollout."""
        calls = []

        def opponent(observation: np.ndarray) -> np.ndarray:
            calls.append(1)
            return np.zeros(3) if len(calls) < 3 else np.array([0.0, np.inf, 0.0])

        env = PaddleTennisEnv(opponent_controller=opponent, contact_shaping=0.25)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            for _ in range(2):
                _obs, _reward, term, trunc, info = env.step(_zero_action())
                assert not (term or trunc)
            env._pending_shaping = 0.25
            qpos_before = env.data.qpos.copy()
            obs_before = env._last_finite_observation.copy()
            echoed, reward, term, trunc, info = env.step(_zero_action())
            assert term and not trunc
            assert reward == pytest.approx(-env.unsafe_physics_penalty - 0.25)
            assert info["rew_unsafe"] == -env.unsafe_physics_penalty
            assert info["rew_shaping_clawback"] == -0.25
            assert info["term_nonfinite"] == 1.0
            assert info["termination_reason_name"] == "nonfinite_state"
            assert np.array_equal(env.data.qpos, qpos_before)
            assert np.array_equal(echoed, obs_before)
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
            assert np.array_equal(env_clipped.data.ctrl, env_limit.data.ctrl)
        finally:
            env_clipped.close()
            env_limit.close()

    def test_nonfinite_action_terminates_without_physics(self):
        env = PaddleTennisEnv()
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
            qpos_before = env.data.qpos.copy()
            echoed, reward, term, trunc, info = env.step(np.array([np.nan, 0.0, 0.0]))
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
            missing = [key for key in _PADDLE_TENNIS_CSV_KEYS if key not in info]
            assert not missing
        finally:
            env.close()


#: The recipe's adopted env kwargs (continuous n-point play + escrows).
_RECIPE_KWARGS = {
    "points_per_episode": None,
    "contact_shaping": 0.25,
    "reach_shaping": 0.25,
}


def _info_json(info: dict) -> str:
    """One canonical encoding of an info dict (numpy scalars included)."""
    return json.dumps(
        info,
        sort_keys=True,
        default=lambda value: value.item() if hasattr(value, "item") else repr(value),
    )


def _world_spin(env: PaddleTennisEnv) -> np.ndarray:
    """R(q) @ omega_body for the ball, by an independent route."""
    import mujoco

    quaternion = env.data.qpos[env._ball_qposadr + 3 : env._ball_qposadr + 7]
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quaternion)
    body = env.data.qvel[env._ball_dofadr + 3 : env._ball_dofadr + 6]
    return rotation.reshape(3, 3) @ body


class TestObservationProfile:
    """``observation_profile``: the context-blind pilot's policy view
    (2026-10-05 review §4.2). The profile changes the policy
    observation and nothing else; ``observation_for_side`` and
    everything read from it stay on the full layout."""

    def test_physical_names_pin(self):
        """The exact 35-value layout; any change is a new era."""
        expected = (
            *(f"ball_position_{axis}" for axis in "xyz"),
            *(f"ball_linear_velocity_{axis}" for axis in "xyz"),
            *(f"ball_angular_velocity_world_{axis}" for axis in "xyz"),
            *(f"own_paddle_position_{axis}" for axis in "xyz"),
            *(f"own_paddle_velocity_{axis}" for axis in "xyz"),
            *(f"opponent_paddle_position_{axis}" for axis in "xyz"),
            *(f"opponent_paddle_velocity_{axis}" for axis in "xyz"),
            *(f"ball_minus_own_paddle_{axis}" for axis in "xyz"),
            "expected_returner_is_own",
            "ball_side_is_own",
            "bounce_count_scaled",
            "contact_latched_own_racket",
            "contact_latched_opponent_racket",
            "contact_latched_court",
            "contact_latched_net",
            "contact_release_progress_own_racket",
            "contact_release_progress_opponent_racket",
            "contact_release_progress_court",
            "contact_release_progress_net",
        )
        assert PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES == expected
        # The normalized slice is the physical block under both
        # profiles; only the spin's names (and frame) differ there.
        block = PADDLE_TENNIS_NORMALIZED_SLICE
        full_block = PADDLE_TENNIS_OBSERVATION_NAMES[block]
        physical_block = PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES[block]
        differing = [
            index
            for index, (old, new) in enumerate(
                zip(full_block, physical_block, strict=True)
            )
            if old != new
        ]
        assert differing == [6, 7, 8]
        assert PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES[24] == (
            "expected_returner_is_own"
        )
        from courtside_dynamics import envs

        assert envs.PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES is (
            PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES
        )

    def test_dropped_context_dims_are_absent(self):
        dropped = (
            *(f"rally_phase_{phase.name.lower()}" for phase in RallyPhase),
            "own_is_serving",
            "feed_crossed_net",
            "pending_return_crossed_net",
            "bounce_count",
            "rally_count",
            "episode_remaining_fraction",
            "contact_latched_own_racket_net",
            "contact_latched_opponent_racket_net",
            "contact_release_progress_own_racket_net",
            "contact_release_progress_opponent_racket_net",
            *(f"ball_angular_velocity_{axis}" for axis in "xyz"),
        )
        for name in dropped:
            assert name in PADDLE_TENNIS_OBSERVATION_NAMES
            assert name not in PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES
        assert len(set(PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES)) == 35

    def test_validation_default_and_pickle(self):
        env = PaddleTennisEnv()
        try:
            assert env.observation_profile == "full"
            assert env.observation_names == PADDLE_TENNIS_OBSERVATION_NAMES
            assert env.observation_space.shape == (48,)
        finally:
            env.close()
        with pytest.raises(ValueError, match="observation_profile"):
            PaddleTennisEnv(observation_profile="context_blind")
        env = PaddleTennisEnv(observation_profile="physical")
        try:
            assert env.observation_names == PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES
            assert env.observation_space.shape == (35,)
            # The class-level default stays the full layout.
            assert PaddleTennisEnv.observation_names == (
                PADDLE_TENNIS_OBSERVATION_NAMES
            )
            assert env._ezpickle_kwargs["observation_profile"] == "physical"
            clone = pickle.loads(pickle.dumps(env))
            try:
                assert clone.observation_profile == "physical"
                assert clone.observation_space.shape == (35,)
                obs, _ = clone.reset(seed=_SMOKE_SEEDS[0])
                assert obs.shape == (35,)
            finally:
                clone.close()
        finally:
            env.close()

    def test_physical_observation_shape_and_values(self):
        """Reset and step return the 35 values; every kept value is
        the full layout's own number (by name), the bounce count is
        min(count, 2) / 2, and the physical block is the full one
        except the spin."""
        env = PaddleTennisEnv(observation_profile="physical", **_RECIPE_KWARGS)
        full_index = {
            name: index for index, name in enumerate(PADDLE_TENNIS_OBSERVATION_NAMES)
        }
        kept = [
            (index, full_index[name])
            for index, name in enumerate(PADDLE_TENNIS_PHYSICAL_OBSERVATION_NAMES)
            if name in full_index
        ]
        assert len(kept) == 35 - 3 - 1  # all but spin and the scaled count
        bounce_values = set()
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
            for _ in range(600):
                assert obs.shape == (35,)
                assert obs.dtype == np.float64
                full = env.observation_for_side(CourtSide.A)
                for physical_at, full_at in kept:
                    assert obs[physical_at] == full[full_at]
                assert obs[26] == min(full[OBS_BOUNCE_COUNT_INDEX], 2.0) / 2.0
                bounce_values.add(float(obs[26]))
                obs, _, term, trunc, _ = env.step(
                    scripted_ground_opponent(env.observation_for_side(CourtSide.A))
                )
                if term or trunc:
                    break
        finally:
            env.close()
        # The rules count at most one bounce on a live point.
        assert bounce_values == {0.0, 0.5}

    def test_world_frame_spin_matches_rotated_body_spin(self):
        """``ball_angular_velocity_world`` = R(q) @ omega_body,
        mirrored per side exactly like every other vector (side B: x
        and y negate), and the world-frame angular velocity MuJoCo
        itself reports for the ball body after a fresh forward pass."""
        import mujoco

        env = PaddleTennisEnv(observation_profile="physical", **_RECIPE_KWARGS)
        scratch = mujoco.MjData(env.model)
        ball_body = int(env.model.jnt_bodyid[env.model.joint("ball_free").id])
        spinning_steps = 0
        tumbling_steps = 0
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[1])
            for _ in range(900):
                obs, _, term, trunc, _ = env.step(
                    scripted_ground_opponent(env.observation_for_side(CourtSide.A))
                )
                world = _world_spin(env)
                np.testing.assert_allclose(obs[6:9], world, rtol=0.0, atol=1e-9)
                side_b = env._physical_observation_for_side(CourtSide.B)
                np.testing.assert_allclose(
                    side_b[6:9],
                    np.array([-world[0], -world[1], world[2]]),
                    rtol=0.0,
                    atol=1e-9,
                )
                scratch.qpos[:] = env.data.qpos
                scratch.qvel[:] = env.data.qvel
                mujoco.mj_forward(env.model, scratch)
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    env.model,
                    scratch,
                    mujoco.mjtObj.mjOBJ_BODY,
                    ball_body,
                    velocity,
                    0,
                )
                np.testing.assert_allclose(obs[6:9], velocity[:3], atol=1e-9)
                body = env.observation_for_side(CourtSide.A)[6:9]
                if np.linalg.norm(world) > 1.0:
                    spinning_steps += 1
                    if not np.allclose(body, world, atol=1e-3):
                        tumbling_steps += 1
                if term or trunc:
                    break
        finally:
            env.close()
        # Non-vacuous: the ball spins, and its body frame has turned
        # away from the world frame while it does.
        assert spinning_steps > 50
        assert tumbling_steps > 50

    def test_full_profile_is_bit_identical_to_default(self):
        """Explicit ``"full"`` and the default produce the same obs,
        reward, terminated, truncated and info, bit for bit.

        Both envs run the same code, so this pins only the kwarg's
        default. That the full layout itself still matches the
        pre-profile env (b9585e2) is ``TestFullProfileReference``'s
        job."""
        actions = np.random.default_rng(3).uniform(-1.0, 1.0, size=(1500, 3))
        for kwargs in ({}, _RECIPE_KWARGS):
            default = PaddleTennisEnv(**kwargs)
            explicit = PaddleTennisEnv(observation_profile="full", **kwargs)
            try:
                for seed in (_SMOKE_SEEDS[0], None):
                    obs_d, info_d = default.reset(seed=seed)
                    obs_e, info_e = explicit.reset(seed=seed)
                    assert obs_d.tobytes() == obs_e.tobytes()
                    assert _info_json(info_d) == _info_json(info_e)
                    for action in actions:
                        step_d = default.step(action)
                        step_e = explicit.step(action)
                        assert step_d[0].tobytes() == step_e[0].tobytes()
                        assert step_d[1:4] == step_e[1:4]
                        assert _info_json(step_d[4]) == _info_json(step_e[4])
                        if step_d[2] or step_d[3]:
                            break
            finally:
                default.close()
                explicit.close()

    @pytest.mark.parametrize("player", ["random", "oracle"])
    @pytest.mark.parametrize(
        "kwargs", [{}, _RECIPE_KWARGS], ids=["one_point", "recipe_npoint"]
    )
    def test_profile_changes_only_the_policy_observation(self, player, kwargs):
        """Same seeds, same side-A actions: rewards, endings, reset
        and step info are identical across profiles, as are the full
        side-local views; the physical block differs only in the
        spin. The oracle plays side A from the full layout."""
        rng = np.random.default_rng(29)
        full_env = PaddleTennisEnv(**kwargs)
        physical_env = PaddleTennisEnv(observation_profile="physical", **kwargs)
        steps = 0
        try:
            for seed in (_SMOKE_SEEDS[0], _SMOKE_SEEDS[2], None):
                obs_f, info_f = full_env.reset(seed=seed)
                obs_p, info_p = physical_env.reset(seed=seed)
                assert _info_json(info_f) == _info_json(info_p)
                while True:
                    view_f = full_env.observation_for_side(CourtSide.A)
                    view_p = physical_env.observation_for_side(CourtSide.A)
                    assert view_f.tobytes() == view_p.tobytes()
                    assert view_f.tobytes() == obs_f.tobytes()
                    assert np.array_equal(obs_p[:6], obs_f[:6])
                    assert np.array_equal(obs_p[9:24], obs_f[9:24])
                    action = (
                        rng.uniform(-1.0, 1.0, size=3)
                        if player == "random"
                        else scripted_ground_opponent(view_f)
                    )
                    obs_f, rew_f, term_f, trunc_f, info_f = full_env.step(action)
                    obs_p, rew_p, term_p, trunc_p, info_p = physical_env.step(action)
                    steps += 1
                    assert obs_f.shape == (48,)
                    assert obs_p.shape == (35,)
                    assert (rew_f, term_f, trunc_f) == (rew_p, term_p, trunc_p)
                    assert _info_json(info_f) == _info_json(info_p)
                    if term_f or trunc_f:
                        break
        finally:
            full_env.close()
            physical_env.close()
        assert steps > 100

    def test_observation_for_side_stays_full_under_physical(self):
        env = PaddleTennisEnv(observation_profile="physical")
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
            assert obs.shape == (35,)
            for side in (CourtSide.A, CourtSide.B):
                assert env.observation_for_side(side).shape == (
                    FULL_OBSERVATION_SIZE,
                )
            env.step(_zero_action())
            assert env.observation_for_side(CourtSide.B).shape == (
                FULL_OBSERVATION_SIZE,
            )
        finally:
            env.close()
        assert FULL_OBSERVATION_SIZE == len(PADDLE_TENNIS_OBSERVATION_NAMES)

    def test_opponent_reads_the_same_full_view_under_either_profile(self):
        """The opponent controller's inputs and actions are identical
        across profiles (it reads observation_for_side(B), never the
        policy observation)."""

        def recording(log: list):
            def controller(observation: np.ndarray) -> np.ndarray:
                action = scripted_ground_opponent(observation)
                log.append((observation.tobytes(), action.tobytes()))
                return action

            return controller

        logs: dict[str, list] = {"full": [], "physical": []}
        actions = np.random.default_rng(5).uniform(-1.0, 1.0, size=(1500, 3))
        for profile, log in logs.items():
            env = PaddleTennisEnv(
                observation_profile=profile,
                opponent_controller=recording(log),
                **_RECIPE_KWARGS,
            )
            try:
                env.reset(seed=_SMOKE_SEEDS[1])
                for action in actions:
                    *_, term, trunc, _ = env.step(action)
                    if term or trunc:
                        break
            finally:
                env.close()
        assert len(logs["full"]) > 100
        assert all(len(obs) == 48 * 8 for obs, _ in logs["full"])
        assert logs["full"] == logs["physical"]

    def test_ground_controllers_refuse_the_physical_observation(self):
        """A scripted controller fed the 35-value policy observation
        would read contact-tail values at its rally indices; it must
        refuse loudly instead."""
        env = PaddleTennisEnv(observation_profile="physical")
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
            for controller in (scripted_ground_opponent, scripted_hard_slam_witness):
                with pytest.raises(ValueError, match="observation_for_side"):
                    controller(obs)
                full = env.observation_for_side(CourtSide.A)
                assert controller(full).shape == (3,)
        finally:
            env.close()


_FULL_PROFILE_REFERENCE = (
    Path(__file__).parent / "data" / "paddle_tennis_full_profile_b9585e2.json"
)


class TestFullProfileReference:
    """The default ``"full"`` profile against b9585e2 itself.

    ``tests/data/paddle_tennis_full_profile_b9585e2.json`` holds
    chained per-step digests of obs, reward, terminated, truncated,
    info and both full side-local views, recorded from b9585e2's
    source by ``tools/paddle_tennis_full_profile_reference.py``
    (seeded and unseeded resets; random and oracle side-A play; the
    one-point default, the recipe's n-point escrow task, volley-legal
    rules, and a truncating hold-escrow variant). The default env and
    an explicit ``"full"`` env must replay every stream; a
    ``"physical"`` env must replay every stream except its own policy
    observation. Physics bits are comparable only on the MuJoCo build
    the file was recorded with, so the replay skips elsewhere (the
    tool's docstring says how to re-record)."""

    @staticmethod
    def _load() -> dict:
        return json.loads(_FULL_PROFILE_REFERENCE.read_text())

    def test_reference_covers_the_recorded_cases(self):
        """The fixture holds exactly the tool's cases, recorded from
        b9585e2, and its streams are not vacuous: they include
        unseeded resets, terminations and truncations."""
        from tools.paddle_tennis_full_profile_reference import (
            CASES,
            CHECKPOINT_EVERY,
            FIELDS,
            SCHEMA,
        )

        reference = self._load()
        assert reference["schema"] == SCHEMA
        assert reference["recorded_from"] == "b9585e2"
        assert reference["checkpoint_every"] == CHECKPOINT_EVERY
        assert set(reference["cases"]) == set(CASES)
        for name, (env_kwargs, player, seed, steps) in CASES.items():
            case = reference["cases"][name]
            assert (case["env_kwargs"], case["player"], case["seed"]) == (
                env_kwargs,
                player,
                seed,
            )
            assert case["steps"] == steps
            assert set(case["checkpoints"]) == set(FIELDS)
            assert case["resets"] >= 2  # the seeded one plus unseeded
        assert sum(c["terminations"] for c in reference["cases"].values()) > 0
        assert sum(c["truncations"] for c in reference["cases"].values()) > 0

    @pytest.mark.parametrize("profile", [None, "full", "physical"])
    def test_replays_the_b9585e2_streams(self, profile):
        import platform

        import mujoco

        from tools.paddle_tennis_full_profile_reference import (
            FIELDS,
            record_case,
        )

        reference = self._load()
        recorded_on = (reference["mujoco"], reference["machine"])
        if recorded_on != (mujoco.__version__, platform.machine()):
            pytest.skip(
                f"reference recorded under MuJoCo {reference['mujoco']} on "
                f"{reference['machine']}; this is MuJoCo "
                f"{mujoco.__version__} on {platform.machine()} (re-record "
                "from b9585e2: tools/paddle_tennis_full_profile_reference.py)"
            )
        fields = tuple(
            field for field in FIELDS if profile != "physical" or field != "obs"
        )
        every = reference["checkpoint_every"]
        for name, case in reference["cases"].items():
            kwargs = dict(case["env_kwargs"])
            if profile is not None:
                kwargs["observation_profile"] = profile
            replay = record_case(
                lambda kw: PaddleTennisEnv(**kw),
                kwargs,
                case["player"],
                case["seed"],
                case["steps"],
                fields=fields,
            )
            assert (
                replay["resets"],
                replay["terminations"],
                replay["truncations"],
            ) == (case["resets"], case["terminations"], case["truncations"]), name
            for field in fields:
                expected = case["checkpoints"][field]
                actual = replay["checkpoints"][field]
                if actual != expected:
                    window = next(
                        index
                        for index, (want, got) in enumerate(
                            zip(expected, actual, strict=True)
                        )
                        if want != got
                    )
                    pytest.fail(
                        f"{name} (profile={profile!r}): {field!r} diverges "
                        f"from b9585e2 within steps "
                        f"{window * every + 1}-{(window + 1) * every} "
                        f"(recorded with numpy {reference['numpy']}, this "
                        f"is numpy {np.__version__})"
                    )


class TestEpisodePolicyCounters:
    """The episode-cumulative side-A instruments: legal hits, confirmed
    returns, k>=2 conversions, contact depth and deep returns. The
    rules' own counters restart with every point, so under n-point
    play the terminal step used to read only the last (truncation-cut)
    point."""

    @staticmethod
    def _recount(env, seed, *, threshold=None, options=None):
        """Roll one oracle-as-A episode, recounting every counter
        independently from ``env._last_transition`` at each step."""
        threshold = env.deep_contact_depth if threshold is None else threshold
        obs, _ = env.reset(seed=seed, options=options)
        legal = returns = rally = deep = 0
        depth_sum = 0.0
        pending: float | None = None
        max_point_hits = 0
        info: dict = {}
        while True:
            obs, _r, term, trunc, info = env.step(scripted_ground_opponent(obs))
            transition = env._last_transition
            # The diagnosis instrument's ledger rule: a fault touch
            # ends the point, so the legal side-A racket events are
            # the first ones of the step's processed stream.
            legal_now = transition.valid_racket_hits.count(CourtSide.A)
            confirms_now = transition.confirmed_returns.count(CourtSide.A)
            depths = [
                abs(event.position[0])
                for event in transition.processed_events
                if event.kind is RallyEventKind.BALL_RACKET_A
            ][:legal_now]
            # A shot needs >100 ms to cross and land: a side-A hit and
            # a side-A confirm never share one 10 ms step.
            assert not (legal_now and confirms_now)
            k = transition.before.valid_return_count_a
            for _ in range(confirms_now):
                returns += 1
                k += 1
                rally += k >= 2
                if pending is not None and pending >= threshold:
                    deep += 1
                pending = None
            for depth in depths:
                legal += 1
                depth_sum += depth
                pending = depth
            if transition.terminated_now or term or trunc:
                pending = None  # the point ended: its shot is void
            max_point_hits = max(max_point_hits, transition.after.legal_hit_count_a)
            assert info["episode_legal_hit_count_a"] == legal
            assert info["episode_valid_return_count_a"] == returns
            assert info["episode_rally_returns_a"] == rally
            assert info["episode_deep_returns_a"] == deep
            assert info["contact_depth_a"] == (depths[-1] if depths else 0.0)
            assert info["episode_contact_depth_sum_a"] == depth_sum
            assert info["episode_mean_contact_depth_a"] == (
                depth_sum / legal if legal else 0.0
            )
            if term or trunc:
                return info, max_point_hits

    def test_counters_match_an_independent_recount(self):
        env = PaddleTennisEnv(**_RECIPE_KWARGS)
        try:
            # Burned bring-up episodes that cross absorbed boundaries
            # (measured: 1 and 2 completed points before the cap).
            for seed, side in ((_SMOKE_SEEDS[2], "b"), (1003, "a")):
                info, max_point_hits = self._recount(
                    env, seed, options={"serve_side": side}
                )
                assert info["points_played"] >= 1
                assert info["episode_legal_hit_count_a"] >= max_point_hits
                # The per-point rules counter at the terminal step
                # misses every earlier point's hits.
                assert info["episode_legal_hit_count_a"] > info["legal_hit_count_a"]
                assert info["episode_rally_returns_a"] > 0  # k>=2 exercised
                assert 0.0 < info["episode_mean_contact_depth_a"] < 6.5
        finally:
            env.close()

    def test_fault_touches_are_not_legal_hits(self):
        """A side-A racket contact that faults is not a legal hit. The
        recount above never sees one (the oracle never faults), so this
        drives the lead-charge volleyer as side A under ground rules:
        every pre-bounce touch is a volley fault. None may count in
        episode_legal_hit_count_a (the recipe's degenerate-guard key: a
        volley-faulting dead policy must still trip it), set
        contact_depth_a, or enter the mean contact depth."""
        env = PaddleTennisEnv(**_RECIPE_KWARGS)
        try:
            obs, _ = env.reset(seed=1003)
            touches = 0
            info: dict = {}
            while True:
                obs, _r, term, trunc, info = env.step(
                    scripted_lead_charge_opponent(obs)
                )
                transition = env._last_transition
                touches += sum(
                    event.kind is RallyEventKind.BALL_RACKET_A
                    for event in transition.processed_events
                )
                if not transition.valid_racket_hits.count(CourtSide.A):
                    assert info["contact_depth_a"] == 0.0
                if term or trunc:
                    break
            # Measured: 12 side-A racket contacts, none of them legal.
            assert touches > 0
            assert info["episode_legal_hit_count_a"] == 0.0
            assert info["episode_valid_return_count_a"] == 0.0
            assert info["episode_contact_depth_sum_a"] == 0.0
            assert info["episode_mean_contact_depth_a"] == 0.0
        finally:
            env.close()

    def test_contact_depth_sum_gives_the_hit_weighted_batch_mean(self):
        """episode_mean_contact_depth_a reads a 0.0 sentinel on a
        hitless episode, so its across-episode mean (the eval
        callback's terminal _ep_mean) is diluted by every hitless
        episode in the batch. episode_contact_depth_sum_a's _ep_mean
        over episode_legal_hit_count_a's _ep_mean is the hit-weighted
        mean depth the recipe points readers to."""
        env = PaddleTennisEnv(**_RECIPE_KWARGS)
        try:
            hitter, _ = self._recount(env, _SMOKE_SEEDS[0])
            # The lead-charge volleyer's every touch faults (measured
            # on this seed in test_fault_touches_are_not_legal_hits).
            obs, _ = env.reset(seed=1003)
            while True:
                obs, _r, term, trunc, hitless = env.step(
                    scripted_lead_charge_opponent(obs)
                )
                if term or trunc:
                    break
        finally:
            env.close()
        assert hitless["episode_legal_hit_count_a"] == 0.0
        assert hitless["episode_contact_depth_sum_a"] == 0.0
        assert hitless["episode_mean_contact_depth_a"] == 0.0  # sentinel
        hits = hitter["episode_legal_hit_count_a"]
        depth = hitter["episode_mean_contact_depth_a"]
        assert hits > 0
        assert hitter["episode_contact_depth_sum_a"] == pytest.approx(depth * hits)

        def ep_mean(key: str) -> float:
            return (hitter[key] + hitless[key]) / 2

        weighted = ep_mean("episode_contact_depth_sum_a") / ep_mean(
            "episode_legal_hit_count_a"
        )
        assert weighted == pytest.approx(depth)
        # The sentinel halves the per-episode mean's batch average.
        assert ep_mean("episode_mean_contact_depth_a") == pytest.approx(depth / 2)

    def test_deep_returns_respond_to_the_depth_threshold(self):
        totals = {}
        for depth in (1e-6, 5.5, 100.0):
            env = PaddleTennisEnv(deep_contact_depth=depth, **_RECIPE_KWARGS)
            try:
                info, _ = self._recount(env, _SMOKE_SEEDS[0])
                totals[depth] = (
                    info["episode_deep_returns_a"],
                    info["episode_valid_return_count_a"],
                )
            finally:
                env.close()
        deep_all, returns = totals[1e-6]
        assert returns > 0
        assert deep_all == returns  # every hit sits beyond the net plane
        assert totals[100.0][0] == 0.0
        # The default 5.5 m band discriminates on this episode
        # (measured: 2 of 6 confirmed returns struck that deep).
        assert 0.0 < totals[5.5][0] < returns

    def test_counters_restart_at_reset_and_survive_every_ending_path(self):
        env = PaddleTennisEnv(**_RECIPE_KWARGS)
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
            info: dict = {}
            # Bounded by the episode: a counter that never moves must
            # fail here, not spin past truncation into the timeout.
            for _ in range(env.episode_len):
                obs, _r, term, trunc, info = env.step(scripted_ground_opponent(obs))
                assert not (term or trunc), (
                    "episode ended before side A's first confirmed return"
                )
                if info["episode_valid_return_count_a"]:
                    break
            # The precondition: nonzero counters reach the guard step,
            # so "carried" below cannot pass vacuously on zeros.
            assert info["episode_valid_return_count_a"] >= 1.0
            assert info["episode_legal_hit_count_a"] >= 1.0
            assert info["episode_contact_depth_sum_a"] > 0.0
            carried = {
                key: info[key]
                for key in (
                    "episode_legal_hit_count_a",
                    "episode_valid_return_count_a",
                    "episode_rally_returns_a",
                    "episode_contact_depth_sum_a",
                    "episode_mean_contact_depth_a",
                    "episode_deep_returns_a",
                )
            }
            # The early nonfinite guard steps no physics: the counters
            # carry, and no hit happened on that step.
            *_, term, _trunc, info = env.step(np.array([np.nan, 0.0, 0.0]))
            assert term
            assert info["contact_depth_a"] == 0.0
            for key, value in carried.items():
                assert info[key] == value, key
            _obs, _ = env.reset()
            *_, info = env.step(_zero_action())
            for key in (*carried, "contact_depth_a"):
                assert info[key] == 0.0, key
        finally:
            env.close()

    def test_deep_contact_depth_validated_and_pickled(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="deep_contact_depth"):
                PaddleTennisEnv(deep_contact_depth=bad)
        env = PaddleTennisEnv(deep_contact_depth=4.0)
        try:
            assert env.deep_contact_depth == 4.0
            clone = pickle.loads(pickle.dumps(env))
            try:
                assert clone.deep_contact_depth == 4.0
            finally:
                clone.close()
        finally:
            env.close()


class TestTerminationScopes:
    """One scope per key family (the _build_info comment): every
    ``term_*`` flag -- grouped and per-reason -- describes the
    EPISODE's ending; ``termination_reason(_name)``, ``rally_terminal``
    and ``rally_phase`` describe the step's own POINT. At an absorbed
    boundary the per-reason flags used to keep the point's fault
    (``term_volley_return`` True beside ``term_volley`` 0.0), and the
    forced-nonfinite ending read ``termination_reason_name="none"``."""

    @staticmethod
    def _assert_scopes(info, *, terminated, truncated):
        fired = {key for key, value in info.items() if key.startswith("term_") and value}
        if not (terminated or truncated):
            assert fired == set(), fired
            return
        if truncated:
            assert fired == {"term_timeout"}, fired
            return
        reason = TerminationReason(int(info["termination_reason"]))
        assert info["termination_reason_name"] == reason.name.lower()
        group = next(name for name, reasons in _TERM_GROUPS if reason in reasons)
        assert fired == {f"term_{reason.name.lower()}", group}, fired

    def test_absorbed_boundaries_and_episode_endings_agree(self):
        boundary_reasons = set()
        for policy, kwargs in (
            # The volley-capable controller faults VOLLEY_RETURN on
            # every receiving point: a per-reason key the grouped flag
            # does not shadow.
            (scripted_lead_charge_opponent, {"points_per_episode": None}),
            (scripted_ground_opponent, {"points_per_episode": 3}),
        ):
            env = PaddleTennisEnv(**kwargs)
            try:
                obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
                ends = {
                    "point_end_" + name.removeprefix("term_"): 0.0
                    for name, _ in _TERM_GROUPS
                    if name != "term_nonfinite"
                }
                while True:
                    obs, _r, term, trunc, info = env.step(policy(obs))
                    self._assert_scopes(info, terminated=term, truncated=trunc)
                    if env._last_transition.terminated_now:
                        # The point scope names the point's fault, and
                        # the same step books it in its group counter.
                        reason = TerminationReason(int(info["termination_reason"]))
                        assert reason is not TerminationReason.NONE
                        assert info["rally_terminal"] is True
                        assert info["rally_phase"] == int(RallyPhase.TERMINAL)
                        group = next(
                            name for name, reasons in _TERM_GROUPS if reason in reasons
                        )
                        key = "point_end_" + group.removeprefix("term_")
                        assert info[key] == ends[key] + 1.0
                        if not term:
                            boundary_reasons.add(reason)
                    for key in ends:
                        ends[key] = info[key]
                    if term or trunc:
                        break
            finally:
                env.close()
        assert TerminationReason.VOLLEY_RETURN in boundary_reasons

    def test_forced_nonfinite_ending_names_itself_everywhere(self):
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            real_get_obs = env._get_obs

            def poisoned():
                obs = real_get_obs().copy()
                obs[0] = np.nan
                return obs

            env._get_obs = poisoned
            *_, term, trunc, info = env.step(_zero_action())
            assert term and not trunc
            # The rules machine never saw the blow-up (its snapshot
            # reads "none"); the published info must not.
            assert env._last_transition.after.termination_reason is (
                TerminationReason.NONE
            )
            assert info["termination_reason_name"] == "nonfinite_state"
            assert info["termination_reason"] == int(TerminationReason.NONFINITE_STATE)
            assert info["rally_terminal"] is True
            assert info["rally_phase"] == int(RallyPhase.TERMINAL)
            self._assert_scopes(info, terminated=term, truncated=trunc)
        finally:
            env.close()
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            *_, term, trunc, info = env.step(np.array([np.nan, 0.0, 0.0]))
            assert info["termination_reason_name"] == "nonfinite_state"
            self._assert_scopes(info, terminated=term, truncated=trunc)
        finally:
            env.close()

    def test_recipe_consumed_keys_survive(self):
        """The fix renames nothing the recipe logs or evaluates."""
        from courtside_dynamics.recipes import (
            _PADDLE_TENNIS_CSV_KEYS,
            _PADDLE_TENNIS_TERMINAL_EVAL_KEYS,
        )

        env = PaddleTennisEnv(points_per_episode=None)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            *_, info = env.step(_zero_action())
            for key in (*_PADDLE_TENNIS_CSV_KEYS, *_PADDLE_TENNIS_TERMINAL_EVAL_KEYS):
                assert key in info, key
        finally:
            env.close()


class TestRecipeExplorationPackage:
    """The ground-era exploration remedy is part of the training
    definition (docs/paddle_tennis_exploration_20260808.md): auto
    entropy from a safe init with the raised target as the floor,
    plus gSDE. Pinned so a revert to stock SAC (whose auto-tuning
    collapsed ent_coef to 5e-5 in the pilot) is a deliberate new
    comparability decision, not drift."""

    def test_model_kwargs_pin(self):
        from courtside_dynamics.recipes import RECIPES

        assert RECIPES["PaddleTennis"].extra_cfg["model_kwargs"] == {
            "use_sde": True,
            "ent_coef": "auto_0.02",
            "target_entropy": -1.5,
            # Without a multi-step train_freq, SAC resets the gSDE
            # noise matrix every collect (= every step): iid noise.
            "train_freq": (64, "step"),
        }


_SHAPING_COMPONENTS = (
    "rew_return",
    "rew_fault",
    "rew_unsafe",
    "rew_shaping",
    "rew_shaping_clawback",
    "rew_reach",
    "rew_reach_clawback",
    "rew_hold",
    "rew_hold_clawback",
)


class TestContactShaping:
    """The escrow contract of design_paddle_tennis_contact_shaping.md:
    pay at a side-A legal hit, keep on confirm, claw back on EVERY
    ending path — with the default-off stream bit-identical to the
    frozen task."""

    @staticmethod
    def _drive(env, policy, seed, mirror_env=None, options=None):
        """Step ``env`` (and optionally ``mirror_env`` in lockstep)
        with ``policy``; return per-step info sums and totals."""
        obs, _ = env.reset(seed=seed, options=options)
        if mirror_env is not None:
            mirror_obs, _ = mirror_env.reset(seed=seed, options=options)
            np.testing.assert_array_equal(obs, mirror_obs)
        totals = {"reward": 0.0, "mirror_reward": 0.0}
        sums = dict.fromkeys(_SHAPING_COMPONENTS, 0.0)
        confirms = 0
        while True:
            action = policy(obs)
            obs, reward, term, trunc, info = env.step(action)
            for key in _SHAPING_COMPONENTS:
                sums[key] += info[key]
            assert reward == pytest.approx(
                sum(info[key] for key in _SHAPING_COMPONENTS), abs=1e-12
            )
            totals["reward"] += reward
            confirms += int(bool(info["event_valid_return_a"]))
            if mirror_env is not None:
                mirror_obs, mirror_reward, mterm, mtrunc, _ = mirror_env.step(action)
                np.testing.assert_array_equal(obs, mirror_obs)
                assert (term, trunc) == (mterm, mtrunc)
                totals["mirror_reward"] += mirror_reward
            if term or trunc:
                return totals, sums, confirms

    def test_escrow_identity_and_default_bit_identity(self):
        """Shaped-vs-unshaped arms of the same seed are bit-identical
        trajectories, and the escrow's whole undiscounted effect is
        exactly 0.25 x side-A confirms."""
        shaped = PaddleTennisEnv(contact_shaping=0.25)
        unshaped = PaddleTennisEnv()
        try:
            for index, seed in enumerate(_SMOKE_SEEDS):
                totals, sums, confirms = self._drive(
                    shaped,
                    scripted_ground_opponent,
                    seed,
                    unshaped,
                    options=_serve_options(index),
                )
                expected = 0.25 * confirms
                assert sums["rew_shaping"] + sums[
                    "rew_shaping_clawback"
                ] == pytest.approx(expected, abs=1e-12)
                assert totals["reward"] - totals["mirror_reward"] == pytest.approx(
                    expected, abs=1e-12
                )
        finally:
            shaped.close()
            unshaped.close()

    def test_default_off_components_are_exact_zero(self):
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            for _ in range(50):
                *_, term, trunc, info = env.step(_zero_action())
                assert info["rew_shaping"] == 0.0
                assert info["rew_shaping_clawback"] == 0.0
                if term or trunc:
                    break
        finally:
            env.close()

    def test_truncation_claws_back_pending_escrow(self):
        env = PaddleTennisEnv(contact_shaping=0.25, episode_len=3)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            env._pending_shaping = 0.25
            term = trunc = False
            info: dict = {}
            reward = 0.0
            while not (term or trunc):
                _obs, reward, term, trunc, info = env.step(_zero_action())
            # Three zero-action steps cannot end the point (the serve
            # is still in flight), so the cap must truncate.
            assert trunc and not term
            assert info["rew_shaping_clawback"] == -0.25
            assert info["term_timeout"] == 1.0
            assert reward == pytest.approx(-0.25)
            assert env._pending_shaping == 0.0
        finally:
            env.close()

    def test_confirmed_advance_survives_cap_truncation(self):
        """S2 case (b): a confirmed hit's advance is kept — truncation
        with an empty escrow claws back nothing. Asserted explicitly
        (not left to a lucky smoke-seed trajectory)."""
        env = PaddleTennisEnv(contact_shaping=0.25)
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[0])
            confirmed = False
            for _ in range(env.episode_len):
                obs, _reward, term, trunc, info = env.step(
                    scripted_ground_opponent(obs)
                )
                assert not (term or trunc), "point ended before a confirm"
                if bool(info["event_valid_return_a"]) and (env._pending_shaping == 0.0):
                    confirmed = True
                    break
            assert confirmed, "ground pair never confirmed a side-A return"
            env.step_number = env.episode_len
            _obs, reward, term, trunc, info = env.step(_zero_action())
            assert trunc and not term
            assert info["term_timeout"] == 1.0
            assert info["rew_shaping_clawback"] == 0.0
            assert reward == pytest.approx(
                sum(info[key] for key in _SHAPING_COMPONENTS)
            )
        finally:
            env.close()

    def test_nan_action_guard_claws_back_pending_escrow(self):
        """The early-return guard is an ending like any other (the
        design's §2 contract): pending escrow claws back next to the
        unsafe penalty."""
        env = PaddleTennisEnv(contact_shaping=0.25)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            env._pending_shaping = 0.25
            obs, reward, term, trunc, info = env.step(np.array([np.nan, 0.0, 0.0]))
            assert term and not trunc
            assert info["rew_unsafe"] == -2.0
            assert info["rew_shaping_clawback"] == -0.25
            assert reward == pytest.approx(-2.25)
            assert env._pending_shaping == 0.0
            assert reward == pytest.approx(
                sum(info[key] for key in _SHAPING_COMPONENTS)
            )
        finally:
            env.close()

    def test_forced_nonfinite_obs_claws_back_pending_escrow(self):
        """A nonfinite observation after a finite physics step (the
        forced_nonfinite branch) must also claw back."""
        env = PaddleTennisEnv(contact_shaping=0.25)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            env._pending_shaping = 0.25
            real_get_obs = env._get_obs

            def poisoned():
                obs = real_get_obs().copy()
                obs[0] = np.nan
                return obs

            env._get_obs = poisoned
            obs, reward, term, trunc, info = env.step(_zero_action())
            assert term and not trunc
            assert info["rew_unsafe"] == -2.0
            assert info["rew_shaping_clawback"] == -0.25
            assert reward == pytest.approx(
                sum(info[key] for key in _SHAPING_COMPONENTS)
            )
            assert bool(np.isfinite(obs).all())
        finally:
            env.close()

    def test_s1_probe_harness_smoke(self):
        """The S1 tool's identities hold on smoke seeds (the real
        verdict runs on the reserved 5300+ block, never in tests)."""
        from tools.paddle_tennis_shaping_probe import (
            WITNESSES,
            evaluate_criteria,
            run_witness,
        )

        results = {}
        for name, witness in WITNESSES:
            shaped = run_witness(
                witness,
                shaping=0.25,
                episodes=2,
                seed_start=_SMOKE_SEEDS[0],
            )
            unshaped = run_witness(
                witness,
                shaping=0.0,
                episodes=2,
                seed_start=_SMOKE_SEEDS[0],
            )
            results[name] = (shaped, unshaped)
        checks = evaluate_criteria(results)
        # The exact identities and the exploit checks must hold even on
        # smoke seeds; the hard-slam preconditions are block-specific
        # measurements and may legitimately differ off-block.
        for check_name, passed, detail in checks:
            if check_name.startswith("precondition"):
                continue
            assert passed, f"{check_name}: {detail}"

    def test_recipe_adopts_contact_shaping(self):
        """The adopted era (reach design doc §4a): the recipe ships
        the contact escrow at its audited 0.25."""
        from courtside_dynamics.recipes import RECIPES

        assert RECIPES["PaddleTennis"].env_kwargs["contact_shaping"] == 0.25


class TestReachShaping:
    """The escrow contract of design_paddle_tennis_reach_shaping.md:
    pay at side A's live first bounce by proximity, keep on the side-A
    legal hit that takes the opportunity, claw back on EVERY ending
    path (point boundaries included) — with the default-off stream
    bit-identical to the frozen task."""

    @staticmethod
    def _drive(env, policy, seed, mirror_env=None, options=None):
        """Step ``env`` (optionally with a lockstep mirror); return
        totals, per-component sums, and the tracker-computed kept
        escrow (commit-before-pay ordering, the implementation's)."""
        obs, _ = env.reset(seed=seed, options=options)
        if mirror_env is not None:
            mirror_obs, _ = mirror_env.reset(seed=seed, options=options)
            np.testing.assert_array_equal(obs, mirror_obs)
        totals = {"reward": 0.0, "mirror_reward": 0.0}
        sums = dict.fromkeys(_SHAPING_COMPONENTS, 0.0)
        kept = pending = 0.0
        hits = 0
        while True:
            action = policy(obs)
            obs, reward, term, trunc, info = env.step(action)
            for key in _SHAPING_COMPONENTS:
                sums[key] += info[key]
            assert reward == pytest.approx(
                sum(info[key] for key in _SHAPING_COMPONENTS), abs=1e-12
            )
            totals["reward"] += reward
            # The env's escrow rule: a hit keeps any prior pending
            # advance, and a same-step payment is the bounce that hit
            # just took — kept immediately, never escrowed.
            if info["event_valid_racket_hit_a"]:
                hits += 1
                kept += pending + info["rew_reach"]
                pending = 0.0
            else:
                pending += info["rew_reach"]
            if mirror_env is not None:
                mirror_obs, mirror_reward, mterm, mtrunc, _ = mirror_env.step(action)
                np.testing.assert_array_equal(obs, mirror_obs)
                assert (term, trunc) == (mterm, mtrunc)
                totals["mirror_reward"] += mirror_reward
            if term or trunc:
                return totals, sums, kept, hits

    def test_escrow_identity_and_default_bit_identity(self):
        """Shaped-vs-unshaped arms of the same seed are bit-identical
        trajectories, and the escrow's whole undiscounted effect is
        exactly the kept (hit-taken) proximity pay."""
        shaped = PaddleTennisEnv(reach_shaping=0.25)
        unshaped = PaddleTennisEnv()
        try:
            for index, seed in enumerate(_SMOKE_SEEDS):
                totals, sums, kept, _hits = self._drive(
                    shaped,
                    scripted_ground_opponent,
                    seed,
                    unshaped,
                    options=_serve_options(index),
                )
                assert sums["rew_reach"] + sums["rew_reach_clawback"] == pytest.approx(
                    kept, abs=1e-12
                )
                assert totals["reward"] - totals["mirror_reward"] == pytest.approx(
                    kept, abs=1e-12
                )
        finally:
            shaped.close()
            unshaped.close()

    def test_default_off_components_are_exact_zero(self):
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            for _ in range(50):
                *_, term, trunc, info = env.step(_zero_action())
                assert info["rew_reach"] == 0.0
                assert info["rew_reach_clawback"] == 0.0
                if term or trunc:
                    break
        finally:
            env.close()

    def test_payment_matches_the_event_position_formula(self):
        """Each nonzero pay equals shaping x max(0, 1 - d/radius) with
        d recomputed from the qualifying BALL_COURT_A event's recorded
        position and the step-end paddle head — exactly."""
        from courtside_dynamics.envs.tennis_rules import RallyEventKind

        env = PaddleTennisEnv(reach_shaping=0.25, reach_shaping_radius=3.0)
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[1])
            payments = 0
            while True:
                obs, _reward, term, trunc, info = env.step(
                    scripted_ground_opponent(obs)
                )
                if info["rew_reach"] > 0.0:
                    payments += 1
                    transition = env._last_transition
                    position = next(
                        event.position
                        for event in transition.processed_events
                        if event.kind is RallyEventKind.BALL_COURT_A
                    )
                    paddle = env._paddle_position(CourtSide.A)
                    distance = float(
                        np.hypot(position[0] - paddle[0], position[1] - paddle[1])
                    )
                    expected = 0.25 * max(0.0, 1.0 - distance / 3.0)
                    assert info["rew_reach"] == pytest.approx(expected, abs=1e-12)
                if term or trunc:
                    break
            assert payments > 0, "the oracle never received a paid bounce"
        finally:
            env.close()

    def test_truncation_claws_back_pending_reach(self):
        env = PaddleTennisEnv(reach_shaping=0.25, episode_len=3)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            env._pending_reach = 0.125
            term = trunc = False
            info: dict = {}
            reward = 0.0
            while not (term or trunc):
                _obs, reward, term, trunc, info = env.step(_zero_action())
            assert trunc and not term
            assert info["rew_reach_clawback"] == -0.125
            assert reward == pytest.approx(-0.125)
            assert env._pending_reach == 0.0
        finally:
            env.close()

    def test_nan_action_guard_claws_back_pending_reach(self):
        """The early-return guard is an ending like any other: pending
        reach claws back next to the unsafe penalty."""
        env = PaddleTennisEnv(reach_shaping=0.25)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            env._pending_reach = 0.125
            _obs, reward, term, trunc, info = env.step(
                np.array([np.nan, 0.0, 0.0])
            )
            assert term and not trunc
            assert info["rew_reach_clawback"] == -0.125
            assert reward == pytest.approx(-env.unsafe_physics_penalty - 0.125)
            assert env._pending_reach == 0.0
        finally:
            env.close()

    def test_point_boundary_claws_back_pending_reach(self):
        """n-point statue: proximity pays at receiving bounces and is
        clawed back in full at every point boundary — statue economics
        stay exactly the frozen ones."""
        shaped = PaddleTennisEnv(points_per_episode=None, reach_shaping=0.25)
        plain = PaddleTennisEnv(points_per_episode=None)
        try:
            paid_any = False
            for index, seed in enumerate(_SMOKE_SEEDS):
                totals, sums, kept, hits = self._drive(
                    shaped,
                    lambda _obs: _zero_action(),
                    seed,
                    plain,
                    options=_serve_options(index),
                )
                assert hits == 0
                assert kept == 0.0
                paid_any = paid_any or sums["rew_reach"] > 0.0
                assert sums["rew_reach"] + sums["rew_reach_clawback"] == pytest.approx(
                    0.0, abs=1e-12
                )
                assert totals["reward"] == pytest.approx(
                    totals["mirror_reward"], abs=1e-12
                )
            assert paid_any, "the statue never received a paid bounce"
        finally:
            shaped.close()
            plain.close()

    def test_same_step_take_is_kept_not_escrowed(self):
        """The §2 ordering amendment: a payment coexisting with the
        hit that takes it is kept immediately — never pending, so a
        later boundary cannot claw back a tight interception's pay —
        while a hit alone keeps exactly the prior pending advance."""
        env = PaddleTennisEnv(reach_shaping=0.25)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            # Hit and qualifying bounce in the same step: prior
            # pending is kept, the new payment never enters escrow.
            env._pending_reach = 0.125
            assert env._reach_escrow_step(took=True, payment=0.2) == 0.2
            assert env._pending_reach == 0.0
            # Bounce alone: the payment escrows.
            assert env._reach_escrow_step(took=False, payment=0.2) == 0.2
            assert env._pending_reach == 0.2
            # Hit alone: pending kept (cleared without clawback).
            assert env._reach_escrow_step(took=True, payment=0.0) == 0.0
            assert env._pending_reach == 0.0
        finally:
            env.close()

    def test_camper_collects_nothing_net(self):
        """The anti-farming witness: parked near the landing, off the
        ball line — proximity is paid every receiving bounce and
        clawed back in full (no hit ever keeps it)."""
        from courtside_dynamics.envs._paddle_court import (
            scripted_reach_camper_witness,
        )

        env = PaddleTennisEnv(reach_shaping=0.25)
        try:
            paid_any = False
            for index, seed in enumerate(_SMOKE_SEEDS):
                totals, sums, kept, hits = self._drive(
                    env,
                    scripted_reach_camper_witness,
                    seed,
                    options=_serve_options(index),
                )
                assert hits == 0
                assert kept == 0.0
                paid_any = paid_any or sums["rew_reach"] > 0.0
                assert sums["rew_reach"] + sums["rew_reach_clawback"] == pytest.approx(
                    0.0, abs=1e-12
                )
            assert paid_any, "the camper never received a paid bounce"
        finally:
            env.close()

    def test_stacking_with_contact_shaping_is_exact(self):
        """Contact and reach escrows stack additively: the per-seed
        shaped-minus-unshaped total equals kept_reach plus
        0.25 x side-A confirms, exactly."""
        shaped = PaddleTennisEnv(contact_shaping=0.25, reach_shaping=0.25)
        unshaped = PaddleTennisEnv()
        try:
            for index, seed in enumerate(_SMOKE_SEEDS):
                obs, _ = shaped.reset(seed=seed, options=_serve_options(index))
                mirror_obs, _ = unshaped.reset(
                    seed=seed, options=_serve_options(index)
                )
                np.testing.assert_array_equal(obs, mirror_obs)
                total = mirror_total = 0.0
                kept_reach = pending_reach = 0.0
                confirms = 0
                while True:
                    action = scripted_ground_opponent(obs)
                    obs, reward, term, trunc, info = shaped.step(action)
                    mirror_obs, mirror_reward, *_ = unshaped.step(action)
                    np.testing.assert_array_equal(obs, mirror_obs)
                    total += reward
                    mirror_total += mirror_reward
                    confirms += int(bool(info["event_valid_return_a"]))
                    if info["event_valid_racket_hit_a"]:
                        kept_reach += pending_reach + info["rew_reach"]
                        pending_reach = 0.0
                    else:
                        pending_reach += info["rew_reach"]
                    if term or trunc:
                        break
                assert total - mirror_total == pytest.approx(
                    kept_reach + 0.25 * confirms, abs=1e-12
                )
        finally:
            shaped.close()
            unshaped.close()

    def test_kwargs_validated(self):
        with pytest.raises(ValueError):
            PaddleTennisEnv(reach_shaping=-0.1)
        with pytest.raises(ValueError):
            PaddleTennisEnv(reach_shaping=float("nan"))
        for radius in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError):
                PaddleTennisEnv(reach_shaping=0.25, reach_shaping_radius=radius)


class TestHoldShaping:
    """The escrow contract of design_paddle_tennis_postswing_hold.md:
    a side-A legal hit arms a travel-metered window, the opponent's
    return strike pays by how little the paddle wandered, the next
    side-A legal hit (the k=2 hit) keeps the advance, and every ending
    path claws back — with the default-off stream bit-identical."""

    @staticmethod
    def _drive(env, policy, seed, mirror_env=None, options=None):
        """Step ``env`` (optionally with a lockstep mirror); return
        totals, per-component sums, and the tracker-computed kept hold
        escrow (keep-before-arm ordering, the implementation's)."""
        obs, _ = env.reset(seed=seed, options=options)
        if mirror_env is not None:
            mirror_obs, _ = mirror_env.reset(seed=seed, options=options)
            np.testing.assert_array_equal(obs, mirror_obs)
        totals = {"reward": 0.0, "mirror_reward": 0.0}
        sums = dict.fromkeys(_SHAPING_COMPONENTS, 0.0)
        kept = pending = 0.0
        hits = 0
        while True:
            action = policy(obs)
            obs, reward, term, trunc, info = env.step(action)
            for key in _SHAPING_COMPONENTS:
                sums[key] += info[key]
            assert reward == pytest.approx(
                sum(info[key] for key in _SHAPING_COMPONENTS), abs=1e-12
            )
            totals["reward"] += reward
            if info["event_valid_racket_hit_a"]:
                hits += 1
                kept += pending + info["rew_hold"]
                pending = 0.0
            else:
                pending += info["rew_hold"]
            if mirror_env is not None:
                mirror_obs, mirror_reward, mterm, mtrunc, _ = mirror_env.step(action)
                np.testing.assert_array_equal(obs, mirror_obs)
                assert (term, trunc) == (mterm, mtrunc)
                totals["mirror_reward"] += mirror_reward
            if term or trunc:
                return totals, sums, kept, hits

    def test_escrow_identity_and_default_bit_identity(self):
        """Shaped-vs-unshaped arms of the same seed are bit-identical
        trajectories, and the escrow's whole undiscounted effect is
        exactly the kept (next-hit-taken) hold pay — under continuous
        n-point play, so boundary clawback and window disarm are
        exercised at every relaunch."""
        shaped = PaddleTennisEnv(points_per_episode=None, hold_shaping=0.25)
        unshaped = PaddleTennisEnv(points_per_episode=None)
        try:
            kept_any = False
            for index, seed in enumerate(_SMOKE_SEEDS):
                totals, sums, kept, _hits = self._drive(
                    shaped,
                    scripted_ground_opponent,
                    seed,
                    unshaped,
                    options=_serve_options(index),
                )
                assert sums["rew_hold"] + sums["rew_hold_clawback"] == pytest.approx(
                    kept, abs=1e-12
                )
                assert totals["reward"] - totals["mirror_reward"] == pytest.approx(
                    kept, abs=1e-12
                )
                kept_any = kept_any or kept > 0.0
            assert kept_any, "the oracle never kept a hold payment"
        finally:
            shaped.close()
            unshaped.close()

    def test_default_off_components_are_exact_zero(self):
        env = PaddleTennisEnv()
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            for _ in range(50):
                *_, term, trunc, info = env.step(_zero_action())
                assert info["rew_hold"] == 0.0
                assert info["rew_hold_clawback"] == 0.0
                if term or trunc:
                    break
        finally:
            env.close()

    def test_payment_matches_the_travel_formula(self):
        """Each nonzero pay equals shaping x max(0, 1 - travel/budget)
        with the travel re-accumulated externally from the step-end
        paddle-head positions — exactly, including follow-through."""
        env = PaddleTennisEnv(
            points_per_episode=None, hold_shaping=0.25, hold_shaping_travel=4.0
        )
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[1])
            armed = False
            prev_xy = (0.0, 0.0)
            travel = 0.0
            payments = 0
            last_points = 0.0
            while True:
                obs, _reward, term, trunc, info = env.step(
                    scripted_ground_opponent(obs)
                )
                paddle = env._paddle_position(CourtSide.A)
                xy = (float(paddle[0]), float(paddle[1]))
                if armed:
                    travel += float(
                        np.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1])
                    )
                    prev_xy = xy
                    if info["event_valid_racket_hit_b"]:
                        expected = 0.25 * max(0.0, 1.0 - travel / 4.0)
                        assert info["rew_hold"] == pytest.approx(
                            expected, abs=1e-12
                        )
                        payments += int(info["rew_hold"] > 0.0)
                        armed = False
                else:
                    assert info["rew_hold"] == 0.0
                # A side-A legal hit (re-)arms from this step's end;
                # a point boundary then disarms whatever is armed
                # (mirroring the env's arm-then-clawback step order).
                if info["event_valid_racket_hit_a"]:
                    armed = True
                    prev_xy = xy
                    travel = 0.0
                if info["points_played"] > last_points:
                    last_points = info["points_played"]
                    armed = False
                if term or trunc:
                    break
            assert payments > 0, "the oracle never earned a hold payment"
        finally:
            env.close()

    def test_boundary_disarms_the_window(self):
        """Every point ending discards an armed window before the
        relaunch (so the teleport can never leak into travel, and the
        next point's serve-return pays nothing). The window is
        re-armed white-box each step so the armed-at-boundary path is
        exercised deterministically — statue points end in seconds,
        while oracle rallies can fill the cap without one boundary."""
        env = PaddleTennisEnv(points_per_episode=None, hold_shaping=0.25)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            last_points = 0.0
            while True:
                if env._hold_anchor_xy is None:
                    paddle = env._paddle_position(CourtSide.A)
                    env._hold_anchor_xy = (float(paddle[0]), float(paddle[1]))
                _obs, _reward, term, trunc, info = env.step(_zero_action())
                if info["points_played"] > last_points:
                    last_points = info["points_played"]
                    assert env._hold_anchor_xy is None
                    assert env._hold_travel == 0.0
                if term or trunc:
                    break
            assert last_points > 0, "no point boundary was crossed"
        finally:
            env.close()

    def test_statue_never_pays(self):
        """No side-A hit ever arms a window, so the opponent's
        serve returns pay nothing: hold components are exact zeros for
        a statue across full continuous-play episodes."""
        env = PaddleTennisEnv(points_per_episode=None, hold_shaping=0.25)
        try:
            for index, seed in enumerate(_SMOKE_SEEDS):
                _totals, sums, kept, hits = self._drive(
                    env,
                    lambda _obs: _zero_action(),
                    seed,
                    options=_serve_options(index),
                )
                assert hits == 0
                assert kept == 0.0
                assert sums["rew_hold"] == 0.0
                assert sums["rew_hold_clawback"] == 0.0
        finally:
            env.close()

    def test_hit_then_freeze_nets_zero(self):
        """The farming attempt this design must defeat: hit once, then
        freeze perfectly still. The frozen paddle earns near-full hold
        pay at the opponent's return strike — and keeps none of it,
        because no second hit ever follows."""
        env = PaddleTennisEnv(hold_shaping=0.25)
        try:
            paid_any = False
            for index, seed in enumerate(_SMOKE_SEEDS):
                obs, _ = env.reset(seed=seed, options=_serve_options(index))
                frozen = False
                sums = {"rew_hold": 0.0, "rew_hold_clawback": 0.0}
                while True:
                    action = (
                        _zero_action()
                        if frozen
                        else scripted_ground_opponent(obs)
                    )
                    obs, _reward, term, trunc, info = env.step(action)
                    sums["rew_hold"] += info["rew_hold"]
                    sums["rew_hold_clawback"] += info["rew_hold_clawback"]
                    if info["event_valid_racket_hit_a"]:
                        frozen = True
                    if term or trunc:
                        break
                paid_any = paid_any or sums["rew_hold"] > 0.0
                assert sums["rew_hold"] + sums["rew_hold_clawback"] == (
                    pytest.approx(0.0, abs=1e-12)
                )
            assert paid_any, "the freezer never received a hold payment"
        finally:
            env.close()

    def test_truncation_claws_back_pending_hold(self):
        env = PaddleTennisEnv(hold_shaping=0.25, episode_len=3)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            env._pending_hold = 0.125
            term = trunc = False
            info: dict = {}
            reward = 0.0
            while not (term or trunc):
                _obs, reward, term, trunc, info = env.step(_zero_action())
            assert trunc and not term
            assert info["rew_hold_clawback"] == -0.125
            assert reward == pytest.approx(-0.125)
            assert env._pending_hold == 0.0
        finally:
            env.close()

    def test_nan_action_guard_claws_back_pending_hold(self):
        env = PaddleTennisEnv(hold_shaping=0.25)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            env._pending_hold = 0.125
            env._hold_anchor_xy = (0.0, 0.0)
            _obs, reward, term, trunc, info = env.step(
                np.array([np.nan, 0.0, 0.0])
            )
            assert term and not trunc
            assert info["rew_hold_clawback"] == -0.125
            assert reward == pytest.approx(-env.unsafe_physics_penalty - 0.125)
            assert env._pending_hold == 0.0
            assert env._hold_anchor_xy is None
        finally:
            env.close()

    def test_same_step_keep_pin_and_rearm(self):
        """The §2 ordering pins, exercised at the helper level (the
        physical step separation makes the edge unreachable in play):
        a payment sharing the keeping hit's step is kept immediately,
        the prior pending is kept too, and the same hit re-arms a
        fresh window from the current paddle position."""
        from types import SimpleNamespace

        env = PaddleTennisEnv(hold_shaping=0.25, hold_shaping_travel=4.0)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            paddle = env._paddle_position(CourtSide.A)
            xy = (float(paddle[0]), float(paddle[1]))
            env._pending_hold = 0.125
            env._hold_anchor_xy = xy
            env._hold_travel = 0.0
            fake = SimpleNamespace(valid_racket_hits=(CourtSide.A, CourtSide.B))
            payment = env._hold_escrow_step(fake)
            assert payment == pytest.approx(0.25)  # zero travel: full pay
            assert env._pending_hold == 0.0  # prior pending + payment kept
            assert env._hold_anchor_xy == xy  # the hit re-armed the window
            assert env._hold_travel == 0.0
        finally:
            env.close()

    def test_kwargs_validated(self):
        with pytest.raises(ValueError):
            PaddleTennisEnv(hold_shaping=-0.1)
        with pytest.raises(ValueError):
            PaddleTennisEnv(hold_shaping=float("nan"))
        for budget in (0.0, -1.0, float("nan")):
            with pytest.raises(ValueError):
                PaddleTennisEnv(hold_shaping=0.25, hold_shaping_travel=budget)

    def test_recipe_adopts_reach_shaping_and_guards(self):
        """The LR1 ADOPT verdict (design doc §4a): reach escrow on at
        0.25, and a guard set rides with it -- since the 2026-10-05
        instrument batch on the episode-cumulative side-A counters
        (the per-point legal_hit_count_a stays logged)."""
        from courtside_dynamics.recipes import RECIPES

        recipe = RECIPES["PaddleTennis"]
        assert recipe.env_kwargs["reach_shaping"] == 0.25
        extra = recipe.extra_cfg
        assert extra["success_key"] == "episode_rally_returns_a"
        assert "legal_hit_count_a" in extra["info_eval_keys"]
        assert extra["degenerate_guard_keys"] == (
            "episode_legal_hit_count_a_ep_mean",
        )
        assert extra["early_stop_degenerate_evals"] == 5
        assert extra["best_metric_min_delta"] == {
            "episode_rally_returns_a_ep_mean": pytest.approx(0.5 / 30),
            "success_rate": pytest.approx(0.5 / 30),
            "episode_reward_mean": 0.25,
        }
        assert extra["confirm_best_eval"] is True
        assert extra["headline_key"] == "episode_rally_returns_a"


class TestNPointEpisodes:
    """The n-point contract of design_paddle_tennis_npoint.md: the
    frozen default stays bit-identical, boundaries carry paddles
    over, the escrow claws back per point, alternation is strict, and
    the relaunch protocol's hazards stay witnessed."""

    def test_np0_default_bit_identical_lockstep(self):
        """The default arm locksteps an n=3 arm bit-for-bit — obs,
        reward, and the FULL info dict — up to the first point end,
        where the default terminates and the n-point arm absorbs.
        Any drift in the shared step path fails loudly (an
        identically-constructed pair could not detect one)."""
        diverged = 0
        # Burned-block seeds whose one-point episode ends in a rally
        # fault before the cap (measured: 200/178 steps), so the
        # absorption divergence is actually reached. Fresh instances
        # per seed: after an absorption, arm b's alternation state
        # has legitimately advanced past arm a's (the
        # partial-point-consumes-turn rule), so instances cannot be
        # reused across seeds.
        for seed in (1008, 1010):
            a = PaddleTennisEnv()
            b = PaddleTennisEnv(points_per_episode=3)
            try:
                oa, _ = a.reset(seed=seed)
                ob, _ = b.reset(seed=seed)
                np.testing.assert_array_equal(oa, ob)
                while True:
                    action = scripted_ground_opponent(oa)
                    oa, ra, ta, tra, ia = a.step(action)
                    ob, rb, tb, trb, ib = b.step(action)
                    assert ra == rb
                    if ta and not tb:
                        # The first point end: arm a ends its episode
                        # while arm b absorbs the boundary. Both count
                        # the completed point.
                        diverged += 1
                        assert ia["points_played"] == 1.0
                        assert ib["points_played"] == 1.0
                        break
                    np.testing.assert_array_equal(oa, ob)
                    assert (ta, tra) == (tb, trb)
                    assert set(ia) == set(ib)
                    for key, va in ia.items():
                        if isinstance(va, (int, float)):
                            assert va == ib[key], key
                    if ta or tra:
                        break
            finally:
                a.close()
                b.close()
        assert diverged >= 1  # a real absorption was witnessed

    def test_statue_nets_minus_one_per_receiving_point(self):
        env = PaddleTennisEnv(points_per_episode=None, episode_len=1500)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            total = 0.0
            receiving = serving = prev = 0
            while True:
                policy_serves = env._serving_side is CourtSide.A
                _obs, reward, term, trunc, info = env.step(_zero_action())
                total += reward
                if info["points_played"] > prev:
                    prev = int(info["points_played"])
                    if policy_serves:
                        serving += 1
                    else:
                        receiving += 1
                if term or trunc:
                    break
            assert trunc and not term
            assert info["term_timeout"] == 1.0
            assert prev == receiving + serving >= 2
            # -1 per receiving point; the shared +1 from the
            # opponent's serve-return offsets each serving fault.
            assert total == pytest.approx(-receiving)
            # Episode-scoped term_* never fired for the absorbed
            # faults; the durable record is the point_end counters.
            end_counts = sum(
                info[f"point_end_{name}"]
                for name in (
                    "out_of_bounds",
                    "ball_net",
                    "second_bounce",
                    "failed_to_cross",
                    "illegal_hit",
                    "net_touch",
                    "volley",
                )
            )
            assert end_counts == prev
        finally:
            env.close()

    def test_escrow_identity_across_point_boundaries(self):
        env = PaddleTennisEnv(points_per_episode=None, contact_shaping=0.25)
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[1])
            paid = clawed = confirms = 0.0
            prev_points = 0
            boundary_clawbacks = 0
            while True:
                obs, _r, term, trunc, info = env.step(scripted_ground_opponent(obs))
                paid += info["rew_shaping"]
                clawed += info["rew_shaping_clawback"]
                confirms += float(bool(info["event_valid_return_a"]))
                if info["points_played"] > prev_points and not (term or trunc):
                    prev_points = int(info["points_played"])
                    if info["rew_shaping_clawback"] != 0.0:
                        boundary_clawbacks += 1
                if term or trunc:
                    break
            assert paid + clawed == pytest.approx(0.25 * confirms)
            # The identity must actually be exercised ACROSS a
            # boundary: at least one mid-episode point end clawed a
            # pending advance back (a terminal clawback alone would
            # leave the boundary path untested).
            assert boundary_clawbacks >= 1
        finally:
            env.close()

    def test_carryover_paddles_continuous_at_boundaries(self):
        """At each un-nudged boundary the policy paddle moves by at
        most one step of motion — never a re-park jump. The moving
        hard-slam witness (not a statue, whose parked paddle cannot
        distinguish carryover from a re-park) is self-validated
        below: at least one checked boundary catches the paddle
        genuinely away from its reset pose."""
        from courtside_dynamics.envs._paddle_court import (
            scripted_hard_slam_witness,
        )

        env = PaddleTennisEnv(points_per_episode=None)
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[1])
            reset_pose = env._paddle_position(CourtSide.A).copy()
            prev_points = 0
            boundaries = displaced = 0
            while True:
                before = env._paddle_position(CourtSide.A).copy()
                obs, _r, term, trunc, info = env.step(scripted_hard_slam_witness(obs))
                after = env._paddle_position(CourtSide.A)
                if info["points_played"] > prev_points:
                    prev_points = int(info["points_played"])
                    if info["point_serve_nudged"] == 0.0 and not (term or trunc):
                        boundaries += 1
                        if float(np.linalg.norm(before - reset_pose)) > 0.5:
                            displaced += 1
                        assert float(np.linalg.norm(after - before)) < 0.5, (
                            "paddle jumped at a point boundary"
                        )
                if term or trunc:
                    break
            assert boundaries >= 1
            # Self-validation: a re-park regression would have
            # tripped the jump bound on these displaced boundaries.
            assert displaced >= 1
        finally:
            env.close()

    def test_alternation_strict_across_points_and_episodes(self):
        env = PaddleTennisEnv(points_per_episode=None, episode_len=1200)
        try:
            servers: list[str] = []
            for index in range(2):
                # Alternation carries across episode boundaries on the
                # UNSEEDED reset (the vectorized auto-reset path); a
                # seeded reset restarts it at side A by contract.
                env.reset(seed=_SMOKE_SEEDS[0] if index == 0 else None)
                servers.append(env._serving_side.name)
                prev = 0
                while True:
                    _obs, _r, term, trunc, info = env.step(_zero_action())
                    if info["points_played"] > prev and not (term or trunc):
                        prev = int(info["points_played"])
                        servers.append(env._serving_side.name)
                    if term or trunc:
                        break
            for first, second in zip(servers, servers[1:], strict=False):
                assert first != second, servers
        finally:
            env.close()

    def test_nudge_clears_launch_envelope(self):
        env = PaddleTennisEnv(points_per_episode=None)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            target = env._paddle_position(CourtSide.A) + np.array([0.05, 0.0, 0.05])
            assert not env._clear_launch_envelope(target)
            env._nudge_paddle_clear(target)
            env.set_state(env.data.qpos.copy(), env.data.qvel.copy())
            assert env._clear_launch_envelope(target)
        finally:
            env.close()

    @staticmethod
    def _slides_within_range(env) -> bool:
        for side in "ab":
            for axis in "xyz":
                joint = env.model.joint(f"player_{side}_slide_{axis}")
                value = float(env.data.qpos[int(joint.qposadr[0])])
                if not float(joint.range[0]) <= value <= float(joint.range[1]):
                    return False
        return True

    def test_nudge_clamps_to_slide_ranges_near_a_limit(self):
        """Regression: the radial nudge wrote slide targets past the
        joint limits (here x, with the paddle parked at its deepest
        reach and the origin just in front of it), which the solver
        snaps back toward the ball. The clamped nudge must stay in
        range AND clear, taking the smallest feasible exit."""
        env = PaddleTennisEnv(points_per_episode=None)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            slide_x = env.model.joint("player_a_slide_x")
            qpos = env.data.qpos.copy()
            qpos[int(slide_x.qposadr[0])] = float(slide_x.range[0])
            env.set_state(qpos, env.data.qvel.copy())
            head = env._paddle_position(CourtSide.A)
            assert head[0] == pytest.approx(-6.4)
            target = head + np.array([0.1, 0.0, 0.1])
            assert not env._clear_launch_envelope(target)
            env._nudge_paddle_clear(target)
            assert self._slides_within_range(env)
            env.set_state(env.data.qpos.copy(), env.data.qvel.copy())
            assert env._clear_launch_envelope(target)
            moved = env._paddle_position(CourtSide.A) - head
            # The minimal feasible exit here is straight down.
            assert moved[0] == pytest.approx(0.0, abs=1e-12)
            assert moved[1] == pytest.approx(0.0, abs=1e-12)
            assert moved[2] < 0.0
            assert float(np.linalg.norm(moved)) < 0.45
        finally:
            env.close()

    def test_relaunch_reverifies_a_nudge_at_a_limit(self):
        """The relaunch protocol end to end on a serve origin parked in
        front of the paddle's deepest reach (zero jitter: every redraw
        is blocked): the nudge fires once, the slides stay in range,
        and the launched ball starts clear of both heads."""
        serve = PaddleCourtServe(start_distance_from_net=6.3, position_noise=(0, 0, 0))
        env = PaddleTennisEnv(points_per_episode=None, serve_config=serve)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            slide_x = env.model.joint("player_a_slide_x")
            qpos = env.data.qpos.copy()
            qpos[int(slide_x.qposadr[0])] = float(slide_x.range[0])
            env.set_state(qpos, env.data.qvel.copy())
            env._serving_side = CourtSide.A
            env._launch_point(mid_episode=True)
            assert env._point_serve_nudged == 1
            assert self._slides_within_range(env)
            ball = env._ball_position()
            np.testing.assert_allclose(ball, (-6.3, 0.0, serve.height))
            assert env._clear_launch_envelope(ball)
        finally:
            env.close()

    def test_relaunch_redraws_when_a_nudge_leaves_the_envelope_blocked(
        self, monkeypatch
    ):
        """The post-nudge re-verify loop, which the real nudge never
        reaches (it always clears): should a nudge leave a head inside
        the launch envelope, the relaunch redraws against where the
        paddles now stand. The nudge is stalled (it moves nothing) and
        the draws are scripted: the protocol's 1 + _SERVE_REDRAWS
        pre-nudge draws all sit on the policy's paddle head, and every
        later one is the real, clear draw. The relaunch must nudge
        once, redraw after it, and launch the ball clear of both
        heads with the paddles left where they stood."""
        env = PaddleTennisEnv(points_per_episode=None)
        try:
            env.reset(seed=_SMOKE_SEEDS[0])
            env._serving_side = CourtSide.A
            head = env._paddle_position(CourtSide.A).copy()
            blocked = head + np.array([0.05, 0.0, 0.05])
            pre_nudge = 1 + env._SERVE_REDRAWS
            real_draw = env._draw_serve
            draws: list[np.ndarray] = []
            nudges: list[np.ndarray] = []

            def scripted_draw():
                position, velocity = real_draw()
                if len(draws) < pre_nudge:
                    position = blocked.copy()
                draws.append(position.copy())
                return position, velocity

            def stalled_nudge(position):
                nudges.append(position.copy())

            monkeypatch.setattr(env, "_draw_serve", scripted_draw)
            monkeypatch.setattr(env, "_nudge_paddle_clear", stalled_nudge)
            env._launch_point(mid_episode=True)

            assert len(nudges) == 1
            np.testing.assert_array_equal(nudges[0], blocked)
            assert env._point_serve_nudged == 1
            # The stalled nudge left the head inside the envelope, so
            # the loop redrew; its first draw is clear and ends it.
            assert len(draws) == pre_nudge + 1
            assert not env._clear_launch_envelope(blocked)
            ball = env._ball_position()
            np.testing.assert_allclose(ball, draws[-1])
            assert env._clear_launch_envelope(ball)
            np.testing.assert_allclose(env._paddle_position(CourtSide.A), head)
        finally:
            env.close()

    def test_hard_slam_far_side_deaths_cause_no_spurious_faults(self):
        """NP1 relaunch-hazard witness: points ending with the ball
        beyond the far baseline must not open the next point with a
        reverse-crossing (illegal-hit-group) fault."""
        from courtside_dynamics.envs._paddle_court import (
            scripted_hard_slam_witness,
        )

        env = PaddleTennisEnv(points_per_episode=None)
        try:
            obs, _ = env.reset(seed=_SMOKE_SEEDS[2])
            last_ball_x = float(env._ball_position()[0])
            prev_points = 0
            far_side_ends = 0
            while True:
                obs, _r, term, trunc, info = env.step(scripted_hard_slam_witness(obs))
                if info["points_played"] > prev_points:
                    prev_points = int(info["points_played"])
                    # Pre-boundary ball position: past the far (+x)
                    # baseline at 6.5 m — the hazard's precondition.
                    if last_ball_x > 6.5:
                        far_side_ends += 1
                last_ball_x = float(env._ball_position()[0])
                if term or trunc:
                    break
            assert info["points_played"] >= 2
            # Self-validation: the witness really died far-side (a
            # statue banks out_of_bounds endings with zero far-side
            # deaths, so the counter alone proves nothing).
            assert far_side_ends >= 2
            assert info["point_end_illegal_hit"] == 0.0
        finally:
            env.close()

    def test_instrument_segments_points_and_measures_recovery(self):
        from courtside_dynamics.training.paddle_diagnosis import (
            run_player,
        )

        statue = lambda observation: np.zeros(3)  # noqa: E731
        traces, travels = run_player(
            statue,
            episodes=2,
            seed_start=_SMOKE_SEEDS[0],
            env_fn=lambda: PaddleTennisEnv(points_per_episode=None),
        )
        assert len(traces) > 2  # multiple points per episode
        assert travels  # inter-point recovery measured
        receiving = [t for t in traces if not t.serve_side_is_policy]
        assert receiving
        for trace in receiving:
            if trace.termination == "second_bounce":
                assert trace.ender == "policy_never_reached"

    def test_recipe_adopts_npoint_continuous_play(self):
        """The LR1 ADOPT verdict: the recipe plays continuous n-point
        (the ENV default stays 1 — the frozen one-point task is
        untouched for direct construction and the NP0 lockstep)."""
        from courtside_dynamics.recipes import RECIPES

        assert RECIPES["PaddleTennis"].env_kwargs["points_per_episode"] is None
        env = PaddleTennisEnv()
        try:
            assert env.points_per_episode == 1
        finally:
            env.close()

    def test_np3_floor_constants_pin(self):
        """The n-point certification's pre-registered contract: the
        reserved block and the floors frozen from NP2's band before
        the block was opened (design doc §4a). Constants only — the
        reserved seeds are never drawn here."""
        from tools.paddle_tennis_npoint_probe import (
            NP3_CERT_EPISODES,
            NP3_CERT_SEED_START,
            NP3_COMPLETED_POINTS_FLOOR,
            NP3_MEAN_CROSSINGS_FLOOR,
            NP3_NUDGE_RATE_CEILING,
            PROBE_SEED_START,
        )

        assert NP3_CERT_SEED_START == 4300
        assert NP3_CERT_EPISODES == 100
        assert NP3_MEAN_CROSSINGS_FLOOR == 9.0
        assert NP3_COMPLETED_POINTS_FLOOR == 50
        assert NP3_NUDGE_RATE_CEILING == 0.02
        assert PROBE_SEED_START == 5400

    def test_l2_toml_spelling_reaches_both_envs(self, tmp_path):
        """The L2 pilot's TOML contract: TOML has no null, so the
        run-config loader's ``"none"`` sentinel is the fill-the-cap
        spelling — it must reach the training AND eval env
        constructors as Python ``None``."""
        from courtside_dynamics.recipes import build_train_config

        config = tmp_path / "npoint_pilot.toml"
        config.write_text(
            '[env]\npoints_per_episode = "none"\ncontact_shaping = 0.25\n'
        )
        cfg = build_train_config(
            "PaddleTennis",
            total_timesteps=1_000,
            log_dir=str(tmp_path / "run"),
            config_file=str(config),
        )
        for factory in (cfg.env_fn, cfg.eval_env_fn or cfg.env_fn):
            env = factory()
            try:
                assert env.points_per_episode is None
                assert env.contact_shaping == 0.25
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

        result = certify_frozen_env(episodes=2, seed_start=_SMOKE_SEEDS[0])
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
                    obs, _, term, trunc, _ = env.step(scripted_ground_opponent(obs))
                    steps.append(obs)
                    if term or trunc:
                        break
                return np.concatenate(steps)
            finally:
                env.close()

        first = run(_SMOKE_SEEDS[2])
        second = run(_SMOKE_SEEDS[2])
        assert np.array_equal(first, second)

    def test_independent_instances_match_bit_for_bit(self):
        """Two independently constructed envs, same seed and actions:
        observations, rewards, terminations and truncations agree
        exactly -- for the frozen default and for the recipe's n-point
        + escrow kwargs, across both serve slots (the seeded reset,
        then the unseeded alternation)."""
        actions = np.random.default_rng(11).uniform(-1.0, 1.0, size=(400, 3))

        def stream(kwargs: dict) -> list:
            env = PaddleTennisEnv(**kwargs)
            try:
                rows: list = []
                for seed in (_SMOKE_SEEDS[0], None):
                    obs, _ = env.reset(seed=seed)
                    rows.append(obs.tobytes())
                    for action in actions:
                        obs, reward, term, trunc, _ = env.step(action)
                        rows.append((obs.tobytes(), reward, term, trunc))
                        if term or trunc:
                            break
                return rows
            finally:
                env.close()

        for kwargs in ({}, _RECIPE_KWARGS):
            assert stream(kwargs) == stream(kwargs)


@pytest.fixture(scope="module")
def drill_library_path(tmp_path_factory) -> str:
    """A small harvested k=2 library built the shipped tool's way: the
    ground oracle drives side A on burned bring-up seeds and the
    tool's own ``_snapshot_env`` records qualifying k=2 instants on
    policy-receiving points. Schema-identical to
    ``tools/paddle_tennis_k2_harvest.py``'s artifact, so this also
    pins the library contract the env consumes."""
    from tools.paddle_tennis_k2_harvest import SCHEMA, _snapshot_env

    env = PaddleTennisEnv(points_per_episode=None)
    entries: list[dict] = []
    try:
        seed = _SMOKE_SEEDS[0]
        # The walk stays inside the burned bring-up block 1000-1119
        # (the module ledger comment): ledger compliance is
        # structural, not empirical.
        while len(entries) < 6 and seed <= 1119:
            # Land on a policy-receiving first point (a seeded reset
            # would otherwise serve A; this is the same seed's draw the
            # old reset-twice idiom reached through the alternation).
            obs, info = env.reset(seed=seed, options={"serve_side": "b"})
            seed += 1
            hit_a = struck_b = False
            prev_points = 0
            for _ in range(1500):
                obs, _r, term, trunc, step_info = env.step(
                    scripted_ground_opponent(obs)
                )
                if term or trunc:
                    break
                if int(step_info["points_played"]) != prev_points:
                    prev_points = int(step_info["points_played"])
                    hit_a = struck_b = False
                if bool(step_info["event_valid_racket_hit_a"]):
                    hit_a, struck_b = True, False
                if bool(step_info["event_valid_racket_hit_b"]) and hit_a:
                    struck_b = True
                ball_x = float(env.data.qpos[env._ball_qposadr])
                ball_vx = float(env.data.qvel[env._ball_dofadr])
                if (
                    struck_b
                    and env._serving_side is CourtSide.B
                    and ball_x > 0.05
                    and ball_vx < -1.0
                ):
                    entry = _snapshot_env(env, obs)
                    ball = entry["qpos"][
                        env._ball_qposadr : env._ball_qposadr + 3
                    ]
                    if all(
                        float(np.linalg.norm(entry[key] - ball))
                        >= env._SERVE_CLEARANCE
                        for key in ("head_a", "head_b")
                    ):
                        entry["seed"] = seed - 1
                        entry["point"] = prev_points
                        entries.append(entry)
                        hit_a = struck_b = False
                        if len(entries) >= 6:
                            break
    finally:
        env.close()
    assert len(entries) == 6, (
        "harvest exhausted the burned bring-up block 1000-1119 before "
        "filling the library"
    )
    library = {
        "schema": SCHEMA,
        "git_sha": "test",
        "model_path": "scripted_ground_opponent",
        "model_sha256": "test",
        "vec_normalize_sha256": "test",
        "seed_start": _SMOKE_SEEDS[0],
        "episodes": seed - _SMOKE_SEEDS[0],
        "continuation_steps": 0,
        "clearance_dropped": 0,
        "crossings_unarmed": 0,
        "entries": entries,
    }
    path = tmp_path_factory.mktemp("drill") / "k2_test_library.pkl"
    with open(path, "wb") as f:
        pickle.dump(library, f)
    return str(path)


class TestK2Drill:
    """The k=2 drill mechanism (docs/design_paddle_tennis_k2_drill.md
    §2, shipped default-off ahead of the battery freeze): KD0
    bit-identity including the RNG-stream discipline, loud validation
    of half-configured pairs and malformed libraries, both D2 launch
    arms' fidelity, policy-receiving-only eligibility with serve
    alternation preserved, info/reset provenance, and the D5
    clearance fallback."""

    def test_kd0_default_off_bit_identical_lockstep(self):
        """KD0, kwarg-neutrality half: an env with the drill kwargs
        at their explicit defaults locksteps a default-constructed
        env bit-for-bit — obs, reward, terminations, and the FULL
        info dict — across episodes starting from both serve slots.
        (Both envs run the same code, so this alone cannot see an
        extra RNG draw on the shared path — that half of KD0 is
        certified against an independent twin generator by
        ``test_kd0_rng_stream_certificate``.)"""
        a = PaddleTennisEnv(points_per_episode=None)
        b = PaddleTennisEnv(
            points_per_episode=None,
            drill_library=None,
            drill_fraction=0.0,
            drill_context="full",
        )
        try:
            for index, seed in enumerate(_SMOKE_SEEDS[:2]):
                oa, ia = a.reset(seed=seed, options=_serve_options(index))
                ob, ib = b.reset(seed=seed, options=_serve_options(index))
                np.testing.assert_array_equal(oa, ob)
                assert ia == ib
                while True:
                    oa, ra, ta, tra, ia = a.step(_zero_action())
                    ob, rb, tb, trb, ib = b.step(_zero_action())
                    np.testing.assert_array_equal(oa, ob)
                    assert (ra, ta, tra) == (rb, tb, trb)
                    assert set(ia) == set(ib)
                    for key, value in ia.items():
                        if isinstance(value, (int, float)):
                            assert value == ib[key], key
                    if ta or tra:
                        break
        finally:
            a.close()
            b.close()

    def test_kd0_rng_stream_certificate(self, drill_library_path):
        """KD0's RNG-discipline half, certified against an
        INDEPENDENT twin generator (a same-code lockstep cannot see a
        draw both arms share): after a reset, the env generator's
        state must equal a twin that replayed exactly the draws the
        launch is specified to consume — the four `_draw_serve`
        uniforms for an undrilled launch (drill off OR ineligible
        policy-serving slot: the drill must add nothing), and the
        eligibility uniform plus the entry-index integer, with NO
        serve draws, for a drilled launch (the §2a pinned stream
        order). Any extra, missing, or reordered draw at the launch
        site moves the state and fails."""
        from gymnasium.utils import seeding

        def _twin_after_serve_draws(seed: int, env: PaddleTennisEnv):
            rng, _ = seeding.np_random(seed)
            serve = env.serve_config
            rng.uniform(
                low=-np.asarray(serve.position_noise),
                high=np.asarray(serve.position_noise),
            )
            rng.uniform(
                serve.speed - serve.speed_noise,
                serve.speed + serve.speed_noise,
            )
            rng.uniform(
                serve.elevation_degrees - serve.elevation_noise_degrees,
                serve.elevation_degrees + serve.elevation_noise_degrees,
            )
            rng.uniform(
                serve.lateral_degrees - serve.lateral_noise_degrees,
                serve.lateral_degrees + serve.lateral_noise_degrees,
            )
            return rng

        seed = _SMOKE_SEEDS[0]
        env = PaddleTennisEnv(points_per_episode=None)
        try:
            env.reset(seed=seed)
            twin = _twin_after_serve_draws(seed, env)
            assert env.np_random.bit_generator.state == twin.bit_generator.state
        finally:
            env.close()

        env = PaddleTennisEnv(
            points_per_episode=None,
            drill_library=drill_library_path,
            drill_fraction=1.0,
        )
        try:
            _obs, info = env.reset(seed=seed)  # a seeded reset serves A
            assert info["serve_side"] == "a"
            twin = _twin_after_serve_draws(seed, env)
            assert env.np_random.bit_generator.state == twin.bit_generator.state

            # The policy-receiving slot, forced (a seeded reset
            # serves A by contract).
            _obs, info = env.reset(seed=seed, options={"serve_side": "b"})
            assert info["serve_side"] == "b"
            assert info["drill_point"] == 1.0
            rng, _ = seeding.np_random(seed)
            rng.random()
            drawn = int(rng.integers(len(env._drill_entries)))
            assert drawn == int(info["drill_entry_index"])
            assert env.np_random.bit_generator.state == rng.bit_generator.state
        finally:
            env.close()

    def test_kd0_stream_shared_until_first_eligible_launch(
        self, drill_library_path
    ):
        """The §2 RNG discipline, drill-on side: a drill-on env
        locksteps the default env bit-for-bit through the whole
        policy-SERVING first point (an ineligible launch draws
        nothing), then diverges exactly at the policy-receiving
        relaunch, which substitutes a library scenario for the drawn
        serve."""
        a = PaddleTennisEnv(points_per_episode=None)
        b = PaddleTennisEnv(
            points_per_episode=None,
            drill_library=drill_library_path,
            drill_fraction=1.0,
        )
        try:
            seed = _SMOKE_SEEDS[2]
            oa, ia = a.reset(seed=seed)
            ob, ib = b.reset(seed=seed)
            assert ia["serve_side"] == ib["serve_side"] == "a"
            assert ib["drill_point"] == 0.0
            np.testing.assert_array_equal(oa, ob)
            while True:
                oa, ra, ta, tra, ia = a.step(_zero_action())
                ob, rb, tb, trb, ib = b.step(_zero_action())
                assert (ra, ta, tra) == (rb, tb, trb)
                assert not (ta or tra)  # divergence must precede the cap
                if int(ia["points_played"]) == 1:
                    break
                np.testing.assert_array_equal(oa, ob)
            # The boundary step describes the (identical, undrilled)
            # first point in both arms, but its relaunch observation
            # already differs: b launched a harvested scenario.
            assert ia["drill_point"] == ib["drill_point"] == 0.0
            assert not np.array_equal(oa, ob)
            _oa, _ra, _ta, _tra, ia = a.step(_zero_action())
            _ob, _rb, _tb, _trb, ib = b.step(_zero_action())
            assert ia["drill_point"] == 0.0
            assert ib["drill_point"] == 1.0
            assert ib["drill_entry_index"] >= 0.0
            assert ib["drill_fallback_count"] == 0.0
        finally:
            a.close()
            b.close()

    def test_validation_rejects_half_configured_and_malformed(
        self, drill_library_path, tmp_path
    ):
        """Cardinal rule 1: every inconsistent configuration and every
        malformed library refuses at construction, never mid-run."""
        with pytest.raises(ValueError, match="enabled together"):
            PaddleTennisEnv(drill_fraction=0.5)
        with pytest.raises(ValueError, match="enabled together"):
            PaddleTennisEnv(drill_library=drill_library_path)
        with pytest.raises(ValueError, match="drill_fraction"):
            PaddleTennisEnv(
                drill_library=drill_library_path, drill_fraction=1.5
            )
        with pytest.raises(ValueError, match="drill_context"):
            PaddleTennisEnv(
                drill_library=drill_library_path,
                drill_fraction=0.5,
                drill_context="mixed",
            )

        def _write(name: str, library: dict) -> str:
            path = tmp_path / name
            with open(path, "wb") as f:
                pickle.dump(library, f)
            return str(path)

        with open(drill_library_path, "rb") as f:
            library = pickle.load(f)

        bad_schema = _write(
            "bad_schema.pkl", {**library, "schema": "other-schema"}
        )
        with pytest.raises(ValueError, match="schema"):
            PaddleTennisEnv(drill_library=bad_schema, drill_fraction=0.5)

        empty = _write("empty.pkl", {**library, "entries": []})
        with pytest.raises(ValueError, match="no entries"):
            PaddleTennisEnv(drill_library=empty, drill_fraction=0.5)

        # A full-arm-only field missing: the feed arm builds, the
        # full arm refuses (the per-arm required-key sets).
        stripped_entries = [dict(entry) for entry in library["entries"]]
        del stripped_entries[0]["sampler"]
        stripped = _write(
            "stripped.pkl", {**library, "entries": stripped_entries}
        )
        PaddleTennisEnv(
            drill_library=stripped, drill_fraction=0.5
        ).close()
        with pytest.raises(ValueError, match="missing"):
            PaddleTennisEnv(
                drill_library=stripped,
                drill_fraction=0.5,
                drill_context="full",
            )

        # D3: an entry harvested from a non-policy-receiving point
        # does not fit the slot the drill launches into.
        wrong_side_entries = [dict(entry) for entry in library["entries"]]
        wrong_side_entries[1]["serving_side"] = CourtSide.A
        wrong_side = _write(
            "wrong_side.pkl", {**library, "entries": wrong_side_entries}
        )
        with pytest.raises(ValueError, match="non-policy-receiving"):
            PaddleTennisEnv(drill_library=wrong_side, drill_fraction=0.5)

        # A library harvested under a different rally-rule profile
        # would mix two task rules inside one env — refused loudly
        # for both arms (the fixture was harvested under "fault").
        for context in ("feed", "full"):
            with pytest.raises(ValueError, match="mix rally rules"):
                PaddleTennisEnv(
                    volley_rule="legal",
                    drill_library=drill_library_path,
                    drill_fraction=0.5,
                    drill_context=context,
                )

    def test_feed_arm_launches_harvested_state_as_fresh_feed(
        self, drill_library_path
    ):
        """Arm (a): the harvested joint state lands exactly (ball AND
        both paddles — design D4), the rally context reads a fresh
        side-B feed, and the reset-info serve keys carry the
        harvested launch ball state (the extended provenance
        contract)."""
        env = PaddleTennisEnv(
            points_per_episode=None,
            drill_library=drill_library_path,
            drill_fraction=1.0,
        )
        try:
            obs, info = env.reset(seed=_SMOKE_SEEDS[0], options={"serve_side": "b"})
            assert info["serve_side"] == "b"
            assert info["drill_point"] == 1.0
            index = int(info["drill_entry_index"])
            assert index >= 0
            entry = env._drill_entries[index]
            np.testing.assert_array_equal(env.data.qpos, entry["qpos"])
            np.testing.assert_array_equal(env.data.qvel, entry["qvel"])
            adr, dof = env._ball_qposadr, env._ball_dofadr
            np.testing.assert_array_equal(
                np.asarray(info["serve_ball_position"]),
                entry["qpos"][adr : adr + 3],
            )
            np.testing.assert_array_equal(
                np.asarray(info["serve_ball_velocity"]),
                entry["qvel"][dof : dof + 3],
            )
            names = list(PADDLE_TENNIS_OBSERVATION_NAMES)
            assert obs[names.index("rally_phase_initial_feed")] == 1.0
            assert obs[names.index("own_is_serving")] == 0.0
            assert obs[names.index("ball_side_is_own")] == 0.0
            assert obs[names.index("feed_crossed_net")] == 0.0
            assert obs[names.index("rally_count")] == 0.0
        finally:
            env.close()

    def test_full_arm_reproduces_the_recorded_observation(
        self, drill_library_path
    ):
        """Arm (b): the launch observation equals the harvest-recorded
        one on every component except ``episode_remaining_fraction``
        (index 35), which reads the drilled episode's OWN clock — the
        documented departure. The restored machine's harvested
        crossing count must not leak into the episode counter."""
        env = PaddleTennisEnv(
            points_per_episode=None,
            drill_library=drill_library_path,
            drill_fraction=1.0,
            drill_context="full",
        )
        try:
            obs, info = env.reset(seed=_SMOKE_SEEDS[0], options={"serve_side": "b"})
            assert info["drill_point"] == 1.0
            entry = env._drill_entries[int(info["drill_entry_index"])]
            deviation = np.abs(obs - entry["obs"])
            deviation[35] = 0.0
            # §3a measured ≤ 8e-6 (max 7.903e-6) at scale on the registered library.
            assert float(deviation.max()) <= 1e-4
            assert obs[35] == 1.0  # a fresh episode clock at reset
            names = list(PADDLE_TENNIS_OBSERVATION_NAMES)
            assert obs[names.index("rally_phase_return_in_flight")] == 1.0
            assert obs[names.index("feed_crossed_net")] == 1.0
            assert obs[names.index("rally_count")] >= 1.0
            # The continuity offset absorbed the restored machine's
            # harvested crossing count (negative for a mid-rally
            # entry) while the published completed-point counter
            # stayed untouched — proven through a REAL step, where an
            # unrebased env would report the harvested count.
            snapshot = env._rules.snapshot()
            harvested = max(
                0,
                int(snapshot.net_crossing_count)
                - int(snapshot.feed_crossed_net),
            )
            assert harvested >= 1
            assert env._crossings_offset == -harvested
            assert env._crossings_base == 0
            _o, _r, _t, _tr, sinfo = env.step(_zero_action())
            assert sinfo["crossings"] == 0.0
            assert sinfo["completed_point_crossings"] == 0.0
        finally:
            env.close()

    def test_eligibility_policy_receiving_only_alternation_preserved(
        self, drill_library_path
    ):
        """D3 at fraction 1.0: every policy-receiving point launches
        drilled, every policy-serving point stays a drawn serve, and
        the serve-alternation ledger is untouched — read entirely
        from the info stream (the boundary step describes its own
        just-ended point, the pre-flip convention)."""
        env = PaddleTennisEnv(
            points_per_episode=None,
            drill_library=drill_library_path,
            drill_fraction=1.0,
        )
        try:
            obs, info = env.reset(seed=_SMOKE_SEEDS[1])
            assert info["serve_side"] == "a"
            assert info["drill_point"] == 0.0
            point_flags: list[tuple[float, float]] = []
            prev_points = 0
            while True:
                obs, _r, term, trunc, step_info = env.step(_zero_action())
                if int(step_info["points_played"]) != prev_points:
                    prev_points = int(step_info["points_played"])
                    point_flags.append(
                        (
                            step_info["serve_side_is_policy"],
                            step_info["drill_point"],
                        )
                    )
                if term or trunc:
                    break
            assert step_info["drill_fallback_count"] == 0.0
            assert len(point_flags) >= 4
            for position, (serve_policy, drilled) in enumerate(point_flags):
                assert serve_policy == (1.0 if position % 2 == 0 else 0.0)
                assert drilled == 1.0 - serve_policy
        finally:
            env.close()

    def test_launch_falls_back_to_a_drawn_serve_on_clearance_violation(
        self, drill_library_path, tmp_path
    ):
        """D5: the launch-time clearance check is unreachable for a
        well-formed library; a violating entry falls back to a drawn
        serve, loudly counted, instead of launching the ball into a
        paddle head."""
        probe = PaddleTennisEnv()
        adr = probe._ball_qposadr
        probe.close()
        with open(drill_library_path, "rb") as f:
            library = pickle.load(f)
        violating_entries = [dict(entry) for entry in library["entries"]]
        for entry in violating_entries:
            entry["head_a"] = entry["qpos"][adr : adr + 3].copy()
            # Plant the harvested A-paddle far from home: pristine
            # code never touches state before the clearance check, so
            # only an ordering violation can teleport the paddle
            # there — making the carryover assertion below a sharp
            # discriminator (the oracle-harvested slides sit near
            # home, which a lax teleport would not distinguish).
            entry["qpos"] = entry["qpos"].copy()
            entry["qpos"][0:3] = (-3.0, 2.0, 0.5)
        path = tmp_path / "violating.pkl"
        with open(path, "wb") as f:
            pickle.dump({**library, "entries": violating_entries}, f)

        env = PaddleTennisEnv(
            points_per_episode=None,
            drill_library=str(path),
            drill_fraction=1.0,
        )
        try:
            obs, info = env.reset(seed=_SMOKE_SEEDS[0], options={"serve_side": "b"})
            assert info["serve_side"] == "b"
            assert info["drill_point"] == 0.0
            assert info["drill_entry_index"] == -1.0
            # A genuine side-B serve draw, not a harvested state.
            assert np.asarray(info["serve_ball_position"])[0] > 0.0
            _obs, _r, _t, _tr, step_info = env.step(_zero_action())
            assert step_info["drill_fallback_count"] == 1.0

            # Mid-episode fallback preserves carryover: the clearance
            # check runs BEFORE any state mutation (§2a), so a
            # violating entry must leave the paddles where play left
            # them — never teleported to the harvested configuration.
            obs, info = env.reset(seed=_SMOKE_SEEDS[1])
            assert info["serve_side"] == "a"
            prev_points = 0
            head_before = None
            while True:
                head_before = env._paddle_position(CourtSide.A)
                obs, _r, term, trunc, step_info = env.step(_zero_action())
                assert not (term or trunc)
                if int(step_info["points_played"]) > prev_points:
                    break
            assert step_info["drill_fallback_count"] == 1.0
            head_after = env._paddle_position(CourtSide.A)
            assert float(np.linalg.norm(head_after - head_before)) < 0.5
            assert head_after[0] < 0.0  # never the harvested B-side pose

            # The fallback counter is per-episode (reset_model zeroes
            # it): a fresh policy-serving episode reads 0 (a leaked
            # counter would carry the previous episode's fallback),
            # and the next policy-receiving episode counts only its
            # OWN reset-launch fallback. A seeded reset serves A.
            _obs, info = env.reset(seed=_SMOKE_SEEDS[0])
            assert info["serve_side"] == "a"
            _o, _r, _t, _tr, step_info = env.step(_zero_action())
            assert step_info["drill_fallback_count"] == 0.0
            _obs, info = env.reset(seed=_SMOKE_SEEDS[0], options={"serve_side": "b"})
            assert info["serve_side"] == "b"
            _o, _r, _t, _tr, step_info = env.step(_zero_action())
            assert step_info["drill_fallback_count"] == 1.0
        finally:
            env.close()

    def test_full_arm_mid_episode_continuity(self, drill_library_path):
        """The full arm through real mid-episode play: restored rules
        machine + reattached sampler + continuity offset survive
        actual steps. The crossings counter never jumps at a drilled
        launch (non-decreasing, at most +1 per step), and the
        published ``completed_point_crossings`` keeps its
        completed-point meaning — never negative, always the
        crossings value of the most recent point-completing step."""
        env = PaddleTennisEnv(
            points_per_episode=None,
            drill_library=drill_library_path,
            drill_fraction=1.0,
            drill_context="full",
        )
        try:
            obs, info = env.reset(seed=_SMOKE_SEEDS[1])
            assert info["serve_side"] == "a"
            prev_crossings = 0.0
            last_boundary_crossings = 0.0
            prev_points = 0
            drilled_points = 0
            while True:
                obs, _r, term, trunc, sinfo = env.step(_zero_action())
                crossings = sinfo["crossings"]
                assert crossings >= prev_crossings
                assert crossings - prev_crossings <= 1.0
                assert sinfo["completed_point_crossings"] >= 0.0
                if int(sinfo["points_played"]) != prev_points:
                    prev_points = int(sinfo["points_played"])
                    last_boundary_crossings = crossings
                    if sinfo["drill_point"] == 1.0:
                        drilled_points += 1
                assert (
                    sinfo["completed_point_crossings"]
                    == last_boundary_crossings
                )
                prev_crossings = crossings
                if term or trunc:
                    break
            assert drilled_points >= 1  # a full-arm launch was played out
            assert sinfo["drill_fallback_count"] == 0.0
        finally:
            env.close()

    def test_intermediate_fraction_mixes_drilled_and_undrilled(
        self, drill_library_path
    ):
        """0 < drill_fraction < 1 (the §5 pilot's regime): the
        eligibility draw's false path actually executes — some
        policy-receiving points drill and some launch drawn serves,
        with clean provenance on both — while policy-serving points
        stay untouched."""
        env = PaddleTennisEnv(
            points_per_episode=None,
            drill_library=drill_library_path,
            drill_fraction=0.5,
        )
        try:
            receiving_flags: list[tuple[float, float]] = []
            # Both serve slots on one seed (the same seed twice with
            # no options would replay the identical episode).
            for index in range(2):
                obs, info = env.reset(
                    seed=_SMOKE_SEEDS[2], options=_serve_options(index)
                )
                prev_points = 0
                while True:
                    obs, _r, term, trunc, sinfo = env.step(_zero_action())
                    if int(sinfo["points_played"]) != prev_points:
                        prev_points = int(sinfo["points_played"])
                        if sinfo["serve_side_is_policy"] == 0.0:
                            receiving_flags.append(
                                (
                                    sinfo["drill_point"],
                                    sinfo["drill_entry_index"],
                                )
                            )
                        else:
                            assert sinfo["drill_point"] == 0.0
                            assert sinfo["drill_entry_index"] == -1.0
                    if term or trunc:
                        break
                assert sinfo["drill_fallback_count"] == 0.0
            drilled = [f for f in receiving_flags if f[0] == 1.0]
            undrilled = [f for f in receiving_flags if f[0] == 0.0]
            assert drilled, receiving_flags
            assert undrilled, receiving_flags
            assert all(entry >= 0.0 for _flag, entry in drilled)
            assert all(entry == -1.0 for _flag, entry in undrilled)
        finally:
            env.close()

    def test_rally_returns_restart_at_every_drilled_launch_in_both_arms(
        self, drill_library_path
    ):
        """episode_rally_returns_a counts the policy's k>=2 confirms
        per LAUNCHED point, so the two drill arms count the same
        thing. Arm "full" restores the harvested rules machine, whose
        valid_return_count_a already holds the harvest's own side-A
        returns (1-6 on this library); reading k from that snapshot
        counted the policy's FIRST return of a drilled point as a
        conversion under arm "full" only. The oracle plays side A
        through drilled policy-receiving points on both arms, and a
        recount whose k restarts at every launch must match the env
        on every step, and the two arms' episode totals agree."""
        restored: dict[str, list[int]] = {}
        totals: dict[str, tuple[float, float]] = {}
        for context in ("feed", "full"):
            env = PaddleTennisEnv(
                points_per_episode=None,
                drill_library=drill_library_path,
                drill_fraction=1.0,
                drill_context=context,
            )
            try:
                obs, info = env.reset(
                    seed=_SMOKE_SEEDS[2], options={"serve_side": "b"}
                )
                assert info["drill_point"] == 1.0
                k = valid = rally = prev_points = 0
                restored[context] = []
                while True:
                    obs, _r, term, trunc, sinfo = env.step(
                        scripted_ground_opponent(obs)
                    )
                    transition = env._last_transition
                    for side in transition.confirmed_returns:
                        if side is not CourtSide.A:
                            continue
                        valid += 1
                        k += 1
                        rally += k >= 2
                        if k == 1 and sinfo["drill_point"] == 1.0:
                            # The policy's first return of a drilled
                            # point: what the rules machine had
                            # already counted for side A this point.
                            restored[context].append(
                                transition.before.valid_return_count_a
                            )
                    assert sinfo["episode_valid_return_count_a"] == valid
                    assert sinfo["episode_rally_returns_a"] == rally
                    if int(sinfo["points_played"]) != prev_points:
                        prev_points = int(sinfo["points_played"])
                        k = 0  # the next point launches from k=0
                    if term or trunc:
                        break
                assert sinfo["drill_fallback_count"] == 0.0
                assert rally >= 1  # k>=2 conversions exercised
                totals[context] = (
                    sinfo["episode_valid_return_count_a"],
                    sinfo["episode_rally_returns_a"],
                )
            finally:
                env.close()
        # Both arms played a drilled point the policy returned; only
        # the full arm's restored machine had already counted side-A
        # returns there -- the case the snapshot-read k got wrong.
        assert restored["feed"] and restored["full"], restored
        assert all(count == 0 for count in restored["feed"]), restored
        assert all(count >= 1 for count in restored["full"]), restored
        # Measured on this seed: each arm plays a drilled point with
        # one policy return, then a drawn point with five (6 returns,
        # 4 conversions); the snapshot-read k gave arm "full" 5.
        assert totals["full"] == totals["feed"], totals

    def test_ezpickle_round_trip_keeps_drill_kwargs(
        self, drill_library_path
    ):
        """The standing EzPickle regression class (the WallBall
        precedent): a pickled clone must keep the drill configuration
        and reload the library — a kwarg dropped from the EzPickle
        call would silently run vectorized workers with the drill
        off."""
        env = PaddleTennisEnv(
            drill_library=drill_library_path,
            drill_fraction=0.5,
            drill_context="full",
        )
        try:
            clone = pickle.loads(pickle.dumps(env))
            try:
                assert clone.drill_library == drill_library_path
                assert clone.drill_fraction == 0.5
                assert clone.drill_context == "full"
                assert len(clone._drill_entries) == len(env._drill_entries)
            finally:
                clone.close()
        finally:
            env.close()

    def test_recipe_pins_the_eval_task_drill_free(
        self, drill_library_path, tmp_path
    ):
        """D6: a run config's [env] drill kwargs reach the training
        env and never the eval env — selection, periodic/final eval,
        and the checkpoint diagnosis stay on the frozen task."""
        from courtside_dynamics.recipes import build_train_config

        config = tmp_path / "drill_pilot.toml"
        config.write_text(
            f'[env]\ndrill_library = "{drill_library_path}"\n'
            "drill_fraction = 0.5\n"
        )
        cfg = build_train_config(
            "PaddleTennis",
            total_timesteps=1_000,
            log_dir=str(tmp_path / "run"),
            config_file=str(config),
        )
        train_env = cfg.env_fn()
        try:
            assert train_env.drill_fraction == 0.5
            assert train_env.drill_library_sha256 is not None
        finally:
            train_env.close()
        eval_env = (cfg.eval_env_fn or cfg.env_fn)()
        try:
            assert eval_env.drill_fraction == 0.0
            assert eval_env.drill_library is None
            assert eval_env._drill_entries is None
        finally:
            eval_env.close()
