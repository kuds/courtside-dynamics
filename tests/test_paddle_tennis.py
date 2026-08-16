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
            for seed in _SMOKE_SEEDS:
                obs, _ = env.reset(seed=seed)
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

    def test_bad_opponent_action_fails_loudly(self):
        env = PaddleTennisEnv(opponent_controller=lambda observation: np.zeros(2))
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
)


class TestContactShaping:
    """The escrow contract of design_paddle_tennis_contact_shaping.md:
    pay at a side-A legal hit, keep on confirm, claw back on EVERY
    ending path — with the default-off stream bit-identical to the
    frozen task."""

    @staticmethod
    def _drive(env, policy, seed, mirror_env=None):
        """Step ``env`` (and optionally ``mirror_env`` in lockstep)
        with ``policy``; return per-step info sums and totals."""
        obs, _ = env.reset(seed=seed)
        if mirror_env is not None:
            mirror_obs, _ = mirror_env.reset(seed=seed)
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
            for seed in _SMOKE_SEEDS:
                totals, sums, confirms = self._drive(
                    shaped, scripted_ground_opponent, seed, unshaped
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

    def test_recipe_does_not_enable_shaping_yet(self):
        """The frozen task ships with shaping OFF until the L1 pilot
        verdict (design doc §3); the recipe must not flip it early."""
        from courtside_dynamics.recipes import RECIPES

        env_kwargs = RECIPES["PaddleTennis"].env_kwargs
        assert "contact_shaping" not in env_kwargs


#: Off-line camper for the reach-shaping anti-farming witness: parked
#: near the serve landing depth (side-local x ~= -4.55) but offset ~1 m
#: laterally, so bounces pay proximity while the ball never reaches the
#: paddle face.
_REACH_CAMPER_ACTION = np.array([-0.61, 0.33, 0.0])


class TestReachShaping:
    """The escrow contract of design_paddle_tennis_reach_shaping.md:
    pay at side A's live first bounce by proximity, keep on the side-A
    legal hit that takes the opportunity, claw back on EVERY ending
    path (point boundaries included) — with the default-off stream
    bit-identical to the frozen task."""

    @staticmethod
    def _drive(env, policy, seed, mirror_env=None):
        """Step ``env`` (optionally with a lockstep mirror); return
        totals, per-component sums, and the tracker-computed kept
        escrow (commit-before-pay ordering, the implementation's)."""
        obs, _ = env.reset(seed=seed)
        if mirror_env is not None:
            mirror_obs, _ = mirror_env.reset(seed=seed)
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
                kept += pending
                pending = 0.0
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
            for seed in _SMOKE_SEEDS:
                totals, sums, kept, _hits = self._drive(
                    shaped, scripted_ground_opponent, seed, unshaped
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
            for seed in _SMOKE_SEEDS:
                totals, sums, kept, hits = self._drive(
                    shaped, lambda _obs: _zero_action(), seed, plain
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

    def test_camper_collects_nothing_net(self):
        """The anti-farming witness: parked near the landing, off the
        ball line — proximity is paid every receiving bounce and
        clawed back in full (no hit ever keeps it)."""
        env = PaddleTennisEnv(reach_shaping=0.25)
        try:
            paid_any = False
            for seed in _SMOKE_SEEDS:
                totals, sums, kept, hits = self._drive(
                    env, lambda _obs: _REACH_CAMPER_ACTION, seed
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
            for seed in _SMOKE_SEEDS:
                obs, _ = shaped.reset(seed=seed)
                mirror_obs, _ = unshaped.reset(seed=seed)
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
                        kept_reach += pending_reach
                        pending_reach = 0.0
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

    def test_recipe_does_not_enable_reach_yet(self):
        """Reach shaping ships OFF until the LR1 pilot verdict
        (design doc §4); the recipe must not flip it early."""
        from courtside_dynamics.recipes import RECIPES

        env_kwargs = RECIPES["PaddleTennis"].env_kwargs
        assert "reach_shaping" not in env_kwargs


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
            for seed in _SMOKE_SEEDS[:2]:
                env.reset(seed=seed)
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

    def test_recipe_does_not_enable_npoint_yet(self):
        from courtside_dynamics.recipes import RECIPES

        assert "points_per_episode" not in RECIPES["PaddleTennis"].env_kwargs

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
