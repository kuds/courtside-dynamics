"""Focused tests for held-out humanoid-tennis curriculum promotion."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from courtside_dynamics.envs.tennis_rules import CourtSide
from courtside_dynamics.training.tennis_curriculum import (
    CurriculumEpisodeResult,
    EvaluationProvenance,
    HeldOutCondition,
    HeldOutSeedSuite,
    PromotionConfig,
    assess_curriculum_promotion,
    evaluate_curriculum_stage,
    summarize_curriculum_episodes,
)


def _terminal_info(**overrides: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "stage_success": False,
        "rally_count": 0,
        "valid_return_count": 0,
        "return_opportunity_count": 1,
        "event_count_ball_net": 0,
        "termination_reason_name": "out_of_bounds",
        "term_ball_net": False,
        "term_second_bounce": False,
        "term_out_of_bounds": True,
        "term_humanoid_net": False,
        "term_racket_net": False,
    }
    info.update(overrides)
    return info


class _OneStepEvalEnv:
    """Tiny deterministic Gym-shaped env used to test the evaluator."""

    def __init__(
        self,
        info_for_condition,
        *,
        stage: int = 2,
        stage_name: str = "anchored_randomized_return",
    ) -> None:
        self.info_for_condition = info_for_condition
        self.resets: list[tuple[int, str]] = []
        self.closed = False
        self.condition: tuple[int, str] | None = None
        self.curriculum_metadata = {
            "curriculum_enabled": True,
            "curriculum_stage": stage,
            "curriculum_stage_name": stage_name,
            "fixture": "one_step",
        }

    def reset(self, *, seed: int, options: dict[str, Any]):
        side = options["serve_side"]
        self.condition = (seed, side)
        self.resets.append(self.condition)
        sign = 1.0 if side == "a" else -1.0
        position = np.asarray(options.get("ball_position", (-1.0 * sign, 0.0, 1.3)))
        velocity = np.asarray(options.get("ball_velocity", (6.0 * sign, 0.0, 3.0)))
        reset_info = {
            "curriculum_stage": self.curriculum_metadata["curriculum_stage"],
            "curriculum_stage_name": self.curriculum_metadata[
                "curriculum_stage_name"
            ],
            "initial_ball_qpos": (*position, 1.0, 0.0, 0.0, 0.0),
            "initial_ball_qvel": (*velocity, 0.0, 0.0, 0.0),
        }
        return np.asarray([seed, 0.0 if side == "a" else 1.0]), reset_info

    def step(self, action):
        del action
        assert self.condition is not None
        info = self.info_for_condition(*self.condition)
        return np.zeros(2), 1.0 if info.get("stage_success") else -1.0, True, False, info

    def close(self) -> None:
        self.closed = True


def _episode(
    index: int,
    *,
    success: bool,
    rally_count: int = 0,
    valid_returns: int = 0,
    opportunities: int = 1,
    faults: tuple[str, ...] = (),
    termination_reason: str | None = None,
) -> CurriculumEpisodeResult:
    return CurriculumEpisodeResult(
        condition=HeldOutCondition(
            seed=60_000 + index // 2,
            serving_side=CourtSide.A if index % 2 == 0 else CourtSide.B,
        ),
        steps=1,
        total_reward=1.0 if success else -1.0,
        initial_ball_qpos=(
            float(index + 1),
            float(index % 2),
            1.3,
            1.0,
            0.0,
            0.0,
            0.0,
        ),
        initial_ball_qvel=(6.0, 0.0, 3.0, 0.0, 0.0, 0.0),
        stage_success=success,
        rally_count=rally_count,
        valid_return_count=valid_returns,
        return_opportunity_count=opportunities,
        terminated=True,
        truncated=False,
        termination_reason=(
            termination_reason
            if termination_reason is not None
            else ("stage_success" if success else "out_of_bounds")
        ),
        fault_flags=tuple(sorted(faults)),
    )


def _provenance(
    stage: int,
    stage_name: str,
    *,
    policy_id: str = "policy-sha256:test",
    normalization_id: str = "raw",
) -> EvaluationProvenance:
    normalized = normalization_id != "raw"
    return EvaluationProvenance(
        curriculum_stage=stage,
        curriculum_stage_name=stage_name,
        curriculum_fingerprint=f"{stage + 1:064x}",
        evaluation_suite_fingerprint="a" * 64,
        canonical_evaluation_spec_id=(
            f"humanoid-tennis-stage-{stage}-promotion-v1"
        ),
        environment_class="tests.OneStepEnv",
        policy_id=policy_id,
        policy_state_sha256="b" * 64,
        policy_artifact_sha256="c" * 64,
        normalization_id=normalization_id,
        normalization_artifact_sha256="d" * 64 if normalized else None,
        observation_transform_name="tests.normalize" if normalized else None,
        observation_transform_sha256="e" * 64 if normalized else None,
        package_version="test",
        package_source_sha256="f" * 64,
        git_sha="abc123",
        runtime_versions=(("mujoco", "test"), ("python", "test")),
        runtime_fingerprint="0" * 64,
    )


def _summary(stage: int, successes: int, total: int = 100):
    stage_name = {
        0: "anchored_racket_intercept",
        1: "anchored_return",
        2: "anchored_randomized_return",
    }.get(stage, f"stage_{stage}")
    episodes = tuple(
        _episode(index, success=index < successes) for index in range(total)
    )
    return summarize_curriculum_episodes(
        stage_name,
        episodes,
        provenance=_provenance(stage, stage_name),
    )


def test_default_suite_is_50_exact_mirrored_launches():
    suite = HeldOutSeedSuite()
    assert suite.base_seed == 50_000
    assert suite.launch_count == 50
    assert suite.mirror_each is True
    assert suite.episode_count == 100
    assert [(item.seed, item.serving_side) for item in suite.conditions[:4]] == [
        (50_000, CourtSide.A),
        (50_000, CourtSide.B),
        (50_001, CourtSide.A),
        (50_001, CourtSide.B),
    ]
    assert suite.conditions[0].local_position_offset == (
        suite.conditions[1].local_position_offset
    )
    assert len({item.local_position_offset for item in suite.conditions}) == 50
    assert [(item.seed, item.serving_side) for item in suite.conditions[-2:]] == [
        (50_049, CourtSide.A),
        (50_049, CourtSide.B),
    ]


def test_unmirrored_suite_alternates_sides():
    suite = HeldOutSeedSuite(
        base_seed=10,
        launch_count=3,
        mirror_each=False,
        max_position_offset=(0.0, 0.0, 0.0),
        max_velocity_offset=(0.0, 0.0, 0.0),
    )
    assert suite.episode_count == 3
    assert suite.conditions == (
        HeldOutCondition(10, CourtSide.A),
        HeldOutCondition(11, CourtSide.B),
        HeldOutCondition(12, CourtSide.A),
    )


def test_promotion_config_defaults_and_validation():
    config = PromotionConfig()
    assert config.success_threshold == 0.80
    assert config.prior_retention_threshold == 0.75
    assert config.min_episodes == 100
    assert config.min_unique_initial_states == 100
    assert config.prior_replay_probability == 0.20
    assert config.require_zero_unsafe is True

    with pytest.raises(ValueError):
        PromotionConfig(success_threshold=1.1)
    with pytest.raises(ValueError):
        PromotionConfig(prior_retention_threshold=-0.1)
    with pytest.raises(ValueError):
        PromotionConfig(prior_replay_probability=float("nan"))
    with pytest.raises(ValueError):
        PromotionConfig(min_episodes=0)
    with pytest.raises(ValueError):
        PromotionConfig(min_unique_initial_states=0)


def test_evaluator_uses_exact_schedule_and_ratio_of_sums():
    outcomes = {
        (100, "a"): _terminal_info(
            stage_success=True,
            rally_count=0,
            valid_return_count=1,
            return_opportunity_count=1,
            event_count_ball_net=1,
            term_out_of_bounds=False,
            termination_reason_name="stage_success",
        ),
        (100, "b"): _terminal_info(
            rally_count=1,
            valid_return_count=0,
            return_opportunity_count=3,
            term_out_of_bounds=False,
            term_second_bounce=True,
            termination_reason_name="second_bounce",
        ),
        (101, "a"): _terminal_info(
            stage_success=True,
            rally_count=2,
            valid_return_count=9,
            return_opportunity_count=9,
            term_out_of_bounds=False,
            term_humanoid_net=True,
            termination_reason_name="stage_success",
        ),
        (101, "b"): _terminal_info(
            rally_count=10,
            valid_return_count=0,
            return_opportunity_count=87,
            term_out_of_bounds=False,
            term_racket_net=True,
            termination_reason_name="racket_b_net",
        ),
    }
    created: list[_OneStepEvalEnv] = []

    def factory():
        env = _OneStepEvalEnv(lambda seed, side: outcomes[(seed, side)])
        created.append(env)
        return env

    suite = HeldOutSeedSuite(
        base_seed=100,
        launch_count=2,
        max_position_offset=(0.0, 0.0, 0.0),
        max_velocity_offset=(0.0, 0.0, 0.0),
    )
    summary = evaluate_curriculum_stage(
        lambda observation: np.zeros(1, dtype=np.float32),
        factory,
        policy_id="fixture-policy-v1",
        normalization_id="raw",
        stage_name="anchored_randomized_return",
        suite=suite,
    )

    assert len(created) == 1
    assert created[0].closed is True
    assert created[0].resets == [
        (100, "a"),
        (100, "b"),
        (101, "a"),
        (101, "b"),
    ]
    assert summary.conditions == suite.conditions
    assert summary.provenance.curriculum_stage == 2
    assert summary.provenance.policy_id == "fixture-policy-v1"
    assert summary.provenance.normalization_id == "raw"
    assert len(summary.provenance.curriculum_fingerprint) == 64
    assert len(summary.provenance.evaluation_suite_fingerprint) == 64
    assert len(summary.provenance.package_source_sha256) == 64
    assert len(summary.provenance.runtime_fingerprint) == 64
    assert {
        "python",
        "mujoco",
        "gymnasium",
        "numpy",
        "torch",
        "stable_baselines3",
    } <= dict(summary.provenance.runtime_versions).keys()
    assert summary.provenance.canonical_evaluation_spec_id is None
    assert summary.success_count == 2
    assert summary.success_rate == 0.5
    assert summary.valid_return_count == 10
    assert summary.return_opportunity_count == 100
    # Ratio of sums: 10 / 100, not the mean of per-episode ratios.
    assert summary.valid_return_rate == pytest.approx(0.1)
    assert summary.rally_min == 0.0
    assert summary.rally_p50 == 1.5
    assert summary.rally_p90 == pytest.approx(7.6)
    assert summary.rally_max == 10.0
    assert summary.fault_rate("ball_net") == 0.25
    assert summary.fault_rate("second_bounce") == 0.25
    assert summary.fault_rate("out_of_bounds") == 0.0
    assert summary.fault_rate("humanoid_net") == 0.25
    assert summary.fault_rate("racket_net") == 0.25
    assert {item.name: item.rate for item in summary.termination_rates} == {
        "racket_b_net": 0.25,
        "second_bounce": 0.25,
    }


def test_predict_compatible_policy_is_deterministic(tmp_path):
    class PredictPolicy:
        def __init__(self) -> None:
            self.deterministic_values: list[bool] = []

        def predict(self, observation, *, deterministic: bool):
            del observation
            self.deterministic_values.append(deterministic)
            return np.zeros(1, dtype=np.float32), None

        def get_parameters(self):
            return {"policy": {"weight": np.asarray([1.0, 2.0])}}

        @classmethod
        def load(cls, _path, *, device):
            assert device == "cpu"
            return cls()

    policy = PredictPolicy()
    policy_path = tmp_path / "policy.zip"
    policy_path.write_bytes(b"checkpoint fixture")
    env = _OneStepEvalEnv(
        lambda _seed, _side: _terminal_info(
            stage_success=True,
            valid_return_count=1,
            return_opportunity_count=1,
            term_out_of_bounds=False,
            termination_reason_name="stage_success",
        ),
        stage=0,
        stage_name="anchored_racket_intercept",
    )
    summary = evaluate_curriculum_stage(
        policy,
        lambda: env,
        policy_id="predict-policy-v1",
        policy_artifact_path=policy_path,
        normalization_id="raw",
        stage_name="anchored_racket_intercept",
        suite=HeldOutSeedSuite(launch_count=1),
    )
    assert summary.success_rate == 1.0
    assert policy.deterministic_values == [True, True]
    assert summary.provenance.policy_state_sha256 is not None
    assert summary.provenance.policy_artifact_sha256 is not None


def test_missing_terminal_success_fails_loudly_and_closes_env():
    info = _terminal_info()
    del info["stage_success"]
    env = _OneStepEvalEnv(
        lambda _seed, _side: info,
        stage=0,
        stage_name="anchored_racket_intercept",
    )
    with pytest.raises(KeyError, match="stage_success"):
        evaluate_curriculum_stage(
            lambda observation: np.zeros(1),
            lambda: env,
            policy_id="fixture-policy-v1",
            normalization_id="raw",
            stage_name="anchored_racket_intercept",
            suite=HeldOutSeedSuite(launch_count=1),
        )
    assert env.closed is True


def test_policy_artifact_must_match_evaluated_policy(tmp_path):
    class MismatchedPolicy:
        def __init__(self, weight: float = 1.0) -> None:
            self.weight = weight

        def predict(self, _observation, *, deterministic: bool):
            assert deterministic
            return np.zeros(1, dtype=np.float32), None

        def get_parameters(self):
            return {"policy": {"weight": np.asarray([self.weight])}}

        @classmethod
        def load(cls, _path, *, device):
            assert device == "cpu"
            return cls(weight=2.0)

    policy_path = tmp_path / "policy.zip"
    policy_path.write_bytes(b"mismatched checkpoint fixture")
    env = _OneStepEvalEnv(
        lambda _seed, _side: _terminal_info(),
        stage=0,
        stage_name="anchored_racket_intercept",
    )
    with pytest.raises(ValueError, match="do not match the evaluated policy"):
        evaluate_curriculum_stage(
            MismatchedPolicy(),
            lambda: env,
            policy_id="mismatched-policy",
            policy_artifact_path=policy_path,
            normalization_id="raw",
            stage_name="anchored_racket_intercept",
        )
    assert env.closed is True


def test_evaluator_applies_and_records_observation_transform(tmp_path):
    import gymnasium as gym
    from gymnasium.spaces import Box
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class NormalizationEnv(gym.Env):
        observation_space = Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
        action_space = Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            del seed, options
            return np.zeros(2), {}

        def step(self, action):
            del action
            return np.zeros(2), 0.0, False, False, {}

    seen: list[np.ndarray] = []

    def policy(observation):
        seen.append(np.asarray(observation).copy())
        return np.zeros(1, dtype=np.float32)

    env = _OneStepEvalEnv(
        lambda _seed, _side: _terminal_info(
            stage_success=True,
            term_out_of_bounds=False,
            termination_reason_name="stage_success",
        ),
        stage=0,
        stage_name="anchored_racket_intercept",
    )
    suite = HeldOutSeedSuite(
        base_seed=123,
        launch_count=1,
        max_position_offset=(0.0, 0.0, 0.0),
        max_velocity_offset=(0.0, 0.0, 0.0),
    )
    normalization_path = tmp_path / "vec_normalize.pkl"
    normalizer = VecNormalize(
        DummyVecEnv([NormalizationEnv]),
        training=False,
        norm_obs=True,
        norm_reward=False,
    )
    normalizer.obs_rms.mean = np.asarray([100.0, -2.0])
    normalizer.obs_rms.var = np.asarray([4.0, 9.0])
    expected = [
        normalizer.normalize_obs(np.asarray([123.0, 0.0])),
        normalizer.normalize_obs(np.asarray([123.0, 1.0])),
    ]
    normalizer.save(str(normalization_path))
    normalizer.close()
    summary = evaluate_curriculum_stage(
        policy,
        lambda: env,
        policy_id="normalized-policy-v1",
        normalization_id="vecnormalize-sha256:abc",
        normalization_artifact_path=normalization_path,
        stage_name="anchored_racket_intercept",
        suite=suite,
    )

    assert len(seen) == len(expected)
    for actual, expected_observation in zip(seen, expected, strict=True):
        np.testing.assert_allclose(actual, expected_observation)
    assert summary.provenance.normalization_id == "vecnormalize-sha256:abc"
    assert summary.provenance.normalization_artifact_sha256 is not None
    assert summary.provenance.observation_transform_name is not None
    assert summary.provenance.observation_transform_name.endswith(
        "VecNormalize.normalize_obs"
    )


def test_evaluator_rejects_ambiguous_normalization_contracts(tmp_path):
    import pickle

    def factory():
        pytest.fail("invalid arguments must fail before env creation")

    def policy(_observation):
        return np.zeros(1, dtype=np.float32)

    with pytest.raises(ValueError, match="non-raw normalization"):
        evaluate_curriculum_stage(
            policy,
            factory,
            policy_id="policy-v1",
            normalization_id="vecnormalize:abc",
        )
    with pytest.raises(ValueError, match="cannot use"):
        evaluate_curriculum_stage(
            policy,
            factory,
            policy_id="policy-v1",
            normalization_id="raw",
            normalization_artifact_path=tmp_path / "unused.pkl",
        )
    invalid_path = tmp_path / "not_vec_normalize.pkl"
    invalid_path.write_bytes(pickle.dumps({"not": "a normalizer"}))
    with pytest.raises(TypeError, match="must contain an SB3 VecNormalize"):
        evaluate_curriculum_stage(
            policy,
            factory,
            policy_id="policy-v1",
            normalization_id="vecnormalize:abc",
            normalization_artifact_path=invalid_path,
        )


def test_evaluator_binds_stage_to_environment_and_closes_on_mismatch():
    env = _OneStepEvalEnv(
        lambda _seed, _side: _terminal_info(),
        stage=0,
        stage_name="anchored_racket_intercept",
    )
    with pytest.raises(ValueError, match="contradicts environment stage"):
        evaluate_curriculum_stage(
            lambda _observation: np.zeros(1, dtype=np.float32),
            lambda: env,
            policy_id="policy-v1",
            normalization_id="raw",
            stage_name="anchored_return",
        )
    assert env.closed is True


def test_default_suite_produces_100_unique_physical_initial_states():
    env = _OneStepEvalEnv(
        lambda _seed, _side: _terminal_info(
            stage_success=True,
            term_out_of_bounds=False,
            termination_reason_name="stage_success",
        ),
        stage=0,
        stage_name="anchored_racket_intercept",
    )
    summary = evaluate_curriculum_stage(
        lambda _observation: np.zeros(1, dtype=np.float32),
        lambda: env,
        policy_id="policy-v1",
        normalization_id="raw",
        stage_name="anchored_racket_intercept",
    )

    assert len(summary.episodes) == 100
    assert summary.unique_initial_state_count == 100


def test_current_and_prior_threshold_boundaries_are_exact():
    current = _summary(2, 80)
    stage0 = _summary(0, 75)
    stage1 = _summary(1, 75)
    report = assess_curriculum_promotion(
        current,
        prior_stages=(stage0, stage1),
    )
    assert report.current_stage_passed is True
    assert report.prior_retention_passed is True
    assert report.eligible_for_manual_promotion is True
    assert report.to_dict()["advisory_only"] is True
    assert "explicitly" in report.reasons[-1]

    current_fail = assess_curriculum_promotion(
        _summary(2, 79),
        prior_stages=(stage0, stage1),
    )
    assert current_fail.current_stage_passed is False
    assert current_fail.eligible_for_manual_promotion is False

    retention_fail = assess_curriculum_promotion(
        current,
        prior_stages=(stage0, _summary(1, 74)),
    )
    assert retention_fail.current_stage_passed is True
    assert retention_fail.prior_retention_passed is False
    assert retention_fail.eligible_for_manual_promotion is False


def test_promotion_rejects_too_few_episodes_for_any_gate():
    with pytest.raises(ValueError, match="requires at least 100"):
        assess_curriculum_promotion(
            _summary(1, 99, total=99),
            prior_stages=(_summary(0, 80),),
        )

    with pytest.raises(ValueError, match="requires at least 100"):
        assess_curriculum_promotion(
            _summary(2, 80),
            prior_stages=(
                _summary(0, 80),
                _summary(1, 50, total=50),
            ),
        )


def test_promotion_requires_every_implemented_predecessor():
    current = _summary(2, 80)
    with pytest.raises(ValueError, match=r"missing=\[0, 1\]"):
        assess_curriculum_promotion(current)
    with pytest.raises(ValueError, match=r"missing=\[0\]"):
        assess_curriculum_promotion(
            current,
            prior_stages=(_summary(1, 80),),
        )


def test_promotion_rejects_noncanonical_or_unverified_evidence():
    current = _summary(0, 80)
    noncanonical = replace(
        current,
        provenance=replace(
            current.provenance,
            canonical_evaluation_spec_id=None,
        ),
    )
    with pytest.raises(ValueError, match="canonical promotion spec"):
        assess_curriculum_promotion(noncanonical)

    unverified = replace(
        current,
        provenance=replace(
            current.provenance,
            policy_state_sha256=None,
        ),
    )
    with pytest.raises(ValueError, match="fingerprintable live SB3 policy"):
        assess_curriculum_promotion(unverified)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("policy_id", "different-policy"),
        ("policy_state_sha256", "1" * 64),
        ("policy_artifact_sha256", "2" * 64),
        ("evaluation_suite_fingerprint", "3" * 64),
        ("environment_class", "different.Env"),
        ("package_version", "different-version"),
        ("package_source_sha256", "4" * 64),
        ("git_sha", "different-sha"),
        ("runtime_fingerprint", "5" * 64),
    ],
)
def test_promotion_rejects_incompatible_evidence_provenance(field, value):
    current = _summary(1, 80)
    prior = _summary(0, 80)
    incompatible = replace(
        prior,
        provenance=replace(prior.provenance, **{field: value}),
    )
    with pytest.raises(ValueError, match="same policy"):
        assess_curriculum_promotion(
            current,
            prior_stages=(incompatible,),
        )


def test_promotion_rejects_pseudoreplicated_initial_states():
    first = _episode(0, success=True)
    episodes = tuple(
        replace(
            _episode(index, success=index < 80),
            initial_ball_qpos=first.initial_ball_qpos,
            initial_ball_qvel=first.initial_ball_qvel,
        )
        for index in range(100)
    )
    summary = summarize_curriculum_episodes(
        "anchored_racket_intercept",
        episodes,
        provenance=_provenance(0, "anchored_racket_intercept"),
    )
    assert summary.unique_initial_state_count == 1
    with pytest.raises(ValueError, match="only 1 unique initial states"):
        assess_curriculum_promotion(summary)


def test_unsafe_episode_blocks_promotion_even_at_success_threshold():
    episodes = tuple(
        _episode(
            index,
            success=index < 80,
            faults=("unsafe",) if index == 0 else (),
        )
        for index in range(100)
    )
    summary = summarize_curriculum_episodes(
        "anchored_return",
        episodes,
        provenance=_provenance(1, "anchored_return"),
    )
    assert summary.success_rate == 0.80
    assert summary.fault_rate("unsafe") == 0.01
    stage0 = _summary(0, 80)
    report = assess_curriculum_promotion(summary, prior_stages=(stage0,))
    assert report.current_stage_passed is False
    assert report.eligible_for_manual_promotion is False
    assert any("unsafe rate" in reason for reason in report.reasons)

    opted_out = assess_curriculum_promotion(
        summary,
        prior_stages=(stage0,),
        config=PromotionConfig(require_zero_unsafe=False),
    )
    assert opted_out.eligible_for_manual_promotion is True


def test_public_results_are_frozen_and_json_compatible():
    summary = _summary(0, 80)
    report = assess_curriculum_promotion(summary)
    with pytest.raises(FrozenInstanceError):
        report.eligible_for_manual_promotion = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        summary.success_rate = 0.0  # type: ignore[misc]

    payload = report.to_dict()
    assert payload["current"]["unique_initial_state_count"] == 100
    assert payload["current"]["provenance"]["policy_id"] == (
        "policy-sha256:test"
    )
    assert payload["current"]["provenance"]["git_sha"] == "abc123"
    assert payload["current"]["provenance"]["runtime_versions"] == {
        "mujoco": "test",
        "python": "test",
    }
    assert payload["current"]["provenance"]["canonical_evaluation_spec_id"] == (
        "humanoid-tennis-stage-0-promotion-v1"
    )
    assert payload["current"]["episodes"][0]["seed"] == 60_000
    assert payload["current"]["episodes"][0]["serving_side"] == "a"
    assert payload["current"]["rally_distribution"] == {
        "min": 0.0,
        "p50": 0.0,
        "p90": 0.0,
        "max": 0.0,
    }


def test_invalid_episode_and_duplicate_stage_names_are_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        _episode(
            0,
            success=True,
            valid_returns=2,
            opportunities=1,
        )

    current = _summary(1, 80)
    with pytest.raises(ValueError, match="unique"):
        assess_curriculum_promotion(
            current,
            prior_stages=(_summary(1, 75),),
        )
