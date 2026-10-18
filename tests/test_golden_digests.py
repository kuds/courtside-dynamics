"""The golden env-stream digests: the gate every env refactor must pass.

``tests/golden/env_digests.json`` holds a digest of every CI case in
:mod:`courtside_dynamics.digests` (obs, reward, terminated, truncated,
qpos and qvel at every reset and step, every info item, and the
spaces and value types), recorded by ``tools/golden_digest.py
--write``. A failure here is an era break or physics drift: declare it
(new task version or a CHANGELOG/DECISIONS entry) before re-recording,
never re-record to make the test pass.

The streams cannot see everything a refactor might move. Float addition
is not associative, and a real PaddleTennis step rarely puts three
nonzero terms on one reward (never all nine), so the association tests
below force all of them at once with adversarial magnitudes and pin
the env's left-to-right order directly.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import gymnasium
import numpy as np
import pytest

from courtside_dynamics.digests import (
    CI_CASES,
    DECLARED_DRIFT,
    FULL_CASES,
    MISMATCH_EXPLANATION,
    PADDLE_TENNIS_REWARD_INFO_KEYS,
    RESET_MODES,
    StreamCase,
    _numpy_umath,
    _oracle_policy,
    digest_mismatches,
    expected_digest,
    format_mismatch,
    info_value_bytes,
    iter_stream,
    left_to_right_sum,
    numpy_simd_dispatch,
    order_sensitive_swaps,
    stream_digest,
    version_info,
)
from courtside_dynamics.envs import BallBalanceEnv, CourtSide, PaddleTennisEnv
from courtside_dynamics.envs._paddle_court import scripted_lead_charge_opponent
from courtside_dynamics.envs.tennis_rules import (
    RallyEvent,
    RallyEventKind,
    TerminationReason,
)
from tools import golden_digest
from tools._checkout import checkout_package_mismatch

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "env_digests.json"

#: The nonfinite guard's reward terms, in the order its sum adds them.
_NONFINITE_TERMS = (
    "rew_unsafe",
    "rew_shaping_clawback",
    "rew_reach_clawback",
    "rew_hold_clawback",
)


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)


def _swaps_but_the_first(count: int) -> list[tuple[int, int]]:
    """Every transposition of ``count`` terms except ``(0, 1)``.

    Exchanging the first two terms can never move a left-to-right sum
    (float addition commutes), so it is the one reordering that is
    harmless by construction.
    """
    return [pair for pair in itertools.combinations(range(count), 2) if pair != (0, 1)]


@pytest.fixture(scope="module")
def golden() -> dict:
    return golden_digest.load_golden(GOLDEN_PATH)


def test_golden_file_is_the_tools_file():
    assert golden_digest.GOLDEN_PATH == GOLDEN_PATH


@pytest.mark.parametrize("set_name", ["ci", "full"])
def test_golden_records_exactly_the_defined_cases(golden, set_name):
    """No stale entry, no unrecorded case: either drifts silently."""
    cases = {"ci": CI_CASES, "full": FULL_CASES}[set_name]
    assert sorted(golden["sets"][set_name]) == sorted(case.name for case in cases)
    # Entries recorded before version_info() gained numpy_simd lack it.
    required = version_info().keys() - {"numpy_simd"}
    for name, entry in golden["sets"][set_name].items():
        assert "stream_sha256" in entry, f"{name} has no primary digest"
        assert required <= entry["versions"].keys() <= version_info().keys(), name
        for label, slot in entry.get("declared_drift", {}).items():
            assert name in DECLARED_DRIFT[label], (name, label)
            assert slot["reason"] == DECLARED_DRIFT[label][name]


@pytest.mark.parametrize("case", CI_CASES, ids=lambda case: case.name)
def test_stream_digest_matches_golden(golden, case):
    live = version_info()
    expected, recorded, drift = expected_digest(
        golden["sets"]["ci"][case.name], case.name, live
    )
    lines = digest_mismatches(expected, stream_digest(case))
    assert not lines, format_mismatch("ci", case.name, lines, recorded, live, drift)


def test_paddle_cases_cover_all_nine_reward_terms():
    paddle = [case for case in CI_CASES if case.recipe == "PaddleTennis"]
    assert len(paddle) == 14
    for case in paddle:
        assert case.required_info_keys == PADDLE_TENNIS_REWARD_INFO_KEYS
        # Every step's reward must be their left-to-right sum, bit for bit.
        assert case.reward_terms == PADDLE_TENNIS_REWARD_INFO_KEYS
        assert case.info_keys is None  # every info item is digested


@pytest.mark.parametrize(
    "name, clawbacks, order_sensitive",
    [
        (
            "paddle_tennis/recipe/oracle_nan_contact_escrow",
            {"rew_shaping_clawback": -0.25},
            [],
        ),
        (
            "paddle_tennis/hold/oracle_nan_reach_hold_escrow",
            {
                "rew_reach_clawback": -0.17534702856592194,
                "rew_hold_clawback": -0.009889609459489718,
            },
            [],
        ),
        (
            # Moving the hold clawback ahead of the unsafe penalty, the
            # (zero) contact clawback or the reach clawback moves the bits.
            "paddle_tennis/hold/oracle_nan_order_sensitive",
            {
                "rew_reach_clawback": -0.17532310086941538,
                "rew_hold_clawback": -0.019691534990717713,
            },
            [(0, 3), (1, 3), (2, 3)],
        ),
    ],
)
def test_nan_witness_lands_on_a_pending_escrow(name, clawbacks, order_sensitive):
    """The NaN action must hit while the named escrows hold a payment.

    Re-proves the witness calibration from the stream itself: the NaN
    step ends the episode through the nonfinite guard, pays the unsafe
    penalty, and claws back exactly what was pending, summed in the
    guard's own order. The last witness's reward depends on that order.
    """
    (case,) = [case for case in CI_CASES if case.name == name]
    nan_step = next(
        record for record in iter_stream(case) if record.step == case.nan_action_step
    )
    assert nan_step.action is not None and np.isnan(nan_step.action).all()
    assert nan_step.terminated and not nan_step.truncated
    info = nan_step.info
    assert info["term_nonfinite"] == 1.0
    assert info["rew_unsafe"] == -2.0
    for key in _NONFINITE_TERMS[1:]:
        assert info[key] == clawbacks.get(key, 0.0), key
    terms = [info[key] for key in _NONFINITE_TERMS]
    assert _bits(nan_step.reward) == _bits(left_to_right_sum(terms))
    assert order_sensitive_swaps(terms) == order_sensitive


def test_an_order_sensitive_reward_stays_in_the_ci_streams(golden):
    """The gate keeps at least one reward whose bits depend on term order.

    ``reward_order_sensitive_steps`` is gated like every digest field, so
    a refactor that moves the witness off its order-sensitive ending
    fails the stream test, and a re-record that loses it fails here.
    """
    ci = golden["sets"]["ci"]
    paddle = [case.name for case in CI_CASES if case.reward_terms]
    assert paddle
    for name in paddle:
        assert "reward_order_sensitive_steps" in ci[name], name
    assert ci["paddle_tennis/hold/oracle_nan_order_sensitive"][
        "reward_order_sensitive_steps"
    ] >= 1


#: Adversarial magnitudes for the nine step() terms, signed as the env
#: signs them. Searched so that exchanging any two terms except the
#: first pair moves the left-to-right sum's bits; the test re-proves it.
_STEP_TERMS = {
    "return_reward": 6.770116276963124,
    "fault_penalty": 63214760.71991999,
    "unsafe_physics_penalty": 63608102571851.62,
    "contact_shaping": 0.019183686867182706,
    "pending_shaping": 186694694260156.75,
    "reach": 24761436067753.547,
    "pending_reach": 64.20557484534038,
    "hold": 156040855067978.56,
    "pending_hold": 19.886466241801962,
}


def test_step_reward_adds_its_nine_terms_left_to_right(monkeypatch):
    """Pin step()'s summation order on a step with every term live.

    The L6/M11 refactors must re-associate this sum and are gated on
    "digests identical", which the natural streams cannot enforce. No
    real step pays all nine (a side-A hit keeps the reach and hold
    escrows), so the step is forced: the rules transition reports side
    B's return confirmed, side A's legal hit and a fault that completes
    the point; the reach and hold escrow updates pay fixed amounts and
    leave the pending escrows as set; and the n-point relaunch after
    the fault observes NaN, so the unsafe penalty lands on the same
    step as the fault. The reward must be the env's left-to-right sum
    of the nine info terms: math.fsum (and sum(), compensated since
    Python 3.12) gives other bits here.
    """
    terms = _STEP_TERMS
    env = PaddleTennisEnv(points_per_episode=None, hold_shaping=0.25)
    try:
        env.reset(seed=0)
        for name in (
            "return_reward",
            "fault_penalty",
            "unsafe_physics_penalty",
            "contact_shaping",
        ):
            monkeypatch.setattr(env, name, terms[name])
        monkeypatch.setattr(env, "_pending_shaping", terms["pending_shaping"])
        monkeypatch.setattr(env, "_pending_reach", terms["pending_reach"])
        monkeypatch.setattr(env, "_pending_hold", terms["pending_hold"])
        monkeypatch.setattr(
            env, "_reach_escrow_step", lambda *, took, payment: terms["reach"]
        )
        monkeypatch.setattr(env, "_hold_escrow_step", lambda _transition: terms["hold"])

        real_advance = env._rules.advance

        def forced_advance(events, **kwargs):
            real = real_advance(events, **kwargs)
            assert not real.before.terminated
            hit = RallyEvent(
                kind=RallyEventKind.BALL_RACKET_A, substep=0, position=(-4.0, 0.0, 0.3)
            )
            return dataclasses.replace(
                real,
                valid_racket_hits=(CourtSide.A,),
                valid_racket_hit_events=(hit,),
                confirmed_returns=(CourtSide.B,),
                first_bounces=(),
                after=dataclasses.replace(
                    real.after, termination_reason=TerminationReason.SECOND_BOUNCE
                ),
            )

        monkeypatch.setattr(env._rules, "advance", forced_advance)
        relaunched: list[bool] = []
        real_launch = env._launch_point
        real_get_obs = env._get_obs

        def launch(*, mid_episode):
            real_launch(mid_episode=mid_episode)
            relaunched.append(mid_episode)

        def get_obs():
            obs = real_get_obs()
            return np.full_like(obs, np.nan) if relaunched else obs

        monkeypatch.setattr(env, "_launch_point", launch)
        monkeypatch.setattr(env, "_get_obs", get_obs)
        _obs, reward, terminated, truncated, info = env.step(np.zeros(3))
    finally:
        env.close()

    assert relaunched == [True]
    assert terminated and not truncated
    observed = [info[key] for key in PADDLE_TENNIS_REWARD_INFO_KEYS]
    assert observed == [
        terms["return_reward"],
        -terms["fault_penalty"],
        -terms["unsafe_physics_penalty"],
        terms["contact_shaping"],
        -(terms["pending_shaping"] + terms["contact_shaping"]),
        terms["reach"],
        -terms["pending_reach"],
        terms["hold"],
        -terms["pending_hold"],
    ]
    # The magnitudes are adversarial: every other order gives other bits.
    assert order_sensitive_swaps(observed) == _swaps_but_the_first(9)
    assert _bits(math.fsum(observed)) != _bits(left_to_right_sum(observed))
    assert _bits(reward) == _bits(left_to_right_sum(observed))


def test_nonfinite_guard_adds_its_four_terms_left_to_right(monkeypatch):
    """Pin the nonfinite guard's own sum: unsafe, then the three clawbacks.

    The magnitudes make every reordering but the first pair visible, and
    the nine-key sum the digests check agrees, since the guard's other
    five terms are exact zeros.
    """
    env = PaddleTennisEnv()
    try:
        env.reset(seed=0)
        monkeypatch.setattr(env, "unsafe_physics_penalty", 0.002363)
        monkeypatch.setattr(env, "_pending_shaping", 10.38)
        monkeypatch.setattr(env, "_pending_reach", 665.0)
        monkeypatch.setattr(env, "_pending_hold", 0.2018)
        _obs, reward, terminated, truncated, info = env.step(np.full(3, np.nan))
    finally:
        env.close()
    assert terminated and not truncated
    terms = [info[key] for key in _NONFINITE_TERMS]
    assert terms == [-0.002363, -10.38, -665.0, -0.2018]
    assert order_sensitive_swaps(terms) == _swaps_but_the_first(4)
    assert _bits(reward) == _bits(left_to_right_sum(terms))
    nine = [info[key] for key in PADDLE_TENNIS_REWARD_INFO_KEYS]
    assert _bits(reward) == _bits(left_to_right_sum(nine))


def test_digest_refuses_a_reward_that_is_not_its_term_sum():
    """The canonical-sum check raises, naming the step and the terms."""

    class Drifted(PaddleTennisEnv):
        def step(self, a):
            obs, reward, terminated, truncated, info = super().step(a)
            return obs, reward + 1e-9, terminated, truncated, info

    case = StreamCase(
        name="drifted",
        env_cls=Drifted,
        policy="zero",
        seeds=(0,),
        steps=3,
        required_info_keys=PADDLE_TENNIS_REWARD_INFO_KEYS,
        reward_terms=PADDLE_TENNIS_REWARD_INFO_KEYS,
    )
    with pytest.raises(ValueError, match="step 0 .* is not the left-to-right sum"):
        stream_digest(case)


def test_left_to_right_sum_is_not_sum():
    terms = [2.0**53, 1.0, -(2.0**53)]
    assert left_to_right_sum(terms) == 0.0  # 2**53 + 1.0 rounds the 1.0 away
    assert math.fsum(terms) == 1.0
    assert order_sensitive_swaps(terms) == [(0, 2), (1, 2)]
    assert order_sensitive_swaps([0.1, 0.2, 0.3]) == [(0, 2)]
    assert order_sensitive_swaps([0.5, -2.0, 0.0, 0.0]) == []  # two nonzero terms


@pytest.mark.parametrize(
    "name", ["paddle_tennis/drill_feed/random", "paddle_tennis/drill_full/random"]
)
def test_drill_cases_cover_every_launch_outcome(golden, name):
    """Both arms record drilled points, undrilled policy-receiving points
    and the clearance fallback; the counts are gated digest fields."""
    (case,) = [case for case in CI_CASES if case.name == name]
    assert case.drill_library and case.env_kwargs["drill_fraction"] == 0.5
    entry = golden["sets"]["ci"][name]
    assert entry["drill_drilled_steps"] > 0
    assert entry["drill_undrilled_receiving_steps"] > 0
    assert entry["drill_fallbacks"] > 0


def test_legal_volley_case_drives_the_lead_charge_oracle():
    (case,) = [c for c in CI_CASES if c.name == "paddle_tennis/legal_volley/oracle"]
    env = case.make_env()
    try:
        assert env.unwrapped.volley_rule == "legal"
        assert _oracle_policy(env) is scripted_lead_charge_opponent
    finally:
        env.close()


def test_reset_modes_pick_the_seed_each_episode_resets_with():
    each = StreamCase(name="x", recipe="BallBalance", policy="zero", seeds=(7,), steps=1)
    once = dataclasses.replace(each, reset="seed_once")
    assert RESET_MODES == ("seed_each", "seed_once")
    assert each.reset == "seed_each"  # the default keeps every older golden
    assert [each.reset_seed(7, k) for k in range(3)] == [7, 8, 9]
    assert [once.reset_seed(7, k) for k in range(3)] == [7, None, None]


def test_seed_once_carries_the_rng_and_the_serve_alternation():
    """The training path: one seeded reset, then ``seed=None`` (SB3).

    Reseeding every episode restarts PaddleTennis' serve alternation at
    side A and redraws the serve from a fresh RNG; seeding once lets
    both carry, so later episodes serve from alternating sides and
    their serves come from the carried RNG stream.
    """

    def resets(mode: str) -> list:
        case = StreamCase(
            name="x", env_cls=PaddleTennisEnv, policy="random", seeds=(0,),
            steps=700, reset=mode,
        )
        return [record for record in iter_stream(case) if record.step is None]

    each, once = resets("seed_each"), resets("seed_once")
    assert [r.info["serve_side"] for r in each] == ["a", "a", "a"]
    assert [r.info["serve_side"] for r in once] == ["a", "b", "a", "b"]
    # The first reset is seeded identically in both modes ...
    assert np.array_equal(each[0].obs, once[0].obs)
    assert np.array_equal(each[0].qpos, once[0].qpos)
    # ... and a later serve from the same side comes from another draw.
    assert not np.array_equal(each[2].qpos, once[2].qpos)


def test_ci_runs_the_training_path_resets():
    """Seed-once cases: PaddleTennis random and oracle over at least three
    episodes (two unseeded resets), plus one WallBall case."""
    once = {case.name: case for case in CI_CASES if case.reset == "seed_once"}
    assert sorted(once) == [
        "paddle_tennis/recipe/oracle_seed_once",
        "paddle_tennis/recipe/random_seed_once",
        "wall_ball_true_baseline/oracle_seed_once",
    ]
    golden = golden_digest.load_golden(GOLDEN_PATH)["sets"]["ci"]
    for name, case in once.items():
        assert case.reset_options == ()  # SB3's auto-resets pass none
        assert golden[name]["episodes"] >= 3, name
    assert once["paddle_tennis/recipe/random_seed_once"].policy == "random"
    assert once["paddle_tennis/recipe/oracle_seed_once"].policy == "oracle"


def test_version_info_records_the_numpy_simd_path():
    simd = version_info()["numpy_simd"]
    assert simd == numpy_simd_dispatch()
    assert simd.startswith("baseline ") and "; dispatch " in simd


#: CI cases replayed with numpy's SIMD dispatch switched off, one per
#: env family. The numpy float64 math the env code calls (np.sin,
#: np.cos, np.hypot, np.sqrt) does not move with the dispatch; the
#: loops that do (exp, log1p, arctan2, ...) appear nowhere in it. This
#: keeps that true as the code changes.
_DISPATCH_CASES = (
    "paddle_tennis/recipe/oracle_nan_contact_escrow",
    "wall_ball_true_baseline/oracle",
    "ball_bounce/oracle",
    "ball_balance/random",
    "humanoid_tennis_coop/random",
)


def test_streams_do_not_depend_on_numpy_simd_dispatch():
    """The goldens must match on a runner without this CPU's SIMD paths.

    numpy picks AVX-512, AVX2 or baseline loops at import, and several
    float64 transcendentals (exp, log1p, arctan2, ...) round differently
    on each (``digests.numpy_simd_dispatch``). A stream that called one
    would match its goldens only on CPUs with the recording's SIMD, and
    GitHub's runners vary. Every dispatch target is disabled in a
    subprocess, so this runs the baseline path whatever the CPU, and
    the streams must not move.
    """
    disabled = " ".join(_numpy_umath().__cpu_dispatch__)
    if not disabled:
        pytest.skip("this numpy build dispatches nothing at runtime")
    env = {
        **os.environ,
        "NPY_DISABLE_CPU_FEATURES": disabled,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    argv = [sys.executable, str(REPO_ROOT / "tools" / "golden_digest.py")]
    for name in _DISPATCH_CASES:
        argv += ["--case", name]
    result = subprocess.run(
        argv, capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=300
    )
    assert "; dispatch none" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(_DISPATCH_CASES)}/{len(_DISPATCH_CASES)} cases match" in result.stdout


#: CI cases replayed under each OpenBLAS kernel: the ones whose env code
#: once reached BLAS (norm, dot, matrix-vector) on every step.
_BLAS_CASES = (
    "paddle_tennis/recipe/oracle_nan_contact_escrow",
    "wall_ball_true_baseline/oracle",
    "ball_bounce/oracle",
    "humanoid_tennis_coop/random",
)

#: OpenBLAS kernels to force, with the numpy CPU features each needs.
_BLAS_KERNELS = {
    "Nehalem": (),
    "Haswell": ("AVX2", "FMA3"),
    "SkylakeX": ("AVX512_SKX",),
}


def _openblas_picks_kernels_at_run_time() -> bool:
    config = np.show_config(mode="dicts") or {}
    blas = config.get("Build Dependencies", {}).get("blas", {})
    return "openblas" in str(blas.get("name", "")) and "DYNAMIC_ARCH" in str(
        blas.get("openblas configuration", "")
    )


@pytest.mark.parametrize("kernel", sorted(_BLAS_KERNELS))
def test_streams_do_not_depend_on_the_openblas_kernel(kernel):
    """The goldens must match whichever BLAS kernels the CPU picks.

    numpy's OpenBLAS picks its kernels from the CPU at run time, and on
    3-vectors they round differently: the AVX-512 ``ddot`` and
    ``A.T @ v`` against the AVX2 ones, the AVX2 ``A @ v`` against the
    pre-AVX2 one. Goldens recorded on an AVX-512 machine through
    ``np.linalg.norm`` failed on an AVX2 runner. The env code does that
    math in ``envs._linalg`` (``tests/test_env_linalg.py`` keeps BLAS
    calls out of it); this forces each kernel the CPU can run, and the
    streams must not move.
    """
    if not _openblas_picks_kernels_at_run_time():
        pytest.skip("numpy's BLAS is not an OpenBLAS that picks kernels at run time")
    features = _numpy_umath().__cpu_features__
    missing = [name for name in _BLAS_KERNELS[kernel] if not features.get(name)]
    if missing:
        pytest.skip(f"this CPU cannot run the {kernel} kernels (no {' '.join(missing)})")
    env = {
        **os.environ,
        "OPENBLAS_CORETYPE": kernel,
        "OPENBLAS_VERBOSE": "2",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    argv = [sys.executable, str(REPO_ROOT / "tools" / "golden_digest.py")]
    for name in _BLAS_CASES:
        argv += ["--case", name]
    result = subprocess.run(
        argv, capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=300
    )
    output = result.stdout + result.stderr
    assert f"Core: {kernel}" in output, output
    assert result.returncode == 0, output
    assert f"{len(_BLAS_CASES)}/{len(_BLAS_CASES)} cases match" in result.stdout


def test_stream_digest_is_sensitive_to_the_nan_injection():
    clean = StreamCase(
        name="probe", recipe="BallBalance", policy="zero", seeds=(0,), steps=40
    )
    poisoned = StreamCase(
        name="probe",
        recipe="BallBalance",
        policy="zero",
        seeds=(0,),
        steps=40,
        nan_action_step=10,
    )
    a, b = stream_digest(clean), stream_digest(poisoned)
    assert a["stream_sha256"] != b["stream_sha256"]
    assert a == stream_digest(clean)  # and deterministic in-process


def test_info_values_hash_by_type_shape_and_content():
    """Non-scalar info values are digested, never skipped."""
    encoded = info_value_bytes
    assert encoded((1.0, 2.0)) != encoded([1.0, 2.0])  # tuple vs list
    assert encoded((1.0, 2.0)) != encoded((1.0, 2.5))
    assert encoded(np.zeros(3)) != encoded(np.zeros(3, dtype=np.float32))
    assert encoded(np.zeros(3)) != encoded(np.zeros((1, 3)))
    assert encoded(np.ones(3)) != encoded(np.ones(3) * 2.0)
    assert encoded({"a": 1, "b": (2,)}) == encoded({"b": (2,), "a": 1})
    assert encoded(frozenset({"x", "y"})) == encoded(frozenset({"y", "x"}))
    assert encoded(((1, 2), ())) != encoded(((1,), (2,)))  # nesting is framed
    assert encoded(1) != encoded(1.0) and encoded(np.float64(0.5)) == encoded(0.5)
    with pytest.raises(TypeError, match="no canonical form"):
        encoded(object())
    with pytest.raises(TypeError, match="object-dtype"):
        encoded(np.array([object()]))
    with pytest.raises(TypeError, match="str keys"):
        encoded({1: "a"})


def test_info_digest_and_census_see_non_scalar_items():
    """An array-valued item moving changes info_sha256 and the census."""

    class Instrumented(BallBalanceEnv):
        scale = 1.0
        key = "probe_vector"

        def step(self, a):
            obs, reward, terminated, truncated, info = super().step(a)
            return obs, reward, terminated, truncated, {
                **info,
                self.key: np.arange(3.0) * self.scale,
            }

    def digest(**attrs):
        env_cls = type("Probe", (Instrumented,), attrs)
        case = StreamCase(name="x", env_cls=env_cls, policy="zero", seeds=(0,), steps=5)
        return stream_digest(case)

    base = digest()
    assert base["info_key_count"] == 1
    scaled, renamed = digest(scale=2.0), digest(key="probe_vec")
    assert scaled["stream_sha256"] == renamed["stream_sha256"] == base["stream_sha256"]
    assert scaled["info_sha256"] != base["info_sha256"]
    assert renamed["info_keys_sha256"] != base["info_keys_sha256"]


def test_interface_digest_sees_types_and_spaces_the_stream_hash_erases():
    class Float32Reward(BallBalanceEnv):
        def step(self, a):
            obs, reward, terminated, truncated, info = super().step(a)
            return obs, np.float32(reward), terminated, truncated, info

    class WiderActions(BallBalanceEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            space = self.action_space
            self.action_space = gymnasium.spaces.Box(
                low=space.low, high=space.high * 2.0, dtype=space.dtype
            )

    def digest(env_cls):
        return stream_digest(
            StreamCase(name="x", env_cls=env_cls, policy="zero", seeds=(0,), steps=5)
        )

    base = digest(BallBalanceEnv)
    for variant in (Float32Reward, WiderActions):
        moved = digest(variant)
        assert moved["stream_sha256"] == base["stream_sha256"], variant
        assert moved["interface_sha256"] != base["interface_sha256"], variant


def test_gates_refuse_another_checkouts_package(tmp_path, monkeypatch):
    """A worktree importing the main checkout's package must stop loudly."""
    assert checkout_package_mismatch(REPO_ROOT) is None
    problem = checkout_package_mismatch(tmp_path)
    assert problem is not None
    assert f"PYTHONPATH={(tmp_path / 'src').resolve()}" in problem
    monkeypatch.setattr(golden_digest, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="another checkout's code"):
        golden_digest.main(["--case", "ball_balance/zero"])


def test_case_validation_is_loud():
    with pytest.raises(ValueError, match="exactly one"):
        StreamCase(
            name="x",
            recipe="BallBalance",
            env_cls=BallBalanceEnv,
            policy="zero",
            seeds=(0,),
            steps=1,
        )
    with pytest.raises(ValueError, match="unknown recipe"):
        StreamCase(name="x", recipe="Nope", policy="zero", seeds=(0,), steps=1)
    with pytest.raises(ValueError, match="policy must be one of"):
        StreamCase(name="x", recipe="BallBalance", policy="best", seeds=(0,), steps=1)
    with pytest.raises(ValueError, match="reset must be one of"):
        StreamCase(
            name="x",
            recipe="BallBalance",
            policy="zero",
            seeds=(0,),
            steps=1,
            reset="seed_never",
        )
    with pytest.raises(ValueError, match="nan_action_step"):
        StreamCase(
            name="x",
            recipe="BallBalance",
            policy="zero",
            seeds=(0,),
            steps=5,
            nan_action_step=5,
        )
    with pytest.raises(ValueError, match="must be required info keys"):
        StreamCase(
            name="x",
            recipe="PaddleTennis",
            policy="zero",
            seeds=(0,),
            steps=1,
            reward_terms=("rew_return",),
        )
    with pytest.raises(ValueError, match="builds the library itself"):
        StreamCase(
            name="x",
            recipe="PaddleTennis",
            policy="zero",
            seeds=(0,),
            steps=1,
            env_kwargs={"drill_library": "elsewhere.pkl"},
            drill_library=True,
        )


def test_policies_without_a_meaning_refuse_loudly():
    for policy in ("oracle", "camper"):
        case = StreamCase(
            name="x", env_cls=BallBalanceEnv, policy=policy, seeds=(0,), steps=2
        )
        with pytest.raises(ValueError, match="no (scripted oracle|camper)"):
            stream_digest(case)


def test_missing_required_info_key_raises():
    case = StreamCase(
        name="x",
        env_cls=BallBalanceEnv,
        policy="zero",
        seeds=(0,),
        steps=3,
        required_info_keys=("rew_return",),
    )
    with pytest.raises(KeyError, match="rew_return"):
        stream_digest(case)


def test_check_reports_a_mismatch_with_versions_and_explanation(tmp_path, capsys):
    golden = json.loads(GOLDEN_PATH.read_text())
    entry = golden["sets"]["ci"]["ball_balance/zero"]
    entry["stream_sha256"] = "0" * 64
    tampered = tmp_path / "env_digests.json"
    tampered.write_text(json.dumps(golden))
    code = golden_digest.main(
        ["--check", "--golden", str(tampered), "--case", "ball_balance/zero"]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "GOLDEN DIGEST MISMATCH: ci/ball_balance/zero" in err
    assert f"expected '{'0' * 64}'" in err
    assert f"recorded under: {entry['versions']}" in err
    assert f"running under:  {version_info()}" in err
    assert MISMATCH_EXPLANATION in err


def test_write_reports_changes_and_keeps_unchanged_provenance(tmp_path, capsys):
    path = tmp_path / "env_digests.json"
    path.write_text(GOLDEN_PATH.read_text())
    golden = json.loads(path.read_text())
    golden["sets"]["ci"]["ball_balance/random"]["info_sha256"] = "f" * 64
    path.write_text(json.dumps(golden))
    cases = ["ball_balance/zero", "ball_balance/random"]
    argv = ["--write", "--golden", str(path)]
    for name in cases:
        argv += ["--case", name]
    assert golden_digest.main(argv) == 0
    out = capsys.readouterr().out
    assert "unchanged                        ball_balance/zero" in out
    assert "CHANGED (info_sha256)" in out
    written = json.loads(path.read_text())["sets"]["ci"]
    original = json.loads(GOLDEN_PATH.read_text())["sets"]["ci"]
    # Unchanged entries keep their recording versions verbatim.
    assert written["ball_balance/zero"] == original["ball_balance/zero"]
    assert written["ball_balance/random"]["versions"] == version_info()
    assert written.keys() == original.keys()  # a --case write keeps the rest


def test_check_is_deterministic_across_processes():
    """Another interpreter, another hash seed: the same digests."""
    env = {
        **os.environ,
        "PYTHONHASHSEED": "1234",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "golden_digest.py"),
            "--case",
            "paddle_tennis/recipe/oracle_nan_contact_escrow",
            "--case",
            "ball_bounce/oracle",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2/2 cases match" in result.stdout


def test_digests_never_import_sb3_or_torch():
    """The digest reads the raw env: no training stack in the process."""
    code = (
        "import sys\n"
        "from courtside_dynamics.digests import CI_CASES, stream_digest\n"
        "stream_digest(next(c for c in CI_CASES if c.recipe == 'PaddleTennis'"
        " and c.nan_action_step is not None))\n"
        "leaked = sorted(m for m in ('torch', 'stable_baselines3') if m in sys.modules)\n"
        "assert not leaked, leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
