"""Golden per-step digests of the env streams (design M1 + L0).

Every refactor of the env and rules code must leave the task's streams
bit-identical unless it declares an era break, and "bit-identical" has
to be something a machine checks, not something a reviewer believes.
This module defines the check: a :class:`StreamCase` names an env, a
scripted driving policy and a seed/step budget, and
:func:`stream_digest` rolls it and hashes everything the learner (and
the physics under it) can see.

- ``stream_sha256`` covers, at every reset and every step, the
  observation (as float64 bytes), the reward (float64), ``terminated``
  and ``truncated``, and the raw MuJoCo ``qpos``/``qvel``. This is the
  design's ``dyn_fp``.
- ``info_sha256`` covers the info dicts separately (sorted scalar
  numeric/bool/str items), so a pure instrument change reads as an info
  change and never as physics drift. The PaddleTennis cases require all
  nine ``rew_*`` decomposition keys at every step.

Episodes reset with ``seed = stream seed + episode index`` until the
step budget is spent; each seed's stream rolls on a freshly constructed
env, so a multi-seed digest is the concatenation of independent streams.

The golden values live in ``tests/golden/env_digests.json``;
``tools/golden_digest.py`` writes and checks them and
``tests/test_golden_digests.py`` gates CI on the fast subset. A mismatch
is an era break or physics drift that must be declared -- never
something to re-record silently (see :data:`MISMATCH_EXPLANATION`).

This module imports numpy, MuJoCo, Gymnasium and the package's envs,
recipe registry and scripted policies only -- never SB3 or torch -- so a
digest reads the raw env, with no wrapper or normalizer in between.
"""

from __future__ import annotations

import hashlib
import platform
import struct
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import gymnasium
import mujoco
import numpy as np

from courtside_dynamics.envs import (
    BallBounceEnv,
    PaddleTennisEnv,
    WallBallEnv,
)
from courtside_dynamics.envs._paddle_court import (
    scripted_ground_opponent,
    scripted_lead_charge_opponent,
    scripted_reach_camper_witness,
)
from courtside_dynamics.recipes import RECIPES
from courtside_dynamics.scripted_policies import (
    ball_bounce_oracle_action,
    wall_ball_baseline_oracle_action,
    wall_ball_oracle_action,
)

#: Driving policies a :class:`StreamCase` may name.
#:
#: - ``"zero"``: the zero action every step.
#: - ``"random"``: ``np.random.default_rng(stream seed)`` drawing uniformly
#:   inside the action space, cast to the space's dtype.
#: - ``"oracle"``: the env's scripted feasibility controller (see
#:   :func:`_oracle_policy`).
#: - ``"camper"``: a constant action that parks the paddle somewhere
#:   meaningful for the env's shaping (see :func:`_camper_policy`).
POLICIES = ("zero", "random", "oracle", "camper")

#: The nine per-step reward-decomposition keys PaddleTennis publishes;
#: they sum exactly to the step reward. Every PaddleTennis case requires
#: all nine at every step, so an escrow that stops publishing its term
#: fails loudly instead of dropping out of the info digest.
PADDLE_TENNIS_REWARD_INFO_KEYS = (
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

#: Printed with every golden-digest mismatch, by the tool and the test.
MISMATCH_EXPLANATION = (
    "A golden digest mismatch is an era break or physics drift, never "
    "noise: the env stream (obs, reward, terminated, truncated, qpos, "
    "qvel) or its info dicts changed for a seeded scripted rollout. If "
    "the recorded and live versions differ, suspect dependency drift "
    "(MuJoCo, numpy) and prove it before anything else. If the change "
    "is intended, it must be DECLARED -- a new task version, or a "
    "CHANGELOG and docs/DECISIONS.md entry naming the era break -- and "
    "only then re-recorded with `python tools/golden_digest.py --write` "
    "(add --full for the local set) in the same commit. Never re-record "
    "to make this check pass."
)


@dataclass(frozen=True)
class StreamCase:
    """One seeded, scripted env stream to digest.

    Exactly one of ``recipe`` and ``env_cls`` names the env. A recipe's
    ``env_kwargs`` are used verbatim (``render_mode`` included: rendering
    is lazy and never feeds the streams), with ``env_kwargs`` layered on
    top as overrides; an ``env_cls`` is built from ``env_kwargs`` alone.

    ``steps`` is the per-seed budget of ``env.step`` calls. Within a
    seed's stream, episode ``k`` resets with ``seed + k`` and, when
    ``reset_options`` is non-empty, ``reset_options[k % len]``.

    ``nan_action_step`` replaces the policy's action with an all-NaN
    one at that 0-based ``env.step`` index of every seed's stream
    (episode boundaries do not restart the count): the nonfinite guard
    witness.

    ``info_keys`` selects the info items to digest; ``None`` digests
    every scalar numeric/bool/str item. ``required_info_keys`` must be
    present (and scalar) in every step's info, or the digest raises.
    """

    name: str
    policy: str
    seeds: tuple[int, ...]
    steps: int
    recipe: str | None = None
    env_cls: type | None = None
    env_kwargs: Mapping[str, Any] = field(default_factory=dict)
    reset_options: tuple[Mapping[str, Any], ...] = ()
    nan_action_step: int | None = None
    info_keys: tuple[str, ...] | None = None
    required_info_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (self.recipe is None) == (self.env_cls is None):
            raise ValueError(
                f"{self.name}: exactly one of recipe and env_cls must be set"
            )
        if self.recipe is not None and self.recipe not in RECIPES:
            raise ValueError(f"{self.name}: unknown recipe {self.recipe!r}")
        if self.policy not in POLICIES:
            raise ValueError(
                f"{self.name}: policy must be one of {POLICIES}, "
                f"got {self.policy!r}"
            )
        if not self.seeds:
            raise ValueError(f"{self.name}: seeds must not be empty")
        if self.steps <= 0:
            raise ValueError(f"{self.name}: steps must be positive")
        if self.nan_action_step is not None and not (
            0 <= self.nan_action_step < self.steps
        ):
            raise ValueError(
                f"{self.name}: nan_action_step must lie in [0, steps)"
            )

    def make_env(self) -> gymnasium.Env:
        """Construct a fresh env for one seed's stream."""
        if self.recipe is not None:
            recipe = RECIPES[self.recipe]
            kwargs = {**recipe.env_kwargs, **self.env_kwargs}
            return recipe.env_cls(**kwargs)
        assert self.env_cls is not None  # __post_init__ guarantees it
        return self.env_cls(**self.env_kwargs)


@dataclass(frozen=True)
class StreamStep:
    """One record of a case's stream: a reset (``step`` is None) or a step."""

    seed: int
    episode: int
    step: int | None
    action: np.ndarray | None
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    qpos: np.ndarray
    qvel: np.ndarray


# -- policies ---------------------------------------------------------------

Policy = Callable[[np.ndarray], np.ndarray]


def _wall_ball_oracle(env: WallBallEnv) -> Policy:
    """The scripted_policies oracle matching a WallBall env's geometry.

    - A fenced full-workspace mapping (the depth-curriculum family:
      GoalRally, TrueBaseline) plans inside the fence and inverts the
      env's actual mapping, per ``wall_ball_baseline_oracle_action``'s
      decoupling contract.
    - An unfenced ``one_bounce`` lane (WallBallBaseline, Bootstrap) is
      the baseline oracle's calibration case: lane == mapping.
    - Anything else is the legacy open/volley paddle, driven by
      ``wall_ball_oracle_action`` on the env's live pivot and range.
    """
    home = float(env.paddle_home_x)
    # Resolved to a concrete world-space span at construction (None
    # selects the frozen historical default there, never here).
    target_range = env.paddle_x_target_range
    assert target_range is not None
    mapping = (float(target_range[0]), float(target_range[1]))
    fence = env.paddle_x_fence
    if fence is not None:
        lane = (float(fence[0]), float(fence[1]))

        def fenced(obs: np.ndarray) -> np.ndarray:
            return wall_ball_baseline_oracle_action(
                obs,
                paddle_x_target_range=lane,
                paddle_home_x=home,
                action_mapping_range=(mapping[0], mapping[1]),
                action_mapping_home=home,
            )

        return fenced
    if env.rally_style == "one_bounce":

        def lane_oracle(obs: np.ndarray) -> np.ndarray:
            return wall_ball_baseline_oracle_action(
                obs,
                paddle_x_target_range=(mapping[0], mapping[1]),
                paddle_home_x=home,
            )

        return lane_oracle

    def legacy(obs: np.ndarray) -> np.ndarray:
        return wall_ball_oracle_action(
            obs,
            paddle_home_x=home,
            paddle_x_target_range=(mapping[0], mapping[1]),
        )

    return legacy


def _oracle_policy(env: gymnasium.Env) -> Policy:
    """The env's scripted feasibility controller, applied to the policy obs.

    PaddleTennis is driven by the same rule-matched controller the env
    gives its side-B opponent (the ground oracle under
    ``volley_rule="fault"``, the lead-charge port under ``"legal"``),
    applied to the side-A observation: the policy and the opponent
    share one mirrored interface. Envs without a scripted oracle raise.
    """
    base = env.unwrapped
    if isinstance(base, PaddleTennisEnv):
        if base.volley_rule == "fault":
            return scripted_ground_opponent
        return scripted_lead_charge_opponent
    if isinstance(base, BallBounceEnv):
        return ball_bounce_oracle_action
    if isinstance(base, WallBallEnv):
        return _wall_ball_oracle(base)
    raise ValueError(f"no scripted oracle for {type(base).__name__}")


def _camper_policy(env: gymnasium.Env) -> Policy:
    """A constant parked action, where one exercises something new.

    PaddleTennis: the frozen reach-shaping anti-farming witness
    (``scripted_reach_camper_witness``): x action -0.61 on the side-A
    piecewise map parks the paddle head at side-local x ~ -4.55, the
    serve's landing depth, and y action 0.33 holds it ~1 m off the ball
    line, so qualifying first bounces pay reach proximity that the
    clawback must then take back, with no hit to keep it. A camper "at
    the home target" would be the zero action exactly (zero maps to the
    home pivot by construction of the piecewise map) and so would
    duplicate the zero digest; the campaign design's camper witness is
    this one. Other envs have no such constant and raise.
    """
    if isinstance(env.unwrapped, PaddleTennisEnv):
        return scripted_reach_camper_witness
    raise ValueError(
        f"no camper witness for {type(env.unwrapped).__name__}: its zero "
        f"action already holds the home target"
    )


def _make_policy(case: StreamCase, env: gymnasium.Env, seed: int) -> Policy:
    space = env.action_space
    assert isinstance(space, gymnasium.spaces.Box)
    if case.policy == "zero":
        zero = np.zeros(space.shape, dtype=space.dtype)
        return lambda _obs: zero.copy()
    if case.policy == "random":
        rng = np.random.default_rng(seed)
        return lambda _obs: rng.uniform(space.low, space.high).astype(space.dtype)
    if case.policy == "oracle":
        return _oracle_policy(env)
    return _camper_policy(env)


# -- streams ----------------------------------------------------------------


def _physics_state(env: gymnasium.Env) -> tuple[np.ndarray, np.ndarray]:
    """Copies of the raw MuJoCo ``qpos``/``qvel`` behind ``env``."""
    data = env.unwrapped.data
    return data.qpos.copy(), data.qvel.copy()


def iter_stream(case: StreamCase) -> Iterator[StreamStep]:
    """Roll ``case`` and yield every reset and step record, in order."""
    for seed in case.seeds:
        env = case.make_env()
        try:
            policy = _make_policy(case, env, seed)
            nan_action = np.full(
                env.action_space.shape, np.nan, dtype=np.float64
            )
            episode = 0
            step = 0
            while step < case.steps:
                options = (
                    dict(case.reset_options[episode % len(case.reset_options)])
                    if case.reset_options
                    else None
                )
                obs, info = env.reset(seed=seed + episode, options=options)
                qpos, qvel = _physics_state(env)
                yield StreamStep(
                    seed=seed,
                    episode=episode,
                    step=None,
                    action=None,
                    obs=np.asarray(obs),
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info=info,
                    qpos=qpos,
                    qvel=qvel,
                )
                done = False
                while not done and step < case.steps:
                    action = (
                        nan_action
                        if step == case.nan_action_step
                        else policy(obs)
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    qpos, qvel = _physics_state(env)
                    yield StreamStep(
                        seed=seed,
                        episode=episode,
                        step=step,
                        action=np.asarray(action),
                        obs=np.asarray(obs),
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info=info,
                        qpos=qpos,
                        qvel=qvel,
                    )
                    step += 1
                    done = bool(terminated or truncated)
                episode += 1
        finally:
            env.close()


def _f64(value: Any) -> bytes:
    return np.ascontiguousarray(value, dtype=np.float64).tobytes()


def _scalar_bytes(value: Any) -> bytes | None:
    """Canonical tagged bytes for a scalar info value; None if not scalar.

    Numeric values hash by class (bool / int / float), not by their
    concrete Python or numpy type: ``np.float64(x)`` and ``float(x)``
    are the same stream, while ``1`` and ``1.0`` are not (CSV writers
    and aggregators see the difference).
    """
    if isinstance(value, (bool, np.bool_)):
        return b"?" + (b"\x01" if value else b"\x00")
    if isinstance(value, (int, np.integer)):
        return b"i" + str(int(value)).encode()
    if isinstance(value, (float, np.floating)):
        return b"f" + struct.pack("<d", float(value))
    if isinstance(value, str):
        encoded = value.encode()
        return b"s" + struct.pack("<q", len(encoded)) + encoded
    if value is None:
        return b"n"
    return None


def _info_bytes(
    case: StreamCase, record: StreamStep, keys_seen: set[str]
) -> bytes:
    info = record.info
    if record.step is not None:
        missing = [key for key in case.required_info_keys if key not in info]
        if missing:
            raise KeyError(
                f"{case.name}: step {record.step} (seed {record.seed}) info "
                f"lacks required key(s) {missing}"
            )
    if case.info_keys is None:
        keys = sorted(info)
    else:
        if record.step is not None:
            missing = [key for key in case.info_keys if key not in info]
            if missing:
                raise KeyError(
                    f"{case.name}: step {record.step} (seed {record.seed}) "
                    f"info lacks digested key(s) {missing}"
                )
        keys = sorted(key for key in case.info_keys if key in info)
    chunks = [b"r" if record.step is None else b"s"]
    for key in keys:
        encoded = _scalar_bytes(info[key])
        if encoded is None:
            if key in case.required_info_keys or case.info_keys is not None:
                raise TypeError(
                    f"{case.name}: info[{key!r}] is not a scalar: "
                    f"{type(info[key]).__name__}"
                )
            continue
        keys_seen.add(key)
        chunks.append(key.encode() + b"\x00" + encoded)
    return b"\x1e".join(chunks) + b"\n"


def stream_digest(case: StreamCase) -> dict[str, Any]:
    """Roll ``case`` and return its digest record (JSON-serializable).

    ``stream_sha256`` and ``info_sha256`` are the gated values; the
    counts, ``reward_sum`` and the info key census exist to make a
    mismatch readable (did the episode structure move? the reward? the
    info schema?).
    """
    stream = hashlib.sha256()
    info_hash = hashlib.sha256()
    keys_seen: set[str] = set()
    steps = episodes = terminations = truncations = 0
    reward_sum = 0.0
    for record in iter_stream(case):
        if record.step is None:
            episodes += 1
            stream.update(b"R" + struct.pack("<qq", record.seed, record.episode))
            stream.update(_f64(record.obs))
        else:
            steps += 1
            terminations += record.terminated
            truncations += record.truncated
            reward_sum += record.reward
            stream.update(b"S")
            stream.update(_f64(record.obs))
            stream.update(struct.pack("<d", record.reward))
            stream.update(bytes((record.terminated, record.truncated)))
        stream.update(_f64(record.qpos))
        stream.update(_f64(record.qvel))
        info_hash.update(_info_bytes(case, record, keys_seen))
    missing = sorted(set(case.required_info_keys) - keys_seen)
    if missing:
        raise KeyError(f"{case.name}: required info keys never digested: {missing}")
    return {
        "stream_sha256": stream.hexdigest(),
        "info_sha256": info_hash.hexdigest(),
        "info_keys_sha256": hashlib.sha256(
            "\n".join(sorted(keys_seen)).encode()
        ).hexdigest(),
        "info_key_count": len(keys_seen),
        "steps": steps,
        "episodes": episodes,
        "terminations": terminations,
        "truncations": truncations,
        "reward_sum": reward_sum,
    }


def version_info() -> dict[str, str]:
    """The library versions a golden digest was recorded under."""
    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "machine": platform.machine(),
    }


def drift_label(versions: Mapping[str, str]) -> str:
    """The declared-drift slot a version set reads: ``"mujoco X.Y"``."""
    major, minor = str(versions["mujoco"]).split(".")[:2]
    return f"mujoco {major}.{minor}"


#: Stream differences between MuJoCo versions that are known, measured
#: and documented, keyed by drift label, then case name. A golden entry
#: carries one extra digest per declared slot (recorded under that
#: MuJoCo) and every other version must reproduce its primary digest.
#: Everything absent here is held bit-identical across versions.
DECLARED_DRIFT: dict[str, dict[str, str]] = {
    "mujoco 3.16": {
        "humanoid_tennis_coop/random": (
            "MuJoCo 3.16.0's mj_setConst refreshes body bounding volumes; "
            "the racket body's BVH stops culling contacts in the enlarged "
            "stringbed's outer strip, so random flailing diverges from "
            "3.15 at step 49. The reason mujoco is capped <3.16; see "
            "docs/DECISIONS.md, 'MuJoCo is capped below 3.16'."
        ),
    },
}

#: Keys of a golden entry that describe the recording, not the stream.
_RECORDING_KEYS = frozenset({"versions", "declared_drift", "reason"})


def expected_digest(
    entry: Mapping[str, Any], name: str, live_versions: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, str], str | None]:
    """Resolve a golden entry for the live versions.

    Returns ``(digest, recorded_versions, drift_label_or_None)``: the
    declared-drift slot when the live MuJoCo has one for ``name``, the
    primary digest otherwise. A golden drift slot that the code no
    longer declares raises -- the golden file must not carry an
    exemption nobody owns.
    """
    label = drift_label(live_versions)
    slots = entry.get("declared_drift", {})
    for slot in slots:
        if name not in DECLARED_DRIFT.get(slot, {}):
            raise ValueError(
                f"golden entry {name!r} carries an undeclared drift slot "
                f"{slot!r}; declare it in DECLARED_DRIFT or remove it"
            )
    chosen = slots[label] if label in slots else entry
    digest = {k: v for k, v in chosen.items() if k not in _RECORDING_KEYS}
    return digest, dict(chosen.get("versions", {})), (
        label if label in slots else None
    )


def digest_mismatches(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[str]:
    """Human-readable lines for every field that differs (empty = match)."""
    return [
        f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]


def format_mismatch(
    set_name: str,
    name: str,
    lines: list[str],
    recorded_versions: Mapping[str, str],
    live_versions: Mapping[str, str],
    drift: str | None,
) -> str:
    """The loud failure text shared by the tool and the test."""
    against = f" (declared-drift slot {drift!r})" if drift else ""
    return "\n".join(
        (
            f"GOLDEN DIGEST MISMATCH: {set_name}/{name}{against}",
            *(f"  {line}" for line in lines),
            f"  recorded under: {dict(recorded_versions)}",
            f"  running under:  {dict(live_versions)}",
            f"  {MISMATCH_EXPLANATION}",
        )
    )


# -- the cases ----------------------------------------------------------------

#: The recipe env plus the post-swing hold arm (the hold-escrow line's
#: probe configuration, docs/design_paddle_tennis_postswing_hold.md).
_HOLD_ARM = {"hold_shaping": 0.25}
#: The frozen one-point episode (the env's own default; the recipe plays
#: continuous n-point). Seeded resets always serve side A, so the
#: one-point cases alternate the serve through reset options, exactly
#: like the recipe's paired evaluation.
_ONE_POINT = {"points_per_episode": 1}
_ALTERNATE_SERVE = ({"serve_side": "a"}, {"serve_side": "b"})

# NaN-with-pending-escrow witnesses (campaign design L0). The steps were
# chosen from the seed-0 oracle stream's own info, and the witness test
# re-proves them from the NaN step's info on every run:
#
# - Recipe env: the side-A legal hit at step index 215 pays
#   rew_shaping = +0.25 into the contact escrow; its return confirms only
#   at index 318. A NaN action at 250 must claw back
#   rew_shaping_clawback = -0.25 through the nonfinite guard.
# - Hold arm: the opponent's return strike at index 325 pays
#   rew_hold = +0.00988960945948972 into the hold escrow, and the live
#   first bounce at 427 pays rew_reach = +0.17534702856592194 into the
#   reach escrow; side A's next legal hit (which would keep both) lands
#   only at 435. A NaN action at 430 must claw back both at once.
_NAN_CONTACT_ESCROW_STEP = 250
_NAN_REACH_HOLD_ESCROW_STEP = 430


def _paddle_case(name: str, **kwargs: Any) -> StreamCase:
    return StreamCase(
        name=name,
        recipe="PaddleTennis",
        required_info_keys=PADDLE_TENNIS_REWARD_INFO_KEYS,
        **kwargs,
    )


def _cases(*, seeds: tuple[int, ...], steps: int, humanoid_steps: int) -> tuple[
    StreamCase, ...
]:
    """One case list, scaled: the CI subset and the full local set."""
    witness_steps = 600
    return (
        *(
            _paddle_case(
                f"paddle_tennis/recipe/{policy}",
                policy=policy,
                seeds=seeds,
                steps=steps,
            )
            for policy in ("zero", "random", "oracle", "camper")
        ),
        _paddle_case(
            "paddle_tennis/one_point/oracle",
            policy="oracle",
            seeds=seeds,
            steps=steps,
            env_kwargs=_ONE_POINT,
            reset_options=_ALTERNATE_SERVE,
        ),
        _paddle_case(
            "paddle_tennis/hold/oracle",
            policy="oracle",
            seeds=seeds,
            steps=steps,
            env_kwargs=_HOLD_ARM,
        ),
        # The witnesses' NaN steps are calibrated on seed 0 only.
        _paddle_case(
            "paddle_tennis/recipe/oracle_nan_contact_escrow",
            policy="oracle",
            seeds=(0,),
            steps=witness_steps,
            nan_action_step=_NAN_CONTACT_ESCROW_STEP,
        ),
        _paddle_case(
            "paddle_tennis/hold/oracle_nan_reach_hold_escrow",
            policy="oracle",
            seeds=(0,),
            steps=witness_steps,
            env_kwargs=_HOLD_ARM,
            nan_action_step=_NAN_REACH_HOLD_ESCROW_STEP,
        ),
        *(
            StreamCase(
                name=f"wall_ball_true_baseline/{policy}",
                recipe="WallBallTrueBaseline",
                policy=policy,
                seeds=seeds,
                steps=steps,
            )
            for policy in ("random", "oracle")
        ),
        *(
            StreamCase(
                name=f"ball_bounce/{policy}",
                recipe="BallBounce",
                policy=policy,
                seeds=seeds,
                steps=steps,
            )
            for policy in ("random", "oracle")
        ),
        *(
            StreamCase(
                name=f"ball_balance/{policy}",
                recipe="BallBalance",
                policy=policy,
                seeds=seeds,
                steps=steps,
            )
            for policy in ("zero", "random")
        ),
        StreamCase(
            name="humanoid_tennis_coop/random",
            recipe="HumanoidTennisCoopSmoke",
            policy="random",
            seeds=seeds[:2],
            steps=humanoid_steps,
        ),
    )


#: The CI gate: one seed, two 1,500-step PaddleTennis episodes per arm.
CI_CASES = _cases(seeds=(0,), steps=3_000, humanoid_steps=300)

#: The local refactor gate: five spaced seeds (spaced so no two streams
#: share an episode reset seed) at the design's 3,000 steps per seed.
FULL_CASES = _cases(
    seeds=(0, 100, 200, 300, 400), steps=3_000, humanoid_steps=600
)

#: Named case sets, as the tool and the golden file key them.
CASE_SETS: dict[str, tuple[StreamCase, ...]] = {
    "ci": CI_CASES,
    "full": FULL_CASES,
}

__all__ = [
    "CASE_SETS",
    "CI_CASES",
    "DECLARED_DRIFT",
    "FULL_CASES",
    "MISMATCH_EXPLANATION",
    "PADDLE_TENNIS_REWARD_INFO_KEYS",
    "POLICIES",
    "StreamCase",
    "StreamStep",
    "digest_mismatches",
    "drift_label",
    "expected_digest",
    "format_mismatch",
    "iter_stream",
    "stream_digest",
    "version_info",
]
