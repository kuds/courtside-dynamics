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
- ``info_sha256`` covers the info dicts separately: every item, with
  tuples, lists, arrays and mappings hashed recursively by type, dtype
  and shape (:func:`info_value_bytes`), so a pure instrument change
  reads as an info change and never as physics drift. An info value
  of a type with no canonical form raises instead of being skipped.
  The PaddleTennis cases require all nine ``rew_*`` decomposition keys
  at every step.
- ``interface_sha256`` covers what the float64 stream hash erases: the
  observation and action spaces (dtype, shape, bounds) and, at every
  record, the observation's dtype and shape and the concrete types of
  reward, ``terminated`` and ``truncated``.
- ``reward_terms`` (PaddleTennis: the nine ``rew_*`` keys) must
  reproduce every step's reward bit for bit as a left-to-right float
  sum in the listed order, or the digest raises. Float addition is not
  associative, so this pins the order a refactor must keep; the record
  counts the steps whose value actually depends on it
  (``reward_order_sensitive_steps``), and the witness cases keep that
  count above zero. ``tests/test_golden_digests.py`` pins the
  association of both reward sums on forced adversarial terms as well,
  since the natural streams rarely put three terms on one step.

A case resets in one of two modes (:data:`RESET_MODES`) until the step
budget is spent. ``"seed_each"`` (the default) reseeds every episode
with ``seed = stream seed + episode index``, so each episode is
reproducible on its own. ``"seed_once"`` is the path training runs:
SB3's VecEnvs seed the first reset and auto-reset with ``seed=None``
from then on, so the env's RNG -- and any state it keeps across resets,
such as PaddleTennis' serve alternation -- carries from one episode
into the next. Each seed's stream rolls on a freshly constructed env,
so a multi-seed digest is the concatenation of independent streams.
A ``drill_library`` case plays the k=2 drill from a tiny library that
:func:`build_drill_library` harvests deterministically in-process.

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

import copy
import functools
import hashlib
import importlib
import itertools
import pickle
import platform
import struct
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium
import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv

from courtside_dynamics.envs import (
    BallBounceEnv,
    CourtSide,
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

#: How a :class:`StreamCase` reseeds its episodes.
#:
#: - ``"seed_each"``: episode ``k`` resets with ``seed=stream seed + k``.
#: - ``"seed_once"``: episode 0 resets with the stream seed and every
#:   later episode with ``seed=None``, as SB3's ``DummyVecEnv`` does
#:   after its first reset, so RNG and cross-episode env state carry.
RESET_MODES = ("seed_each", "seed_once")

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
    "qvel), its info dicts or its interface (spaces, value types) "
    "changed for a seeded scripted rollout. If "
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
    seed's stream, episode ``k`` resets with ``seed + k`` under
    ``reset="seed_each"`` (the default), or with ``seed`` for ``k = 0``
    and ``None`` after it under ``reset="seed_once"`` (the training
    path; see :data:`RESET_MODES`). Either way, when ``reset_options``
    is non-empty, episode ``k`` passes ``reset_options[k % len]``; SB3's
    auto-resets pass none, so the ``seed_once`` cases leave it empty.

    ``nan_action_step`` replaces the policy's action with an all-NaN
    one at that 0-based ``env.step`` index of every seed's stream
    (episode boundaries do not restart the count): the nonfinite guard
    witness.

    ``info_keys`` selects the info items to digest; ``None`` digests
    every item. ``required_info_keys`` must be present (and scalar) in
    every step's info, or the digest raises.

    ``reward_terms`` names info items whose left-to-right float sum, in
    that order, must equal every step's reward bit for bit (see the
    module doc); they must also be required keys.

    ``drill_library`` builds the digests' own k=2 drill library
    (:func:`build_drill_library`) and passes its path as the env's
    ``drill_library`` kwarg; ``env_kwargs`` set the fraction and arm.
    The digest then also counts drilled steps, undrilled steps of
    policy-receiving points, and clearance fallbacks, so a re-record
    that stops exercising one of the three launch outcomes is visible.
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
    reward_terms: tuple[str, ...] = ()
    drill_library: bool = False
    reset: str = "seed_each"

    def __post_init__(self) -> None:
        if (self.recipe is None) == (self.env_cls is None):
            raise ValueError(
                f"{self.name}: exactly one of recipe and env_cls must be set"
            )
        if self.reset not in RESET_MODES:
            raise ValueError(
                f"{self.name}: reset must be one of {RESET_MODES}, "
                f"got {self.reset!r}"
            )
        unrequired = sorted(set(self.reward_terms) - set(self.required_info_keys))
        if unrequired:
            raise ValueError(
                f"{self.name}: reward_terms {unrequired} must be required info keys"
            )
        if self.drill_library and "drill_library" in self.env_kwargs:
            raise ValueError(
                f"{self.name}: drill_library=True builds the library itself; "
                f"drop the env_kwargs path"
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
        overrides = dict(self.env_kwargs)
        if self.drill_library:
            overrides["drill_library"] = drill_library_path()
        if self.recipe is not None:
            recipe = RECIPES[self.recipe]
            kwargs = {**recipe.env_kwargs, **overrides}
            return recipe.env_cls(**kwargs)
        assert self.env_cls is not None  # __post_init__ guarantees it
        return self.env_cls(**overrides)

    def reset_seed(self, seed: int, episode: int) -> int | None:
        """What ``env.reset(seed=...)`` gets for episode ``episode`` of
        the stream seeded ``seed`` (``None``: carry the RNG over)."""
        if self.reset == "seed_each":
            return seed + episode
        return seed if episode == 0 else None


@dataclass(frozen=True)
class StreamStep:
    """One record of a case's stream: a reset (``step`` is None) or a step.

    ``reward``, ``terminated`` and ``truncated`` are cast to float and
    bool; ``interface`` keeps what the casts (and the float64 stream
    hash) erase -- see :func:`_interface_bytes`.
    """

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
    interface: bytes = b""


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


def _box_action_space(env: gymnasium.Env) -> gymnasium.spaces.Box:
    """``env``'s action space, which every digested env declares as a Box."""
    space = env.action_space
    if not isinstance(space, gymnasium.spaces.Box):
        raise TypeError(
            f"{type(env.unwrapped).__name__}: the digests drive Box action "
            f"spaces only, got a {type(space).__name__}"
        )
    return space


def scripted_policy(name: str, env: gymnasium.Env, seed: int) -> Policy:
    """The driving policy ``name`` (:data:`POLICIES`) for ``env``.

    The digests' frozen definitions, shared with the probe CLI
    (``experiments.probe``) so a probe drives exactly the policy a
    golden stream does: ``random`` draws from
    ``np.random.default_rng(seed)``.
    """
    if name not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {name!r}")
    space = _box_action_space(env)
    if name == "zero":
        zero = np.zeros(space.shape, dtype=space.dtype)
        return lambda _obs: zero.copy()
    if name == "random":
        rng = np.random.default_rng(seed)
        return lambda _obs: rng.uniform(space.low, space.high).astype(space.dtype)
    if name == "oracle":
        return _oracle_policy(env)
    return _camper_policy(env)


def _make_policy(case: StreamCase, env: gymnasium.Env, seed: int) -> Policy:
    return scripted_policy(case.policy, env, seed)


# -- the drill library ---------------------------------------------------------

#: The library schema PaddleTennisEnv consumes
#: (``tools/paddle_tennis_k2_harvest.py`` writes the registered one).
_DRILL_LIBRARY_SCHEMA = "k2-drill-library-v0"
#: The harvest walks the test suite's burned bring-up block
#: (1000-1119), never a ledger block; two entries fill on seed 1000.
_DRILL_HARVEST_SEEDS = range(1000, 1120)
_DRILL_HARVEST_ENTRIES = 2


def _drill_snapshot(env: PaddleTennisEnv, obs: np.ndarray) -> dict[str, Any]:
    """Everything either launch arm reads, as the harvest tool records it."""
    sampler = env._event_sampler
    return {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "qacc_warmstart": env.data.qacc_warmstart.copy(),
        "obs": np.asarray(obs, dtype=np.float64).copy(),
        "rules": copy.deepcopy(env._rules),
        # The compiled model is shared, never copied: the launching env
        # attaches its own.
        "sampler": copy.deepcopy(sampler, memo={id(sampler.model): None}),
        "serving_side": env._serving_side,
        "head_a": env._paddle_position(CourtSide.A),
        "head_b": env._paddle_position(CourtSide.B),
    }


def _drill_entry_is_clear(env: PaddleTennisEnv, entry: Mapping[str, Any]) -> bool:
    """The launch-clearance test ``PaddleTennisEnv._launch_point`` applies."""
    ball = entry["qpos"][env._ball_qposadr : env._ball_qposadr + 3]
    return all(
        float(np.linalg.norm(entry[key] - ball)) >= env._SERVE_CLEARANCE
        for key in ("head_a", "head_b")
    )


def build_drill_library() -> dict[str, Any]:
    """A tiny k=2 drill library, harvested deterministically in-process.

    The ground oracle drives side A through default-kwarg continuous
    play on points side B serves, and the state is recorded at the k=2
    instant: after A's legal hit and B's legal return, with the ball
    back over the net heading for A (the criterion of
    ``tools/paddle_tennis_k2_harvest.py`` and of the test suite's drill
    fixture). Entries carry what both launch arms read, so one library
    serves the ``"feed"`` and ``"full"`` cases. A third entry copies
    the first with the ball on side A's paddle head: it fails the
    launch clearance, so the drill's loud fallback (a drawn serve,
    counted in ``drill_fallback_count``) is digested too.

    Nothing here is written to the repository: the library is a pure
    function of the env code, rebuilt in every process that digests a
    drill case, so a change to the harvested states is a digest change
    like any other.
    """
    env = PaddleTennisEnv(points_per_episode=None)
    entries: list[dict[str, Any]] = []
    try:
        for seed in _DRILL_HARVEST_SEEDS:
            obs, _info = env.reset(seed=seed, options={"serve_side": "b"})
            hit_a = struck_b = False
            points = 0
            for _ in range(env.episode_len):
                obs, _reward, terminated, truncated, info = env.step(
                    scripted_ground_opponent(obs)
                )
                if terminated or truncated:
                    break
                if int(info["points_played"]) != points:
                    points = int(info["points_played"])
                    hit_a = struck_b = False
                if info["event_valid_racket_hit_a"]:
                    hit_a, struck_b = True, False
                if info["event_valid_racket_hit_b"] and hit_a:
                    struck_b = True
                ball_x = float(env.data.qpos[env._ball_qposadr])
                ball_vx = float(env.data.qvel[env._ball_dofadr])
                if not (
                    struck_b
                    and env._serving_side is CourtSide.B
                    and ball_x > 0.05
                    and ball_vx < -1.0
                ):
                    continue
                entry = _drill_snapshot(env, obs)
                if _drill_entry_is_clear(env, entry):
                    entries.append(entry)
                    hit_a = struck_b = False
                    if len(entries) == _DRILL_HARVEST_ENTRIES:
                        break
            if len(entries) == _DRILL_HARVEST_ENTRIES:
                break
        else:
            raise RuntimeError(
                f"the digest drill harvest exhausted seeds "
                f"{_DRILL_HARVEST_SEEDS.start}-{_DRILL_HARVEST_SEEDS.stop - 1} "
                f"with {len(entries)} of {_DRILL_HARVEST_ENTRIES} entries"
            )
        blocked = copy.deepcopy(entries[0])
        blocked["qpos"][env._ball_qposadr : env._ball_qposadr + 3] = blocked["head_a"]
        assert not _drill_entry_is_clear(env, blocked)
        entries.append(blocked)
    finally:
        env.close()
    return {"schema": _DRILL_LIBRARY_SCHEMA, "entries": entries}


@functools.cache
def _drill_library_directory() -> tempfile.TemporaryDirectory[str]:
    # Cached, so the directory lives until interpreter exit, when its
    # finalizer removes it.
    directory = tempfile.TemporaryDirectory(prefix="courtside_digest_drill_")
    path = Path(directory.name) / "digest_drill_library.pkl"
    path.write_bytes(pickle.dumps(build_drill_library()))
    return directory


def drill_library_path() -> str:
    """The file of :func:`build_drill_library`, harvested once per process."""
    return str(Path(_drill_library_directory().name) / "digest_drill_library.pkl")


# -- streams ----------------------------------------------------------------


def _physics_state(env: gymnasium.Env) -> tuple[np.ndarray, np.ndarray]:
    """Copies of the raw MuJoCo ``qpos``/``qvel`` behind ``env``."""
    base = env.unwrapped
    if not isinstance(base, MujocoEnv):
        raise TypeError(
            f"{type(base).__name__} is not a MujocoEnv: the stream digest "
            f"hashes the raw MuJoCo qpos/qvel"
        )
    return base.data.qpos.copy(), base.data.qvel.copy()


def iter_stream(case: StreamCase) -> Iterator[StreamStep]:
    """Roll ``case`` and yield every reset and step record, in order."""
    for seed in case.seeds:
        env = case.make_env()
        try:
            policy = _make_policy(case, env, seed)
            nan_action = np.full(
                _box_action_space(env).shape, np.nan, dtype=np.float64
            )
            episode = 0
            step = 0
            while step < case.steps:
                options = (
                    dict(case.reset_options[episode % len(case.reset_options)])
                    if case.reset_options
                    else None
                )
                obs, info = env.reset(
                    seed=case.reset_seed(seed, episode), options=options
                )
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
                    interface=_interface_bytes(
                        obs,
                        spaces=(env.observation_space, env.action_space),
                    ),
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
                        interface=_interface_bytes(
                            obs, values=(reward, terminated, truncated)
                        ),
                    )
                    step += 1
                    done = bool(terminated or truncated)
                episode += 1
        finally:
            env.close()


def _f64(value: Any) -> bytes:
    return np.ascontiguousarray(value, dtype=np.float64).tobytes()


def _type_name(value: Any) -> bytes:
    """A version-stable name for a value's concrete type.

    A numpy scalar names its dtype (numpy 2 renamed ``np.bool_``'s class
    to ``numpy.bool``); anything else its module and qualified name.
    """
    if isinstance(value, np.generic):
        return f"numpy:{value.dtype.str}".encode()
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}".encode()


def _array_bytes(array: np.ndarray) -> bytes:
    """An array's dtype, shape and raw bytes; object arrays refuse."""
    if array.dtype.hasobject:
        raise TypeError("an object-dtype array has no canonical bytes")
    header = f"{array.dtype.str}{array.shape}".encode()
    return header + b"\x00" + np.ascontiguousarray(array).tobytes()


def _space_bytes(space: gymnasium.Space[Any]) -> bytes:
    """A Box space's dtype, shape and bounds; other spaces refuse loudly."""
    if not isinstance(space, gymnasium.spaces.Box):
        raise TypeError(f"no canonical form for a {type(space).__name__} space")
    return b"Box" + _array_bytes(space.low) + _array_bytes(space.high)


def _interface_bytes(
    obs: Any,
    *,
    spaces: tuple[gymnasium.Space[Any], gymnasium.Space[Any]] | None = None,
    values: tuple[Any, Any, Any] | None = None,
) -> bytes:
    """What the float64 stream hash erases, for one record.

    The raw observation's type, dtype and shape at every record; at a
    reset, the observation and action spaces; at a step, the concrete
    types of reward, ``terminated`` and ``truncated`` before the casts
    (``np.float32(1.0)`` and ``1.0`` hash the same stream bytes).
    """
    raw = np.asarray(obs)
    chunks = [_type_name(obs), f"{raw.dtype.str}{raw.shape}".encode()]
    if spaces is not None:
        chunks += [_space_bytes(space) for space in spaces]
    if values is not None:
        chunks += [_type_name(value) for value in values]
    return _framed(chunks)


def _framed(chunks: Iterable[bytes]) -> bytes:
    """Count- and length-prefixed concatenation: unambiguous nesting."""
    items = list(chunks)
    return struct.pack("<q", len(items)) + b"".join(
        struct.pack("<q", len(item)) + item for item in items
    )


def _scalar_bytes(value: Any) -> bytes | None:
    """Canonical tagged bytes for a scalar info value; None if not scalar.

    Numeric values hash by class (bool / int / float), not by their
    concrete Python or numpy type: ``np.float64(x)`` and ``float(x)``
    are the same stream, while ``1`` and ``1.0`` are not (CSV writers
    and aggregators see the difference). A ``str`` subclass (a
    ``StrEnum`` member) hashes as its string value.
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


def info_value_bytes(value: Any) -> bytes:
    """Canonical tagged bytes for any info value; unknown types raise.

    Scalars hash as :func:`_scalar_bytes` does. Containers hash
    recursively, every element length-prefixed: tuples and lists under
    distinct tags, numpy arrays with their dtype and shape, mappings by
    sorted string key, sets in sorted element order. Anything else
    raises ``TypeError``, so a new kind of info value fails loudly
    instead of dropping out of the digest.
    """
    scalar = _scalar_bytes(value)
    if scalar is not None:
        return scalar
    if isinstance(value, np.ndarray):
        return b"a" + _array_bytes(value)
    if isinstance(value, (tuple, list)):
        tag = b"t" if isinstance(value, tuple) else b"l"
        return tag + _framed(info_value_bytes(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return b"e" + _framed(sorted(info_value_bytes(item) for item in value))
    if isinstance(value, Mapping):
        keys = list(value)
        if not all(isinstance(key, str) for key in keys):
            raise TypeError(f"a mapping info value needs str keys, got {keys!r}")
        return b"m" + _framed(
            key.encode() + b"\x00" + info_value_bytes(value[key])
            for key in sorted(keys)
        )
    raise TypeError(
        f"no canonical form for an info value of type {type(value).__name__}; "
        f"teach digests.info_value_bytes about it"
    )


def left_to_right_sum(terms: Sequence[float]) -> float:
    """``((t0 + t1) + t2) + ...``: the association the reward sums use.

    Deliberately neither ``sum()`` (compensated since Python 3.12) nor
    ``math.fsum``: either can differ from the env's own expression in
    the last bit.
    """
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _float_bits(value: float) -> bytes:
    return struct.pack("<d", value)


def order_sensitive_swaps(terms: Sequence[float]) -> list[tuple[int, int]]:
    """Every transposition ``(i, j)`` of ``terms`` that moves the sum's bits.

    :func:`left_to_right_sum` of ``terms`` with positions ``i`` and
    ``j`` exchanged differs from the canonical sum exactly for the
    returned pairs. ``(0, 1)`` never does (float addition commutes), and
    with fewer than three nonzero terms no pair does.
    """
    canonical = _float_bits(left_to_right_sum(terms))
    swaps = []
    for i, j in itertools.combinations(range(len(terms)), 2):
        swapped = list(terms)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        if _float_bits(left_to_right_sum(swapped)) != canonical:
            swaps.append((i, j))
    return swaps


def _reward_is_order_sensitive(case: StreamCase, record: StreamStep) -> bool:
    """Check a step's reward against its canonical term sum (raises).

    Returns whether that sum's value depends on the summation order.
    """
    terms = [record.info[key] for key in case.reward_terms]
    canonical = left_to_right_sum(terms)
    if _float_bits(float(canonical)) != _float_bits(record.reward):
        raise ValueError(
            f"{case.name}: step {record.step} (seed {record.seed}) reward "
            f"{record.reward!r} is not the left-to-right sum {canonical!r} of "
            f"{dict(zip(case.reward_terms, terms, strict=True))}: the reward "
            f"decomposition or its summation order changed"
        )
    nonzero = sum(1 for term in terms if term != 0.0)
    return nonzero >= 3 and bool(order_sensitive_swaps(terms))


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
        value = info[key]
        if key in case.required_info_keys and _scalar_bytes(value) is None:
            raise TypeError(
                f"{case.name}: info[{key!r}] is not a scalar: "
                f"{type(value).__name__}"
            )
        try:
            encoded = info_value_bytes(value)
        except TypeError as error:
            raise TypeError(f"{case.name}: info[{key!r}]: {error}") from None
        keys_seen.add(key)
        chunks.append(key.encode() + b"\x00" + encoded)
    return b"\x1e".join(chunks) + b"\n"


def stream_digest(case: StreamCase) -> dict[str, Any]:
    """Roll ``case`` and return its digest record (JSON-serializable).

    ``stream_sha256``, ``info_sha256`` and ``interface_sha256`` are the
    gated values; the counts, ``reward_sum``, the info key census (every
    key seen at any reset or step), ``reward_order_sensitive_steps``
    for a case with ``reward_terms`` and the ``drill_*`` counts for a
    drill case exist to make a mismatch readable (did the episode
    structure move? the reward? the info schema?) and are gated too.
    """
    stream = hashlib.sha256()
    info_hash = hashlib.sha256()
    interface = hashlib.sha256()
    keys_seen: set[str] = set()
    steps = episodes = terminations = truncations = order_sensitive = 0
    drilled = undrilled_receiving = 0
    fallbacks: dict[tuple[int, int], float] = {}
    reward_sum = 0.0
    for record in iter_stream(case):
        # Info first: it raises the readable error for a missing key.
        info_hash.update(_info_bytes(case, record, keys_seen))
        interface.update(record.interface)
        if case.reward_terms and record.step is not None:
            order_sensitive += _reward_is_order_sensitive(case, record)
        if case.drill_library and record.step is not None:
            info = record.info
            drilled += info["drill_point"] == 1.0
            undrilled_receiving += (
                info["drill_point"] == 0.0 and info["serve_side_is_policy"] == 0.0
            )
            # An episode counter: its last value is the episode's total.
            fallbacks[(record.seed, record.episode)] = info["drill_fallback_count"]
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
    missing = sorted(set(case.required_info_keys) - keys_seen)
    if missing:
        raise KeyError(f"{case.name}: required info keys never digested: {missing}")
    digest: dict[str, Any] = {
        "stream_sha256": stream.hexdigest(),
        "info_sha256": info_hash.hexdigest(),
        "interface_sha256": interface.hexdigest(),
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
    if case.reward_terms:
        digest["reward_order_sensitive_steps"] = order_sensitive
    if case.drill_library:
        digest["drill_drilled_steps"] = drilled
        digest["drill_undrilled_receiving_steps"] = undrilled_receiving
        digest["drill_fallbacks"] = int(sum(fallbacks.values()))
    return digest


def _numpy_umath() -> Any:
    """numpy's compiled umath module, which carries the CPU-dispatch tables."""
    try:
        return importlib.import_module("numpy._core._multiarray_umath")
    except ImportError:  # numpy 1.x before the numpy._core alias
        return importlib.import_module("numpy.core._multiarray_umath")


def numpy_simd_dispatch() -> str:
    """The SIMD code paths numpy's ufuncs run on in this process.

    numpy compiles some ufunc loops for several instruction sets and
    picks one at import from the CPU (and ``NPY_DISABLE_CPU_FEATURES``),
    and the paths do not round alike. On numpy 2.5.3 (x86-64) the
    float64 ``exp``, ``log``, ``log1p``, ``expm1``, ``tan``, ``arctan``,
    ``arctan2``, ``arcsin``, ``arccos``, ``power`` and ``cbrt`` give
    other last bits without AVX-512, and ``tanh`` without AVX2; float64
    ``sin``, ``cos``, ``sqrt`` and ``hypot`` do not move. The env
    streams call none of the moving loops, and
    ``tests/test_golden_digests.py`` replays CI cases with every
    dispatch target disabled (``docs/DECISIONS.md``, "Golden digests and
    the freeze harness gate every refactor"). This records the path
    anyway, so a mismatch report says which one ran: the compiled
    baseline and the dispatch targets this CPU enables, e.g.
    ``"baseline X86_V2; dispatch X86_V3 X86_V4"``.
    """
    umath = _numpy_umath()
    features: Mapping[str, bool] = umath.__cpu_features__
    enabled = [name for name in umath.__cpu_dispatch__ if features.get(name)]
    return (
        f"baseline {' '.join(umath.__cpu_baseline__) or 'none'}; "
        f"dispatch {' '.join(enabled) or 'none'}"
    )


def version_info() -> dict[str, str]:
    """The library versions (and numpy SIMD path) a golden digest was
    recorded under.

    ``numpy_simd`` is newer than the first entries: a golden entry's
    ``versions`` may lack it, and then the recording's dispatch is
    unknown (``tests/test_golden_digests.py`` allows exactly that).
    """
    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "numpy_simd": numpy_simd_dispatch(),
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
#: The original freeze's rally rules (volleys confirm returns); the
#: oracle is then the lead-charge port the env gives side B.
_LEGAL_VOLLEY = {"volley_rule": "legal"}
#: The k=2 drill on half the policy-receiving launches, per arm. Under
#: random play the seed-0 stream holds drilled, undrilled-eligible and
#: fallback launches in both arms; tests/test_golden_digests.py asserts
#: that coverage from the stream.
_DRILL_ARMS = {
    context: {"drill_fraction": 0.5, "drill_context": context}
    for context in ("feed", "full")
}

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
# - Hold arm, seed 1 (an order-sensitive ending): the opponent's strike
#   at index 342 pays rew_hold = +0.019691534990717713 and the live first
#   bounce at 442 pays rew_reach = +0.17532310086941538; side A's next
#   legal hit lands at 449. The NaN action at 443 claws back both, and
#   its reward -2.0 - 0.17532... - 0.01969... rounds differently when
#   the hold clawback moves ahead of the unsafe penalty or the reach
#   clawback: no other digested step depends on the summation order.
_NAN_CONTACT_ESCROW_STEP = 250
_NAN_REACH_HOLD_ESCROW_STEP = 430
_NAN_ORDER_SENSITIVE_SEED = 1
_NAN_ORDER_SENSITIVE_STEP = 443

#: The recipe's training-path streams (``reset="seed_once"``): two
#: 1,500-step episodes and 200 steps into a third, so two unseeded
#: resets carry the env RNG and the serve alternation. On seed 0 the
#: oracle's three episodes serve from sides A, B, A; every seeded reset
#: serves from A.
_SEED_ONCE_PADDLE_STEPS = 3_200


def _paddle_case(name: str, **kwargs: Any) -> StreamCase:
    return StreamCase(
        name=name,
        recipe="PaddleTennis",
        required_info_keys=PADDLE_TENNIS_REWARD_INFO_KEYS,
        reward_terms=PADDLE_TENNIS_REWARD_INFO_KEYS,
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
        _paddle_case(
            "paddle_tennis/legal_volley/oracle",
            policy="oracle",
            seeds=seeds,
            steps=steps,
            env_kwargs=_LEGAL_VOLLEY,
        ),
        *(
            _paddle_case(
                f"paddle_tennis/drill_{context}/random",
                policy="random",
                seeds=seeds,
                steps=steps,
                env_kwargs=kwargs,
                drill_library=True,
            )
            for context, kwargs in _DRILL_ARMS.items()
        ),
        # The witnesses' NaN steps are calibrated on one seed only.
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
        _paddle_case(
            "paddle_tennis/hold/oracle_nan_order_sensitive",
            policy="oracle",
            seeds=(_NAN_ORDER_SENSITIVE_SEED,),
            steps=witness_steps,
            env_kwargs=_HOLD_ARM,
            nan_action_step=_NAN_ORDER_SENSITIVE_STEP,
        ),
        # The training path: SB3 seeds the first reset only.
        *(
            _paddle_case(
                f"paddle_tennis/recipe/{policy}_seed_once",
                policy=policy,
                seeds=seeds,
                steps=_SEED_ONCE_PADDLE_STEPS,
                reset="seed_once",
            )
            for policy in ("random", "oracle")
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
        StreamCase(
            name="wall_ball_true_baseline/oracle_seed_once",
            recipe="WallBallTrueBaseline",
            policy="oracle",
            seeds=seeds,
            steps=steps,
            reset="seed_once",
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


#: The CI gate: one seed, two 1,500-step PaddleTennis episodes per arm
#: (three for the seed-once arms).
CI_CASES = _cases(seeds=(0,), steps=3_000, humanoid_steps=300)

#: The local refactor gate: five spaced seeds (spaced so no two streams
#: share an episode reset seed) at the design's 3,000 steps per seed;
#: the humanoid case takes the first two at 600 steps, and the NaN
#: witnesses stay on the one seed each is calibrated on.
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
    "RESET_MODES",
    "StreamCase",
    "StreamStep",
    "build_drill_library",
    "digest_mismatches",
    "drift_label",
    "drill_library_path",
    "expected_digest",
    "format_mismatch",
    "info_value_bytes",
    "iter_stream",
    "left_to_right_sum",
    "numpy_simd_dispatch",
    "order_sensitive_swaps",
    "stream_digest",
    "version_info",
]
