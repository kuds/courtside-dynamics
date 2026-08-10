"""PaddleTennis behavioral diagnosis: the exchange/positioning instrument.

Package home of the instrument first shipped as
``tools/paddle_tennis_diagnosis_probe.py`` (which remains the CLI).
It walks every point's rules-event stream into per-shot, per-exchange,
and positioning records, so a policy's failure mode is separable into
hypotheses (see the tool's docstring and
``docs/paddle_tennis_diagnosis_20260808.md``, where this instrument
separated the ground-era pilot's plateau into "one memorized
serve-return macro, no general ball-reaching" -- H1 -- and rejected
the stroke-authority and opponent-asymmetry explanations).

Living in the package (not ``tools/``) so training can run it:
:class:`DiagnosisProbeCallback` replays the instrument against the
live model at every checkpoint save, writing one report per
checkpoint plus a cached policy-independent oracle reference row into
``reports/diagnosis/``. The callback is exception-isolated -- a
diagnosis failure must never kill a training run (the video-callback
lesson) -- and disables itself after a failure rather than failing
every checkpoint.
"""

from __future__ import annotations

import dataclasses
import os
from collections import Counter
from collections.abc import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from courtside_dynamics.envs._paddle_court import (
    PADDLE_COURT,
    scripted_ground_opponent,
)
from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
from courtside_dynamics.envs.tennis_rules import (
    CourtSide,
    RallyEventKind,
    TerminationReason,
)

P_DIAG_SEED_START = 5200
P_DIAG_EPISODES = 100

#: Exchange-survival table depth (policy hits per point worth reporting).
_MAX_EXCHANGE = 5
#: Steps excluded after the policy's own hit before the recovery-hold
#: window opens (~0.3 s: the swing follow-through is not "wandering").
FOLLOW_THROUGH_STEPS = 30


def native_checkpoint_policy(
    model_path: str, vec_normalize_path: str
) -> Callable[[np.ndarray], np.ndarray]:
    """A saved PaddleTennis SB3 SAC checkpoint behind its normalizer.

    Unlike the P5 tool's champion loader there is no shim: the
    checkpoint was trained on this env's own 48-value observation.
    The saved ``SelectiveVecNormalize`` carries the recipe's excluded
    tail (indices 24-47), so ``normalize_obs`` reproduces training
    normalization exactly.
    """
    import pickle

    from stable_baselines3 import SAC

    model = SAC.load(model_path, device="cpu")
    with open(vec_normalize_path, "rb") as handle:
        normalizer = pickle.load(handle)
    normalizer.training = False

    def act(observation: np.ndarray) -> np.ndarray:
        normalized = normalizer.normalize_obs(observation[None, :].astype(np.float64))[
            0
        ]
        action, _ = model.predict(normalized, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    return act


@dataclasses.dataclass(slots=True)
class ShotRecord:
    """One racket hit and what became of the ball it struck."""

    hitter: CourtSide
    exchange_index: int  # 1-based count of THIS hitter's hits this point
    crossed: bool = False
    landing_depth: float | None = None  # |x| in the landing half
    landing_lateral: float | None = None
    landed_in: bool | None = None
    outcome: str = "open"  # in | out | net | opponent_hit | terminal


@dataclasses.dataclass(slots=True)
class EpisodeTrace:
    """Instrumented record of one point."""

    serve_side_is_policy: bool
    shots: list[ShotRecord]
    policy_hits: int
    crossings: int
    termination: str
    ender: str  # attribution label, see _attribute_ender
    recovery_travel: list[float]  # per policy hit: hold-window XY path (m)
    ready_errors: list[float]  # per incoming first bounce on policy side
    touched_after_bounce: list[bool]


def _attribute_ender(
    termination: TerminationReason,
    open_shot: ShotRecord | None,
    expected_returner: CourtSide | None,
    *,
    feed_landed_in: bool,
    feed_receiver: CourtSide,
) -> str:
    """Name who/what ended the point, from the rules' perspective.

    OUT_OF_BOUNDS needs the shot ledger's own facts: the rules give
    it priority over SECOND_BOUNCE, so a never-reached ball that
    bounced IN and then skipped out terminates OUT_OF_BOUNDS -- the
    fault belongs to the receiver who never reached it, NOT to the
    (good) shot. Attributing by hitter alone inverted H1 evidence
    into H3 (adversarial review, 2026-08-08).
    """
    if termination is TerminationReason.NONE:
        return "cap"
    prefix_by_side = {CourtSide.A: "policy", CourtSide.B: "opponent"}
    if termination is TerminationReason.OUT_OF_BOUNDS:
        if open_shot is not None:
            if open_shot.landed_in:
                receiver = open_shot.hitter.opponent
                return f"{prefix_by_side[receiver]}_never_reached"
            return f"{prefix_by_side[open_shot.hitter]}_shot_out"
        if feed_landed_in:
            return f"{prefix_by_side[feed_receiver]}_never_reached"
        return "feed_out"
    if termination is TerminationReason.BALL_NET:
        if open_shot is not None:
            return f"{prefix_by_side[open_shot.hitter]}_shot_net"
        return "feed_net"
    if termination is TerminationReason.SECOND_BOUNCE:
        if expected_returner is not None:
            return f"{prefix_by_side[expected_returner]}_never_reached"
        return "second_bounce"
    if termination is TerminationReason.VOLLEY_RETURN:
        if expected_returner is not None:
            return f"{prefix_by_side[expected_returner]}_volley_fault"
        return "volley_fault"
    return termination.name.lower()


def run_episode(
    env: PaddleTennisEnv,
    policy: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> tuple[list[EpisodeTrace], list[float]]:
    """Walk an episode's event stream into per-POINT traces.

    Reads ``env._last_transition`` (the rules transition of the step
    just taken) for ordered semantic events with positions -- the same
    private surface the P3 harness reads on the prototype scene; the
    info dict does not carry event positions.

    Under the frozen one-point default this returns exactly one trace
    (and no inter-point travels). Under n-point play
    (docs/design_paddle_tennis_npoint.md) the rules machine still
    emits a terminated transition at each point's end -- the env
    absorbs it -- so the walk segments on those transitions: one
    trace per point (the truncation-cut partial point included, with
    its cap-style ender, exactly like a cap-truncated one-point
    episode), plus the era's new metric, one **inter-point recovery
    travel** per boundary: the policy paddle's XY path length from
    the point's end until the next feed's arrival (its first
    bounce; a feed that nets or dies without bouncing bounds its
    window at that point's end instead, and the cap bounds the
    last window at episode end).
    """
    observation, reset_info = env.reset(seed=seed)
    serve_side_is_policy = reset_info["serve_side_is_policy"] >= 0.5
    feed_receiver = CourtSide.B if serve_side_is_policy else CourtSide.A

    traces: list[EpisodeTrace] = []
    interpoint_travels: list[float] = []
    interpoint_active = False
    interpoint_travel = 0.0
    point_crossings_start = 0

    shots: list[ShotRecord] = []
    open_shot: ShotRecord | None = None
    hits_by_side = {CourtSide.A: 0, CourtSide.B: 0}
    recovery_travel: list[float] = []
    ready_errors: list[float] = []
    touched_after_bounce: list[bool] = []
    feed_landed_in = False
    feed_bounced = False

    # Recovery-hold window over the policy paddle: opens
    # FOLLOW_THROUGH_STEPS after its own legal hit (excluding the
    # swing), closes at the opponent's next legal hit or terminal.
    # Measures whether the player holds ground while its shot is away
    # -- the honest "wandering" number (the first draft's hit-to-bounce
    # window included the swing and the legitimate run-up, so the
    # designed hold-ground oracle itself measured as zipping).
    window_countdown = -1  # steps until the window opens; -1 = closed
    window_active = False
    window_travel = 0.0
    awaiting_touch = False
    previous_paddle = observation[9:12].copy()

    def close_window() -> None:
        nonlocal window_active, window_countdown, window_travel
        if window_active:
            recovery_travel.append(window_travel)
        window_active = False
        window_countdown = -1
        window_travel = 0.0

    info: dict = {}
    last_nudges = 0
    while True:
        observation, _, terminated, truncated, info = env.step(policy(observation))
        raw = env.observation_for_side(CourtSide.A)
        paddle = raw[9:12]
        # A sanctioned relaunch nudge is a referee teleport, not
        # policy motion: its step contributes to neither travel
        # metric (the step's genuine motion is bounded by the
        # measured 0.123 m boundary jump, versus the up-to-0.45 m
        # teleport it would overcount).
        nudges_now = int(info.get("point_serve_nudged", 0))
        nudged_step = nudges_now > last_nudges
        last_nudges = nudges_now
        if window_countdown > 0:
            window_countdown -= 1
            if window_countdown == 0:
                window_active = True
        elif window_active and not nudged_step:
            window_travel += float(np.linalg.norm(paddle[:2] - previous_paddle[:2]))
        if interpoint_active and not nudged_step:
            interpoint_travel += float(np.linalg.norm(paddle[:2] - previous_paddle[:2]))
        previous_paddle = paddle.copy()

        transition = env._last_transition
        assert transition is not None
        # A racket event that the rules rejected as a terminal fault
        # (volley, double hit, wrong hitter...) is NOT a made hit:
        # only sides listed in valid_racket_hits earn ledger entries
        # (the first draft counted fault touches, inflating exchange
        # survival exactly where the H1 verdict looks).
        legal_credits = Counter(transition.valid_racket_hits)
        for event in transition.processed_events:
            if event.kind is RallyEventKind.BALL_RACKET_A:
                if legal_credits[CourtSide.A] <= 0:
                    continue
                legal_credits[CourtSide.A] -= 1
                hits_by_side[CourtSide.A] += 1
                if open_shot is not None and open_shot.outcome == "open":
                    open_shot.outcome = "opponent_hit"
                open_shot = ShotRecord(
                    hitter=CourtSide.A,
                    exchange_index=hits_by_side[CourtSide.A],
                )
                shots.append(open_shot)
                close_window()
                window_countdown = FOLLOW_THROUGH_STEPS
                if awaiting_touch:
                    touched_after_bounce.append(True)
                    awaiting_touch = False
            elif event.kind is RallyEventKind.BALL_RACKET_B:
                if legal_credits[CourtSide.B] <= 0:
                    continue
                legal_credits[CourtSide.B] -= 1
                hits_by_side[CourtSide.B] += 1
                if open_shot is not None and open_shot.outcome == "open":
                    open_shot.outcome = "opponent_hit"
                open_shot = ShotRecord(
                    hitter=CourtSide.B,
                    exchange_index=hits_by_side[CourtSide.B],
                )
                shots.append(open_shot)
                # The ball is coming back: the hold window ends.
                close_window()
            elif event.kind in (
                RallyEventKind.NET_CROSSING_TO_A,
                RallyEventKind.NET_CROSSING_TO_B,
            ):
                if open_shot is not None:
                    open_shot.crossed = True
            elif event.kind in (
                RallyEventKind.BALL_COURT_A,
                RallyEventKind.BALL_COURT_B,
            ):
                if event.position is None:
                    continue
                in_bounds = bool(
                    event.in_bounds
                    if event.in_bounds is not None
                    else PADDLE_COURT.is_in_bounds(event.position[0], event.position[1])
                )
                incoming_first_bounce = False
                if (
                    open_shot is not None
                    and open_shot.crossed
                    and open_shot.outcome == "open"
                ):
                    open_shot.landing_depth = abs(float(event.position[0]))
                    open_shot.landing_lateral = float(event.position[1])
                    open_shot.landed_in = in_bounds
                    open_shot.outcome = "in" if in_bounds else "out"
                    incoming_first_bounce = (
                        open_shot.hitter is CourtSide.B and in_bounds
                    )
                elif not feed_bounced and open_shot is None:
                    # The serve's own first bounce.
                    feed_bounced = True
                    feed_landed_in = in_bounds
                    incoming_first_bounce = feed_receiver is CourtSide.A and in_bounds
                    if interpoint_active:
                        # The new feed has arrived: the inter-point
                        # recovery window closes (npoint design §2).
                        interpoint_travels.append(interpoint_travel)
                        interpoint_active = False
                        interpoint_travel = 0.0
                if incoming_first_bounce and event.kind is RallyEventKind.BALL_COURT_A:
                    # A playable ball just bounced on the policy's
                    # side: ready-position snapshot. Guarded to
                    # incoming FIRST bounces only -- the first draft
                    # also fired on the policy's own failed-to-cross
                    # landings and second bounces, polluting the
                    # ready-error and touch-rate baselines.
                    error = float(
                        np.hypot(
                            paddle[0] - float(event.position[0]),
                            paddle[1] - float(event.position[1]),
                        )
                    )
                    ready_errors.append(error)
                    awaiting_touch = True
        point_ended = bool(transition.terminated_now and not terminated)
        if point_ended or terminated or truncated:
            # Finalize the current point's trace (boundary, episode
            # fault, or cap truncation mid-point).
            if awaiting_touch:
                touched_after_bounce.append(False)
                awaiting_touch = False
            close_window()
            after = transition.after
            termination = after.termination_reason
            if (
                open_shot is not None
                and open_shot.outcome == "open"
                and termination is TerminationReason.BALL_NET
            ):
                open_shot.outcome = "net"
            # The env's forced-nonfinite backstop ends the episode
            # while the rules snapshot still reads "none"; the trace
            # must carry the authoritative episode-ending flag so
            # unsafe endings are countable from traces alone.
            termination_name = str(info["termination_reason_name"])
            if terminated and float(info.get("term_nonfinite", 0.0)):
                termination_name = "nonfinite_state"
            traces.append(
                EpisodeTrace(
                    serve_side_is_policy=serve_side_is_policy,
                    shots=shots,
                    policy_hits=hits_by_side[CourtSide.A],
                    crossings=int(info["crossings"]) - point_crossings_start,
                    termination=termination_name,
                    ender=_attribute_ender(
                        termination,
                        open_shot,
                        after.expected_returner,
                        feed_landed_in=feed_landed_in,
                        feed_receiver=feed_receiver,
                    ),
                    recovery_travel=recovery_travel,
                    ready_errors=ready_errors,
                    touched_after_bounce=touched_after_bounce,
                )
            )
            point_crossings_start = int(info["crossings"])
            if point_ended and not (terminated or truncated):
                # A new point was relaunched inside this step: reset
                # the per-point state and open the inter-point
                # recovery window (env._serving_side already names
                # the new server).
                serve_side_is_policy = env._serving_side is CourtSide.A
                feed_receiver = CourtSide.B if serve_side_is_policy else CourtSide.A
                shots = []
                open_shot = None
                hits_by_side = {CourtSide.A: 0, CourtSide.B: 0}
                recovery_travel = []
                ready_errors = []
                touched_after_bounce = []
                feed_landed_in = False
                feed_bounced = False
                # One travel entry per boundary: if the previous
                # window never met its feed's first bounce (the feed
                # netted/died first), it closes here at the point's
                # end instead of being silently dropped.
                if interpoint_active:
                    interpoint_travels.append(interpoint_travel)
                interpoint_active = True
                interpoint_travel = 0.0
        if terminated or truncated:
            break

    if interpoint_active:
        interpoint_travels.append(interpoint_travel)

    return traces, interpoint_travels


def _shot_row(shots: list[ShotRecord], label: str) -> str:
    n = len(shots)
    if n == 0:
        return f"  {label:<26} n=0"
    crossed = sum(s.crossed for s in shots)
    landed = [s for s in shots if s.landed_in is not None]
    landed_in = [s for s in shots if s.landed_in]
    depths = [s.landing_depth for s in landed_in if s.landing_depth is not None]
    out_depths = [
        s.landing_depth
        for s in landed
        if s.landed_in is False and s.landing_depth is not None
    ]
    return (
        f"  {label:<26} n={n:<4} crossed {crossed / n:4.0%} "
        f"in {len(landed_in) / n:4.0%}"
        + (f"  in-depth {float(np.mean(depths)):.2f}" if depths else "")
        + (f"  out-depth {float(np.mean(out_depths)):.2f}" if out_depths else "")
    )


def report(
    traces: list[EpisodeTrace],
    label: str,
    *,
    interpoint_travels: list[float] | None = None,
) -> str:
    lines = [f"player: {label}  ({len(traces)} points)"]

    policy_shots = [s for t in traces for s in t.shots if s.hitter is CourtSide.A]
    opponent_shots = [s for t in traces for s in t.shots if s.hitter is CourtSide.B]
    lines.append("shot ledger:")
    lines.append(_shot_row(policy_shots, "policy, all"))
    receive = [t for t in traces if not t.serve_side_is_policy]
    serving = [t for t in traces if t.serve_side_is_policy]
    for split_label, split in (("receiving", receive), ("serving", serving)):
        split_shots = [s for t in split for s in t.shots if s.hitter is CourtSide.A]
        for k in (1, 2, 3):
            lines.append(
                _shot_row(
                    [s for s in split_shots if s.exchange_index == k],
                    f"policy {split_label}, hit #{k}",
                )
            )
    lines.append(_shot_row(opponent_shots, "opponent, all"))

    # Survival split by serve side: when receiving, hit #1 is the
    # solved feed return; when serving, hit #1 additionally requires
    # the opponent's feed return to succeed first. Pooling the halves
    # smears the exchange-2 cliff the H1 verdict looks for.
    for split_label, split in (("receiving", receive), ("serving", serving)):
        if not split:
            continue
        survival = [
            sum(t.policy_hits >= k for t in split) / len(split)
            for k in range(1, _MAX_EXCHANGE + 1)
        ]
        lines.append(
            f"exchange survival, policy {split_label} "
            f"({len(split)} points): "
            + "  ".join(f"k={k}:{p:.0%}" for k, p in enumerate(survival, 1))
        )

    enders = Counter(t.ender for t in traces)
    lines.append(
        "point enders: " + ", ".join(f"{k} {v}" for k, v in enders.most_common())
    )

    travel = [x for t in traces for x in t.recovery_travel]
    ready = [x for t in traces for x in t.ready_errors]
    touched = [x for t in traces for x in t.touched_after_bounce]
    if travel:
        lines.append(
            f"recovery hold (post-swing, while own shot is away): "
            f"travel mean {np.mean(travel):.2f} m "
            f"(p90 {np.percentile(travel, 90):.2f})"
        )
    if ready:
        lines.append(
            f"ready position: bounce-time error mean "
            f"{np.mean(ready):.2f} m (p90 {np.percentile(ready, 90):.2f}); "
            f"touched after bounce {np.mean(touched):.0%} "
            f"of {len(touched)}"
        )
    if interpoint_travels:
        lines.append(
            f"inter-point recovery (point end to next feed arrival): "
            f"travel mean {np.mean(interpoint_travels):.2f} m "
            f"(p90 {np.percentile(interpoint_travels, 90):.2f}) "
            f"over {len(interpoint_travels)} boundaries"
        )
    lines.append(
        f"crossings mean {np.mean([t.crossings for t in traces]):.2f}; "
        "terminations: "
        + ", ".join(
            f"{k} {v}" for k, v in Counter(t.termination for t in traces).most_common(4)
        )
    )
    return "\n".join(lines)


def run_player(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    episodes: int,
    seed_start: int,
    env_fn: Callable[[], PaddleTennisEnv] = PaddleTennisEnv,
) -> tuple[list[EpisodeTrace], list[float]]:
    """All point traces plus inter-point recovery travels.

    One trace per point (== one per episode under the frozen
    one-point default, where the travel list is always empty).
    """
    env = env_fn()
    traces: list[EpisodeTrace] = []
    travels: list[float] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            episode_traces, episode_travels = run_episode(env, policy, seed)
            traces.extend(episode_traces)
            travels.extend(episode_travels)
        return traces, travels
    finally:
        env.close()


class DiagnosisProbeCallback(BaseCallback):
    """Run the diagnosis instrument against the live model at checkpoints.

    ``save_freq`` uses callback-call units (the caller converts env
    steps, matching ``CheckpointCallback``). The oracle reference row
    is measured once, lazily, at the first trigger (it is
    policy-independent). Reports land in ``save_dir`` as
    ``diagnosis_probe_<num_timesteps>.txt`` plus
    ``diagnosis_probe_oracle.txt``.

    ``env_fn`` is the probe-env factory; ``train()`` passes the run's
    resolved evaluation factory so a TOML ``[env]`` override reaches
    the diagnosis env too -- the instrument must measure the task the
    run actually trains on, never a silently different stock env.

    Any exception is caught, logged, and disables the callback: a
    broken diagnosis (wrong env for the instrument, a corrupt
    normalizer) must cost one warning, never the training run.
    """

    def __init__(
        self,
        *,
        save_dir: str,
        save_freq: int,
        episodes: int = 30,
        seed_start: int = P_DIAG_SEED_START,
        env_fn: Callable[[], PaddleTennisEnv] = PaddleTennisEnv,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        if save_freq <= 0:
            raise ValueError("save_freq must be positive")
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        self.save_dir = save_dir
        self.save_freq = int(save_freq)
        self.episodes = int(episodes)
        self.seed_start = int(seed_start)
        self.env_fn = env_fn
        self._disabled = False
        self._oracle_written = False

    def _policy(self) -> Callable[[np.ndarray], np.ndarray]:
        vec_normalize = self.model.get_vec_normalize_env()

        def act(observation: np.ndarray) -> np.ndarray:
            normalized = observation
            if vec_normalize is not None:
                batch = vec_normalize.normalize_obs(
                    observation[None, :].astype(np.float64)
                )
                # Box observations normalize to an array, never a dict.
                assert isinstance(batch, np.ndarray)
                normalized = batch[0]
            action, _ = self.model.predict(normalized, deterministic=True)
            return np.asarray(action, dtype=np.float64)

        return act

    def _write(self, filename: str, text: str) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")

    def _run_probe(self) -> None:
        if not self._oracle_written:
            traces, travels = run_player(
                scripted_ground_opponent,
                episodes=self.episodes,
                seed_start=self.seed_start,
                env_fn=self.env_fn,
            )
            self._write(
                "diagnosis_probe_oracle.txt",
                report(
                    traces,
                    "ground oracle (reference)",
                    interpoint_travels=travels,
                ),
            )
            self._oracle_written = True
        traces, travels = run_player(
            self._policy(),
            episodes=self.episodes,
            seed_start=self.seed_start,
            env_fn=self.env_fn,
        )
        self._write(
            f"diagnosis_probe_{self.num_timesteps}.txt",
            report(
                traces,
                f"checkpoint@{self.num_timesteps}",
                interpoint_travels=travels,
            ),
        )
        if self.verbose:
            print(f"[diagnosis] wrote probe report at {self.num_timesteps} steps")

    def _on_step(self) -> bool:
        if self._disabled or self.n_calls % self.save_freq != 0:
            return True
        try:
            self._run_probe()
        except Exception as error:  # noqa: BLE001 -- isolation by design
            self._disabled = True
            print(
                "[diagnosis] probe failed and is disabled for the "
                f"rest of the run: {error!r}"
            )
        return True
