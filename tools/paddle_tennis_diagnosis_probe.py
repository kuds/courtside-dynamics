"""PaddleTennis ground-era diagnosis: where do the learned rallies die?

The ground-era pilot (run ``20260808_022106``) solved the serve return
(success 0.97) and plateaued at ~1.3 crossings — 6x below the scripted
band (7.78) — with ``out_of_bounds`` ending 83-100% of episodes, and
the run video shows the paddle wandering after its first hit. This
probe turns that video impression into numbers by instrumenting every
exchange of every point, for the learned checkpoint AND the ground
oracle on the same seeds (the oracle row calibrates every metric):

- **Shot ledger** (per racket hit): did the shot cross, where did it
  land (depth, lateral, in/out), or did it die in the net — split by
  hitter and by exchange index, so "the policy's second stroke lands
  long" and "the oracle mishandles the policy's returns" are separate
  rows, not one confounded taxonomy.
- **Exchange survival**: P(the point reaches the policy's k-th hit).
- **Point-ender attribution**: who struck the ball that ended the
  point, and how (shot out / shot net / receiver never reached it /
  volley fault / cap).
- **Recovery hold** (the "zipping all over the court" number): paddle
  XY path length from 0.3 s after the policy's own legal hit (swing
  follow-through excluded) until the opponent's next legal hit — a
  trained player holds ground while its shot is away (the oracle's
  hold-low recovery); an untrained tail wanders.
- **Ready-position error**: when an incoming ball bounces on the
  policy's side, the paddle's XY distance from the bounce point at
  bounce time, and whether the policy touches the ball at all before
  the point ends.

Hypotheses this separates (the pilot cannot): H1 credit starvation
(the policy's returns land out, so exchange 2+ is never rewarded and
its behavior there is untrained noise), H2 stroke miscalibration only
(returns land out but positioning is fine), H3 opponent asymmetry
(policy returns land in; the oracle's replies land out).

Usage::

    # Reference row (local, no checkpoint):
    python tools/paddle_tennis_diagnosis_probe.py --oracle

    # The learned checkpoint (run where the artifacts are mounted):
    python tools/paddle_tennis_diagnosis_probe.py \
        --model .../model/best_model.zip \
        --vec-normalize .../model/best_vec_normalize.pkl

Seeds default to calibration block **5200-5299** (recorded in the
diagnosis snapshot's ledger); the reserved block 4100-4199 is never
touched here.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from collections.abc import Callable

import numpy as np

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
        normalized = normalizer.normalize_obs(
            observation[None, :].astype(np.float64)
        )[0]
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
) -> EpisodeTrace:
    """Play one point and walk its event stream into an EpisodeTrace.

    Reads ``env._last_transition`` (the rules transition of the step
    just taken) for ordered semantic events with positions -- the same
    private surface the P3 harness reads on the prototype scene; the
    info dict does not carry event positions.
    """
    observation, reset_info = env.reset(seed=seed)
    serve_side_is_policy = reset_info["serve_side_is_policy"] >= 0.5
    feed_receiver = (
        CourtSide.B if serve_side_is_policy else CourtSide.A
    )

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
    while True:
        observation, _, terminated, truncated, info = env.step(
            policy(observation)
        )
        raw = env.observation_for_side(CourtSide.A)
        paddle = raw[9:12]
        if window_countdown > 0:
            window_countdown -= 1
            if window_countdown == 0:
                window_active = True
        elif window_active:
            window_travel += float(
                np.linalg.norm(paddle[:2] - previous_paddle[:2])
            )
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
                    else PADDLE_COURT.is_in_bounds(
                        event.position[0], event.position[1]
                    )
                )
                incoming_first_bounce = False
                if (
                    open_shot is not None
                    and open_shot.crossed
                    and open_shot.outcome == "open"
                ):
                    open_shot.landing_depth = abs(
                        float(event.position[0])
                    )
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
                    incoming_first_bounce = (
                        feed_receiver is CourtSide.A and in_bounds
                    )
                if (
                    incoming_first_bounce
                    and event.kind is RallyEventKind.BALL_COURT_A
                ):
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
        if terminated or truncated:
            break

    if awaiting_touch:
        touched_after_bounce.append(False)
    close_window()

    after = env._last_transition.after
    termination = after.termination_reason
    if (
        open_shot is not None
        and open_shot.outcome == "open"
        and termination is TerminationReason.BALL_NET
    ):
        open_shot.outcome = "net"

    return EpisodeTrace(
        serve_side_is_policy=serve_side_is_policy,
        shots=shots,
        policy_hits=hits_by_side[CourtSide.A],
        crossings=int(info["crossings"]),
        termination=info["termination_reason_name"],
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


def _shot_row(shots: list[ShotRecord], label: str) -> str:
    n = len(shots)
    if n == 0:
        return f"  {label:<26} n=0"
    crossed = sum(s.crossed for s in shots)
    landed = [s for s in shots if s.landed_in is not None]
    landed_in = [s for s in shots if s.landed_in]
    depths = [
        s.landing_depth for s in landed_in if s.landing_depth is not None
    ]
    out_depths = [
        s.landing_depth
        for s in landed
        if s.landed_in is False and s.landing_depth is not None
    ]
    return (
        f"  {label:<26} n={n:<4} crossed {crossed / n:4.0%} "
        f"in {len(landed_in) / n:4.0%}"
        + (
            f"  in-depth {float(np.mean(depths)):.2f}"
            if depths
            else ""
        )
        + (
            f"  out-depth {float(np.mean(out_depths)):.2f}"
            if out_depths
            else ""
        )
    )


def report(traces: list[EpisodeTrace], label: str) -> str:
    lines = [f"player: {label}  ({len(traces)} episodes)"]

    policy_shots = [
        s for t in traces for s in t.shots if s.hitter is CourtSide.A
    ]
    opponent_shots = [
        s for t in traces for s in t.shots if s.hitter is CourtSide.B
    ]
    lines.append("shot ledger:")
    lines.append(_shot_row(policy_shots, "policy, all"))
    receive = [t for t in traces if not t.serve_side_is_policy]
    serving = [t for t in traces if t.serve_side_is_policy]
    for split_label, split in (("receiving", receive), ("serving", serving)):
        split_shots = [
            s
            for t in split
            for s in t.shots
            if s.hitter is CourtSide.A
        ]
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
            f"({len(split)} eps): "
            + "  ".join(
                f"k={k}:{p:.0%}" for k, p in enumerate(survival, 1)
            )
        )

    enders = Counter(t.ender for t in traces)
    lines.append(
        "point enders: "
        + ", ".join(f"{k} {v}" for k, v in enders.most_common())
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
    lines.append(
        f"crossings mean {np.mean([t.crossings for t in traces]):.2f}; "
        "terminations: "
        + ", ".join(
            f"{k} {v}"
            for k, v in Counter(
                t.termination for t in traces
            ).most_common(4)
        )
    )
    return "\n".join(lines)


def run_player(
    policy: Callable[[np.ndarray], np.ndarray],
    *,
    episodes: int,
    seed_start: int,
) -> list[EpisodeTrace]:
    env = PaddleTennisEnv()
    try:
        return [
            run_episode(env, policy, seed)
            for seed in range(seed_start, seed_start + episodes)
        ]
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=P_DIAG_EPISODES)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=P_DIAG_SEED_START,
        help="first calibration seed (see the snapshot's ledger)",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="instrument the ground oracle (reference row; the "
        "default when no --model is given)",
    )
    parser.add_argument("--model", help="SB3 SAC checkpoint")
    parser.add_argument(
        "--vec-normalize", help="frozen normalizer (.pkl)"
    )
    args = parser.parse_args(argv)
    if args.model and not args.vec_normalize:
        parser.error("--model requires --vec-normalize")

    if args.model:
        policy = native_checkpoint_policy(
            args.model, args.vec_normalize
        )
        label = f"checkpoint:{args.model}"
    else:
        policy = scripted_ground_opponent
        label = "ground oracle (reference)"

    traces = run_player(
        policy, episodes=args.episodes, seed_start=args.seed_start
    )
    print(report(traces, label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
