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
import sys

from courtside_dynamics.envs._paddle_court import (
    scripted_ground_opponent,
)
from courtside_dynamics.training.paddle_diagnosis import (
    P_DIAG_EPISODES,
    P_DIAG_SEED_START,
    native_checkpoint_policy,
    report,
    run_player,
)


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

    traces, travels = run_player(
        policy, episodes=args.episodes, seed_start=args.seed_start
    )
    print(report(traces, label, interpoint_travels=travels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
