"""PH1 — post-swing-hold incentive-ordering probe (pre-registered).

The scripted witness matrix from
``docs/design_paddle_tennis_postswing_hold.md`` §3: witnesses play
side A on fresh calibration seeds 6200+ under ``hold_shaping=0.25``
(travel budget 4.0 m) and ``0.0``, plus a stacked arm
(``hold + reach + contact`` at 0.25 each) for the oracle. Shaping is
reward-side only, so the arms of one seed are bit-identical
trajectories and every criterion is an exact per-seed identity:

- every witness, per episode: ``sum(rew_hold) + sum(rew_hold_clawback)
  == kept`` exactly, where ``kept`` is the tracker-computed sum of
  hold payments whose follow-up side-A legal hit (the k=2 hit)
  happened — keep-before-arm ordering, the implementation's;
- every witness, per seed: shaped-minus-unshaped total reward equals
  the same ``kept`` exactly (plus the sibling escrows' exact terms on
  the stacked arm);
- every positive payment coincides with an opponent legal racket hit
  (``event_valid_racket_hit_b``) — pay never fires without the return
  strike that defines the window's end;
- statue, camper, and volley-patting: **zero pay everywhere** — no
  side-A legal hit ever arms a window (the escrow is strictly
  post-swing; serve returns against a non-hitter pay nothing);
- ``hit_then_freeze`` (the farming attempt this design must defeat):
  collects near-full pay at the opponent's return strike in a healthy
  fraction of episodes and keeps **exactly zero** of it — no second
  hit, no realized reward;
- ground oracle: the only witness that completes second exchanges —
  kept pay strictly positive, riding on real k=2 hits;
- stacked arm (oracle): shaped-minus-unshaped total equals
  ``kept_hold + kept_reach + 0.25 x side-A confirms`` exactly — the
  three escrows compose additively.

Usage:
    python tools/paddle_tennis_hold_probe.py [--episodes N]
        [--seed-start S]
"""
from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Callable

import numpy as np

from courtside_dynamics.envs._paddle_court import (
    scripted_ground_opponent,
    scripted_hard_slam_witness,
    scripted_net_patting_opponent,
    scripted_reach_camper_witness,
    scripted_statue_witness,
)
from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv

#: Fresh block burned by this probe
#: (docs/design_paddle_tennis_postswing_hold.md §3/§5).
PROBE_SEED_START = 6200
PROBE_EPISODES = 100
HOLD = 0.25
TRAVEL_BUDGET = 4.0

_IDENTITY_TOL = 1e-9
_RESERVED_BLOCKS = (
    (4100, 4199),
    (4300, 4399),
    (5200, 5299),
    (5300, 5399),
    (5400, 5499),
    (5500, 5599),
    (5600, 6199),
)


@dataclasses.dataclass
class WitnessTotals:
    episodes: int = 0
    paid: float = 0.0
    clawed: float = 0.0
    kept: float = 0.0
    hits: int = 0
    confirms: int = 0
    episodes_paid: int = 0
    episodes_kept: int = 0
    reward_shaped: float = 0.0
    reward_unshaped: float = 0.0
    identity_worst: float = 0.0
    pay_without_strike: int = 0


def _run_witness(
    name: str,
    policy: Callable[[np.ndarray], np.ndarray],
    episodes: int,
    seed_start: int,
    *,
    stack_siblings: bool = False,
    freeze_after_first_hit: bool = False,
) -> WitnessTotals:
    shaped = PaddleTennisEnv(
        hold_shaping=HOLD,
        hold_shaping_travel=TRAVEL_BUDGET,
        reach_shaping=0.25 if stack_siblings else 0.0,
        contact_shaping=0.25 if stack_siblings else 0.0,
    )
    unshaped = PaddleTennisEnv()
    totals = WitnessTotals()
    zero = np.zeros(3, dtype=np.float64)
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = shaped.reset(seed=seed)
            mirror_obs, _ = unshaped.reset(seed=seed)
            np.testing.assert_array_equal(obs, mirror_obs)
            episode_paid = episode_clawed = 0.0
            kept_hold = pending_hold = 0.0
            kept_reach = pending_reach = 0.0
            shaped_total = unshaped_total = 0.0
            confirms = 0
            frozen = False
            while True:
                action = zero if frozen else policy(obs)
                obs, reward, term, trunc, info = shaped.step(action)
                mirror_obs, mirror_reward, mterm, mtrunc, _ = unshaped.step(action)
                np.testing.assert_array_equal(obs, mirror_obs)
                assert (term, trunc) == (mterm, mtrunc)
                shaped_total += reward
                unshaped_total += mirror_reward
                confirms += int(bool(info["event_valid_return_a"]))
                # Mirror the env's escrow rule for both escrows: a
                # side-A legal hit keeps any prior pending advance,
                # and a payment sharing that hit's step is kept
                # immediately, never escrowed.
                if info["event_valid_racket_hit_a"]:
                    totals.hits += 1
                    kept_hold += pending_hold + info["rew_hold"]
                    pending_hold = 0.0
                    kept_reach += pending_reach + info["rew_reach"]
                    pending_reach = 0.0
                    if freeze_after_first_hit:
                        frozen = True
                else:
                    pending_hold += info["rew_hold"]
                    pending_reach += info["rew_reach"]
                if info["rew_hold"] > 0.0 and not info["event_valid_racket_hit_b"]:
                    totals.pay_without_strike += 1
                episode_paid += info["rew_hold"]
                episode_clawed += info["rew_hold_clawback"]
                if term or trunc:
                    break
            identity_gap = abs((episode_paid + episode_clawed) - kept_hold)
            totals.identity_worst = max(totals.identity_worst, identity_gap)
            if identity_gap > _IDENTITY_TOL:
                raise AssertionError(
                    f"{name} seed {seed}: paid+clawed != kept "
                    f"({episode_paid + episode_clawed} vs {kept_hold})"
                )
            expected_delta = kept_hold + (
                kept_reach + 0.25 * confirms if stack_siblings else 0.0
            )
            if abs((shaped_total - unshaped_total) - expected_delta) > _IDENTITY_TOL:
                raise AssertionError(
                    f"{name} seed {seed}: shaped-unshaped total "
                    f"{shaped_total - unshaped_total} != expected {expected_delta}"
                )
            totals.episodes += 1
            totals.paid += episode_paid
            totals.clawed += episode_clawed
            totals.kept += kept_hold
            totals.confirms += confirms
            totals.episodes_paid += int(episode_paid > 0.0)
            totals.episodes_kept += int(kept_hold > 0.0)
            totals.reward_shaped += shaped_total
            totals.reward_unshaped += unshaped_total
        return totals
    finally:
        shaped.close()
        unshaped.close()


def _refuse_reserved(seed_start: int, episodes: int) -> None:
    span = range(seed_start, seed_start + episodes)
    for low, high in _RESERVED_BLOCKS:
        if any(low <= seed <= high for seed in span):
            raise SystemExit(
                f"seed range [{seed_start}, {seed_start + episodes}) intersects "
                f"reserved/burned block {low}-{high}; refuse to run"
            )


def main() -> None:
    global HOLD, TRAVEL_BUDGET
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=PROBE_EPISODES)
    parser.add_argument("--seed-start", type=int, default=PROBE_SEED_START)
    parser.add_argument("--hold-shaping", type=float, default=HOLD)
    parser.add_argument("--travel-budget", type=float, default=TRAVEL_BUDGET)
    args = parser.parse_args()
    _refuse_reserved(args.seed_start, args.episodes)
    HOLD = args.hold_shaping
    TRAVEL_BUDGET = args.travel_budget

    # (name, policy, stacked, freeze_after_first_hit)
    witnesses: list[
        tuple[str, Callable[[np.ndarray], np.ndarray], bool, bool]
    ] = [
        ("statue", scripted_statue_witness, False, False),
        ("camper", scripted_reach_camper_witness, False, False),
        ("hit_then_freeze", scripted_ground_opponent, False, True),
        ("ground_oracle", scripted_ground_opponent, False, False),
        ("hard_slam", scripted_hard_slam_witness, False, False),
        ("volley_patting", scripted_net_patting_opponent, False, False),
        ("oracle_stacked", scripted_ground_opponent, True, False),
    ]

    rows = []
    failures: list[str] = []
    for name, policy, stacked, freeze in witnesses:
        totals = _run_witness(
            name,
            policy,
            args.episodes,
            args.seed_start,
            stack_siblings=stacked,
            freeze_after_first_hit=freeze,
        )
        rows.append((name, totals))
        if totals.pay_without_strike:
            failures.append(
                f"{name}: {totals.pay_without_strike} pays without a return strike"
            )
        if name in ("statue", "camper", "volley_patting") and totals.paid != 0.0:
            failures.append(
                f"{name}: paid {totals.paid} != 0 (a window armed without a hit)"
            )
        if name == "hit_then_freeze":
            if totals.kept != 0.0:
                failures.append(
                    f"hit_then_freeze: kept {totals.kept} != 0 (farmable)"
                )
            if totals.episodes_paid < totals.episodes * 0.25:
                failures.append(
                    f"hit_then_freeze: paid in only "
                    f"{totals.episodes_paid}/{totals.episodes} episodes"
                )
        if name in ("ground_oracle", "oracle_stacked") and totals.kept <= 0.0:
            failures.append(f"{name}: kept {totals.kept} — no second exchange banked")

    print(
        "PH1 post-swing-hold witnesses "
        f"(episodes={args.episodes}, seeds {args.seed_start}+)"
    )
    header = (
        f"  {'witness':<16} {'paid':>9} {'clawed':>9} {'kept':>9} "
        f"{'hits':>5} {'conf':>5} {'eps paid':>9} {'eps kept':>9} "
        f"{'mean rew (shaped)':>18}"
    )
    print(header)
    for name, t in rows:
        print(
            f"  {name:<16} {t.paid:>9.3f} {t.clawed:>9.3f} {t.kept:>9.3f} "
            f"{t.hits:>5d} {t.confirms:>5d} "
            f"{t.episodes_paid:>4d}/{t.episodes:<4d} "
            f"{t.episodes_kept:>4d}/{t.episodes:<4d} "
            f"{t.reward_shaped / max(t.episodes, 1):>18.3f}"
        )
    print(
        f"  worst per-episode identity gap: "
        f"{max(t.identity_worst for _, t in rows):.2e}"
    )
    if failures:
        for failure in failures:
            print(f"  [FAIL] {failure}")
        raise SystemExit(1)
    print("PH1 verdict: PASS (all identities exact; farming defeated)")


if __name__ == "__main__":
    main()
