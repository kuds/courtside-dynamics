"""RS1 — reach-shaping incentive-ordering probe (pre-registered).

The scripted witness matrix from
``docs/design_paddle_tennis_reach_shaping.md`` §3: five witnesses play
side A on calibration seeds 5500+ under ``reach_shaping=0.25`` (radius
3.0 m) and ``0.0``, plus a stacked arm (``contact_shaping=0.25`` +
``reach_shaping=0.25``) for the oracle. Because shaping is reward-side
only, the arms of the same seed produce bit-identical trajectories, so
every criterion is an exact per-seed identity — no statistics:

- every witness, per episode: ``sum(rew_reach) + sum(rew_reach_clawback)
  == kept`` exactly, where ``kept`` is the tracker-computed sum of
  proximity payments whose opportunity a side-A legal hit then took
  (commit-before-pay ordering, the implementation's);
- every witness, per seed: shaped-minus-unshaped total reward equals
  the same ``kept`` exactly (the escrow's whole undiscounted effect);
- every positive payment coincides with a processed first-bounce event
  (``event_first_bounce``) — pay never fires without a live bounce;
- statue and off-line camper: ``kept == 0`` on every episode (net
  exactly zero — camping is not farmable), with the camper collecting
  a payment in >= 50% of episodes (the witness actually exercises the
  pay path) and the statue in >= 1 episode of the block;
- ground oracle: keeps >= 90% of what it is paid (it takes nearly
  every opportunity), and every kept advance rides on a real hit;
- hard-slam: keeps >= 50% of what it is paid (touch-then-out still
  banks the reach advance — reach pays positioning, not aim);
- volley-patting: whatever it is paid nets to its kept sum exactly
  (pre-bounce faults create no live bounce, so pay only ever follows
  the balls it fails to volley);
- stacked arm (oracle): shaped-minus-unshaped total equals
  ``kept_reach + 0.25 x side-A confirms`` exactly — the two escrows
  compose additively.

Usage:
    python tools/paddle_tennis_reach_probe.py [--episodes N]
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
)
from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv

#: Reserved for this probe (docs/design_paddle_tennis_reach_shaping.md
#: §3): fresh calibration block. 5200-5299 is the diagnosis block,
#: 5300-5399 the contact-shaping block, 5400-5499 the n-point block;
#: 4100-4199 and 4300-4399 stay reserved.
PROBE_SEED_START = 5500
PROBE_EPISODES = 100
REACH = 0.25
RADIUS = 3.0

_IDENTITY_TOL = 1e-9
_RESERVED_BLOCKS = ((4100, 4199), (4300, 4399), (5200, 5299), (5300, 5399), (5400, 5499))


def _statue(_observation: np.ndarray) -> np.ndarray:
    return np.zeros(3)


def _camper(_observation: np.ndarray) -> np.ndarray:
    """Parked near the serve landing depth, ~1 m off the ball line."""
    return np.array([-0.61, 0.33, 0.0])


@dataclasses.dataclass
class WitnessTotals:
    episodes: int = 0
    paid: float = 0.0
    clawed: float = 0.0
    kept: float = 0.0
    hits: int = 0
    confirms: int = 0
    episodes_paid: int = 0
    reward_shaped: float = 0.0
    reward_unshaped: float = 0.0
    identity_worst: float = 0.0
    pay_without_bounce: int = 0


def _run_witness(
    name: str,
    policy: Callable[[np.ndarray], np.ndarray],
    episodes: int,
    seed_start: int,
    *,
    stack_contact: bool = False,
) -> WitnessTotals:
    shaped = PaddleTennisEnv(
        reach_shaping=REACH,
        reach_shaping_radius=RADIUS,
        contact_shaping=0.25 if stack_contact else 0.0,
    )
    unshaped = PaddleTennisEnv()
    totals = WitnessTotals()
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = shaped.reset(seed=seed)
            mirror_obs, _ = unshaped.reset(seed=seed)
            np.testing.assert_array_equal(obs, mirror_obs)
            episode_paid = episode_clawed = 0.0
            kept = pending = 0.0
            shaped_total = unshaped_total = 0.0
            confirms = 0
            while True:
                action = policy(obs)
                obs, reward, term, trunc, info = shaped.step(action)
                mirror_obs, mirror_reward, mterm, mtrunc, _ = unshaped.step(action)
                np.testing.assert_array_equal(obs, mirror_obs)
                assert (term, trunc) == (mterm, mtrunc)
                shaped_total += reward
                unshaped_total += mirror_reward
                confirms += int(bool(info["event_valid_return_a"]))
                if info["event_valid_racket_hit_a"]:
                    totals.hits += 1
                    kept += pending
                    pending = 0.0
                if info["rew_reach"] > 0.0 and not info["event_first_bounce"]:
                    totals.pay_without_bounce += 1
                pending += info["rew_reach"]
                episode_paid += info["rew_reach"]
                episode_clawed += info["rew_reach_clawback"]
                if term or trunc:
                    break
            identity_gap = abs((episode_paid + episode_clawed) - kept)
            totals.identity_worst = max(totals.identity_worst, identity_gap)
            if identity_gap > _IDENTITY_TOL:
                raise AssertionError(
                    f"{name} seed {seed}: paid+clawed != kept "
                    f"({episode_paid + episode_clawed} vs {kept})"
                )
            expected_delta = kept + (0.25 * confirms if stack_contact else 0.0)
            if abs((shaped_total - unshaped_total) - expected_delta) > _IDENTITY_TOL:
                raise AssertionError(
                    f"{name} seed {seed}: shaped-unshaped total "
                    f"{shaped_total - unshaped_total} != expected {expected_delta}"
                )
            totals.episodes += 1
            totals.paid += episode_paid
            totals.clawed += episode_clawed
            totals.kept += kept
            totals.confirms += confirms
            totals.episodes_paid += int(episode_paid > 0.0)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=PROBE_EPISODES)
    parser.add_argument("--seed-start", type=int, default=PROBE_SEED_START)
    args = parser.parse_args()
    _refuse_reserved(args.seed_start, args.episodes)

    witnesses: list[tuple[str, Callable[[np.ndarray], np.ndarray], bool]] = [
        ("statue", _statue, False),
        ("camper", _camper, False),
        ("ground_oracle", scripted_ground_opponent, False),
        ("hard_slam", scripted_hard_slam_witness, False),
        ("volley_patting", scripted_net_patting_opponent, False),
        ("oracle_stacked", scripted_ground_opponent, True),
    ]

    rows = []
    failures: list[str] = []
    for name, policy, stacked in witnesses:
        totals = _run_witness(
            name, policy, args.episodes, args.seed_start, stack_contact=stacked
        )
        rows.append((name, totals))
        if totals.pay_without_bounce:
            failures.append(f"{name}: {totals.pay_without_bounce} pays without a bounce")
        if name in ("statue", "camper") and totals.kept != 0.0:
            failures.append(f"{name}: kept {totals.kept} != 0 (farmable)")
        if name == "camper" and totals.episodes_paid < totals.episodes * 0.5:
            failures.append(
                f"camper: paid in only {totals.episodes_paid}/{totals.episodes} episodes"
            )
        if name == "statue" and totals.episodes_paid < 1:
            failures.append("statue: never paid — the pay path was not exercised")
        if name == "ground_oracle" and totals.kept < 0.9 * totals.paid:
            failures.append(
                f"ground_oracle: kept {totals.kept:.3f} < 90% of paid {totals.paid:.3f}"
            )
        if name == "hard_slam" and totals.kept < 0.5 * totals.paid:
            failures.append(
                f"hard_slam: kept {totals.kept:.3f} < 50% of paid {totals.paid:.3f}"
            )

    print("RS1 reach-shaping witnesses "
          f"(episodes={args.episodes}, seeds {args.seed_start}+)")
    header = (
        f"  {'witness':<15} {'paid':>9} {'clawed':>9} {'kept':>9} "
        f"{'hits':>5} {'conf':>5} {'eps paid':>9} {'mean rew (shaped)':>18}"
    )
    print(header)
    for name, t in rows:
        print(
            f"  {name:<15} {t.paid:>9.3f} {t.clawed:>9.3f} {t.kept:>9.3f} "
            f"{t.hits:>5d} {t.confirms:>5d} "
            f"{t.episodes_paid:>4d}/{t.episodes:<4d} "
            f"{t.reward_shaped / max(t.episodes, 1):>18.3f}"
        )
    print(f"  worst per-episode identity gap: "
          f"{max(t.identity_worst for _, t in rows):.2e}")
    if failures:
        for failure in failures:
            print(f"  [FAIL] {failure}")
        raise SystemExit(1)
    print("RS1 verdict: PASS (all identities exact; anti-farming witnessed)")


if __name__ == "__main__":
    main()
