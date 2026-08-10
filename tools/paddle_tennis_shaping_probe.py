"""S1 — contact-shaping incentive-ordering probe (pre-registered).

The scripted witness matrix from
``docs/design_paddle_tennis_contact_shaping.md`` §3: four witnesses
play side A on calibration seeds 5300+ under ``contact_shaping=0.25``
and ``0.0``. Because shaping is reward-side only, the two arms of the
same seed produce bit-identical trajectories, so every criterion is an
exact per-seed identity — no statistics:

- witness-validity precondition (hard-slam), as the decidable
  operationalization of the design's frozen wording ("touches the
  ball in >= 50% of episodes and lands the majority of its strokes
  out"): *touch* := at least one LEGAL hit per episode
  (``event_valid_racket_hit_a`` — a faulting graze counts as no
  touch, the conservative direction), and *strokes out* := majority
  of legal hits unconfirmed (the event-level superset of landed-out
  that also covers net faults and shots still in flight at the cap;
  the design doc records both substitutions);
- statue: exactly zero shaping paid and clawed back;
- every witness, per episode:
  ``sum(rew_shaping) + sum(rew_shaping_clawback)
  == shaping x side-A confirmed returns``;
- every witness, per seed: shaped-minus-unshaped total reward equals
  the same quantity (the escrow's whole undiscounted effect);
- volley-patting: exactly zero shaping paid (fault contacts open no
  escrow, witnessed at the reward level).

Usage:
    python tools/paddle_tennis_shaping_probe.py [--episodes N]
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

#: Reserved for this probe (docs/design_paddle_tennis_contact_shaping.md
#: §3): fresh calibration block; 5200+ is the diagnosis block and
#: 4100-4199 stays reserved for the registered run's held-out gate.
PROBE_SEED_START = 5300
PROBE_EPISODES = 100
SHAPING = 0.25

_IDENTITY_TOL = 1e-9


def _statue(_observation: np.ndarray) -> np.ndarray:
    return np.zeros(3)


WITNESSES: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("statue", _statue),
    ("hard_slam", scripted_hard_slam_witness),
    ("ground_oracle", scripted_ground_opponent),
    ("volley_patting", scripted_net_patting_opponent),
)


@dataclasses.dataclass
class EpisodeRow:
    seed: int
    steps: int
    total_reward: float
    shaping_paid: float
    clawback: float
    hits_a: int
    confirms_a: int


def run_episode(
    env: PaddleTennisEnv,
    witness: Callable[[np.ndarray], np.ndarray],
    seed: int,
) -> EpisodeRow:
    observation, _ = env.reset(seed=seed)
    total = shaping = clawback = 0.0
    hits = confirms = steps = 0
    while True:
        observation, reward, terminated, truncated, info = env.step(
            witness(observation)
        )
        steps += 1
        total += float(reward)
        shaping += float(info["rew_shaping"])
        clawback += float(info["rew_shaping_clawback"])
        hits += int(bool(info["event_valid_racket_hit_a"]))
        confirms += int(bool(info["event_valid_return_a"]))
        if terminated or truncated:
            return EpisodeRow(
                seed=seed,
                steps=steps,
                total_reward=total,
                shaping_paid=shaping,
                clawback=clawback,
                hits_a=hits,
                confirms_a=confirms,
            )


def run_witness(
    witness: Callable[[np.ndarray], np.ndarray],
    *,
    shaping: float,
    episodes: int,
    seed_start: int,
) -> list[EpisodeRow]:
    env = PaddleTennisEnv(contact_shaping=shaping)
    try:
        return [
            run_episode(env, witness, seed)
            for seed in range(seed_start, seed_start + episodes)
        ]
    finally:
        env.close()


def evaluate_criteria(
    results: dict[str, tuple[list[EpisodeRow], list[EpisodeRow]]],
) -> list[tuple[str, bool, str]]:
    """The pre-registered S1 criteria as (name, passed, detail) rows.

    ``results`` maps witness name to (shaped rows, unshaped rows).
    """
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append((name, passed, detail))

    shaped_slam, _ = results["hard_slam"]
    touch_rate = np.mean([row.hits_a > 0 for row in shaped_slam])
    total_hits = sum(row.hits_a for row in shaped_slam)
    total_confirms = sum(row.confirms_a for row in shaped_slam)
    add(
        "precondition: hard_slam touches >= 50% of episodes",
        bool(touch_rate >= 0.5),
        f"touch rate {touch_rate:.0%}",
    )
    add(
        "precondition: hard_slam majority of hits unconfirmed",
        bool(total_confirms < 0.5 * max(total_hits, 1)),
        f"{total_confirms} confirms of {total_hits} hits",
    )

    shaped_statue, _ = results["statue"]
    statue_paid = sum(row.shaping_paid for row in shaped_statue)
    statue_claw = sum(row.clawback for row in shaped_statue)
    add(
        "statue: zero shaping paid and clawed back",
        statue_paid == 0.0 and statue_claw == 0.0,
        f"paid {statue_paid}, clawed {statue_claw}",
    )

    shaped_pat, _ = results["volley_patting"]
    pat_paid = sum(row.shaping_paid for row in shaped_pat)
    add(
        "volley_patting: zero shaping paid (faults open no escrow)",
        pat_paid == 0.0,
        f"paid {pat_paid}",
    )

    for name, (shaped, unshaped) in results.items():
        worst_escrow = 0.0
        worst_total = 0.0
        identical = True
        for srow, urow in zip(shaped, unshaped, strict=True):
            expected = SHAPING * srow.confirms_a
            worst_escrow = max(
                worst_escrow,
                abs(srow.shaping_paid + srow.clawback - expected),
            )
            worst_total = max(
                worst_total,
                abs(srow.total_reward - urow.total_reward - expected),
            )
            identical = identical and (
                srow.steps == urow.steps
                and srow.hits_a == urow.hits_a
                and srow.confirms_a == urow.confirms_a
            )
        add(
            f"{name}: per-episode escrow identity",
            worst_escrow <= _IDENTITY_TOL,
            f"worst |paid+clawback - 0.25*confirms| = {worst_escrow:.2e}",
        )
        add(
            f"{name}: per-seed total-reward identity",
            worst_total <= _IDENTITY_TOL,
            f"worst |shaped-unshaped - 0.25*confirms| = {worst_total:.2e}",
        )
        add(
            f"{name}: arms bit-identical (steps/hits/confirms)",
            identical,
            "trajectory-invariance witnessed",
        )
    return checks


def report(
    results: dict[str, tuple[list[EpisodeRow], list[EpisodeRow]]],
) -> str:
    lines = ["S1 contact-shaping probe"]
    for name, (shaped, _unshaped) in results.items():
        hits = sum(r.hits_a for r in shaped)
        confirms = sum(r.confirms_a for r in shaped)
        paid = sum(r.shaping_paid for r in shaped)
        claw = sum(r.clawback for r in shaped)
        mean_total = np.mean([r.total_reward for r in shaped])
        lines.append(
            f"  {name:<15} hits {hits:>4}  confirms {confirms:>4}  "
            f"paid {paid:>7.2f}  clawed {claw:>8.2f}  "
            f"mean total reward {mean_total:>7.3f}"
        )
    lines.append("")
    checks = evaluate_criteria(results)
    for name, passed, detail in checks:
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name} ({detail})")
    verdict = all(passed for _name, passed, _detail in checks)
    lines.append("")
    lines.append(f"S1 verdict: {'PASS' if verdict else 'FAIL'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=PROBE_EPISODES)
    parser.add_argument("--seed-start", type=int, default=PROBE_SEED_START)
    args = parser.parse_args()

    results: dict[str, tuple[list[EpisodeRow], list[EpisodeRow]]] = {}
    for name, witness in WITNESSES:
        shaped = run_witness(
            witness,
            shaping=SHAPING,
            episodes=args.episodes,
            seed_start=args.seed_start,
        )
        unshaped = run_witness(
            witness,
            shaping=0.0,
            episodes=args.episodes,
            seed_start=args.seed_start,
        )
        results[name] = (shaped, unshaped)
        print(f"[{name}] {args.episodes} episodes x 2 arms done")
    print()
    print(report(results))


if __name__ == "__main__":
    main()
