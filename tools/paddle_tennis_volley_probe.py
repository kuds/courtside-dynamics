"""PaddleTennis ground-rules probe: does the bounce rule kill the loop?

The first learned GPU run (``20260803_004559``, the volley-era record)
maximized cooperative return rate with a close-net volley loop -- a
net crossing every ~14 control steps, 91.6% of time in
``return_in_flight``, crossings still climbing at 37.6 when the budget
ended. This probe measures the proposed fix (the
``require_bounce_before_return`` rules profile, env kwarg
``volley_rule="fault"``) against three scripted players on a shared
seed block:

- **ground**  -- the era's reference controller
  (``ground_lead_charge_local_action``: run-up wait behind the
  predicted landing, soft calibrated ground stroke, hold-low
  recovery);
- **volley**  -- the frozen P1 ``lead_charge`` port, whose returns
  are largely volleys (measured in bring-up: its home-column
  pre-positioning intercepts the serve's descent);
- **patting** -- the adversarial net-play witness
  (``net_patting_local_action``), a scripted approximation of the
  learned loop's style (it volleys every return at close range; it
  does not reproduce the learned loop's full cadence or persistence,
  which remain documented by the GPU run itself).

Each player plays BOTH sides (same controller through the P4 mirror)
under both rule profiles, 100 episodes per cell.

Pre-registered adoption criteria (committed before the probe block
was drawn; bring-up calibration used the burned 1000-block):

- **A. Kill**: under ``fault``, the volley and patting rows collapse
  to mean crossings <= 0.2 with volley_return >= 80% of terminations.
- **B. Feasibility**: under ``fault``, the ground row sustains mean
  crossings >= 2.6 (the volley-era certification floor) with a
  >=1-crossing rate >= 0.60.
- **C. Cadence floor**: the ground row's cadence is >= 60 control
  steps per crossing (no fast loop is even representable).
- **D. Rule-neutrality**: the ground row's mean crossings under
  ``fault`` and ``legal`` agree within two combined standard errors
  (the rule leaves non-volley play untouched).

Adopt ``volley_rule="fault"`` as the era default iff all four hold.

Seeds: calibration block **5100-5199** (recorded in the ground-rules
snapshot's ledger). The reserved blocks 4100-4199 (volley-era learned
gate) and 4200-4299 (designated for the ground-era held-out
certification) are never touched here.

Usage::

    python tools/paddle_tennis_volley_probe.py            # full matrix
    python tools/paddle_tennis_volley_probe.py --quick    # smoke
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from collections.abc import Callable

import numpy as np

from courtside_dynamics.envs._paddle_court import (
    scripted_ground_opponent,
    scripted_lead_charge_opponent,
    scripted_net_patting_opponent,
)
from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv

PROBE_SEED_START = 5100
PROBE_EPISODES = 100

PLAYERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ground": scripted_ground_opponent,
    "volley": scripted_lead_charge_opponent,
    "patting": scripted_net_patting_opponent,
}


@dataclasses.dataclass(slots=True)
class CellResult:
    """One player x rule cell of the probe matrix."""

    player: str
    volley_rule: str
    episodes: int
    mean_crossings: float
    std_crossings: float
    ge1_rate: float
    mean_returns: float
    cadence_steps_per_crossing: float
    volley_fault_fraction: float
    truncated: int
    terminations: Counter

    def row(self) -> str:
        top = ", ".join(
            f"{name} {count}"
            for name, count in self.terminations.most_common(3)
        )
        cadence = (
            f"{self.cadence_steps_per_crossing:.0f}"
            if np.isfinite(self.cadence_steps_per_crossing)
            else "-"
        )
        return (
            f"| {self.player} | {self.volley_rule} "
            f"| {self.mean_crossings:.2f} | {self.ge1_rate:.0%} "
            f"| {self.mean_returns:.2f} | {cadence} "
            f"| {self.volley_fault_fraction:.0%} | {self.truncated} "
            f"| {top} |"
        )


def run_cell(
    player: str,
    volley_rule: str,
    *,
    episodes: int,
    seed_start: int,
) -> CellResult:
    """Play ``episodes`` seeded points, the same controller both sides."""
    controller = PLAYERS[player]
    env = PaddleTennisEnv(
        volley_rule=volley_rule, opponent_controller=controller
    )
    crossings: list[int] = []
    returns: list[float] = []
    steps: list[int] = []
    truncated_count = 0
    terminations: Counter = Counter()
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, _ = env.reset(seed=seed)
            step_count = 0
            while True:
                observation, _, terminated, truncated, info = env.step(
                    controller(observation)
                )
                step_count += 1
                if terminated or truncated:
                    break
            crossings.append(int(info["crossings"]))
            returns.append(float(info["valid_return_count"]))
            steps.append(step_count)
            truncated_count += int(truncated)
            terminations[info["termination_reason_name"]] += 1
    finally:
        env.close()

    values = np.asarray(crossings, dtype=np.float64)
    total_crossings = int(values.sum())
    return CellResult(
        player=player,
        volley_rule=volley_rule,
        episodes=episodes,
        mean_crossings=float(values.mean()),
        std_crossings=(
            float(values.std(ddof=1)) if episodes > 1 else 0.0
        ),
        ge1_rate=float((values >= 1).mean()),
        mean_returns=float(np.mean(returns)),
        cadence_steps_per_crossing=(
            float(np.sum(steps)) / total_crossings
            if total_crossings
            else float("inf")
        ),
        volley_fault_fraction=(
            terminations["volley_return"] / episodes
        ),
        truncated=truncated_count,
        terminations=terminations,
    )


def evaluate_criteria(cells: dict[tuple[str, str], CellResult]) -> list[str]:
    """Apply the pre-registered adoption criteria; return verdict lines."""
    lines = []
    kill_ok = True
    for player in ("volley", "patting"):
        cell = cells[(player, "fault")]
        ok = (
            cell.mean_crossings <= 0.2
            and cell.volley_fault_fraction >= 0.8
        )
        kill_ok &= ok
        lines.append(
            f"A kill [{player}]: crossings {cell.mean_crossings:.2f} "
            f"<= 0.2 and volley faults {cell.volley_fault_fraction:.0%} "
            f">= 80% -> {'PASS' if ok else 'FAIL'}"
        )
    ground_fault = cells[("ground", "fault")]
    feasible = (
        ground_fault.mean_crossings >= 2.6
        and ground_fault.ge1_rate >= 0.60
    )
    lines.append(
        f"B feasibility: ground/fault crossings "
        f"{ground_fault.mean_crossings:.2f} >= 2.6 and ge1 "
        f"{ground_fault.ge1_rate:.0%} >= 60% -> "
        f"{'PASS' if feasible else 'FAIL'}"
    )
    cadence_ok = ground_fault.cadence_steps_per_crossing >= 60.0
    lines.append(
        f"C cadence: {ground_fault.cadence_steps_per_crossing:.0f} "
        f"steps/crossing >= 60 -> {'PASS' if cadence_ok else 'FAIL'}"
    )
    ground_legal = cells[("ground", "legal")]
    combined_se = float(
        np.hypot(
            ground_fault.std_crossings
            / np.sqrt(ground_fault.episodes),
            ground_legal.std_crossings
            / np.sqrt(ground_legal.episodes),
        )
    )
    delta = abs(
        ground_fault.mean_crossings - ground_legal.mean_crossings
    )
    neutral = delta <= 2.0 * combined_se
    lines.append(
        f"D rule-neutrality: |{ground_fault.mean_crossings:.2f} - "
        f"{ground_legal.mean_crossings:.2f}| = {delta:.2f} <= "
        f"2 x SE ({2.0 * combined_se:.2f}) -> "
        f"{'PASS' if neutral else 'FAIL'}"
    )
    verdict = kill_ok and feasible and cadence_ok and neutral
    lines.append(
        "VERDICT: "
        + (
            "ADOPT volley_rule='fault' as the era default"
            if verdict
            else "DO NOT ADOPT -- record and diagnose"
        )
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", type=int, default=PROBE_EPISODES
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=PROBE_SEED_START,
        help="first calibration seed (see the snapshot's ledger)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="4-episode smoke matrix"
    )
    args = parser.parse_args(argv)
    episodes = 4 if args.quick else args.episodes

    cells: dict[tuple[str, str], CellResult] = {}
    for player in PLAYERS:
        for volley_rule in ("fault", "legal"):
            cells[(player, volley_rule)] = run_cell(
                player,
                volley_rule,
                episodes=episodes,
                seed_start=args.seed_start,
            )

    print(
        "Ground-rules probe "
        f"({episodes} episodes/cell, seeds {args.seed_start}-"
        f"{args.seed_start + episodes - 1}; same controller both sides)"
    )
    print()
    print(
        "| player | rule | crossings | >=1 | returns | steps/x "
        "| volley faults | truncated | top terminations |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(cells):
        print(cells[key].row())
    print()
    for line in evaluate_criteria(cells):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
