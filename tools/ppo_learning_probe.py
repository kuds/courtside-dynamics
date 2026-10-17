"""PPO learning probe: does the BallBalance PPO profile still learn?

PPO plan Step 12 (``docs/design_ppo_first_class_20261016.md``), the
manual learning bar beside the CI smoke test (the 1-D reach in
``tests/test_train.py``). It trains BallBalance with its recipe's PPO
profile exactly as ``build_train_config`` resolves it (Step 3: 8
workers x 512-step rollouts, minibatch 256, 10 epochs, lr 3e-4,
target_kl 0.03, log_std_init -1, ent_coef 0, reward normalization on)
through ``train()`` for at most 330 s of wall clock, then scores the
protected best model behind its own normalizer, deterministically, on
20 held-out seeds (70,000-70,019, far from the run's derived training
and evaluation seeds). PASS when the mean reaches 700 of the 750-step
episode. The bar needs learned balancing: the zero action balances
about 470 +/- 18 steps, and std-1 exploration noise drops the ball in
about 120. The audit measured 750 on all 20 seeds by 320k steps.

It runs for minutes of CPU, so it is a nightly or workflow_dispatch
job, never CI. Exit status 0 on PASS, 1 on FAIL. The run directory is
kept (a fresh temporary one unless ``--log-dir`` names one) so a FAIL
can be read from its ``stage_summary.txt``, whose Training Health
section carries the on-policy flags.

Usage::

    OMP_NUM_THREADS=1 python tools/ppo_learning_probe.py
    python tools/ppo_learning_probe.py --log-dir /tmp/probe --json probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

try:
    from tools._checkout import checkout_package_mismatch
except ModuleNotFoundError:  # run as a script: tools/ itself is on sys.path
    from _checkout import checkout_package_mismatch  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent

RECIPE = "BallBalance"
#: The wall-clock training cap, in seconds (the audit's measured run
#: reached 750/750 inside it on four shared CPUs).
WALL_CLOCK_CAP_SECONDS = 330.0
#: Held-out scoring: seeds ``[start, start + episodes)``.
HELD_OUT_SEED_START = 70_000
HELD_OUT_EPISODES = 20
#: The bar on the held-out mean (episodes are 750 steps).
PASS_AT = 700.0


class WallClockCap(BaseCallback):
    """Stop ``learn()`` once ``seconds`` of wall clock have passed.

    Sets ``stop_reason`` like the trainer's own guards, so the cap is
    named in ``stage_summary.txt``.
    """

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = float(seconds)
        self.stop_reason: str | None = None
        self._start = 0.0

    def _on_training_start(self) -> None:
        self._start = time.monotonic()

    def _on_step(self) -> bool:
        if time.monotonic() - self._start <= self.seconds:
            return True
        self.stop_reason = (
            f"wall_clock_cap: the probe's {self.seconds:.0f} s training cap"
        )
        return False


def score_best_model(
    log_dir: str, *, seed_start: int, episodes: int
) -> list[float]:
    """Deterministic returns of ``log_dir``'s best model on held-out seeds.

    The model loads through the algorithm its run recorded
    (``algos.load_run_model``) and acts on observations normalized by
    the ``best_vec_normalize.pkl`` saved with it, on the recipe's
    evaluation environment.
    """
    from courtside_dynamics.recipes import make_eval_env_fn
    from courtside_dynamics.training.algos import load_run_model
    from courtside_dynamics.training.artifacts import locate_artifact

    model_path = locate_artifact(log_dir, "best_model")
    normalizer_path = locate_artifact(log_dir, "best_vec_normalize")
    if model_path is None or normalizer_path is None:
        raise FileNotFoundError(
            f"{log_dir} holds no best_model.zip / best_vec_normalize.pkl pair"
        )
    model = load_run_model(model_path, "PPO", device="cpu")
    with open(normalizer_path, "rb") as handle:
        normalizer = pickle.load(handle)
    normalizer.training = False
    env_fn = make_eval_env_fn(RECIPE)
    returns = []
    for seed in range(seed_start, seed_start + episodes):
        env = env_fn()
        try:
            obs, _ = env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                action, _ = model.predict(
                    normalizer.normalize_obs(obs[None]), deterministic=True
                )
                obs, reward, terminated, truncated, _ = env.step(action[0])
                total += float(reward)
                done = bool(terminated or truncated)
        finally:
            env.close()
        returns.append(total)
    return returns


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--log-dir",
        default=None,
        help="run directory (default: a fresh temporary directory, kept)",
    )
    parser.add_argument("--seed", type=int, default=0, help="training seed")
    parser.add_argument(
        "--seconds",
        type=float,
        default=WALL_CLOCK_CAP_SECONDS,
        help="wall-clock training cap",
    )
    parser.add_argument("--json", default=None, help="write the result here")
    args = parser.parse_args(argv)

    problem = checkout_package_mismatch(REPO_ROOT)
    if problem is not None:
        print(problem, file=sys.stderr)
        return 2

    from courtside_dynamics.recipes import build_train_config
    from courtside_dynamics.training import train

    log_dir = args.log_dir or tempfile.mkdtemp(prefix="ppo_learning_probe_")
    if os.path.isdir(log_dir) and os.listdir(log_dir):
        parser.error(f"--log-dir {log_dir} is not empty")
    cap = WallClockCap(args.seconds)
    cfg = build_train_config(
        RECIPE,
        algo="PPO",
        log_dir=log_dir,
        seed=args.seed,
        record_video=False,
        video_freq=0,
        checkpoint_freq=0,
        extra_callbacks=(cap,),
    )
    print(
        f"[probe] {RECIPE} PPO profile: n_envs={cfg.n_envs} "
        f"model_kwargs={cfg.model_kwargs} eval_freq={cfg.eval_freq:,} "
        f"cap={args.seconds:.0f} s -> {log_dir}",
        flush=True,
    )
    started = time.monotonic()
    model = train(cfg)
    trained_seconds = time.monotonic() - started

    returns = score_best_model(
        log_dir, seed_start=HELD_OUT_SEED_START, episodes=HELD_OUT_EPISODES
    )
    mean = statistics.fmean(returns)
    verdict = "PASS" if mean >= PASS_AT else "FAIL"
    result: dict[str, Any] = {
        "recipe": RECIPE,
        "algo": "PPO",
        "seed": args.seed,
        "log_dir": log_dir,
        "model_kwargs": cfg.model_kwargs,
        "n_envs": cfg.n_envs,
        "wall_clock_cap_seconds": args.seconds,
        "train_seconds": round(trained_seconds, 1),
        "timesteps_trained": int(model.num_timesteps),
        "sb3_n_updates": int(model._n_updates),
        "stop_reason": cap.stop_reason,
        "held_out_seeds": [
            HELD_OUT_SEED_START,
            HELD_OUT_SEED_START + HELD_OUT_EPISODES - 1,
        ],
        "held_out_returns": returns,
        "held_out_mean": mean,
        "held_out_min": min(returns),
        "pass_at": PASS_AT,
        "verdict": verdict,
    }
    print(
        f"[probe] trained {model.num_timesteps:,} steps "
        f"({model._n_updates} SB3 updates) in {trained_seconds:.0f} s; "
        f"best model on seeds {HELD_OUT_SEED_START}-"
        f"{HELD_OUT_SEED_START + HELD_OUT_EPISODES - 1}: mean {mean:.1f} "
        f"(min {min(returns):.0f}, max {max(returns):.0f}); "
        f"bar {PASS_AT:.0f}: {verdict}"
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
