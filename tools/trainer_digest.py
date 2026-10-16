"""SAC freeze, part 2: tiny seeded train() jobs on HEAD vs a base ref.

docs/design_ppo_first_class_20261016.md, Step 0 (the design's
``tools/sac_freeze_digest.py``; its ``tests/test_sac_freeze.py`` is
part 1, shipped as ``tools/recipe_snapshot.py`` plus
``tests/test_recipe_snapshot.py``). Every PR that touches
the training stack must leave the SAC paths bit-identical unless it
declares the change, and the recipe snapshot alone cannot prove that:
it freezes the *configuration*, not what ``train()`` does with it. This
tool runs the same tiny seeded jobs twice -- once against this checkout's
``src/`` (the working tree, uncommitted edits included) and once against
a temporary ``git worktree`` of ``--base`` -- and compares hashes of what
each run produced:

- the final policy ``state_dict`` and, for the SAC family,
  ``log_ent_coef``;
- the final VecNormalize ``obs_rms`` / ``ret_rms`` and its settings
  (read from the pickled stats, never the pickle bytes);
- the monitor CSVs (every column but the wall-clock ``t``);
- ``evaluations.npz``; ``eval_info.csv`` without wall-clock rows;
- the selection fields of ``best_model_meta.json`` (the artifact
  sha256s are dropped: SB3 zips embed timestamps);
- the checkpoint-diagnosis reports (PaddleTennis);
- ``config.json``'s ``resolved_model`` (hyperparameters read back from
  the model), so a change the float32 math rounds away still shows;
- the global numpy, torch and Python RNG states after ``train()``.

Both sides run on this machine, each job in its own subprocess with one
OpenMP/MKL thread, so no cross-platform float drift can enter the
comparison. The sides run under different hash seeds (``PYTHONHASHSEED``
1 for HEAD, 0 for the base): real runs draw a random one, so a training
path that comes to depend on str-hash order (iterating a set of names)
shows up as a mismatch instead of matching under a shared seed. The
worktree lives in a fresh temporary directory (``--tmp-root`` or the
system default) and is removed afterwards, together with the run
directories, unless ``--keep``.

``--base-patch FILE`` applies a patch inside the temporary worktree
only, never to this checkout: the self-test that a deliberate
perturbation is caught.

Usage::

    python tools/trainer_digest.py --base main
    python tools/trainer_digest.py --base HEAD   # determinism self-check
    python tools/trainer_digest.py --base HEAD --base-patch perturb.diff
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

#: eval_info.csv metrics that measure wall clock rather than the run,
#: by exact name. None exists today: the file holds (timestep, metric,
#: value) rows of task metrics only. A substring filter would silently
#: drop real metrics (term_timeout_ep_mean, point_end_second_bounce_*),
#: so a future wall-clock metric must be named here, and until it is,
#: it shows up loudly as a mismatch.
_WALL_CLOCK_METRICS: frozenset[str] = frozenset()


# -- jobs (run inside the subprocess, against that side's package) -----------


def _job_sac_ball_balance(log_dir: str, algo: str = "SAC") -> Any:
    """SAC on BallBalance: 2 envs, 2k steps, four evaluations."""
    from courtside_dynamics.recipes import (
        build_train_config,
        make_env_fn,
        make_eval_env_fn,
    )

    short = {"episode_len": 200}
    return build_train_config(
        "BallBalance",
        algo=algo,
        log_dir=log_dir,
        seed=0,
        n_envs=2,
        total_timesteps=2_000,
        eval_freq=500,
        n_eval_episodes=2,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        env_fn=make_env_fn("BallBalance", env_overrides=short),
        eval_env_fn=make_eval_env_fn("BallBalance", env_overrides=short),
        model_kwargs={"buffer_size": 10_000, "batch_size": 64},
    )


def _job_demosac_off_ball_balance(log_dir: str) -> Any:
    """DemoSAC with demonstrations off: must train exactly like SAC."""
    return _job_sac_ball_balance(log_dir, algo="DEMOSAC")


def _job_sac_paddle_tennis_quick(log_dir: str) -> Any:
    """The PaddleTennis recipe's SAC quick test, shrunk to a tiny budget.

    Keeps the recipe's whole training path -- gSDE, train_freq 64,
    auto entropy with the raised target, the normalizer's excluded
    rally/contact tail, headline selection with confirm_best, paired
    evaluation, monitor info keywords and the checkpoint diagnosis --
    on 300-step episodes and a 20k replay buffer.
    """
    from courtside_dynamics.recipes import (
        RECIPES,
        build_train_config,
        make_env_fn,
        make_eval_env_fn,
    )

    short = {"episode_len": 300}
    model_kwargs = {
        **RECIPES["PaddleTennis"].extra_cfg["model_kwargs"],
        "buffer_size": 20_000,
    }
    return build_train_config(
        "PaddleTennis",
        algo="SAC",
        log_dir=log_dir,
        quick_test=True,
        seed=0,
        n_envs=2,
        total_timesteps=1_536,
        eval_freq=768,
        checkpoint_freq=768,
        n_eval_episodes=2,
        reward_eval_episodes=1,
        checkpoint_diagnosis={"episodes": 1, "seed_start": 5200},
        video_freq=0,
        record_video=False,
        env_fn=make_env_fn("PaddleTennis", env_overrides=short),
        eval_env_fn=make_eval_env_fn("PaddleTennis", env_overrides=short),
        model_kwargs=model_kwargs,
    )


def _job_ppo_ball_balance(log_dir: str) -> Any:
    """PPO on BallBalance: 2 envs x 128-step rollouts, 4 updates."""
    from courtside_dynamics.recipes import (
        build_train_config,
        make_env_fn,
        make_eval_env_fn,
    )

    short = {"episode_len": 200}
    return build_train_config(
        "BallBalance",
        algo="PPO",
        log_dir=log_dir,
        seed=0,
        n_envs=2,
        total_timesteps=1_024,
        eval_freq=512,
        n_eval_episodes=2,
        checkpoint_freq=0,
        video_freq=0,
        record_video=False,
        env_fn=make_env_fn("BallBalance", env_overrides=short),
        eval_env_fn=make_eval_env_fn("BallBalance", env_overrides=short),
        model_kwargs={"n_steps": 128, "batch_size": 64, "n_epochs": 2},
    )


JOBS: dict[str, Callable[[str], Any]] = {
    "sac_ball_balance": _job_sac_ball_balance,
    "demosac_off_ball_balance": _job_demosac_off_ball_balance,
    "sac_paddle_tennis_quick": _job_sac_paddle_tennis_quick,
    "ppo_ball_balance": _job_ppo_ball_balance,
}


# -- hashing (inside the subprocess) -------------------------------------------


def _sha(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True).encode())


def _array_bytes(array: Any) -> bytes:
    import numpy as np

    array = np.asarray(array)
    header = f"{array.dtype.str}{array.shape}".encode()
    return header + np.ascontiguousarray(array).tobytes()


def _locate(log_dir: str, name: str) -> Path | None:
    from courtside_dynamics.training.artifacts import locate_artifact

    found = locate_artifact(log_dir, name)
    return None if found is None else Path(found)


def _hash_state_dict(state_dict: Any) -> str:
    chunks = []
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().numpy()
        chunks.append(key.encode() + b"\x00" + _array_bytes(tensor))
    return _sha(*chunks)


def _hash_normalizer(path: Path | None) -> dict[str, str]:
    if path is None:
        return {"vecnormalize_obs_rms": "absent", "vecnormalize_ret_rms": "absent"}
    with open(path, "rb") as handle:
        normalizer = pickle.load(handle)

    def rms(stats: Any) -> str:
        return _sha(
            _array_bytes(stats.mean),
            _array_bytes(stats.var),
            repr(float(stats.count)).encode(),
        )

    settings = {
        key: repr(getattr(normalizer, key, None))
        for key in (
            "norm_obs",
            "norm_reward",
            "clip_obs",
            "clip_reward",
            "gamma",
            "epsilon",
            "normalize_obs_excluded_indices",
        )
    }
    return {
        "vecnormalize_obs_rms": rms(normalizer.obs_rms),
        "vecnormalize_ret_rms": rms(normalizer.ret_rms),
        "vecnormalize_settings": _json_sha(settings),
    }


def _hash_monitors(log_dir: str) -> str:
    monitor_dir = _locate(log_dir, "monitor_dir")
    if monitor_dir is None:
        return "absent"
    chunks = []
    for path in sorted(monitor_dir.glob("*.monitor.csv")):
        with open(path, encoding="utf-8") as handle:
            handle.readline()  # the JSON header carries the wall-clock t_start
            rows = list(csv.DictReader(handle))
        kept = [{k: v for k, v in row.items() if k != "t"} for row in rows]
        chunks.append(path.name.encode() + _json_sha(kept).encode())
    return _sha(*chunks) if chunks else "absent"


def _hash_evaluations(log_dir: str) -> str:
    import numpy as np

    path = _locate(log_dir, "evaluations")
    if path is None:
        return "absent"
    with np.load(path) as data:
        return _sha(*(key.encode() + _array_bytes(data[key]) for key in sorted(data)))


def _hash_eval_info(log_dir: str) -> str:
    path = _locate(log_dir, "eval_info_csv")
    if path is None:
        return "absent"
    with open(path, encoding="utf-8") as handle:
        rows = [
            row for row in csv.reader(handle) if row[1] not in _WALL_CLOCK_METRICS
        ]
    return _json_sha(rows)


def _hash_best_meta(log_dir: str) -> str:
    path = _locate(log_dir, "best_model_meta")
    if path is None:
        return "absent"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.pop("artifacts", None)  # zip sha256s embed timestamps
    return _json_sha(meta)


def _hash_resolved_model(log_dir: str) -> str:
    """config.json's ``resolved_model``: the kwargs read back from the model.

    Catches a hyperparameter change that the float32 training math
    cannot see (a float64 ULP on the learning rate rounds to the same
    float32 step size, so every trained output stays bit-identical).
    """
    path = _locate(log_dir, "config")
    if path is None:
        return "absent"
    return _json_sha(json.loads(path.read_text(encoding="utf-8"))["resolved_model"])


def _hash_diagnosis(log_dir: str) -> str:
    diagnosis_dir = _locate(log_dir, "diagnosis_dir")
    if diagnosis_dir is None:
        return "absent"
    failures = diagnosis_dir / "diagnosis_probe_failures.txt"
    if failures.exists():
        return "FAILED: " + failures.read_text(encoding="utf-8").strip()
    files = sorted(diagnosis_dir.glob("*.txt"))
    return _sha(*(p.name.encode() + p.read_bytes() for p in files)) if files else "absent"


def run_job(name: str, src: str, log_dir: str, out: str) -> None:
    """Subprocess entry point: train job ``name`` and write its hashes."""
    import random

    import numpy as np
    import torch

    import courtside_dynamics

    package = Path(courtside_dynamics.__file__).resolve()
    if Path(src).resolve() not in package.parents:
        raise RuntimeError(
            f"imported courtside_dynamics from {package}, not from {src}: "
            f"the comparison would be meaningless"
        )
    from courtside_dynamics.training import train

    started = time.monotonic()
    cfg = JOBS[name](log_dir)
    model = train(cfg)
    components: dict[str, str] = {
        "policy_state_dict": _hash_state_dict(model.policy.state_dict()),
        "log_ent_coef": (
            _sha(_array_bytes(model.log_ent_coef.detach().cpu().numpy()))
            if getattr(model, "log_ent_coef", None) is not None
            else "n/a"
        ),
        **_hash_normalizer(_locate(log_dir, "vec_normalize")),
        "monitor_csvs": _hash_monitors(log_dir),
        "evaluations_npz": _hash_evaluations(log_dir),
        "eval_info_csv": _hash_eval_info(log_dir),
        "best_model_meta": _hash_best_meta(log_dir),
        "resolved_model": _hash_resolved_model(log_dir),
        "diagnosis_reports": _hash_diagnosis(log_dir),
        "numpy_rng": _json_sha(
            [repr(part) if not hasattr(part, "tolist") else part.tolist()
             for part in np.random.get_state()]
        ),
        "torch_rng": _sha(torch.get_rng_state().numpy().tobytes()),
        "python_rng": _json_sha(repr(random.getstate())),
        "num_timesteps": str(model.num_timesteps),
    }
    Path(out).write_text(
        json.dumps(
            {
                "job": name,
                "package": str(package.parent),
                "seconds": round(time.monotonic() - started, 1),
                "components": components,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


# -- orchestration ---------------------------------------------------------------


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


#: ``PYTHONHASHSEED`` per side: different on purpose (module doc).
_HASH_SEEDS = {"head": "1", "base": "0"}


def _job_env(side: str, src: Path) -> dict[str, str]:
    """A job subprocess's environment: ``src`` first, one thread, its hash seed."""
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(src), *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": _HASH_SEEDS[side],
    }


def _launch(job: str, side: str, src: Path, scratch: Path) -> dict[str, Any]:
    out = scratch / f"{side}_{job}.json"
    log_dir = scratch / "runs" / side / job
    env = _job_env(side, src)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-job", job,
            "--src", str(src),
            "--log-dir", str(log_dir),
            "--out", str(out),
        ],
        cwd=scratch,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout).strip().splitlines()[-25:]}
    return json.loads(out.read_text(encoding="utf-8"))


def compare_runs(
    jobs: list[str], head: dict[str, Any], base: dict[str, Any]
) -> tuple[list[tuple[str, str, str, str, str]], bool]:
    """Table rows ``(job, component, head, base, status)`` and all-match."""
    rows: list[tuple[str, str, str, str, str]] = []
    ok = True
    for job in jobs:
        sides = (head[job], base[job])
        if any("error" in side for side in sides):
            ok = False
            for label, side in zip(("HEAD", "base"), sides, strict=True):
                if "error" in side:
                    rows.append((job, f"{label} job FAILED", "", "", "ERROR"))
            continue
        a, b = (side["components"] for side in sides)
        for component in sorted(set(a) | set(b)):
            x, y = a.get(component, "missing"), b.get(component, "missing")
            failed = x.startswith("FAILED") or y.startswith("FAILED")
            status = "match" if x == y and not failed else "MISMATCH"
            ok = ok and status == "match"
            rows.append((job, component, x[:16], y[:16], status))
    return rows, ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", help="git ref to compare against")
    parser.add_argument(
        "--job", action="append", dest="jobs", choices=sorted(JOBS),
        help="run only this job (repeatable)",
    )
    parser.add_argument("--base-patch", type=Path, help="apply inside the base worktree")
    parser.add_argument("--parallel", type=int, default=2, help="concurrent jobs")
    parser.add_argument("--tmp-root", type=Path, help="parent of the temporary dir")
    parser.add_argument("--keep", action="store_true", help="keep the temporary dir")
    # Internal: the per-job subprocess.
    parser.add_argument("--run-job", help=argparse.SUPPRESS)
    parser.add_argument("--src", help=argparse.SUPPRESS)
    parser.add_argument("--log-dir", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.run_job:
        run_job(args.run_job, args.src, args.log_dir, args.out)
        return 0
    if not args.base:
        parser.error("--base is required")

    base_sha = _git("rev-parse", "--verify", f"{args.base}^{{commit}}")
    head_sha = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    jobs = args.jobs or list(JOBS)
    if args.tmp_root is not None:
        args.tmp_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="trainer_digest_", dir=args.tmp_root))
    worktree = scratch / "base"
    started = time.monotonic()
    print(f"HEAD side: this checkout's src/ at {head_sha[:12]}"
          f"{' + uncommitted changes' if dirty else ''}")
    print(f"base side: worktree of {args.base} at {base_sha[:12]}"
          f"{f' + patch {args.base_patch}' if args.base_patch else ''}")
    try:
        _git("worktree", "add", "--detach", str(worktree), base_sha)
        if args.base_patch is not None:
            _git("apply", str(args.base_patch.resolve()), cwd=worktree)
        sides = {"head": REPO_ROOT / "src", "base": worktree / "src"}
        tasks = [(job, side) for job in jobs for side in sides]
        with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
            futures = {
                task: pool.submit(_launch, task[0], task[1], sides[task[1]], scratch)
                for task in tasks
            }
            results = {task: future.result() for task, future in futures.items()}
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT, capture_output=True,
        )
        subprocess.run(["git", "worktree", "prune"], cwd=REPO_ROOT, capture_output=True)
        if args.keep:
            print(f"kept {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)

    head = {job: results[(job, "head")] for job in jobs}
    base = {job: results[(job, "base")] for job in jobs}
    rows, ok = compare_runs(jobs, head, base)
    widths = [max(len(row[i]) for row in rows + [("job", "component", "HEAD", "base", "")])
              for i in range(5)]
    print()
    for row in [("job", "component", "HEAD", "base", "status"), *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))
    for job in jobs:
        for label, side in (("HEAD", head[job]), ("base", base[job])):
            if "error" in side:
                print(f"\n{job} ({label}) failed:\n  " + "\n  ".join(side["error"]))
    seconds = {
        job: f"{head[job].get('seconds', '?')}s / {base[job].get('seconds', '?')}s"
        for job in jobs
    }
    print(f"\njob seconds (HEAD / base): {seconds}")
    print(f"total {time.monotonic() - started:.0f}s: "
          + ("every component matches" if ok else "MISMATCH -- the training path "
             "changed; declare it (CHANGELOG / DECISIONS) or fix it"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
