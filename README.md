# Courtside Dynamics

[![CI](https://github.com/kuds/courtside-dynamics/actions/workflows/ci.yml/badge.svg)](https://github.com/kuds/courtside-dynamics/actions/workflows/ci.yml)

A progression of MuJoCo environments and deep-RL training scripts aimed at
a single long-term goal: **teach humanoid agents to rally and play tennis**.

The curriculum climbs from simple ball control up to full racket-and-wall
rallying, and (eventually) humanoid tennis players. Each environment is a
stepping stone; each one is trained with Stable-Baselines3 on the same
shared pipeline.

## Environments

| Env id                                  | Source                                    | Description                                                                 |
|-----------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| `CourtsideDynamics/BallBalance-v0`      | `courtside_dynamics.envs.ball_balance`    | Keep a ball on a 6-DOF tray.                                                |
| `CourtsideDynamics/BallBounce-v0`       | `courtside_dynamics.envs.ball_bounce`     | Juggle a ball on a 6-DOF paddle.                                            |
| `CourtsideDynamics/WallBall-v2`         | `courtside_dynamics.envs.wall_ball`       | Rally a ball against a wall with a 5-DOF racket and a gated wall-hit reward. |
| `CourtsideDynamics/HumanoidTennisCoop-v0` | `courtside_dynamics.envs.humanoid_tennis` | Centralized cooperative two-G1 physical rally environment.                   |

![](/Images/sac_ball_balance.gif)
![](/Images/sac_ball_bounce.gif)
![](/Images/sac_wall_ball.gif)

### Humanoid tennis development status

Phases 1–4 provide a regulation court, physical tennis ball and net, two
29-DoF Unitree G1 models, rigid right-wrist rackets, ordered substep contact
events, a deterministic cooperative rally state machine, and the registered
centralized Gymnasium environment. It exposes one normalized 58-value action
vector; zero is the two-player standing-reference PD hold:

- player A: `[0:29]`;
- player B: `[29:58]`;
- player A right arm/racket subset: `[22:29]`;
- player B right arm/racket subset: `[51:58]`.

The centralized observation has 299 labeled values with stable public slices:
player A proprioception `[0:71]`, player B proprioception `[71:142]`, physical
racket A `[142:157]`, physical racket B `[157:172]`, ball
position/velocity/spin `[172:181]`, useful ball-relative coordinates
`[181:193]`, rally state `[193:221]`, contact-latch state `[221:241]`, and the
active-action mask `[241:299]`. The mask is all ones in the default
free-standing environment and selects the learned controls in a constrained
curriculum instance, so every supported stage retains the same 58/299 API.

The G1 simulation assets are pinned from MuJoCo Menagerie under BSD-3-Clause;
see `THIRD_PARTY_NOTICES.md`. The physical G1 hardware is proprietary, and its
inclusion does not imply Unitree endorsement. [LATENT](https://zzk273.github.io/LATENT/)
is useful research evidence for G1 tennis and a future motion-prior training
path, but its code is not a dependency of this Gymnasium/SB3 implementation.

The rule reducer validates a return only after it crosses the net and is then
volleyed or lands in bounds, suppresses duplicate contact episodes, and reports
one explicit fault reason for double bounces, out balls, illegal hits, net
contacts, or unsafe simulation. The shared default reward is +1 only for that
confirmed legal return, -1 for an ordinary fault, and -2 for unsafe/non-finite
physics; survival, feed crossings, first bounces, and unconfirmed racket taps
pay zero. Optional hit shaping is escrowed and clawed back on a failed return.

Reset feeds alternate sides, use bounded seeded noise, start 1.1–1.5 m high
near the baseline, and have nonzero velocity toward the other player. Reset and
step `info` retain the serve side and full initial ball qpos/qvel for recording.
`scripted_policies.run_humanoid_tennis_oracle` supplies a mirrored deterministic
physics/rules integration harness: a real ball hits a real wrist-mounted racket,
crosses back over the net, and lands in bounds without mid-rally state writes.
It is a feasibility fixture, not evidence of learned humanoid tennis.
The constrained `run_humanoid_tennis_stage0_oracle` and
`run_humanoid_tennis_stage1_oracle` similarly prove a physical intercept and a
timed target return on both mirrored sides. The Stage 1 timing is deliberately
sensitive and is not presented as a robust scripted tennis policy.

Phase 4 adds immutable, fixed-per-environment curriculum presets. Stage 0 is a
fixed-pelvis two-shoulder intercept of a slow deterministic physical feed;
Stage 1 requires a fixed-pelvis right-arm shot to land in a generous target;
Stage 2 adds bounded seeded launch randomization. The learned player alternates
with the serve side, and the observation's action mask exposes the active slice.
Stage 0 activates only shoulder pitch/roll (`[22:24]` for A or `[51:53]` for
B); Stages 1–2 activate the seven-value right-arm slice shown above.
Both G1 pelvises are held by real MuJoCo weld constraints in these anchored
tasks, all inactive joints receive standing-reference PD targets, and early
contact forgiveness enlarges only the massless physical stringbed dimensions.

The registry describes planned stages 3–5, but selecting one raises an explicit
`NotImplementedError`; no fake partner or unvalidated foot constraint is
substituted. Stage 6 exposes the existing two-free-standing environment mode
without a training recipe. This milestone does not claim that two free-standing
humanoids can be trained end to end with vanilla PPO or SAC.

Curriculum parameters stay fixed for an environment instance. The v0
observation includes the action mask but not target, launch-distribution, or
racket-scale context, so reset-time stage mixing would be partially observed.
The promotion evaluator instead uses a pinned, mirrored low-discrepancy launch
suite with 100 distinct initial ball states and an advisory 80% current-stage /
75% prior-stage-retention gate, with zero unsafe episodes required by default.
Evaluation evidence records the exact curriculum and held-out-suite
fingerprints, environment, package source/Git revision, live SB3 parameter
hash, checkpoint hash, observation-normalization artifact hash, and exact
Python/MuJoCo/Gymnasium/NumPy/Torch/SB3 runtime versions. Only the
canonical preset, recipe horizon, default rule/contact settings, and standard
held-out suite are promotion-eligible; easier custom variants remain useful
experiments but cannot pass the gate. Normalized policies must supply their
frozen `VecNormalize` snapshot; the evaluator loads and applies that exact
artifact rather than accepting a separate transform callback. Raw observations
cannot be mislabeled as normalized. Promotion requires fresh
evidence for every earlier implemented stage from that same policy and
normalizer. Automatic checkpoint promotion, replay-buffer migration, and mixed
prior-level rehearsal are not implemented.

`HumanoidTennisCoopSmoke` is intentionally only a 10,000-step integration
recipe for exercising SB3, checkpoints, compact video CSV recording, and eval
metrics. Its eval output includes fault-type episode rates and the
min/p50/p90/max rally-length distribution; it is not a learning baseline.

Three experimental fixed-stage recipes are available:
`HumanoidTennisStage0Intercept`, `HumanoidTennisStage1AnchoredReturn`, and
`HumanoidTennisStage2RandomizedReturn`. They default to experimental PPO runs;
their budgets and SB3 settings are starting points, not evidence of learned
convergence. SAC remains selectable explicitly, but its entropy tuner sees all
58 dimensions and is not mask-aware; PPO also wastes exploration on inactive
coordinates, so neither choice establishes full-task feasibility.

## Layout

```
courtside-dynamics/
├── pyproject.toml                        # deps + package metadata
├── src/courtside_dynamics/
│   ├── assets/                           # MJCF, racket, court, and licensed robot assets
│   ├── envs/                             # Gymnasium environments and tennis curriculum contracts
│   ├── callbacks/
│   │   ├── video_record.py               # unified video + CSV recorder
│   │   └── info_dict_eval.py             # per-episode info aggregates -> TB/CSV
│   ├── training/
│   │   ├── train.py                      # SAC / PPO training entry point
│   │   ├── algos.py                      # algo-name -> SB3 class registry
│   │   ├── artifacts.py                  # config.json / stage_summary.txt writers + artifact registry
│   │   ├── monitor_log.py                # wall-clock-ordered monitor CSV loader
│   │   └── tennis_curriculum.py          # held-out metrics and advisory promotion gate
│   ├── recipes.py                        # env+algo presets used by the notebook
│   ├── notebook_utils.py                 # Drive mount, plots, replay, run report, artifact audit
│   ├── scripted_policies.py              # hand-coded oracles for env validation
│   └── colab_setup.py                    # Colab EGL bootstrap
├── tests/                                # env, training, callback, recipe, notebook-helper tests
└── notebooks/sb3_training.ipynb          # one Colab driver for the whole curriculum
```

## Installation

Courtside Dynamics supports Python 3.11 through 3.13.

```bash
# From a clone of this repo
pip install -e ".[train,notebooks]"
```

The base install pulls `mujoco`, `gymnasium`, `numpy`, and the small
`imageio`/`packaging` runtime dependencies imported by Gymnasium's MuJoCo
module. The `train` extra adds `stable-baselines3`, `torch`, `tensorboard`,
`pandas`, `matplotlib`, and `moviepy` for video recording. The `notebooks`
extra adds `mediapy`; it intentionally does
not install Jupyter itself (wherever the notebook runs, Jupyter already
exists -- and on Colab installing it conflicts with `google-colab`'s
pinned `jupyter-server`). The `dev` extra adds `pytest`, `ruff`, and
`mypy` for working on the repo.

## Quick start

The supported path is a recipe: it fills in the per-env defaults
(constructor kwargs, training budget, success metric, CSV columns) that
the notebook relies on.

```python
from courtside_dynamics.recipes import build_train_config
from courtside_dynamics.training import train

cfg = build_train_config("WallBall", algo="SAC", log_dir="./logs/WallBall")
model = train(cfg)
```

Or configure everything by hand:

```python
from courtside_dynamics.envs import BallBounceEnv
from courtside_dynamics.training import TrainConfig, train

cfg = TrainConfig(
    env_fn=lambda: BallBounceEnv(render_mode="rgb_array", min_force=100.0),
    algo="SAC",
    total_timesteps=1_500_000,
    log_dir="./logs/BallBounce",
    name_prefix="ball_bounce",
)
model = train(cfg)
```

Or, using the registered gymnasium ids:

```python
import gymnasium
import courtside_dynamics  # noqa: F401  (triggers registration)

env = gymnasium.make("CourtsideDynamics/BallBounce-v0")
```

The cooperative tennis environment uses the same ordinary Gymnasium/SB3
interface (not PettingZoo):

```python
env = gymnasium.make("CourtsideDynamics/HumanoidTennisCoop-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.unwrapped.neutral_action)
```

The first constrained physical task can be constructed directly and checked
with its non-neutral mirrored oracle:

```python
from courtside_dynamics.envs import HumanoidTennisCoopEnv
from courtside_dynamics.scripted_policies import run_humanoid_tennis_stage0_oracle

env = HumanoidTennisCoopEnv(curriculum_config=0, episode_len=150)
result = run_humanoid_tennis_stage0_oracle(env, serving_side="a", seed=0)
assert result.stage_success
```

For promotion, call `training.evaluate_curriculum_stage` with a fixed-stage
environment factory plus stable `policy_id` and `normalization_id` values. Pass
the evaluated SB3 checkpoint as `policy_artifact_path`. When
`normalization_id` is not `"raw"`, also pass the frozen `VecNormalize` snapshot
as `normalization_artifact_path`; the evaluator deserializes it and applies its
own `normalize_obs` method. The IDs are human-readable labels; eligibility uses
hashes derived from the live policy and those files. Then pass the resulting
summary and fresh summaries for every earlier implemented stage to
`training.assess_curriculum_promotion`. The report is JSON-compatible and
advisory only; it never mutates a recipe, environment, checkpoint, or training
run.

Pass `seed=...` to `build_train_config` / `TrainConfig` when comparing
reward or env tweaks: it makes the whole run reproducible (SB3, the
training workers, and every helper env get derived, non-overlapping
seeds), so run-to-run noise doesn't masquerade as a real difference.

## Run artifacts & troubleshooting

Every `train(cfg)` run leaves a self-describing `log_dir` -- you can
answer "how was this model produced, and why did it underperform?" from
disk alone, after the Colab runtime is gone:

| Artifact | What it answers |
|----------|-----------------|
| `config.json` | Exact env/algo/hyperparameters (incl. SB3-resolved defaults), package versions, GPU, git SHA. |
| `stage_summary.txt` | End-of-run report: final/best eval, duration, throughput, device, final `train/*` health metrics. |
| `evaluations.npz`, `eval_info.csv` | Deterministic eval curve + per-episode info aggregates (success rate, bounce/hit counts, termination-cause breakdown). |
| `tensorboard/`, `tensorboard/progress.csv` | Live scalars and their CSV mirror (SAC `ent_coef`/losses, PPO `explained_variance`/`approx_kl`, ...). |
| `monitor/`, `checkpoints/`, `videos/` | Per-episode training returns, periodic snapshots, rollout videos. |
| `best_model.zip` + `best_vec_normalize.pkl` | Best policy plus the obs-normalization stats from the moment it was saved. |

`courtside_dynamics.notebook_utils` turns those into diagnostics:
`print_stage_summary` replays the report, `check_run_artifacts` audits
the directory and explains anything missing (eval never fired, moviepy
absent, ...), and `plot_learning_curve` / `plot_eval_info` /
`plot_training_health` chart the CSVs. The notebook runs all of them
after training.

## Tests

```bash
pip install -e ".[train,dev]"
pytest
```

The suite runs the Stable-Baselines3 `env_checker` on every env plus
random-rollout sanity checks, and a targeted Wall Ball reward suite
(`TestWallBallRewardGate`) that pins down the gated wall reward,
per-cycle paddle bonus, shaping clawback, and termination flags with
an oracle-vs-noop comparison. Callbacks, the training entry point, the
recipe registry, monitor-log loading, and the notebook report/audit
helpers are covered by their own test modules. Humanoid-tennis Phase 1 adds
court/ball/racket calibration, collision-mask, two-G1 namespacing, action-slice,
rigid wrist-mount, checksum/license, and finite-rollout coverage. Phase 2 adds
pure mirrored rally traces plus MuJoCo substep tests for semantic contact
rising edges, release hysteresis, sampled force peaks, net crossings, court
classification, net faults, and unsafe-state detection. Phase 3 locks the
58-action/299-observation Gymnasium API, seeded alternating serves, registration,
reward/info invariants, finite rollouts, rendering, and a mirrored physical
legal-return oracle. Phase 4 adds preset and target-boundary validation, exact
alternating action masks, seeded randomized launches, physical weld stability,
inactive-action/PD-hold invariants, a mirrored Stage 0 active-vs-neutral oracle,
curriculum reward/termination precedence, recipe schemas, and deterministic
held-out promotion metrics.

## Results

Hardware: Google Colab T4 (the results below). The notebook's `n_envs`
guidance assumes the newer L4 runtime; both work.

Ball Balance caps at +1 per step, so the reported reward is dominated by
the episode length (750 here) — a non-crashing policy will score near the
ceiling regardless of the algorithm. (The 751s below predate the fix for
an off-by-one that ran every episode one step past `episode_len`; new
runs cap at exactly 750.)

| Simulation Type | Model Type | Average Reward | Training Time | Total Training Steps |
|-----------------|------------|----------------|---------------|----------------------|
| Ball Balance    | PPO        | 751            | 1:58:54       | 2,000,000            |
| Ball Balance    | SAC        | 751            | 1:52:23       | 2,000,000            |
| Ball Bounce     | SAC        | 9.65           | 1:57:47       | 2,000,000            |
| WallBall-v2     | SAC        | *in progress*  |               |                      |

## Blog Posts
- [Serving Up Some Robotics: Setting Up a Tennis Environment in MuJoCo
](https://www.findingtheta.com/blog/serving-up-some-robotics-setting-up-a-tennis-environment-in-mujoco)
