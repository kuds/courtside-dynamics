# Courtside Dynamics

A progression of MuJoCo environments and deep-RL training scripts aimed at
a single long-term goal: **teach humanoid agents to rally and play tennis**.

The curriculum climbs from simple ball control up to full racket-and-wall
rallying, and (eventually) humanoid tennis players. Each environment is a
stepping stone; each one is trained with Stable-Baselines3 on the same
shared pipeline.

## Environments

| Env id                                  | Source                                    | Description                                                                 |
|-----------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|
| `Humanoid-v5`                           | stock `gymnasium`                         | Baseline locomotion task.                                                   |
| `CourtsideDynamics/BallBalance-v0`      | `courtside_dynamics.envs.ball_balance`    | Keep a ball on a 6-DOF tray.                                                |
| `CourtsideDynamics/BallBounce-v0`       | `courtside_dynamics.envs.ball_bounce`     | Juggle a ball on a 6-DOF paddle.                                            |
| `CourtsideDynamics/WallBall-v0`         | `courtside_dynamics.envs.wall_ball`       | Hit a ball into a wall with a 4-DOF paddle.                                 |
| `CourtsideDynamics/TennisWall-v0`       | `courtside_dynamics.envs.tennis_wall`     | Rally a ball against a wall with a 5-DOF racket and shaped reward.          |
| `CourtsideDynamics/HumanoidTennis-v0`   | *planned*                                 | Full humanoid tennis -- the project north star.                             |

![](/Images/sac_humanoid.gif)
![](/Images/sac_ball_balance.gif)
![](/Images/sac_ball_bounce.gif)
![](/Images/sac_wall_ball.gif)

## Layout

```
courtside-dynamics/
├── pyproject.toml                        # pinned deps + package metadata
├── src/courtside_dynamics/
│   ├── assets/*.xml                      # MJCF model files
│   ├── envs/                             # Gymnasium environments
│   ├── callbacks/video_record.py         # unified video + CSV recorder
│   ├── training/train.py                 # SAC / PPO training entry point
│   └── colab_setup.py                    # Colab EGL bootstrap
├── tests/test_envs.py                    # env_checker + reward sanity
└── *.ipynb                               # experiment notebooks (slim wrappers)
```

## Installation

```bash
# From a clone of this repo
pip install -e ".[train,notebooks]"
```

The base install pulls only `mujoco`, `gymnasium`, and `numpy`. The
`train` extra adds `stable-baselines3`, `torch`, `tensorboard`,
`pandas`, and `matplotlib`. The `notebooks` extra adds `mediapy` and
`jupyter`.

## Quick start

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

## Tests

```bash
pytest
```

The suite runs the Stable-Baselines3 `env_checker` on every env plus a
random-rollout sanity check. The Wall Ball reward-signal test is
currently `xfail(strict=True)` -- see the docstring in
`tests/test_envs.py` for the context.

## Results

Hardware: Google Colab T4.

Ball Balance caps at +1 per step, so the reported reward is dominated by
the episode length (750 here) — a non-crashing policy will score near the
ceiling regardless of the algorithm.

| Simulation Type | Model Type | Average Reward | Training Time | Total Training Steps |
|-----------------|------------|----------------|---------------|----------------------|
| Humanoid-v5     | SAC        | 6579.66        | 4:41:52       | 3,800,000            |
| Ball Balance    | PPO        | 751            | 1:58:54       | 2,000,000            |
| Ball Balance    | SAC        | 751            | 1:52:23       | 2,000,000            |
| Ball Bounce     | SAC        | 9.65           | 1:57:47       | 2,000,000            |
| WallBall-v0     | SAC        | *in progress*  |               |                      |
| TennisWall-v0   |            | *planned*      |               |                      |
