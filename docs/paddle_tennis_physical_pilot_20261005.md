# PaddleTennisPhysical: the context-blind pilot card (2026-10-05)

Kind: **Proposed**. This is a one-page pilot card, not a
pre-registration. The recipe and its instruments are shipped; nothing
has trained yet. Sources: [`repo_review_20261005.md`](repo_review_20261005.md)
§2, §4.2, §7.2 and §7.5, and the `DECISIONS.md` entry "The instrument
batch …".

## Why

The PaddleTennis policy returns the serve (k=1) 90–95% of the time and
converts the second ball (k=2) at most 1%. The pre-freeze diagnostics
([`paddle_tennis_prefreeze_diagnostics_20260830.md`](paddle_tennis_prefreeze_diagnostics_20260830.md)
§4) found the policy gated on the rally-bookkeeping block of its
observation (dims 24–35):

- the same physical k=2 state converts 6.9% when presented as a fresh
  feed, and 1.3–2.0% in its real mid-rally context;
- swapping that block between the two presentations moves the action
  by 1.60 on average;
- the scripted oracle, which reads none of it, plays both
  presentations alike.

The pilot removes that context from the **policy's** observation.

## The bundle

`PaddleTennisPhysical` is the `PaddleTennis` recipe with exactly these
changes, declared together:

| Setting | PaddleTennis | PaddleTennisPhysical |
|---|---|---|
| `observation_profile` | `"full"` (48 values) | `"physical"` (35 values) |
| Normalizer raw tail | indices 24..47 | indices 24..34 |
| `gamma` | 0.99 (SB3 default) | **0.995** |
| Budget | 2M | 3M per seed, run to the end |
| Early-stop patience | 20 evaluations | off |
| `name_prefix` | `paddle_tennis` | `paddle_tennis_physical` |

The `"physical"` policy observation:

- `[0:24]`: the same physical block, except that the ball spin is in
  the world frame (`ball_angular_velocity_world_*`);
- `[24:27]`: `expected_returner_is_own`, `ball_side_is_own`,
  `bounce_count_scaled` = min(bounce_count, 2) / 2;
- `[27:35]`: latch and release-progress state of the four ball contact
  channels.

It drops the rally-phase one-hot, `own_is_serving`, both crossing
flags, `rally_count`, the episode clock, and the four racket–net
contact dims (always zero on this court). SB3 bootstraps time-limit
truncation through `TimeLimit.truncated`, so the clock is not needed
for correctness.

γ = 0.995 is part of the bundle. At 100 Hz control, γ = 0.99 is about
a 100-step horizon, and consecutive policy hits are about 210 steps
apart (0.99^210 ≈ 0.12). WallBall made the same move (DECISIONS,
0.13.0).

Patience is off so that each seed gets the full 3M, as the reference
did. It changes how long a run lasts, not what it learns by a given
step. With the base recipe's patience of 20 evaluations, a run can
stop at eval 40 (1M steps at `eval_freq` 25k). Selection now reads
k≥2 conversions, which sit near zero for most of a run, so new bests
are rare and that stop is likely. The reference selected on crossings,
ran its full 3M, and peaked on k=1 near 2.4M.

## Held fixed

Everything else is inherited from `PaddleTennis` unchanged:

- the task: physics, serve band, the ground oracle as opponent,
  ground rules, continuous n-point play in 1500-step episodes, and the
  contact and reach escrows at 0.25 each;
- SAC with the exploration package (gSDE, `train_freq` 64,
  `ent_coef` `auto_0.02`, target entropy −1.5) and 8 workers;
- selection and success on `episode_rally_returns_a`, with the 0.5/30
  min-deltas and `confirm_best_eval`;
- paired evaluation (seed + 1,000,000, alternating serves), the
  monitor keys, and the degenerate guard: a run that makes no legal
  hit over 5 consecutive flat evaluations still ends early;
- the checkpoint diagnosis: 30 episodes on seeds 5200+, every
  250k steps. Its oracle reference row reads the full layout under
  either profile, so it is identical to a `PaddleTennis` run's.

Only the policy's view changes. Under the same seeds and actions, the
env's rewards, endings and info are bit-identical across the two
profiles (`tests/test_paddle_tennis.py::TestObservationProfile`), and
the default `"full"` profile is bit-identical to the pre-profile env.

## Launch (Colab)

Two runs, one per seed, from `notebooks/sb3_training.ipynb`:

1. Section 1: `REPO_REF = "main"` (once this branch has merged).
2. Section 2: `ENV = "PaddleTennisPhysical"`, `ALGO = "SAC"`,
   `SEED = 0`, `QUICK_TEST = False`. Leave `TOTAL_TIMESTEPS`,
   `N_ENVS`, `EARLY_STOP_PATIENCE` and `MODEL_KWARGS` at `None`: the
   3M budget, patience off and the inherited settings are the
   recipe's. Section 5 should print `total_timesteps=3,000,000` and
   `early_stop_patience=None`.
   `CONFIG_FILE = "auto"` copies `paddle_tennis_physical.toml` into
   the Drive `configs/` folder. Leave that copy unedited.
3. Run all cells.
4. Repeat with `SEED = 1`. Each run gets its own timestamped directory
   under `training_runs/PaddleTennisPhysical/sac/`.

After each run, read k=2 at `best_model` with a cell like this. It
replays the diagnosis on the run's own recorded env (30 episodes,
seeds 5200+), the same instrument and seeds as the in-run reports:

```python
from courtside_dynamics.notebook_utils import score_paddle_stage

reading = score_paddle_stage(
    LOG_DIR,
    bars={
        "k2_receiving": {
            "metric": "k2_receiving_survival",
            "pass_at": 0.05,
            "fail_at": 0.01,
            "higher_is_better": True,
            "gating": False,  # record only; the rule below decides
        },
    },
    report_name="physical_pilot_best_model.json",
)
print(reading["metrics"]["k1_receiving_survival"],
      reading["metrics"]["k2_receiving_survival"])
```

## What to watch

- In `metrics/eval_info.csv`: `episode_rally_returns_a_ep_mean` and
  `success_rate` (the k≥2 headline), and
  `episode_legal_hit_count_a_ep_mean` (is the policy hitting at all?).
- `reports/diagnosis/diagnosis_probe_<steps>.txt`: the "exchange
  survival, policy receiving" line (k=1, k=2, …) at each 250k
  checkpoint, next to `diagnosis_probe_oracle.txt`.
- k=1 should approach the reference band (90–95% receiving). If the
  policy cannot learn k=1 without the context block, the k=2 bar cannot
  pass either. Note it in the write-up as the failure mode.

## Reference to beat

The registered run `20260816_235141` (`PaddleTennis`, 3M steps):
k=1 95% and k=2 at most 1% (final checkpoint 95%/0%; 2.4M checkpoint
95%/1%).

## Decision rule

Read `k2_receiving_survival` at `best_model` on both seeds:

- **ADOPT** if it is **≥ 5% on both seeds**. The context gate was
  binding. Book it in `DECISIONS.md`, and replicate with at least three
  seeds before treating the profile as the next baseline (cardinal
  rule 8).
- **FALSIFIED** if it is **≤ 1% on both seeds**. The gate was not the
  binding constraint. Route to LD1′ demo injection, the fallback
  ([`design_paddle_tennis_demo_injection.md`](design_paddle_tennis_demo_injection.md)).
- **INCONCLUSIVE** otherwise (the seeds split, or either lands between
  1% and 5%). Either run the better seed again at 5M or add a third
  seed (`SEED = 2`) at 3M; the maintainer chooses. The 5M run is a new
  from-scratch run (`TOTAL_TIMESTEPS = 5_000_000`, the same `SEED`) in
  its own run directory. Training has no resume, so it repeats the
  first 3M (not bit for bit on a GPU), and patience is off for it too.

Only the degenerate guard can end a pilot run before its budget.
`stage_summary.txt` records the stop and the steps used. Read such a
seed at its `best_model` like any other, and give the stop step in the
write-up.

## What the pilot cannot attribute

- **Observation vs γ.** The bundle changes both together, so an ADOPT
  cannot say which change did the work. A FALSIFIED does not clear
  either one alone, because each could mask the other. Only a split
  arm (`observation_profile` alone, or γ alone) separates them.
- **Within the observation change.** Dropping the context block also
  changes the spin to the world frame and scales the bounce count. The
  dropped racket–net dims were always zero, so removing them changes
  nothing.
- **Two seeds.** This is a pilot. An ADOPT says the lever is worth a
  registered run, not that a new baseline exists.
