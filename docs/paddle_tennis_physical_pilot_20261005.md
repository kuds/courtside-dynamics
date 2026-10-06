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
truncation through `TimeLimit.truncated`, so the critic's targets stay
correct without the clock. One reward does depend on it: the
truncation step claws back any escrow still pending (−0.25 for a
pending contact escrow), and without the clock the policy cannot see
that step coming. It lands at most once per 1500-step episode. With
the oracle playing side A under the pilot's kwargs, 2 of 6 episodes
(seeds 5200–5205) ended on a −0.25 contact clawback.

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
  min-deltas and `confirm_best_eval` (held fixed against today's
  `PaddleTennis` recipe, not against the reference run, which predates
  it: see "Reference to beat");
- paired evaluation (seed + 1,000,000, alternating serves), the
  monitor keys, and the degenerate guard: a run that makes no legal
  hit over 5 consecutive flat evaluations still ends early;
- the checkpoint diagnosis: 30 episodes on seeds 5200+, every
  250k steps. Its oracle reference row reads the full layout under
  either profile, so it is identical to a `PaddleTennis` run's.

Only the policy's view changes. Under the same seeds and actions, the
env's rewards, endings and info are bit-identical across the two
profiles (`tests/test_paddle_tennis.py::TestObservationProfile`), and
the default `"full"` profile is bit-identical to the pre-profile env
(`TestFullProfileReference` replays streams recorded from b9585e2).

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
3. Insert the reading cell below as a new code cell directly above
   section 10, "Disconnect Colab runtime". Run all ends in that
   section's `disconnect_runtime` call, which releases the runtime, so
   a cell placed after it never runs.
4. Run all cells.
5. Repeat with `SEED = 1`. Each run gets its own timestamped directory
   under `training_runs/PaddleTennisPhysical/sac/`.

The reading cell reads k=2 at `best_model`. It replays the diagnosis
on the run's own recorded env (30 episodes, seeds 5200+), the same
instrument and seeds as the in-run reports, and prints the receiving
points and the conversion count with the rate:

```python
from courtside_dynamics.notebook_utils import score_paddle_stage

reading = score_paddle_stage(
    LOG_DIR,
    bars={
        # The decision rule's bands in the whole percents the diagnosis
        # report prints: PASS = ADOPT band (prints >= 5%), FAIL =
        # FALSIFIED band (prints <= 1%), MIDDLE = between.
        "k2_receiving": {
            "metric": "k2_receiving_survival",
            "pass_at": 0.045,
            "fail_at": 0.015,
            "higher_is_better": True,
            "gating": False,  # record only; the rule reads both seeds
        },
    },
    report_name="physical_pilot_best_model.json",
)
metrics = reading["metrics"]
points = metrics["receiving_points"]
k1 = metrics["k1_receiving_survival"]
k2 = metrics["k2_receiving_survival"]
band = {"PASS": "ADOPT band", "MIDDLE": "between", "FAIL": "FALSIFIED band"}[
    reading["bars"]["k2_receiving"]["verdict"]
]
print(
    f"receiving points {points}: k=1 {k1:.1%}; "
    f"k=2 {round(k2 * points)} of {points} = {k2:.2%} "
    f"(prints {k2:.0%}) -> {band}"
)
```

If the runtime is already gone, read the run from a fresh one. Run
section 1 only, then a cell with
`from courtside_dynamics.notebook_utils import mount_drive`,
`mount_drive()` and `LOG_DIR` set by hand to the finished run,
`"/content/drive/MyDrive/Finding Theta/courtside-dynamics/training_runs/PaddleTennisPhysical/sac/<YYYYMMDD_HHMMSS>"`,
then the reading cell. Do not re-run section 3: `resolve_run_dir`
always creates a new, empty timestamped directory, which the scorer
refuses (`FileNotFoundError`), and it leaves that empty directory next
to the pilot runs.

## What to watch

- In `metrics/eval_info.csv`: `episode_rally_returns_a_ep_mean` and
  `success_rate` (the k≥2 headline), and
  `episode_legal_hit_count_a_ep_mean` (is the policy hitting at all?).
- `reports/diagnosis/diagnosis_probe_<steps>.txt`: the "exchange
  survival, policy receiving" line (k=1, k=2, …) at each 250k
  checkpoint, next to `diagnosis_probe_oracle.txt`. The line gives the
  receiving points too; at about 85, one conversion prints as 1%.
- k=1 should approach the reference band (90–95% receiving). If the
  policy cannot learn k=1 without the context block, the k=2 bar cannot
  pass either. Note it in the write-up as the failure mode.

## Reference to beat

The registered run `20260816_235141` (`PaddleTennis`, from scratch,
3M steps): k=1 95% and k=2 at most 1% at every checkpoint (1% at
eleven, 0% at nineteen; final checkpoint 95%/0%; its `best_model`,
2.4M, 95%/1%).

Those are the in-run report's whole-percent prints. This block holds
about 81–86 receiving points for that lineage, so a printed "1%" is
one conversion, about 1.2% as an exact fraction. The decision rule
below is read in the same units.

The reference's `best_model` was chosen by the recipe of its day:
headline `crossings`, success `legal_hit_count_a`, one scalar 0.25
min-delta, unpaired evaluation. The pilot's is chosen on
`episode_rally_returns_a`, k≥2 conversions themselves, with paired
evaluation. `DECISIONS.md` (the instrument-batch entry) says
`best_model` picks are not comparable across that change. The
per-checkpoint readings above are not affected by selection, and no
reference checkpoint printed above 1%.

## Decision rule

Read `k2_receiving_survival` at `best_model` on each seed with the
reading cell. The thresholds are the spec's ≥ 5% and ≤ 1%, read in
whole percents, the unit the reference was booked in. The bar in the
reading cell scores exactly these bands:

| Band | Exact `k2_receiving_survival` | Report prints | At 85 receiving points |
|---|---|---|---|
| ADOPT band (bar PASS) | ≥ 0.045 | ≥ 5% | 4 or more conversions |
| between (bar MIDDLE) | 0.015 to < 0.045 | 2–4% | 2 or 3 |
| FALSIFIED band (bar FAIL) | < 0.015 | ≤ 1% | 0 or 1 |

- **ADOPT** if **both seeds land in the ADOPT band**. The bundle,
  under today's selection, cleared the bar. Before booking that the
  context gate was binding, run the selection-matched control (see
  "What the pilot cannot attribute") and confirm `best_model` on a
  fresh seed block (lesson 13a). Then book it in `DECISIONS.md`, and
  replicate with at least three seeds before treating the profile as
  the next baseline (cardinal rule 8).
- **FALSIFIED** if **both seeds land in the FALSIFIED band**: no
  better than the reference. The gate was not the binding constraint.
  Route to LD1′ demo injection, the fallback
  ([`design_paddle_tennis_demo_injection.md`](design_paddle_tennis_demo_injection.md)).
  Selection on k≥2 conversions favours k=2 at `best_model`, so a
  FALSIFIED is harder to reach here than under the reference's
  selection.
- **INCONCLUSIVE** otherwise (the seeds split, or either lands
  between). Either run the better seed again at 5M or add a third
  seed (`SEED = 2`) at 3M; the maintainer chooses. The 5M run is a new
  from-scratch run (`TOTAL_TIMESTEPS = 5_000_000`, the same `SEED`) in
  its own run directory. Training has no resume, so it repeats the
  first 3M (not bit for bit on a GPU), and patience is off for it too.

Reading the thresholds:

- **Resolution.** The 30 episodes hold about 80–110 receiving points,
  depending on the policy. One conversion is about 1.2 percentage
  points at 85 and 0.9 at 108, so each band is about one conversion
  wide. Report the receiving points and the conversion count for each
  seed, not only the rate. The cut-offs depend on the point count: 4
  of 85 (4.7%) is in the ADOPT band, 4 of 90 (4.4%) is not.
- **The reading cell decides.** It replays `best_model` on the CPU.
  The in-run report ran on the GPU, and the two can differ by a
  conversion on the same checkpoint and seeds (LH1c's 2.5M checkpoint:
  4 of 85 in-run, 5 of 81 replayed).
- **For the maintainer.** The ≥ 5% and ≤ 1% numbers are the approved
  spec's; this card only fixes their unit, and that choice is yours
  to confirm before launch. Read as exact fractions instead, ≤ 1%
  would put an exact replication of the reference (1 conversion in
  ~85) in INCONCLUSIVE, and FALSIFIED would need zero conversions on
  both seeds. The bar would then need `pass_at` 0.05 and a `fail_at`
  just above 0.01, because a bar FAILs only strictly below `fail_at`.

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
  nothing. Dropping the clock makes the truncation step's escrow
  clawback unforeseeable (see "The bundle"), a small side effect.
- **Selection.** The pilot's `best_model` is chosen on k≥2
  conversions with paired evaluation; the reference's was chosen on
  crossings, unpaired (see "Reference to beat"). A k=2-favouring pick
  alone can cross the ADOPT bar on these seeds: the follow-on LH1c
  run (`20260821_013700`: same observation and γ, plus the hold
  escrow, warm-started from the reference) had a checkpoint that read
  5% in-run and 6.2% (5 of 81) replayed on seeds 5200–5229, and 1.6%
  (3 of 191) on fresh seeds 5230–5299
  ([`paddle_tennis_review_next_steps_20260823.md`](paddle_tennis_review_next_steps_20260823.md)
  §2, PT-K2). The pilot has no arm that isolates this. The control is
  a `PaddleTennis` run under today's recipe with the pilot's seed and
  run length (`ENV = "PaddleTennis"`, `TOTAL_TIMESTEPS = 3_000_000`,
  `EARLY_STOP_PATIENCE = 0`, which turns patience off), read with the
  same cell. The maintainer can run it alongside the pilot, or only
  once an ADOPT needs it. Selection uses the paired
  eval seeds (seed + 1,000,000), not 5200+, so the effect comes
  through correlation, not direct fitting to the diagnosis seeds.
- **Two seeds.** This is a pilot. An ADOPT says the lever is worth a
  registered run, not that a new baseline exists.
