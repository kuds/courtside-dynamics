# Checkpoint selection audit: run 20260727_004014

Paired re-scoring of four checkpoints from the first depth run to clear
stage 2, on the goal geometry with fresh held-out seeds. It was run to
test a specific claim — that the late-run policy beat the shipped
`best_model.zip` on the goal task and the run had therefore selected the
wrong checkpoint. **The claim did not replicate.** The audit is recorded
here because the way it failed is the second instance of the same
statistical error in this campaign, and because it changes what we
believe the depth plateau is.

## What prompted it

`WallBallDepthCurriculum` selects `best_model.zip` on the *matched-stage*
evaluator (`best_metric_keys` on `eval_info`, the stream the performance
gate reads). The unsynced goal-geometry evaluator (`eval_info_final.csv`,
30 episodes at the ladder's final stage) is computed at every evaluation
and never consulted for selection.

Reading that stream after the run, the late-run region looked better on
the goal task than the selected checkpoint:

| checkpoint | goal metric, 8-eval trailing mean |
|---|---:|
| 4,650,000 (`best_model.zip`) | 1.163 |
| 5,500,000 | 1.113 |
| 5,750,000 | 1.283 |
| 6,000,000 (`final_model.zip`) | 1.254 |

The single highest evaluation of the whole run was 1.533 at 5,950,000,
and the best 8-eval window ended at 5,825,000. On that basis the run's
selection criterion was called wrong for the campaign objective, and a
re-audit was proposed.

## Method

`tools`-free scratch harness (`ckpt_audit.py`), reusing
`notebook_utils._rollout_wall_ball_seed` — the exact per-episode function
the shipped long-horizon audit calls — so the numbers are directly
comparable rather than a parallel reimplementation.

* **Geometry.** Constructor kwargs copied verbatim from the run's own
  `config.json` → `evaluation_env` block, with only `episode_len` raised
  to 5,000: fence `(-4.7, -3.0)`, `paddle_start_x` −3.9, `paddle_home_x`
  −1.7, `serve_speed` 7.0. These policies trained against the 0.21.0
  ladder; scoring them on the 0.22.0 ladder that shipped afterwards would
  measure geometry they never saw. `return_shaping_scale` resolves to
  0.0, so the 0.22.0 reward change is inert here — and since the metric
  is `bounce_count` and the policies are deterministic, a reward change
  could not perturb the comparison regardless.
* **Pairing.** All four checkpoints see the same 200 seeds
  (20,000–20,199). Seeds 0–199, 1000–1199 and 10,000–10,049 are burned by
  earlier work including the run's own audit. Pairing was necessary, not
  cosmetic: the effect under test is ~0.09 returns against a per-episode
  SD of ~0.49, so four independent 50-episode means could not resolve it.
* **Normalizers.** Each model loaded with its own `VecNormalize`,
  `training=False`. Pairing was verified two ways: `obs_rms.count`
  matches each checkpoint's timestep (4,650,008 / 5,500,008 / 5,750,008 /
  6,000,008), and the best model's normalizer hashes to
  `5dd123119fd7…`, matching the sha256 bound into `best_model_meta.json`.

## Results

200 held-out seeds, goal geometry, `episode_len` 5,000, deterministic.

| checkpoint | completed returns | ≥2 rate (Wilson 95%) | ≥3 | ep len | sec |
|---|---:|---|---:|---:|---:|
| `best_model.zip` @ 4,650,000 | **1.215** ± 0.032 | 19.5% [15–26] | 2.0% | 270.7 | 2.71 |
| checkpoint @ 5,500,000 | 1.065 ± 0.035 | 15.5% [11–21] | 0.0% | 244.1 | 2.44 |
| checkpoint @ 5,750,000 | **1.300** ± 0.048 | 26.0% [20–32] | 4.0% | 272.8 | 2.73 |
| `final_model.zip` @ 6,000,000 | 1.135 ± 0.035 | 17.0% [12–23] | 1.5% | 263.6 | 2.64 |

Paired against `best_model.zip`, same seeds:

```
@ 5,500,000   delta -0.150 +/- 0.046   t=-3.24   better/worse/tied  21/43/136
@ 5,750,000   delta +0.085 +/- 0.052   t=+1.62   better/worse/tied  35/26/139
@ 6,000,000   delta -0.080 +/- 0.044   t=-1.80   better/worse/tied  18/28/154
```

**Harness validation.** `best_model.zip` scores 1.215 here against the
shipped audit's 1.14 on the disjoint seeds 10,000–10,049 — close
agreement on independent seeds, so the harness measures the same
quantity.

## Findings

**1. The claim is refuted. `best_model.zip` stands.** `final_model` is
*worse*, not better (−0.080, t = −1.80). Only 5,750,000 is nominally
ahead, by +0.085 at t = +1.62 (two-sided p ≈ 0.11) — not significant, and
that is before accounting for it being the argmax of four candidates.
Under Bonferroni over the three comparisons the threshold is p < 0.017;
it is nowhere near.

**2. The test was not underpowered.** It resolved the 5,500,000
checkpoint's −0.150 deficit at t = −3.24, so it detects real differences
of that size. The predicted effect is simply smaller than the one it
caught: resolving +0.085 at α = 0.05 would need ~290 seeds. A 7%
difference is not worth acting on when the entire toolchain already
points at `best_model.zip`.

**3. The prior evidence was a winner's curse.** The trailing means above
were computed from a 30-episode estimator over the same 240 evaluations
from which the maximum was then taken. That is selection on the noise.
See `plan_wall_ball_aligned_deep_stages.md` for the first instance in
this campaign, where a control arm's calibration argmax (94.0% on
calibration) dropped to 89.0% held-out and failed while the stale shipped
value passed at 94.0%. Calibration-to-held-out swings of 3–5 points at
n = 200 are routine here, and every margin that has mattered in this
campaign is that same order.

**4. The plateau is structural, not a selection artifact.** All four
checkpoints, spanning 1.35M steps, sit at 1.07–1.30 completed returns and
2.4–2.7 s episodes. Nothing in that window is meaningfully better at the
goal task than anything else. Whatever caps performance is not the choice
of snapshot.

## Consequences

* **No re-selection.** Keep `best_model.zip`.
* **Adding the goal stream to selection is downgraded, not adopted.** The
  structural observation stands — selection scores stage-3 geometry while
  the goal stream is computed every evaluation and ignored. But the
  evidence that this *costs* anything evaporated: the stage-3-selected
  checkpoint beat two of three goal-region alternatives and tied the
  third. Revisit only if a future run shows the two streams actually
  disagreeing on held-out seeds.
* **The 0.22.0 changes are the right response.** Finding 4 is what you
  would expect if the ceiling is set by runway, action resolution, and a
  reward blind to shot quality — the three mechanisms 0.22.0 targets —
  rather than by which checkpoint is kept.

## Seed ledger

Burned (do not reuse for held-out claims): 0–199, 1000–1199,
10,000–10,049 (run 20260727_004014's own long-horizon audit),
**20,000–20,199 (this audit)**. Reserved and still clean: 2000–2199,
3000–3199.
