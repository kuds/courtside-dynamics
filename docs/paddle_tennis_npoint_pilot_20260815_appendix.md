# Appendix: n-point pilot review workpapers (2026-08-15)

Companion to
[`paddle_tennis_npoint_pilot_20260815_review.md`](paddle_tennis_npoint_pilot_20260815_review.md).
Data sources: the seven Drive run folders under
`training_runs/PaddleTennis/sac/` (stage summaries, config.json
provenance, per-checkpoint diagnosis probes, metrics CSVs), read
2026-08-15.

## A. Cross-run comparison (the six runs before the reviewed one)

| | 20260803_004559 | 20260808_022106 | 20260809_005951 | 20260809_161704 | 20260809_211147 | 20260810_211754 |
|---|---|---|---|---|---|---|
| Git SHA | aec8cd1 | 7db6f59 | 1c1e8e1 | a29883c | b57c3d2 | **2e597b1** (same as failed latest run) |
| Run config | paddle_tennis.toml (a45bec216463) | paddle_tennis.toml (a45bec216463) | paddle_tennis.toml (a45bec216463) | paddle_tennis_shaping_pilot.toml (4c1376ae7b97) | paddle_tennis_shaping_pilot.toml (4c1376ae7b97) | paddle_tennis_npoint_pilot.toml (79f6f3a67a9c) |
| volley_rule | **absent** (kwarg didn't exist yet) | fault | fault | fault | fault | fault |
| contact_shaping | – | – | – | 0.25 | 0.25 | 0.25 |
| points_per_episode | – (kwarg absent) | – | – | – | – | null (continuous n-point play) |
| serve_config / opponent_controller | null / null | null / null | null / null | null / null | null / null | null / null |
| Penalties (return/fault/unsafe) | 1.0 / 1.0 / 2.0 | same | same | same | same | same |
| episode_len / court_style | 1500 / diagnostic | same | same | same | same | same |
| SAC overrides | none (defaults: ent_coef auto, target_entropy −3.0, train_freq 1, no SDE) | none (defaults, same as left) | use_sde, ent_coef auto_0.02, target_entropy −1.5, train_freq (64,step) | same as left | same as left | same as left |
| n_envs | 8 | 8 | 4 | 4 | 4 | 4 |
| Timesteps | 2,000,000 (completed) 70 FPS | 1,750,000 of 6M (early stop) 68 FPS | 1,000,000 (completed) 70 FPS | 1,000,000 (completed) 68 FPS | 2,000,000 (completed) 67 FPS | 1,125,000 of 2M (early stop) **39 FPS** |
| Final eval reward | 46.633 ± 17.576 | 0.233 ± 0.667 | −0.467 ± 0.670 | 0.150 ± 0.774 | 0.633 ± 0.826 | −3.267 ± 0.573 |
| Best eval reward | 41.067 ± 16.992 @2.0M | 0.467 ± 0.499 @1.575M | −0.433 ± 0.616 @500k | 0.325 ± 0.866 @925k | 0.592 ± 0.991 @1.675M | −3.025 ± 0.518 @525k |
| Headline final / best (step) | 37.60 / 37.60 (@2.0M) | 1.27 / 1.37 (@1.25M) | 0.53 / 0.67 (@550k) | 1.07 / 1.23 (@850k) | 1.50 / **1.77 (@1.975M)** | 3.03 / 3.30 (@625k)* |
| Avg ep length | 832.2 | 311.9 | 224.0 | 296.2 | 330.6 | 1500.0 (always timeout) |
| ent_coef @ end | 6.80e-4 | 5.26e-5 | 3.96e-5 | 1.24e-4 | 1.54e-4 | 4.71e-5 |
| policy std @ end | n/a (no SDE, not logged) | n/a | 0.0325 | 0.0357 | 0.0250 | 0.0319 |
| Diagnosis available | none | 1 post-hoc probe on best_model (probe SHA 20bd994, 100 eps, seeds 5200-99) + oracle | per-100k checkpoint probes + oracle (30 eps, seed 5200) | same | same | same (per-point format, 214 points) |
| Final ckpt: policy shots | – | 49 (best@1.25M; hit#1 receiving only; crossed 82%, in 76%) | **2** (both serving, 0% in, out-depth 13.97 m) | 16 (14 rec hit#1: 93% crossed / 57% in) | 23 (15 rec hit#1: 100%/80% in; 8 serve hit#1: 88%/75%) | **1** of 214 points (0% crossed) |
| k=1 / k=2 survival (receiving) | – | 98% / 0% | 0% / 0% | 93% / 0% | **100% / 0%** | 0% / 0% |
| k=1 survival (serving) | – | 0% | 13% | 13% | **53%** | 1% |
| Dominant point ender (final ckpt) | – | policy_never_reached 83/100 | policy_never_reached 28/30 | policy_never_reached 21/30 | policy_never_reached 23/30 | policy_never_reached 182/214 (+cap 30) |
| Touched ball after bounce | – | 37% | 7% | 43% | 50% | **1%** |

*Run 6's headline is not comparable to single-point runs: with points_per_episode=None the 1500-step episode
contains many opponent-served points, so `crossings` accumulates across points (its 3.30 "best" comes almost
entirely from opponent shots, as the probe shows the policy essentially never touches the ball).

### The warm-start source: which run achieved "k=1 mastered"

**20260809_211147** (the 2M-step contact-shaping pilot, git b57c3d2). Its final checkpoint @2,000,000 shows
exchange-survival **k=1 receiving 100%** (15/15) and k=1 serving 53%, with receiving hit #1 100% crossed / 80%
landed in (in-depth 3.99 m); k=2 is still 0% in both roles (the "second-ball wall": after its own shot it holds
8.33 m from the ready position, touched-after-bounce only 50%, so it never reaches the opponent's reply).
Exact settings:

- env: PaddleTennisEnv, episode_len 1500, serve_config null, opponent_controller null, court_style "diagnostic",
  volley_rule "fault", return_reward 1.0, fault_penalty 1.0, unsafe_physics_penalty 2.0, **contact_shaping 0.25**,
  single-point episodes (no points_per_episode kwarg — pre-n-point env).
- run config: paddle_tennis_shaping_pilot.toml (sha256 4c1376ae7b97…), content `[env] contact_shaping = 0.25`.
- SAC: use_sde true, ent_coef "auto_0.02", target_entropy −1.5, train_freq (64,"step"), gradient_steps −1,
  lr 3e-4, gamma 0.99, batch 256, buffer 1M, tau 0.005, net 256x256 ReLU, n_envs 4, seed 0, 2M steps,
  normalize_obs (obs 24-47 excluded), no reward normalization.
- Outcome: final eval 0.633 ± 0.826; best headline crossings_ep_mean 1.77 @1.975M (best model selected there);
  final ent_coef 1.54e-4, std 0.025.

Runner-up: 20260808_022106's best model (step 1.25M, default SAC hyperparams, no shaping, no SDE) already had
k=1 receiving 98% but k=1 serving 0% and lower shot quality (82% crossed / 76% in).

### Cross-run findings bearing on the reviewed run

- **The n-point collapse reproduces across seeds**: seed 0 (20260810_211754) and seed 1 (20260815_015143), both
  at SHA 2e597b1 with identical config, show the same zero-contact, policy_never_reached-dominated behavior from
  the first checkpoint onward. It is a mode/config problem, not a bad seed.
- Switching to n-point continuous play discarded the k=1-mastered progress of run 5 rather than building on it
  (warm_start: null in every run — no run ever warm-started from run 5's best model).
- Entropy collapse (ent_coef → 1e-4..5e-5, std → 0.02-0.036) occurs in **every** run regardless of
  ent_coef=auto (target −3.0) or auto_0.02 (target −1.5); target_entropy is not being honored in effect
  (ent_coef_loss stays non-zero, coef keeps shrinking). The failed run's 5.3e-05 / std 0.018 is the same
  chronic pattern, only slightly more extreme.
- contact_shaping=0.25 was the single change that took k=1 receiving from 0% (run 3) to 93-100% (runs 4-5)
  in the single-point env; in the n-point env the same shaping failed to produce any contact.
- Eval-reward and best-model selection caveats seen before the failed run: run 1's reward/headline were exploit
  inflation; run 6's "best" reward at 525k (−3.03) improved by fault-avoidance while contact stayed near zero —
  exactly the pattern flagged in the 20260815 run.
- Reward scale discontinuity across modes: single-point runs sit near 0 to +0.6 (episodes ~220-370 steps);
  n-point runs sit near −3 to −3.4 (1500-step timeout episodes). Early-stop patience (20 evals) plus this scale
  meant both 6M- and 2M-budget runs terminated near half budget.
- Throughput cost of n-point: 39 FPS vs 67-70 FPS single-point (≈1.7x slower per step).

## B. Reviewed run (20260815_015143): per-checkpoint diagnosis timeline

| Checkpoint | Points | Policy hits n | Crossed% | In% | k=1 recv | k=1 serv | Top-3 point enders (count) | Ready-pos bounce-time err mean (m) | Inter-point recovery travel mean (m) | Crossings mean | Termination mix |
|---|---|---|---|---|---|---|---|---|---|---|---|
| oracle | 63 | 189 | 92% | 88% | 94% | 91% | cap 30, opponent_shot_net 17, policy_shot_net 11 | 0.89 (p90 0.90); touched-after-bounce 98% | 1.95 (p90 3.48) | 5.48 | none 30, ball_net 28, out_of_bounds 5 |
| 100000 | 218 | 6 | 83% | 0% (out-depth 13.79) | 2% | 4% | policy_never_reached 176, cap 30, policy_shot_out 5 | 2.44 (p90 4.28); touched 3% | 8.01 (p90 10.45) | 0.44 | out_of_bounds 173, none 30, second_bounce 9, wrong_hitter 3 |
| 200000 | 218 | 0 | — | — | 0% | 0% | policy_never_reached 182, cap 30, policy_volley_fault 4 | 3.01 (p90 4.58); touched 0% | 8.88 (p90 11.18) | 0.42 | out_of_bounds 174, none 30, second_bounce 8, volley_return 4 |
| 300000 | 214 | 2 | 50% | 0% (out-depth 18.09) | 0% | 2% | policy_never_reached 182, cap 30, failed_to_cross 1 = policy_shot_out 1 | 2.23 (p90 3.67); touched 1% | 8.39 (p90 10.53) | 0.44 | out_of_bounds 178, none 30, second_bounce 5, failed_to_cross 1 |
| 400000 | 214 | 3 | 100% | 0% (out-depth 17.46) | 0% | 3% | policy_never_reached 181, cap 30, policy_shot_out 3 | 2.75 (p90 4.72); touched 2% | 8.97 (p90 10.68) | 0.45 | out_of_bounds 178, none 30, second_bounce 6 |
| 500000 | 214 | 1 | 100% | **100% (in-depth 5.33)** | 0% | 1% | policy_never_reached 183, cap 30, failed_to_cross 1 | 2.63 (p90 4.20); touched 1% | 8.68 (p90 10.53) | 0.44 | out_of_bounds 175, none 30, second_bounce 8, failed_to_cross 1 |
| 600000 | 214 | 1 | 100% | 0% (out-depth 10.12) | 0% | 1% | policy_never_reached 183, cap 30, policy_shot_out 1 | 2.39 (p90 3.73); touched 1% | 8.37 (p90 10.62) | 0.44 | out_of_bounds 178, none 30, second_bounce 6 |
| 700000 | 217 | 1 | 0% | 0% | 0% | 1% | policy_never_reached 182, cap 30, policy_volley_fault 3 | 2.00 (p90 3.34); touched 1% | 8.44 (p90 10.16) | 0.42 | out_of_bounds 175, none 30, second_bounce 7, volley_return 3 |
| 800000 | 215 | 1 (recv hit#1) | 0% | 0% | 1% | 0% | policy_never_reached 182, cap 29, policy_volley_fault 3 | 3.63 (p90 5.61); touched 1% | 7.65 (p90 9.85) | 0.43 | out_of_bounds 175, none 29, second_bounce 7, volley_return 3 |
| 900000 | 217 | 0 | — | — | 0% | 0% | policy_never_reached 181, cap 30, policy_volley_fault 5 | 2.79 (p90 4.63); touched 0% | 9.28 (p90 11.36) | 0.43 | out_of_bounds 173, none 30, second_bounce 9, volley_return 5 |
| 1000000 | 214 | 0 | — | — | 0% | 0% | policy_never_reached 183, cap 30, policy_volley_fault 1 | 2.44 (p90 3.81); touched 0% | 8.39 (p90 10.01) | 0.44 | out_of_bounds 176, none 30, second_bounce 7, volley_return 1 |
| 1100000 | 213 | **7** (all serv hit#1) | 0% | 0% | 0% | 7% | policy_never_reached 176, cap 30, policy_shot_out 4 (also policy_shot_net 2) | 2.44 (p90 3.79); touched 4% | 8.72 (p90 10.61) | 0.44 | out_of_bounds 174, none 30, second_bounce 6, ball_net 2 |
| 1200000 | 216 | 1 | 0% | 0% | 0% | 1% | policy_never_reached 183, cap 30, policy_volley_fault 1 = policy_shot_out 1 = wrong_hitter 1 | 3.55 (p90 5.11); touched 1% | 8.65 (p90 10.47) | 0.43 | out_of_bounds 177, none 30, second_bounce 7, volley_return 1 |
| 1300000 | 214 | 1 | 0% | 0% | 0% | 1% | policy_never_reached 183, cap 30, failed_to_cross 1 | 2.48 (p90 3.86); touched 1% | 9.05 (p90 11.02) | 0.43 | out_of_bounds 176, none 30, second_bounce 7, failed_to_cross 1 |
| 1400000 | 214 | 2 | 50% | 0% (out-depth 15.10) | 0% | 2% | policy_never_reached 181, cap 30, policy_shot_out 2 | 3.59 (p90 5.06); touched 1% | 8.34 (p90 9.98) | 0.44 | out_of_bounds 176, none 30, second_bounce 7, volley_return 1 |
| 1500000 | 214 | 1 | 100% | 0% (out-depth 9.77) | 0% | 1% | policy_never_reached 183, cap 30, policy_shot_out 1 | 3.41 (p90 4.59); touched 1% | 7.83 (p90 10.00) | 0.44 | out_of_bounds 177, none 30, second_bounce 7 |
| 1600000 | 214 | 0 | — | — | 0% | 0% | policy_never_reached 183, cap 30, policy_volley_fault 1 | 3.54 (p90 5.27); touched 0% | 8.30 (p90 10.21) | 0.43 | out_of_bounds 176, none 30, second_bounce 7, volley_return 1 |
| 1700000 | 214 | 0 | — | — | 0% | 0% | policy_never_reached 184, cap 30 (only two enders) | **3.97 (p90 5.24)**; touched 0% | 8.48 (p90 10.69) | 0.43 | out_of_bounds 177, none 30, second_bounce 7 |
| 1800000 | 214 | 0 | — | — | 0% | 0% | policy_never_reached 183, cap 30, policy_volley_fault 1 | 3.70 (p90 5.21); touched 0% | 8.29 (p90 10.72) | 0.44 | out_of_bounds 176, none 30, second_bounce 7, volley_return 1 |
| 1900000 | 214 | 0 | — | — | 0% | 0% | policy_never_reached 184, cap 30 (only two enders) | 3.27 (p90 4.65); touched 0% | 8.82 (p90 10.90) | 0.43 | out_of_bounds 177, none 30, second_bounce 7 |
| 2000000 | 216 | 0 | — | — | 0% | 0% | policy_never_reached 184, cap 30, policy_volley_fault 1 | 3.52 (p90 4.89); touched 0% | 8.65 (p90 10.67) | 0.43 | out_of_bounds 177, none 30, second_bounce 8, volley_return 1 |

Notes on columns: "touched" = "touched after bounce" percentage from the ready-position line. Recovery-hold lines (present only when the policy made at least one shot) are omitted from the table; values where present: 100k 4.10, 300k 4.68, 400k 5.87, 500k 4.11, 600k 1.73, 1.1M 1.35, 1.2M 1.91, 1.4M 3.13, 1.5M 6.23 m (oracle 2.31 m).

### Deviations from the "zero contacts after 100k" narrative

The summary claim "ZERO contacts from 200k through 2M" is an over-simplification. Strictly zero-contact checkpoints are 200k, 900k, 1.0M, 1.6M, 1.7M, 1.8M, 1.9M, 2.0M. Residual sporadic contact persists through the middle of the run:

- 300k: n=2, 400k: n=3, 500k: n=1, 600k: n=1, 700k: n=1, 800k: n=1, 1.1M: **n=7**, 1.2M: n=1, 1.3M: n=1, 1.4M: n=2, 1.5M: n=1. Total 21 contacts across 200k–1.5M (~2,780 probe points), then strictly zero for the final 500k (1.6M–2.0M).
- Every one of these residual contacts except one is a serving hit #1 (the fed ball on the policy's own serve); the sole receiving contact is 800k's single hit. The policy never once achieved hit #2 in any checkpoint.
- Ball quality of residual contacts is terrible: only ONE policy shot in the entire run landed in — the single 500k shot (crossed 100%, in 100%, in-depth 5.33 m). All others were 0% in, with out-depths 9.77–18.09 m (wild over-hits) or 0% crossed (1.1M's 7 contacts all failed to cross; 2 into the net — the only policy_shot_net events after the oracle).
- 1.1M is the largest post-100k anomaly: 7 serving contacts (k=1 serving 7%), 0% crossed, enders policy_shot_out 4 + policy_shot_net 2, touched-after-bounce briefly back up to 4%. This flicker dies out by 1.2M.

### Drift trends

1. **Ready-position bounce-time error worsens.** 100k–1.0M oscillates 2.00–3.63 m (mean ~2.6); 1.2M–2.0M sits at 3.27–3.97 m (mean ~3.55, worst 3.97 at 1.7M). Oracle is 0.89 m. The policy is not merely failing to swing — it is drifting further from where the ball will bounce as training proceeds.
2. **Touched-after-bounce decays to hard zero.** 3% at 100k, 1–4% through 1.5M, exactly 0% at every checkpoint from 1.6M on (and at 200k/900k/1.0M). In the final quarter of training the policy never even makes post-bounce contact with the ball's position.
3. **No recovery/positioning learning.** Inter-point recovery travel is flat at 7.65–9.28 m (oracle 1.95 m) with no downward trend — the policy is ~8–9 m out of position at every feed arrival for the entire run.
4. **Point-ender mix ossifies.** policy_never_reached climbs slightly from 176 (100k) to 184 (1.7M/1.9M/2.0M) of ~214 points and secondary faults (policy_shot_out, wrong_hitter, failed_to_cross) disappear; by 1.7M/1.9M the enders are exactly two categories: policy_never_reached 184 + cap 30. The behavior distribution narrows, consistent with the reported entropy collapse (ent_coef 5.3e-05, policy std 0.018).
5. **Environment/opponent stationarity confirmed.** Opponent line is stable across all checkpoints (n=98–102, crossed 92–95%, in 88–92%, in-depth 3.96–4.05), so metric changes are attributable to the policy, not probe drift.
6. **Crossings mean flat at 0.42–0.45** (oracle 5.48) — essentially all crossings are opponent shots; there was never any rally.
7. **Point throughput inverse-signal:** checkpoints complete 213–218 points in the probe budget vs the oracle's 63, i.e. points end ~3.4x faster because the policy never reaches the ball (terminations dominated by out_of_bounds 173–178 = opponent's in-bounds ball bouncing away untouched, plus second_bounce 5–9).

## C. Training-metrics trajectory (progress.csv / eval_info.csv / evaluations.npz / monitor)

Reviewed run 20260815_015143; files fetched from the Drive run folder.

### C.1 Training health (progress.csv) (progress.csv)

#### Entropy coefficient: collapsed before the first log point
- Configured `auto_0.02` (initial alpha 0.02), `target_entropy=-1.5`. At the FIRST logged train row (step 12,000, n_updates 11,776) `train/ent_coef` was already **9.83e-4** — a >20x collapse inside the first 12k steps. So the thresholds 1e-2 and 1e-3 were both crossed at/before step 12,000 (unobservably early).
- Below **1e-4 at step 372,000**. Minimum **3.38e-5 at 1,452,000**. Final **5.34e-5 at 2,000,000**. It never re-inflated above 2e-4 after step ~200k.
- `ent_coef_loss`: -4.68 at 12k, -0.43 at 372k, -1.38 final — the auto-tuner kept pushing alpha down the whole run.

#### Policy std: started tiny, decayed monotonically, never recovered
- First logged `train/std` = **0.0501 at 12k** — this equals exp(-3), i.e. the SB3 SAC default `log_std_init=-3`; exploration noise was ~0.05 from the outset.
- Monotonic decay: 0.048@204k, 0.0463@372k, 0.0433@504k (first <0.04 at 660k), 0.033@1.0M (first <0.03 at 1,164k), 0.0245@1.5M, first <0.02 at 1,836k, **minimum 0.0183 at 1,992k (final)**. Max value after 1M steps = 0.0327 — no recovery, ever.

#### Losses
- `actor_loss`: -0.202 at 12k, rose to ~+0.30 by 200k and plateaued at 0.27–0.31 for the rest of the run (final +0.29).
- `critic_loss`: microscopic throughout — 1.1e-4 at 12k, run max 2.9e-4 (at 576k), final 3.9e-5. Consistent with a nearly constant-reward, no-signal regime (TD targets trivially predictable because almost nothing ever happens).

#### Rollout reward after collapse
- `rollout/ep_rew_mean` (100-episode window): -3.13 at 12k, -3.36 avg in (0,0.5M], -3.25 in (0.5,1M], -3.20 in (1,1.5M], **-3.34 in (1.5,2M]** (final logged -3.24). Post-collapse (>=372k) linear slope **-0.025 per 1M steps** — i.e. training reward was flat-to-slightly-worse after the entropy collapse; full-run slope +0.025/1M. Any "improvement" is within noise (window means span only -3.45..-3.12 over 2M steps).

### C.2 Behavioral metrics (eval_info.csv, 80 evals every 25k)

Every one of the 80 evals: `episode_length=1500`, `term_timeout_ep_mean=1.0`, all other `term_*` exactly 0 — every eval episode ran to the 1500-step timeout (continuous n-point play, as configured).

Per-episode means, essentially constant for 2M steps (linear slopes all ~0):
- `legal_hit_count_ep_mean`: mean 0.372, range [0.233, 0.500], slope +0.0075/1M. **It never exceeded 0 because it never went to 0 — it was nonzero at all 80 evals but flat**; given the reward evidence below (and the checkpoint probes), these ~0.37 hits/episode are not policy rally contacts improving over time — the count includes serve/feed/opponent-side contacts and never trended. `legal_hit_count_max` per eval: 1–3 (mostly 1–2).
- `rally_count_ep_mean`: mean **0.042** (zero at 21 of 80 evals; max single-eval value 0.133; `rally_count_max` was 1 at almost every eval, run max 3). No rallies, start to finish.
- `bounce_count_ep_mean`: mean 0.424, flat (slope +0.042/1M); `bounce_count_max`=1 at all 80 evals.
- `crossings_ep_mean`: mean 3.04, `crossings_ep_p50`=3.000 at ALL 80 evals, p90 3.4, max 4–5. Flat (slope -0.007/1M).
- `points_played_ep_mean`: mean 6.19/episode (~242 steps per point), drifting slightly DOWN (slope -0.071/1M; 6.63 at 25k -> 6.13 at 2M).
- `completed_point_crossings_ep_mean`: mean 2.91, flat (slope -0.006/1M), range 2.70–3.07.
- `success_rate` = 1.000 at all 80 evals — metric is degenerate/meaningless in this run.
- Phase occupancy (flat all run): awaiting_return 57.5%, return_in_flight 23.8%, initial_feed 18.3%, terminal 0.4%. `serve_side_is_policy_ep_mean` ~0.58.

#### Point enders (per-episode means; sum exactly equals points_played)
| ender | all-run mean | first 10 evals | last 10 evals |
|---|---|---|---|
| out_of_bounds | **5.86 (94.7%)** | 5.907 | 5.850 |
| second_bounce | 0.193 | 0.143 | 0.213 |
| volley | 0.058 | **0.137** | **0.023** |
| ball_net | 0.036 | 0.037 | 0.037 |
| failed_to_cross | 0.025 | 0.030 | 0.017 |
| illegal_hit | 0.015 | **0.050** | **0.010** |
| net_touch | 0 | 0 | 0 |

(Note: eval_info has no `policy_never_reached` category — that label comes from the separate checkpoint-probe tooling; here those points are booked overwhelmingly as `out_of_bounds`.) The only things that changed over 2M steps: illegal_hit fell 0.050->0.010 and volley fell 0.137->0.023 (penalized enders avoided), while second_bounce rose 0.143->0.213. That substitution IS the entire behavioral "learning" of the run.

### C.3 Eval reward vs behavior (evaluations.npz): fault-avoidance signature, quantified

- 80 evals x 30 episodes, all `ep_lengths`=1500. Mean reward: **-3.833 at 25k -> -3.025 at 2M** (best = final). Linear slope **+0.1375 per 1M steps**, corr(mean, steps)=0.494. First half (<=1M): -3.359 +/- 0.172; second half: -3.234 +/- 0.119 (Welch t = 3.72 — the improvement is statistically real but tiny: ~0.3 reward over 2M steps).
- Episode-reward distribution is quantized at integers (whole-point penalties): over all 2,400 eval episodes: -3: 77.8%, -4: 12.5%, -5: 8.4%, -6: 0.5%, -2: 0.8%, plus exactly **3 non-integer episodes** (-1.75 @1,250k; +0.25 @1,350k; -0.75 @2,000k). With `contact_shaping=0.25`, a non-integer total is the fingerprint of a policy ball contact -> **contact-shaping reward fired in 3 of 2,400 eval episodes (0.125%)**, all after 1.2M steps, never twice in one eval.
- The mean-reward gain decomposes entirely as mass moving from -5/-4 to -3: first 10 evals: {-5: 14.3%, -4: 18.0%, -3: 65.7%}; last 10 evals: {-5: 4.0%, -4: 10.7%, -3: 84.3%}. Combined with the flat-zero task metrics of section 2, this is the classic **fault-avoidance signature**: the optimizer reduced penalty events (illegal hits, volleys, extra faults) while never learning to touch the ball. Reward +0.8 (first eval to last) / +0.13 per 1M (trend), rallies +0.00, legal hits +0.00, bounces +0.00.

### C.4 Training-episode reward distribution (monitor CSVs, envs 0 and 2)

- Both envs: 166 episodes, every single one length 1500 (timeout; no early termination the entire run), 249k steps/env x 8 envs = 1.992M ✓.
- env0 rewards: {-3: 130, -4: 19, -5: 12, -6: 3, -2: 2}, mean -3.301; env2: {-3: 124, -4: 27, -5: 14, -2: 1}, mean -3.325. Late-training (last 40 eps) means -3.375 / -3.350 vs first-40 means -3.400 / -3.400 — no meaningful improvement in the training distribution itself.
- **0 of 332 training episodes had a non-integer reward** -> the contact-shaping bonus (0.25/contact) never fired even once in these two training envs across all 2M steps. The replay buffer therefore contained essentially no contact experience for the critic to propagate.

### C.5 Bottom line

1. Exploration was dead on arrival: alpha fell 0.02 -> 9.8e-4 inside the first 12k steps (before the first log), <1e-4 by 372k, ending 5.3e-5; policy std began at the log_std_init=-3 default (0.050) and only shrank, to 0.018. There was never a phase with enough action noise to discover ball contact.
2. Behavior is flat-zero from eval 1 to eval 80: rallies ~0.04/ep, bounce max 1, crossings p50 pinned at 3, ~6.2 points/ep of which 94.7% end out_of_bounds; all episodes (train and eval) end by 1500-step timeout.
3. The +0.8 eval-reward "improvement" (-3.83 -> -3.03, slope +0.14/1M, t=3.72) is fully explained by fewer penalized enders (illegal_hit 0.050->0.010, volley 0.137->0.023) shifting episode rewards from -5/-4 to -3 — fault-avoidance, not play. Contact shaping was earned in 3/2400 eval episodes and 0/332 sampled training episodes.
4. `success_rate`=1.0 at every eval is degenerate and should not be used.

## D. Branch code review (claude/repo-env-review-4096tu at 2e597b1)

Adversarial review of `git diff 4f91053..2e597b1` (~1,800 added lines: env,
recipes, diagnosis, probe tool, tests), every finding traced through the
checkout and several verified by direct simulation.

### D.1 (HIGH, explains zero contact) — continuous play inverts the contact incentive at low skill


**File**: `src/courtside_dynamics/envs/paddle_tennis.py`, `step()` absorbed-boundary
logic (lines ~687–733, `_points_remaining` line 751).

**Mechanism (traced)**: In the frozen one-point env, every episode ends at its first
fault, so ANY policy — statue, volley-faulter, out-slammer — pays exactly one
`-fault_penalty` per episode; the fault-rate term of the return is flat and contact
attempts cost nothing relative to passivity. Under `points_per_episode=None` the
episode return becomes `-(# completed points) + (# confirmed returns) + escrow-net`,
i.e. **point throughput is the dominant reward term**, and points where the policy
intervenes unsuccessfully are much SHORTER than points where it never intervenes:

- a pre-bounce touch is an instant `VOLLEY_RETURN` fault (~39–49 steps/point,
  measured);
- never touching lets the feed/return run to `SECOND_BOUNCE`/roll-out
  (~150–300 steps/point, measured);
- so touching multiplies the per-episode fault count.

**Measured on this exact checkout** (`points_per_episode=None`,
`contact_shaping=0.25`, seeds 5300–5309, mean total episode reward):

| policy | mean episode return | points/episode | steps/point |
|---|---|---|---|
| statue (zeros) | **−4.5** | ~6 | 153–300 |
| net-patting toucher (volleys every feed) | **−13.5** | ~14 | 39–49 |
| hard-slam witness (legal post-bounce touch, shots out) | **−1.6** | — | — |

The reward landscape between "never touch" (−4.5) and "touch legally then miss"
(−1.6) passes through "touch at the wrong time" (−13.5). A noisy early policy near
the ball's descent path samples volley faults, which are punished at ~3x the statue
baseline — a barrier the one-point env did not have (there all three columns would
read ≈ −1). The contact-shaping escrow cannot bridge it: per point it nets exactly
`0.25 × confirms` (NP1 identity, verified in code and by the shipped witnesses), so
a hit-without-confirm nets 0 undiscounted while the accelerated fault turnover is
charged in full. This precisely matches the observed run signature: reward
"improves" by fault-avoidance (moving away from the ball converts −13.5-style
episodes toward −4.5 and serving points to 0 via the opponent's free serve-return
confirm), task metrics stay at zero, and SAC's entropy collapse (F6) then locks the
avoidance policy in. Also consistent: the 100k checkpoint's 6 contacts landing 13m+
out (the hard-slam signature) vanishing by 200k.

Note this is the amendment's *designed* accounting, not a coding slip — the escrow,
alternation, carryover and counters all do exactly what the design doc says (NP0/NP1
witnesses pass) — but the design creates a contact-averse local gradient that can
cause the observed zero-contact collapse. Additional asymmetry: on the policy's
serving points a statue banks the opponent's serve-return confirm (+1) against the
eventual fault (−1) for a net 0 (pinned by
`test_statue_nets_minus_one_per_receiving_point`), so touching the opponent's return
and faulting converts a 0-point into a −1-point AND shortens it.

### D.2 (MEDIUM) — probe "policy hits" counts only

**File**: `src/courtside_dynamics/training/paddle_diagnosis.py:253–266` — a
`BALL_RACKET_A` event earns a ledger entry only when matched against
`transition.valid_racket_hits` credits; volley / premature / double-hit / wrong-hitter
touches are skipped by design. So "zero contacts" in the reports means zero *legal*
hits: a policy that touched the ball only illegally would ALSO print `policy_hits=0`.
In this run the enders (~184/216 `policy_never_reached`, few illegal/volley endings)
show the ball genuinely was not touched, so the headline conclusion stands — but the
number alone cannot distinguish "never touches" from "touches only illegally";
interpret alongside the ender table (which the reports do carry).

### D.3 (VERIFIED NON-BUG) — normalization parity of the eval/diagnosis pipeline:

Explicitly checked, as tasked:

- **Checkpoint-time probe** (`paddle_diagnosis.py:586–601`
  `DiagnosisProbeCallback._policy`): obtains the LIVE training normalizer via
  `self.model.get_vec_normalize_env()` and applies `normalize_obs` to the raw probe
  observation before `model.predict(..., deterministic=True)`. `SelectiveVecNormalize`
  (`train.py:186–233`) keeps the recipe's excluded tail 24–47 raw in `normalize_obs`
  itself, so the excluded-from-normalization assumption is honored identically at
  train and probe time. `VecNormalize.normalize_obs` does not update `obs_rms`, so
  probing mid-training does not perturb stats.
- **Probe env**: `train.py:1200–1213` wires `env_fn=resolved_eval_env_fn`, the run's
  resolved evaluation factory including TOML `[env]` overrides
  (`points_per_episode="none"`, `contact_shaping=0.25`); the probe reports' 216
  points over 30 episodes confirm the n-point env was actually probed.
  `test_l2_toml_spelling_reaches_both_envs` pins the override reaching both envs.
- **Eval envs**: built with `training=False`, `norm_reward=False`
  (`train.py:1026–1033`) and synced each evaluation via `sync_envs_normalization`
  (`callbacks/info_dict_eval.py:368–377`, `train.py:1674`).
- **Offline tool** (`native_checkpoint_policy`, `paddle_diagnosis.py:54–81`): loads
  the checkpoint's own saved `SelectiveVecNormalize` pickle with `training=False`.
- **Empirical**: running `run_episode` on this checkout with the ground oracle on the
  n-point env yields nonzero `policy_hits` (5 and 1 across 2 points, seed 5200), and
  a statue yields 7 points of `policy_never_reached` — the instrument detects
  contacts when they happen and segments points correctly.

Conclusion: the zero-contact number is trustworthy as "zero legal hits by the
deterministic policy under the exact training observation statistics."

### D.4 (MEDIUM, worsener) — relaunch clearance

**File**: `src/courtside_dynamics/envs/paddle_tennis.py:916–922`
(`_clear_launch_envelope`), `_launch_point` 967–975. Under carryover, a paddle parked
anywhere along the new feed's trajectory beyond 0.45 m of the origin — the serving
side's own corridor (feed rises from |x|≈3.25 toward the net) or the receiving side's
descent path — is struck by the feed mid-flight, yielding an instant
illegal-hit/volley fault, −1, and immediate relaunch (the npoint probe's parker
witness comment states the corridor is deliberately not cleared). In one-point mode
paddles re-park at home every episode, where the P3 band was validated clear; under
n-point carryover a policy hovering in the corridor can insta-fault half its points
(its serving points) at high cadence, adding corridor-avoidance pressure on top of
F1. In THIS run's probes illegal-hit endings are rare, so the realized contribution
at the deterministic checkpoints is small, but the stochastic training-time policy
sees this pressure whenever exploration brings it near the ball path — same
gradient direction as F1: away from the ball.

### D.5 (LOW) — `_nudge_paddle_clear` writes

**File**: `src/courtside_dynamics/envs/paddle_tennis.py:924–954`. The sanctioned
nudge adds a world-frame delta projected on the slide-joint axes without clamping to
joint ranges (an at-limit joint gets an out-of-range qpos that the limit constraint
resolves impulsively next step), and `data.ctrl` still targets the pre-nudge
position, so the position servo can drive the paddle back into the just-cleared
envelope during the serve's first ~2 control steps (ball needs ~0.05 s to leave the
0.45 m sphere at 9 m/s). Bounded in practice: nudges are rare (NP2/NP3 measured 0%),
counted in `point_serve_nudged`, and both travel metrics exclude nudged steps.

### D.6 (context, pre-existing) — (MEDIUM context, could_explain_zero_contact=TRUE but NOT introduced by this

**File**: `src/courtside_dynamics/recipes.py` (PaddleTennis recipe, lines
~1107–1200). Whitespace-insensitive diff shows the ONLY semantic recipes changes are
new CSV/eval info keys (`points_played`, `completed_point_crossings`,
`point_serve_nudged`, seven `point_end_*`). The exploration package
(`use_sde=True`, `ent_coef="auto_0.02"`, `target_entropy=-1.5`,
`train_freq=(64,"step")`) predates the diff. The recipe's comment says the stock
run's 5e-5 coefficient was "the tuner resting at the too-low default target (-3.0),
so the fix is the target" — yet this run collapsed to 5.3e-5 AT target −1.5 with
policy std 0.018. With a tanh-squashed Gaussian whose mean saturates (paddle pinned
at a workspace corner), the sampled −log π SAC's dual update measures reads high
(the −log(1−tanh²) correction), so measured entropy sits above target and the
coefficient anneals to zero even as std collapses. The n-point diff neither causes
nor fixes this; it interacts with F1 by removing the exploration needed to cross
F1's barrier. Flagged because the recipe's stated rationale for −1.5 is empirically
falsified by this run.

---

### D.7 Explicitly checked and found correct (no defect)

- **Feed/serve placement in n-point mode**: `_draw_serve` mirrors freshly built
  arrays in place exactly once for the serving side (`mirror_for_side` mutates and
  returns, `_serve.py:100–112`); mid-episode relaunch teleports the ball to the
  drawn cell (ball qpos + quaternion reset, qvel[0:6] zeroed then linear velocity
  set, `paddle_tennis.py:981–990`), the npoint probe's spawn-cell/velocity/clearance
  witnesses pass, and NP3 held-out certification passed. The feed is reachable —
  the shipped ground oracle rallies on the n-point env (11.4 crossings/episode NP2).
- **Position carryover**: mid-episode `qpos/qvel = data.copy()` keeps both paddles
  and their velocities exactly where play left them; ordering is clear → teleport →
  `set_state` (mj_forward) → fresh `RallyStateMachine` → `sampler.reset` — the
  re-prime AFTER the teleport kills the two reproduced hazards (41 m/s deflection
  spawn; stale reverse-crossing). NP0 lockstep bit-identity of the default path and
  carryover-continuity tests pass on this checkout.
- **Reward accounting across boundaries**: escrow pays at hit, clears on the hit's
  confirm (confirm processed before the same-step new hit), claws back on absorbed
  boundaries, terminations, truncations, and the nonfinite guard alike; per-episode
  identity `paid + clawback == 0.25 × confirms_A` is witnessed exactly by NP1 on
  bit-identical shaped/unshaped arms. No double-count or sign error found.
  `_points_remaining` off-by-one checked (n=1 terminates on point 1; n=2 absorbs
  exactly one boundary). `crossings`/`completed_point_crossings` bases update
  coherently at boundaries; the diagnosis instrument's per-point crossings subtract
  the boundary-step base correctly.
- **Observations at point transitions**: the relaunch observation is taken AFTER the
  fresh rules machine and sampler re-prime, so phase one-hot, serving/returner/
  ball-side flags, bounce/rally counts and the contact latch tail (24–47) all
  describe the new point; `episode_remaining_fraction` correctly keeps decaying
  across points. `serve_side_is_policy` in info intentionally reports the ended
  point's server on boundary steps (documented, and the probe reads
  `env._serving_side` post-flip for the new point — consistent).
- **Event detection**: rules machine untouched by the diff; per-point re-instantiation
  restores serve-phase semantics each point; sampler substep counters and the
  machine's `_last_event_substep` both restart at re-prime. A latched contact at
  relaunch (e.g. paddle resting on court) is primed as latched, suppressing a
  spurious rising edge.
- **Probe segmentation**: boundary step's events are processed before the trace is
  finalized; the truncation-cut partial point gets a `cap` ender exactly like a
  cap-truncated one-point episode; absorbed-at-cap (no relaunch) handled; nudged
  steps excluded from both travel metrics; `point_end_*` counter total pinned equal
  to `points_played` by an NP1 witness.
- **`_record_point_end` coverage**: every non-unsafe `TerminationReason` maps into a
  `_TERM_GROUPS` bucket; unsafe reasons are excluded from `point_completed` by
  construction (they terminate the episode).

### D.8 Bottom line

No implementation bug was found that fabricates the zero-contact result — the
instrument, its normalization, and the n-point mechanics all do what the design doc
says (F3). The prime suspect this diff contributes to the failure is F1: the
n-point continuous economics make every unsuccessful contact attempt strictly worse
than passivity (measured −13.5 vs −4.5 vs −1.6 per episode across the
touch-illegally / statue / touch-legally-and-miss witnesses), a barrier that did not
exist in the one-point task the shaping extension was validated on, amplified by the
uncleared serve corridor under carryover (F4) and locked in by the pre-existing
SAC entropy-collapse pathology (F6).
