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

## Deviations from the "zero contacts after 100k" narrative

The summary claim "ZERO contacts from 200k through 2M" is an over-simplification. Strictly zero-contact checkpoints are 200k, 900k, 1.0M, 1.6M, 1.7M, 1.8M, 1.9M, 2.0M. Residual sporadic contact persists through the middle of the run:

- 300k: n=2, 400k: n=3, 500k: n=1, 600k: n=1, 700k: n=1, 800k: n=1, 1.1M: **n=7**, 1.2M: n=1, 1.3M: n=1, 1.4M: n=2, 1.5M: n=1. Total 21 contacts across 200k–1.5M (~2,780 probe points), then strictly zero for the final 500k (1.6M–2.0M).
- Every one of these residual contacts except one is a serving hit #1 (the fed ball on the policy's own serve); the sole receiving contact is 800k's single hit. The policy never once achieved hit #2 in any checkpoint.
- Ball quality of residual contacts is terrible: only ONE policy shot in the entire run landed in — the single 500k shot (crossed 100%, in 100%, in-depth 5.33 m). All others were 0% in, with out-depths 9.77–18.09 m (wild over-hits) or 0% crossed (1.1M's 7 contacts all failed to cross; 2 into the net — the only policy_shot_net events after the oracle).
- 1.1M is the largest post-100k anomaly: 7 serving contacts (k=1 serving 7%), 0% crossed, enders policy_shot_out 4 + policy_shot_net 2, touched-after-bounce briefly back up to 4%. This flicker dies out by 1.2M.

## Drift trends

1. **Ready-position bounce-time error worsens.** 100k–1.0M oscillates 2.00–3.63 m (mean ~2.6); 1.2M–2.0M sits at 3.27–3.97 m (mean ~3.55, worst 3.97 at 1.7M). Oracle is 0.89 m. The policy is not merely failing to swing — it is drifting further from where the ball will bounce as training proceeds.
2. **Touched-after-bounce decays to hard zero.** 3% at 100k, 1–4% through 1.5M, exactly 0% at every checkpoint from 1.6M on (and at 200k/900k/1.0M). In the final quarter of training the policy never even makes post-bounce contact with the ball's position.
3. **No recovery/positioning learning.** Inter-point recovery travel is flat at 7.65–9.28 m (oracle 1.95 m) with no downward trend — the policy is ~8–9 m out of position at every feed arrival for the entire run.
4. **Point-ender mix ossifies.** policy_never_reached climbs slightly from 176 (100k) to 184 (1.7M/1.9M/2.0M) of ~214 points and secondary faults (policy_shot_out, wrong_hitter, failed_to_cross) disappear; by 1.7M/1.9M the enders are exactly two categories: policy_never_reached 184 + cap 30. The behavior distribution narrows, consistent with the reported entropy collapse (ent_coef 5.3e-05, policy std 0.018).
5. **Environment/opponent stationarity confirmed.** Opponent line is stable across all checkpoints (n=98–102, crossed 92–95%, in 88–92%, in-depth 3.96–4.05), so metric changes are attributable to the policy, not probe drift.
6. **Crossings mean flat at 0.42–0.45** (oracle 5.48) — essentially all crossings are opponent shots; there was never any rally.
7. **Point throughput inverse-signal:** checkpoints complete 213–218 points in the probe budget vs the oracle's 63, i.e. points end ~3.4x faster because the policy never reaches the ball (terminations dominated by out_of_bounds 173–178 = opponent's in-bounds ball bouncing away untouched, plus second_bounce 5–9).

## C. Training-metrics trajectory (progress.csv / eval_info.csv)

(see below — added from the metrics analysis)

## D. Branch code review (claude/repo-env-review-4096tu at 2e597b1)

(see below — added from the adversarial review)
