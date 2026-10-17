# Making PPO a first-class algorithm (2026-10-16)

Status (2026-10-17): **Steps 0–14 implemented** on `claude/week1-foundations` (Step 0 as the recipe snapshot plus the trainer digest; see `DECISIONS.md`, "PPO is first-class" and the entries before it). Open: the section 4 "Build" items beyond Step 9 (`reset_log_std_on_advance`, `critic_warmup_rollouts`), scheduling the learning probe (nightly or workflow_dispatch), and probing the WallBall, PaddleTennis and humanoid profiles.

Kind: **Adopted plan — implemented on the week-1 branch (2026-10-17).** The 2026-10-16 PPO audit at `main`@`b9585e2` ran three passes:
- a static pass over every algorithm-dependent path;
- an empirical pass that ran PPO end to end through `train()` on five recipes, including a learning smoke test on BallBalance;
- a pass on test coverage and the gaps.

An adversarial verifier confirmed 33 findings, 20 of them unique. This plan orders the fixes. Each step keeps every SAC path and every env stream bit-identical.

PPO works but is not yet first-class. I re-checked each fact below against main @ b9585e2, with read-only probes in session scratch space (not committed).

- **The core loop is sound.** PPO runs through `train()` with every callback. Time-limit truncation bootstraps correctly. PPO→PPO warm start copies the weights exactly. With explicit hyperparameters, BallBalance PPO scored 750/750 on 20 held-out seeds; the zero action scores 470.
- **Recipes are built around SAC.** Each `Recipe` carries one flat `extra_cfg["model_kwargs"]` (recipes.py:40-55), so PPO is refused on PaddleTennis and WallBallBootstrap and gets SB3 defaults everywhere else. SB3 2.9's SAC now accepts `n_steps`, so `validate_model_kwargs` (algos.py:41) cannot catch a shared key whose meaning differs between the two algorithms.
- **Guards count env steps, not policy updates.** The WallBall-family and PaddleTennis recipes set `early_stop_degenerate_evals=5` at `eval_freq` 25k. With 8 envs × SB3's default 2,048-step rollout, PPO has done only 0 to 7 updates when that stop fires. `quick_test` (25k steps) can report "completed" with zero updates.
- **The only recipes that default to PPO run outside PPO's trust region.** On the humanoid stages, approx_kl reads 0.22–1.71 and clip_fraction 0.72–0.83. At SB3 defaults, gSDE's action noise before clipping has a std of 5.1 on a [-1, 1] action space.
- **Tests and loaders lag behind.** No test runs PPO through selection, the performance gate, checkpoints or the loaders, and several loaders hard-code `SAC.load`.

## 2. Ordered change list

Every new behaviour is gated on `algo_family(cfg.algo) == "PPO"` unless a step says otherwise.

**Step 0 — SAC freeze harness (do this first).** About 3 h.
- `tests/test_sac_freeze.py` plus `tests/golden/recipe_snapshot.json`, captured at b9585e2. For every recipe × {default, SAC, DemoSAC} × quick_test {off, on}, where the resolved algorithm is SAC-family, store the TrainConfig as canonical JSON. Callables are stored by qualname. Env kwargs are read from the factory closure, so no env is built and the test runs in under 2 s.
- `tools/sac_freeze_digest.py --base <ref>` runs small seeded `train()` jobs on HEAD and on a `git worktree` of the base ref, on the same machine (this avoids cross-platform float drift in committed golden values). The jobs:
  - SAC on BallBalance with 2 envs;
  - DemoSAC with demos off;
  - a PaddleTennis-recipe SAC quick run (gSDE, `train_freq` 64, normalizer with excluded indices, headline selection, checkpoint diagnosis).
- The tool hashes the policy state_dict, `log_ent_coef`, VecNormalize stats, monitor r/l columns, `evaluations.npz`, `eval_info.csv`, the selection fields of `best_model_meta`, and the final numpy/torch RNG state. Every PR below must pass it before merge.
- Why the existing tests are not enough: SD0 compares DemoSAC with SAC at the same revision, and the np0/kd0 lockstep tests check env streams at one revision. Neither catches a change that hits SAC and its baseline alike. No step here touches `envs/`, so the monitor hash plus diff review covers the env streams.

**Step 1 — Algorithm families (static-7).** 1.5 h.
- algos.py: derive `OFF_POLICY_ALGOS` and `ON_POLICY_ALGOS` with `issubclass`, and add `algo_family()`.
- Replace the hand-written string tests in `_resolved_norm_reward` (train.py:934), `_config_norm_reward` (:1544), `_prepare_warm_start` (:1559) and the notebooks.
- Tests: the derived sets equal today's. Unit-test both resolvers: `None` resolves to True for PPO and False for SAC and DemoSAC.
- SAC: classification is identical, so every resolved value is identical. One intended notebook-only fix: in sb3_training, `ALGO="DemoSAC"` gets 8 envs instead of 16. Note it in the CHANGELOG.

**Step 2 — Per-algorithm config blocks (tests-gaps-1, static-1, empirical-9).** 5 h.
- Add `Recipe.algo_cfg[family]`. Merge order: extra_cfg → algo block (model_kwargs merged one level deep) → TOML → quick_test → explicit arguments.
- Move the PaddleTennis and WallBallBootstrap SAC bundles into `algo_cfg["SAC"]`, and the humanoid `{use_sde, ent_coef}` bundle into `algo_cfg["PPO"]`.
- run_config.py: add `[train.algo.sac]` and `[train.algo.ppo]` tables, merged only when the family matches.
- Add `DIVERGENT_SHARED_KWARGS = {n_steps, batch_size, ent_coef, use_sde, sde_sample_freq}`. A flat TOML or recipe key from this set is refused when the run's family differs from the recipe default's family. DemoSAC on PaddleTennis still passes. An explicit `model_kwargs=` is exempt. The error hint now says that an explicit `model_kwargs=` replaces the whole bundle.
- Tests:
  - the SAC snapshot is unchanged except the humanoid×SAC entries (an intentional change on a non-campaign path; update test_recipes.py:754);
  - the refusal test at test_recipes.py:771 becomes a PPO build test;
  - per-family TOML tables;
  - the divergent-key refusal.

**Step 3 — PPO starting defaults in `algo_cfg["PPO"]` (values in section 3).** 2 h of code plus about 4 h of manual probes. Test: every recipe builds under PPO and passes Step 4. SAC: the values are keyed to PPO only; the snapshot confirms it.

**Step 4 — On-policy pre-flight checks (static-2, static-8, static-3(1)).** 3 h.
- Add `_validate_on_policy_config(cfg)`, called next to `_validate_callback_config` (before `write_run_config`) and from `build_train_config`. When a key is absent, its default comes from the constructor via `inspect.signature`.
- It refuses:
  - the SAC-only gate levers `advance_update_pause_steps > 0`, `clear_replay_buffer_on_advance` and `reset_entropy_on_advance`;
  - `batch_size <= 1` when advantages are normalized, `n_steps * n_envs <= 1`, and a rollout length that `batch_size` does not divide;
  - `eval_freq // n_envs < n_steps`, and `total_timesteps < n_steps * n_envs`;
  - `policy_kwargs` keys the policy class does not accept.
- Tests: one per refusal, each asserting the log_dir is left untouched. WallBallDepthCurriculum and its Aligned variant under PPO are refused at build time.
- SAC: the function returns immediately for off-policy algorithms.
- This step must ship together with Steps 3 and 5. Otherwise sb3_training's 16-env PPO default and every PPO quick test would be refused.

**Step 5 — `quick_test` for on-policy runs (empirical-2).** 2 h.
- After explicit overrides merge, if the caller did not set `n_steps`, set it to the largest power of two ≤ min(current, 25,000 // (4·n_envs), eval_freq // n_envs). SB3's default 2,048 becomes 512 at 8 envs and 256 at 16; one env keeps 2,048.
- `write_run_summary` warns when `_n_updates == 0`.
- Tests: at least 4 rollouts at 1, 8 and 16 envs. The SAC quick-test snapshot is unchanged.

**Step 6 — Count guards in policy updates (static-3, empirical-1).** 3 h.
- At train.py:2033 and :2161, floor the early-stop `min_evals` and `degenerate_min_evals` at ceil(K · n_steps · n_envs / eval_freq). K is a new recorded field, `min_policy_updates_before_stop` (default 20).
- performance_gate.py: for on-policy runs, apply stage advances in `_on_rollout_end` so no rollout straddles a stage change. `_on_step` (:772) currently applies them mid-rollout.
- `eval_info.csv` gets an `n_updates` column on PPO runs.
- Tests: the PPO degenerate guard cannot fire before K updates; gate stages are applied at a rollout boundary; SAC's callback kwargs equal today's (asserted).

**Step 7 — Reward-normalizer gamma (static-4).** 1 h.
- Pass the resolved model gamma to `SelectiveVecNormalize` (train.py:1944) when the run is on-policy and normalizes rewards, and set it in `_load_warm_start_normalizer` too.
- Record `normalize_reward_resolved` and the normalizer gamma in config.json.
- Test: the pickled normalizer gamma is 0.995. SAC: the branch is on-policy only.

**Step 8 — CPU by default (static-5).** 1 h. In `_build_algo`, set `device="cpu"` when the run is on-policy, uses an ActorCriticPolicy, `require_device` is None and no device key is given. Test with CUDA mocked as available. The SAC path is untouched.

**Step 9 — Warm-start exploration (empirical-4).** 2 h.
- Add `WarmStartConfig.reset_log_std` (default False).
- On-policy: `transfer_log_ent_coef=False` now raises and points to `reset_log_std`. When `reset_log_std` is True, refill `policy.log_std` from the configured `log_std_init` after `load_state_dict`. Provenance lists `policy.log_std` as either transferred or reset.
- SAC family with `reset_log_std` raises. The field is new, so no existing config changes.
- The humanoid notebook's AUTO_ADVANCE passes True.
- Tests cover both branches.

**Step 10 — Loaders follow the recorded algo (static-6).** 3 h.
- Add `algos.load_run_model(path, algo=None)`, which reads config.json. SAC and DemoSAC load through `SAC.load`, which is DemoSAC's designed inference path. PPO loads through `PPO.load`. An explicit algo that disagrees with the record raises.
- Apply it to paddle_diagnosis.py:70 and its four tools, `score_paddle_stage`, `record_best_model_video(algo=None)`, `_load_best_model`, and p5_transfer.py:323.
- Test: `score_paddle_stage` on a real PPO checkpoint. SAC still uses the same `SAC.load` call.

**Step 11 — Provenance and training health (static-9/10/11, empirical-3).** 3 h.
- `_model_info` records `target_kl` (SAC has no such attribute, so it is skipped). On-policy runs also record `sde_sample_freq`, `log_std_init`, `ortho_init`, `n_updates`, and an actual-versus-budget timesteps line.
- New on-policy Training Health flags:
  - approx_kl > max(2·target_kl, 0.05);
  - clip_fraction > 0.3;
  - explained_variance < 0 over the last quarter of training;
  - log std drifting more than 1 nat from its initial value.
- Test: the flags fire on a progress.csv shaped like the humanoid probe. SAC: covered by the attribute skip and the on-policy branches; the digest confirms it.

**Step 12 — Tests that cross the whole path (tests-gaps-9).** 5 h.
- **Parametrized SAC/PPO `train()` test** on BallBalance (episode_len 12, n_steps 8, 1 env). It covers headline selection with confirm_best and eval_seed, final_info_eval, checkpoints with VecNormalize, a one-stage gate and the closing evaluation. It asserts the artifacts, the resolved_model keys, the stop reason and `norm_reward`.
- **PPO determinism:** two seeded runs give identical parameters.
- **CI learning smoke (verified here):**
  - Setup: a 1-D reach env defined in the tests (reward −|goal − x|, truncated at 40 steps, never terminates), run through `train()` with PPO, 4 envs, n_steps 256, batch 256, lr 1e-3, reward normalization on.
  - Result: over 24,576 steps the evaluation mean rose from −15.6 to −1.8 (zero action is about −26.7). That was 240 updates in 10 s on one thread.
  - Assertions: best ≥ −6; approx_kl present and < 0.1; reloading the best model and normalizer pair reproduces its score.
- **Manual learning probe**, `tools/ppo_learning_probe.py`, run nightly or by workflow_dispatch: BallBalance with the Step 3 defaults and a 330 s cap. It passes when the best model scores ≥ 700 on 20 held-out seeds. The earlier run measured 750; the zero action scores 470 ± 18.

**Step 13 — Opt-in SubprocVecEnv (tests-gaps-8).** 3 h. Add `TrainConfig.vec_env` ∈ {"dummy", "subproc"}, default "dummy", applied to training workers only and recorded. Test: a subproc rollout matches the dummy one (observations exact; rewards equal after a float32 cast). SAC's default is unchanged.

**Step 14 — Docs (tests-gaps-13).** 2 h. README:101, the docstrings of `TrainConfig.algo` and `build_train_config`, the per-family tables in the run-config spec, the sb3_training n_envs table, a DECISIONS entry and the CHANGELOG.

Total: about 40 h of engineering, plus wall-clock time for the probes.

## 3. Recommended PPO defaults

These are starting points to tune, not frozen values. Unless noted, all families use clip_range 0.2, gae_lambda 0.95, n_epochs 10, the recipe's own gamma, and eval_freq 25k. Every row passes Step 4's checks at 8 and 16 envs.

| Family | Settings | Rationale |
|---|---|---|
| BallBalance / BallBounce | 8 envs, n_steps 512, batch 256, lr 3e-4, ent_coef 0, log_std_init −1, target_kl 0.03, reward normalization on | Measured: 750/750 by 320k steps, with approx_kl 0.011–0.014. The zero action already balances about 470 steps, while std-1 noise drops the ball in about 120. |
| WallBall family | 8 envs, n_steps 1024, batch 256, lr 3e-4, ent_coef 0, log_std_init −1, target_kl 0.03; gamma 0.995 where the recipe sets it, with a matching normalizer gamma | The longer horizon gives about 11 episodes per rollout, and Step 6 keeps the degenerate stop at 20+ updates. Not yet probed. |
| PaddleTennis | 8 envs, n_steps 1024, batch 256, lr 3e-4, target_kl 0.02, use_sde, sde_sample_freq 64, log_std_init −2, ortho_init False, ent_coef 0, reward normalization off | Matches the SAC recipe's 64-step noise hold. With −2 and no orthogonal init, pre-clip noise std measures about 0.28, against 5.1 at SB3 defaults. The reward is sparse, so no return normalization. Not yet probed. |
| Humanoid Stages 0–2 | 1 env (later 8 envs at n_steps 512, once Step 13 lands), n_steps 2048, batch 256, n_epochs 5, lr 1e-4, target_kl 0.02, use_sde, sde_sample_freq = episode_len (150 / 300), log_std_init −2, ortho_init False, ent_coef 0–0.001, reward normalization off, `reset_log_std` at each stage warm start | target_kl is the essential fix: log_std_init alone measured approx_kl 2.45. Holding noise for a whole episode matches the evidence that a constant random action succeeds about 15% of the time; today's setting draws one noise sample per 13 episodes. Under ent_coef 0.01, every one of the 58 action columns' log std rose by 0.04 to 0.18 within 26.6k steps. Masked dimensions feel only the entropy term, so this drift accumulates across stages. |

## 4. PPO counterparts for SAC-only features

Build:
- **Temperature transfer and the temperature-skip option** → `reset_log_std` (Step 9). Later, a gate option `reset_log_std_on_advance` that reuses the same helper, off by default.
- **Replay clearing and update pause on stage advance** → apply stage changes at rollout boundaries (Step 6). PPO throws its rollout buffer away after every update, so there is nothing to clear.
- **Warm-start warmup** → `critic_warmup_rollouts` (default 0). It freezes the actor and log std for k rollouts after a warm start, so the transferred value head fits the new stage's reward before the policy moves. About 3 h; it matters for the humanoid ladder.
- **SD0 bit-identity test** → the PPO determinism test (Step 12).

Skip:
- **Automatic entropy tuning.** A fixed ent_coef plus target_kl plus the std health flag is the standard PPO approach.
- **In-loop demo injection (DemoSAC).** If PPO needs demonstrations, pretrain an actor with behaviour cloning and use it as a warm-start source through the existing path. A BC auxiliary loss inside the PPO update (DAPG-style) belongs in the GPU stack.
- **Gradient-step compensation and train_freq coercion.** These do not apply to PPO.

## 5. Defer to the GPU / multi-agent stack

- **Massively parallel physics** (MJX / MuJoCo Warp) with a GPU-native PPO: thousands of envs, GPU-resident buffers, observation normalization inside the compute graph, large minibatches and a KL-adaptive learning rate. Do not push SB3 onto the GPU; Step 8 keeps it on CPU.
- **Shared-parameter multi-agent PPO for 2v2:** separate observations and actions per agent, agent IDs, a centralized critic (MAPPO), and self-play opponent pools. In SB3, HumanoidTennisCoop stays a centralized single-agent env (README:93-95), with no PettingZoo adapters.
- **Critics that see privileged information, recurrent policies, and population-based training or large sweeps.**

Carry these forward from this plan into that stack:
- the per-family config schema;
- guards counted in policy updates;
- the health thresholds;
- the warm-start semantics (`reset_log_std`, observation fingerprint, normalizer transfer);
- loaders keyed on the recorded algo;
- the two learning bars, reused as SB3-vs-GPU parity tests.

Also add an export of policy weights plus `obs_rms`, so policies trained on the GPU are scored by the existing CPU MuJoCo evaluation harness.