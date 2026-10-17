# Campaign structure: tasks, levers, instruments and experiments (2026-10-16)

Kind: **Adopted design — week 1 in progress (2026-10-16).** A design study of how an environment that needs many experiments should be organized. It had three inputs:
- an archaeology of every lever in the WallBall and PaddleTennis campaigns, taken from git, docs and the Drive `configs/` folder;
- a prototype that moved contact, reach and hold shaping out of `PaddleTennisEnv` with bit-identical streams;
- a prior-art survey covering Isaac Lab/mjlab manager-based envs, Hydra, RL Zoo and others.

Three proposals were scored by two judges, then synthesized. The L-steps interleave with the M-steps of `design_env_ladder_and_structure_20261016.md`.

**Short answer.** Yes. The problem is structural, not one bug:
- Every training aid (a "lever") was built as a default-off kwarg inside the env class, and none were removed afterwards.
- Each experiment is defined in 5–7 places.

The fix keeps today's envs and `train()` and moves three boundaries:
1. **Tasks become frozen, versioned IDs.**
2. **Levers become small, deletable modules outside the env.** They read one per-step events record that the env publishes.
3. **An experiment is one spec file in git.** It names a parent and exactly one lever, and its `kind` fixes the seeds, the budget and how the verdict is computed.

A standalone harness scores the bare task, with no shaping.

**How this fits the earlier plan.** It refines M0–M12, adding ten L-steps (about 18 net days) between existing M-steps. The runs that plan starts on day 0 never pause.

**Basis.**
- **Core:** the lever-plugin proposal.
- **Ordering:** from the campaign-first proposal.
- **Added rules:** the manager-based proposal's resolver and mjlab rules.

Code claims were re-checked read-only at `2e66444`; the env is unchanged from `b9585e2`.

## 1. Findings

**Already found by this session's review**

| # | Finding | Evidence | Cost |
|---|---|---|---|
| R1 | Validation runs after side effects | Gate values are checked after `config.json` is written and CSVs are rotated. Algo overrides fail late, with wrong hints. Headline-only knobs do nothing without `headline_key`. | Bad launches cost hours and leave debris |
| R2 | Config layering is re-implemented at each consumer | The notebook re-implements the `[env]` layering. 11 tool files and `notebook_utils.py:2589` call `PaddleTennisEnv(` directly. | Consumers drift silently |
| R3 | Run dirs and Drive can't be trusted | `reuse_log_dir` keeps stale CSVs and TensorBoard events. Drive isn't flushed on disconnect. One run lost all its videos. This study adds: 7 of ~48 run dirs are stubs that look live. | Evidence is lost or misread |
| R4 | Nothing is pinned | Notebooks install from `main`. MuJoCo 3.16 drift. Seed-ledger gaps. A TOML cannot express a PPO PaddleTennis run. | Runs can't be reproduced |

**Found by this study**

| # | Finding | Evidence | Cost |
|---|---|---|---|
| S1 | Levers enter the env and never leave | 92% of the 22,154 lines added for 26 levers survive, and only 1 lever was deleted. About 10,900 lines (17% of the tree) serve negative, never-run or retired levers. 6 of 18 named PaddleTennisEnv kwargs are off in every recipe. | Every change re-proves "bit-identical when off" across more arms: 140 doc mentions and 12 one-off lockstep tests. A drill-only path corrupted `completed_point_crossings` (`2668be0`). |
| S2 | The lever is sometimes in none of the 5–7 places | LT1's TOML doesn't contain its lever. Three runs share one TOML sha with different algo settings. Run TOMLs live on Drive, outside git. | Frozen budgets were overridden 4 times: GoalRally 3M vs 6M, LH1b 2M vs 1M (patience 60 vs 20; found by the L1 index, 2026-10-17), LH1c 3M vs 1M, and LT1's unregistered companion (`20260828_113136`) 3M vs 1M; the registered LT1 run trained its frozen 1M |
| S3 | Experiments copy and edit instead of naming a parent and one change | Flipping one boolean meant restating 5 gate stages. 5 TOMLs repeat the same operational block. The L2 "replication" changed both the seed and `n_envs`. | Results can't be attributed to one change |
| S4 | Env IDs don't name the task | Suffixes were dropped in `5337792`. About 4 PaddleTennis task changes ran under one ID. MuJoCo moved 3.11→3.12 between LH1b and LH1c. | Comparisons across runs are unsound |
| S5 | Shaping leaks into selection | `eval_env_overrides` strips only the drill (`recipes.py:1189`). Oracle eval episodes carry +2.48 to +2.98 of shaping on task returns of 10–14. | The tie-break (`min_delta` 0.25) can favour escrow collection over task return |
| S6 | Code comes before the cheap test | Bootstrap (1,151 lines) never ran. The drill (1,706 lines) and DemoSAC (2,151 lines, 44+ days) have no runs. 9 bespoke probe tools (3,264 lines). About 1,390 hand-booked seeds. The cheap paths that did work: the capacity test needed no code, and the PT2 probe killed command-rate before any code. | About 5 days median from idea to verdict, mostly before any run |
| S7 | Prose is the experiment ledger | 35–39% of commits are docs-only. PaddleTennis has 58k words, 41% on levers never adopted. | Answering "what have we tried?" means reading about 100k words |
| S8 | Runs are rare and n=1 | Eval takes 55–65% of wall clock, and 1M SAC steps take 6h17m. 48 days without a run. The first 1M steps of LT1's companion were byte-identical to LT1. | No replication, and a slow learning loop |

## 2. The structure

| Object | What it is | Home | Changes by |
|---|---|---|---|
| **Task version** | Physics, rules, task reward (return, fault, unsafe), termination, observation, episode format | `tasks/` registry, `courtside/PaddleTennis-v2`, plus a lock | A new version only |
| **Eval protocol** | Episodes, seed blocks, headline, selection and tie-break, instruments, device | `paddle-npoint@vN` | A bump, which means re-score, not retrain |
| **Lever** | Reward term, launch event, algo preset or warm-start option | `levers/<name>.py` plus a registry entry | Added as a module; deleted on retirement |
| **Instrument** | Info-only metric: contact depth, k1/k2 survival, term dose | Harness `instruments/` | A protocol bump |
| **Experiment** | Parent, one lever, a kind | `experiments/specs/<family>/vN/*.toml` | Immutable once a run references it |

An era is still (task version, eval protocol).

**Task or lever?** Anything that changes what counts as a point, an ending or the observation is a task version; n-point play was one. Anything that changes only what the learner is paid, where it starts, or how it learns is a lever.

**Layout** (additions to the earlier target layout):
```
src/courtside_dynamics/
  envs/_events.py          CourtStepEvents, contract "courtside.events.v1"
  envs/paddle_tennis.py    fills step_events; public snapshot()/restore_point(); 19 -> 11 kwargs
  tasks/registry.py        TaskVersion, make_task(), locks/, legacy_kwargs_to_spec()
  levers/                  registry + TermStack; contact_escrow.py, reach_escrow.py (adopted)
  experiments/             spec.py (resolve->validate->freeze->hash, no side effects), run.py
                           (spec -> today's TrainConfig), harness.py, protocols.py, select.py,
                           probe.py, verdicts.py, status.py, index.py, ledger.py
  experiments/specs/       algos.toml, paddle_tennis/v2/*.toml, _archive/   (package data)
docs/status/<family>.md    generated
Drive runs/<family>/<spec>/s<seed>/<run_id>/  run_record.json, spec.lock.toml,
      sync_manifest.json, snapshots/, scores/<protocol>.jsonl;  index/runs.jsonl (regenerated)
```
Specs ship as package data, so a Colab install from a tag carries exactly the specs that tag can run.

```python
@dataclass(frozen=True)
class TaskVersion:
    id: str                         # "courtside/PaddleTennis-v2"
    env_cls: type
    kwargs: Mapping[str, Any]       # the whole task; make_task() refuses non-render kwargs
    eval_kwargs: Mapping[str, Any]  # play_env_cfg analogue
    lock: str                       # dyn_fp stream digest (M1) + MuJoCo/numpy, XML sha, frame_skip, timestep
    events: str | None              # "courtside.events.v1"; None = legacy (WallBall)

@dataclass(frozen=True)
class CourtStepEvents:              # (N, S[, 3]); after rules.advance, BEFORE n-point relaunch
    physics_stepped; legal_hit; legal_hit_pos; first_bounce; first_bounce_pos
    confirmed_return; paddle_head; ball_pos; point_ended

class RewardTerm(Protocol):
    channels: tuple[str, ...]       # rew_<name>[_clawback]; CSV schema unchanged
    def reset(self, mask: np.ndarray) -> None: ...
    def __call__(self, ev: CourtStepEvents) -> tuple[np.ndarray, ...]: ...  # (N,) per channel

@dataclass(frozen=True)
class Lever:
    name: str
    kind: Literal["reward_term", "launch_event", "algo", "warm_start"]
    params: type                    # frozen dataclass; unknown key -> "did you mean"
    order: int | None               # summation slot: contact 10, reach 20
    status: Literal["probe", "pilot", "adopted"]   # retired = deleted
    gpu_step_cap: int
    kill_date: date
```

**Where `TermStack` goes.** SB3's `make_vec_env` wraps `Monitor` around what `env_fn` returns, and applies `wrapper_class` outside it (verified). So `TermStack` sits inside `env_fn`, and the Monitor CSVs stay byte-identical. It has two modes: `train` adds the terms to the reward, and `audit` only logs them to `info`.

```toml
[experiment]
id       = "pt2.hold.pilot"
kind     = "pilot"
parent   = "pt2.base-r1"
lever    = "hold_escrow"    # every changed levers.*/algo.*/warm_start.* path must fall under it
question = "Does a post-swing hold escrow lift k>=2 survival at 1M?"

[levers.hold_escrow]
scale  = 0.25
travel = 4.0

[budget]
timesteps = 1_000_000

[seeds]
pool   = "pilot"
values = [...]              # written by `experiments new`; ledger = legacy blocks + spec seeds

[verdict]
metric     = "k2_either_survival"
min_effect = 0.02
```

**Resolver rules.** All of them run before anything is written:
- Unknown or unread keys are errors.
- Every changed path must fall under the declared lever.
- A `[task]` change is refused with "new task version".
- A replication may differ only in `[seeds]`.
- `[algo]` is required, and its defaults come from `algos.toml`, the exploration preset's only home.
- The launcher has no budget, seed, env or algo overrides; quick tests are `kind="smoke"`.

**Fit with the mjlab humanoid track**
- **Terms already have mjlab's shape:** stateful classes with `reset(mask)` returning `(N,)` arrays. One batched instance over 6 envs matched exactly, so the torch twins are mechanical.
- **Escrows take their weight as a parameter and pay lump sums.** A manager must never rescale them by weight×dt, because w·Σp ≠ Σ(w·p). Check mjlab's reward manager before porting.
- **The names map across.** `eval_kwargs` maps to `play_env_cfg`, and spec paths map to tyro overrides.
- **The real cost is the events producer.** The sampler and rules machine must become batched tensors that keep substep ordering, with `point_ended` kept separate from env reset.
- **Doubles adds a player axis** (`events.v2`) at rung entry.
- **Don't build a generic CPU manager layer.** Only the reward table has consumers.

## 3. The life of a lever

| Step | Files touched | Cost | Exit |
|---|---|---|---|
| **0. Idea** | `<L>.probe.toml` (~15 lines: question, bars, cap, kill date). No design doc, env edit or hand-booked seeds. | minutes | the spec resolves |
| **1. Probe** (CPU) | `levers/<L>.py` (20–60 lines) and its test. A missing hook field is added additively, gated on unchanged digests. `experiments.probe` runs the oracle, random and camper witnesses plus the parent's snapshots, with the term in audit mode. | 10–120 CPU-min | PASS goes to pilot; KILL goes to retire; MIDDLE goes to the maintainer. If it must change observations, rules or endings, it is a task version, not a lever. |
| **2. Pilot** | `<L>.pilot.toml` (parent = incumbent); `run_spec()` | 2 seeds, paired; the parent arm is reused by spec hash | continue, stop, extend or invalid. A pilot never adopts. |
| **3. Comparison** | `<L>.cmp.toml` | 3 seeds, paired; usually only the child trains | adopt, reject or inconclusive (one extension to 5 seeds) |
| **4a. Adopt** | `base_r2.toml` = base plus one lever table (~5 lines) | minutes | **It becomes a recipe default, never a task change,** so there is no new era |
| **4b. Retire** | Tag `lever/<L>/final`. Delete the module, its test and its tools. Move its specs to `_archive/` with `code_ref`. Records and verdicts stay. | minutes | Env, tasks and digests are untouched, so nothing is re-proved. `experiments debt` fails while dead code remains. |

Under this regime, hold would have cost about 100 lines, all deleted at its verdict. Today about 1,050 lines and 4,997 words of it remain.

## 4. Experiment kinds and computed verdicts

| Kind | Question | Seeds | Verdict |
|---|---|---|---|
| smoke | Does it run? | scratch | none; never counts as evidence |
| probe | Does the mechanism fire? | probe pool | PASS / KILL / MIDDLE |
| rescore | What do snapshots score under protocol P? | the protocol's own blocks | none; supplies baselines |
| baseline | Where is the incumbent this era? | 3, paired pool | IQM and CI |
| pilot | Is there a signal at 1M? | 2, paired with the parent | continue / stop / extend / invalid |
| comparison | Should it be adopted? | 3–5, paired | adopt / reject / inconclusive |
| claim | Is the rung met? | sealed held-out block | pass / middle / fail, final |
| replication | Was it seed luck? | new seeds only | feeds the parent's pool |

**How verdicts are computed.** `verdicts.py` reuses `score_campaign_bars` (moved from `notebook_utils.py:2389`).

**Pilot verdicts**
- **Invalid:** the lever's audited dose is outside the probe's band, or a seed aborted.
- **Stop:** the mean paired difference is ≤ 0.
- **Continue:** the mean is ≥ `min_effect` and every seed's difference is above 0.
- **Extend:** anything else; one extra seed, allowed once.

**Comparison verdicts.** Compute the IQM per arm, then a stratified-bootstrap 95% CI of the difference.
- **Adopt:** the CI's lower bound is above 0 and the point estimate is ≥ `min_effect`.
- **Reject:** the CI's upper bound is below `min_effect`.
- **Inconclusive:** anything else.

**Rules for every verdict**
- Verdicts read **confirm seeds**, never the scores that selection used.
- Comparisons across eras are refused unless both arms are re-scored under one protocol.
- A manual override is stored with its reason.
- STATUS and the DECISIONS lines are generated.
- Only claims get a one-page prereg. Every other kind gets about 100 words: its question plus its bars.

**Why iteration gets cheap**
- **Snapshots at every eval cadence:** 1.6 MB each, about 64 MB per 1M steps, so any of them can be re-scored offline.
- **Offline scoring is cheap.** Measured here: 81 s per 30×1,500-step selection eval on one CPU core. The in-loop equivalent is inferred at 310–370 s.
- **Projected pilot time:** if selection leaves the loop, a 1M SAC pilot could take about 2.2–2.8 h instead of 6h17m. This has not been measured on Colab.
- **In-loop eval stays for now,** because early stopping, the guards and `PerformanceGate` depend on it, as the earlier plan says. It shrinks only after two checks:
  - Colab's `cpu_count` and 1-vs-2-trainer throughput are measured.
  - An A/B through `train()` with the repo's own callbacks shows the same stop step, or the change is declared.
- **Parent arms are reused by spec hash,** so a comparison usually trains only the child.
- **PPO makes multi-seed pilots routine.** PPO is first-class from M2 and collects 632–832 steps/s on 4 CPU cores, so 3M steps take about 1–1.3 h. Run multi-seed pilots in PPO on CPU, and SAC pilots on the L4.

## 5. PaddleTennis under the new structure

| Constructor arguments (19) | Destination |
|---|---|
| `episode_len`, `frame_skip`, `serve_config`, `opponent_controller`, `volley_rule`, `return_reward`, `fault_penalty`, `unsafe_physics_penalty` | stay in the task |
| `points_per_episode` | stays; pinned to 1 in `PaddleTennis-v1` (historical one-point) and None in `-v2` (current) |
| `court_style` | stays; render only, outside `dyn_fp` |
| `deep_contact_depth` | stays until M11's `InfoSchema`, then becomes a harness instrument |
| `contact_shaping` | `contact_escrow` lever (adopted, order 10) |
| `reach_shaping`, `reach_shaping_radius` | `reach_escrow` lever (adopted, order 20) |
| `hold_shaping`, `hold_shaping_travel` | deleted behind a tag (closed 08-22) |
| `drill_library`, `drill_fraction`, `drill_context` | deleted after `restore_point()` goes public, because the DemoSAC harvest calls `env._drill_entries` and `env._launch_drill` (`tools/paddle_tennis_k2_demo_harvest.py:156-182`) |

That leaves 11 arguments, and the env's default build equals the trained task.

**The other levers**
- **Exploration package:** an `algos.toml` (SAC, v2) entry.
- **n-point play:** the v1→v2 boundary.
- **Temperature skip:** the option and its tests (217 lines) are deleted behind a tag.
- **PT2 (command-rate):** archived.
- **DemoSAC:** stays at `training/demo_sac.py`, the pickle path the earlier plan keeps. It becomes an `algo` lever with a cap and a kill date, waiting for the PPO-reference readout.

**`pt2.base-r1`** is `PaddleTennis-v2`, then contact (0.25), then reach (0.25, r=3.0), plus SAC `sac-explore-v1` (gSDE, `auto_0.02`, target −1.5, `train_freq` 64).

**Extraction prototype**
- **Reach alone, moved out:** 0 differences in obs, reward, terminated, truncated and the reach info keys over 89,278 steps (3 configs × oracle, random and camper; per-env and batched).
- **Contact, reach and hold together, in the historical order:** 0 differences over 36,000 steps. The same held on fresh seeds 7301–7302, and with a NaN action injected while an escrow was pending (0 on 3 seeds, including the −0.0 clawback bytes).
- **Without the hook:** reward differs on 38–106 steps per row (max 0.0051), so the hook is required.
- **One aid at a time is fragile:** it changes the last bit on 25.2% of steps with co-occurring values, so all three move together.
- **Env size:** about +40 hook lines and −500 escrow and drill lines, which takes the env from 1,781 to about 1,280 lines.

**Eras**
- **@v3 (new, step L3).** The eval env sets contact and reach to 0, which is valid because PaddleTennis observations hold no escrow state. The tie-break becomes task reward, and verdicts read confirm seeds.
- **Rescoring.** LR1, the registered run, LH1\* and LT1 (pre-10-05 protocol) and the PPO reference (@v2) are re-scored on CPU under @v3. The k≥2 headline is reward-independent, so it must match across protocols.

**Effect on M12.** Clawbacks are paid on the truncation step (`paddle_tennis.py:897-903`). Once the escrows are external, fixing that is a lever revision, which shrinks M12's v3 to the forecast, the time convention and the spin/bounce fixes.

**WallBall** stays frozen as legacy. Its observation entries 21 and 22 are `pending_advance` and `recoverable_bounce_eligible`, and its reward is summed inside the contact loop, so its aids cannot come out cleanly.
- At M6, register TrueBaseline with its kwargs flattened, gated on digests.
- Archive the recipe chain behind a tag.
- Leave the dead kwargs in place.

## 6. Migration plan (interleaved with M0–M12)

| Step | Slot | Change | Gate | Days | Impact |
|---|---|---|---|---|---|
| L0 | with M1 | Extend the digests: the +hold arm, one-point, the camper witness, the NaN-with-pending-escrow witness, all nine `rew_*` keys | two recordings match; under 2 CPU-min | 0.5 | tooling |
| L1 | after M1 | `run_record.json`; write locally, then sync to Drive with a sha manifest; regenerated index; backfill ~48 legacy run dirs and the archaeology verdicts | the 7 stubs are marked aborted; each of the 15 PaddleTennis runs maps to one lever; a forced disconnect leaves a verifiable manifest | 2.5 | tooling |
| L2 | with M3 | Snapshots, audit mode, a pure `select()`, `device="cpu"` in the protocol, a generic probe CLI | offline scores equal the in-loop rows bitwise on a CPU-trained run; the RS1 numbers reproduce | 2 | tooling |
| L3 | after M3 | Declare `paddle-npoint@v3`; re-score the history | the k≥2 headline is identical under @v2 and @v3 | 1 | era-boundary |
| L4 | with M6 | Register v1; add `legacy_kwargs_to_spec()`; route `notebook_utils.py:2589` and the 11 tools through `make_task` | LR1's run dir re-scores identically | 1.5 | bit-identical |
| L5 | with M7 | Lever registry, resolver rules, kinds, caps, `algos.toml`; convert the ~20 Drive TOMLs and the cell-7 constants into archive specs | the LR1, registered-run and LT1 specs compile to their recorded `config.json`; four bad specs fail before any file is written | 2 | tooling |
| — | after L5 | **First spec-driven run:** `pt2.base-r1` baseline, 3 seeds, @v3 | STATUS shows the IQM and CI | — | runs |
| L6 | before M11 | Events hook; `TermStack` in `env_fn`; contact, reach and hold extracted together | digests identical, including the NaN witness; quick_test CSVs byte-identical | 3 | bit-identical |
| L7 | after L6 | Tag, then delete: the in-env escrows, hold, the drill training path, temperature skip, 5 lever tools, the 12 one-off identity tests. Add an env-signature lock and ban `unwrapped._` in `levers/`. | digests unchanged; the harvest reproduces the `k2-demo-library-v0` sha; `debt` is empty | 2.5 | bit-identical |
| L8 | after a Colab measurement | Trim in-loop eval to early stopping and the guards; select offline over every snapshot | the `train()` A/B shows the same training stream; the stop rule is unchanged or declared | 2.5 | declared change |
| L9 | after L5 | Verdicts, STATUS, `debt`; archive never-adopted lever docs | generated STATUS reproduces the archaeology lever table | 2 | tooling |

**Effort.** About 19.5 days, minus about 1.5 days saved in M11 (which now only decomposes the task terms, `InfoSchema` and the `Controller`). That is about 18 days net on top of M0–M12's ~31.
- **L0–L5 (~9.5 days):** stop the experiment sprawl and take shaping out of selection.
- **L6–L7 (~5.5 days):** stop the env bloat.

**Order**
- **Week 1:** M0, M1+L0, L1, M2, M3+L2, L3.
- **Weeks 2–3:** M5, M6+L4, M7+L5, then the first spec-driven baseline at about day 14.
- **Then:** L6, L7, L9, L8 once measured, and M8–M12.

**Runs never pause.** The earlier plan's day-0 PPO reference and its SAC γ arm run from the pinned tag and are re-scored at L3.

## 7. What not to do

- **No new env kwarg for a lever.** A new kwarg means a new task version, and the signature lock enforces it.
- **No "default off" for negative verdicts.** Tag and delete.
- **Don't extract one escrow while others stay in the env.** Move all three together, in the historical order.
- **Don't delete env kwargs before L4.** `notebook_utils.py:2589` rebuilds historical runs from their recorded kwargs, so every re-score would raise `TypeError`.
- **Don't delete the drill before `restore_point()` reproduces the harvest's library sha.**
- **Don't extract WallBall's aids, and don't turn n-point play, volley rules or termination relaxations into levers.** These are task versions.
- **Don't rescale escrows by weight or dt,** on CPU or on mjlab.
- **No generic CPU manager framework, no Hydra or MLflow, no SQLite on Drive.** Borrow only the mechanics: delta over parent, locked schema, unread-key errors.
- **Don't co-append to one Drive index.** Per-run records are the source of truth.
- **No in-loop eval trimming or async early-stop barrier** before the Colab measurement and the `train()` A/B.
- **Don't move DemoSAC.** Its pickle path stays, and it runs under a kill date.
- **Don't let the migration become another runless stretch.** Runs continue throughout, and the first spec-driven baseline lands at about day 14.