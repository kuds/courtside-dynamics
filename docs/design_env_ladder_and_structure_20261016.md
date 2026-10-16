# Env ladder and repository structure toward humanoid doubles (2026-10-16)

Kind: **Adopted design — week 1 in progress (2026-10-16).** A whole-repo design study, run read-only at `main`@`b9585e2`:
- Three grounding passes covered the current architecture, the project's own lessons, and prior art.
- Four independent proposals (composable framework, incremental migration, learnability and scale first, experiment system) were scored by three judges: goal fit, engineering risk, and research-process fit.
- A synthesis built on the incremental proposal, grafting on the others.

The maintainer then made PPO a binding requirement. Section 8 records that. Week 1 implements M0–M3, and the companion campaign-structure design's L0–L3 runs alongside them. Probes cited below ran in session scratch space and are not committed.

**Basis.** Proposal 2 ("extract, don't rewrite") is the base. It is the only proposal that all three judges ranked in their top two. Grafted onto it:
- from Proposal 3: the brain/body ladder and the feasibility spikes;
- from Proposal 4: the experiment system;
- from Proposal 1: the sanity tasks and the termination contract.

Every claim about current code was re-checked read-only at b9585e2 (SB3 2.9.0, MuJoCo 3.16.0).

## 1. Bottom line

- **Restructure by extraction behind a digest gate, never by rewrite.** Today's task becomes `courtside/PaddleTennis-v2` and is frozen. Each refactor step either reproduces golden digests or ships as a new version beside v2. The campaign never pauses.
- **Target shape.** One per-agent `CourtEnv` core, shaped like a PettingZoo Parallel env. It has:
  - named observations for each seat;
  - a referee for N players on two teams;
  - embodiments behind one `RacketCommand` interface;
  - controllers that read the same named observations a policy reads.

  Single-agent SB3 envs become thin views over the core, and doubles is `n_per_side=2`. Each piece is built only when the rung that needs it starts.
- **Ladder.** Learn the tennis "brain" on cheap CPU paddles: rally, duel, humanoid-scale proxy, rules, singles, doubles. Learn the G1 "body" once on GPU: footwork, then a racket-command striker. Join them last. Every humanoid rung is first certified with a scripted brain, so doubles never waits on k=2.
- **Backend.** Paddle rungs stay on CPU SB3. Humanoid rungs move to GPU MuJoCo (mjlab + rsl_rl) only after a Colab L4 spike passes numeric bars.
- **PPO is first-class from this week.**
  - About 2 days of fixes come first.
  - Next come PPO counterparts for the SAC-only machinery, then MAPPO at the doubles rung.
  - The algorithm is a required experiment field and part of the comparability key.
- **Eras.** An era is the pair (task fingerprint, eval-protocol version). A standalone harness re-scores any checkpoint under any protocol, so new instruments mean re-scoring, not retraining.
- **Train now: yes.**
  - Today: tag the repo, pin the notebooks, and launch a 3-seed PPO reference on the unchanged task. It needs no code change.
  - This week: re-score the August checkpoints and run the GPU spike.
  - The context-blind pilot waits about 1.5 weeks.
  - Skip the humanoid Stage 0 probe and any fresh 20-hour SAC reference run.

## 2. Target structure

| Package | Owns |
|---|---|
| `core/` (numpy only) | `PlayerId`, `ObsLayout`/`NamedObs`, `InfoSchema`, digests |
| `court/` (no MuJoCo, no SB3) | referee, `Scorer`, events, `BallSpec`/`BallForecaster`, scenario draw, reward and termination terms with one `Escrow`, scripted brains |
| `sim/` | MjSpec scene builder, embodiments, geometry read from the compiled model |
| `envs/` | `CourtEnv`; `PaddleTennisEnv` as a thin assembly; WallBall, BallBounce and BallBalance frozen |
| `tasks/` | versioned `TaskSpec` registry, `locks/`, per-(family, algorithm) profiles, diagnostics hooks |
| `training/` | `train()`, `AlgoSpec`, views, `AgentSlotsVecEnv`, MAPPO policy, warm start, demos |
| `experiments/` | specs, seed ledger, eval harness, `compare()`, verdicts, generated `STATUS.md` |
| `gpu/` (optional extra) | mjlab tasks for the humanoid rungs; never imported on the CPU path |

**This layout removes:**
- literal observation indices (`_paddle_court.py:447-449`, `paddle_diagnosis.py:215`, `recipes.py:364-369`);
- the side-baked event enum;
- the duplicate serve draws and escrows;
- geometry copied by hand from XML;
- the paddle hook at `train.py:2091`;
- the notebook constants.

**Import and pickle rules:**
- Imports run one way: `core` → `court` → `sim` → `envs` → `tasks`. Import-linter enforces it.
- Pickled classes keep their paths: `training.train.SelectiveVecNormalize` and `training.demo_sac.DemoSAC`.

```python
@dataclass(frozen=True)
class PlayerId: team: Literal["A", "B"]; slot: int        # "A0", "B1"

class ObsLayout:          # per instance and seat; fields carry group, normalize, critic_only
    def view(self, obs) -> NamedObs: ...                  # obs_view["bounce_count"]
    def select(self, profile: str) -> "ObsLayout": ...    # "full" | "physics_only"
    def normalize_mask(self) -> np.ndarray: ...
    def fingerprint(self) -> str: ...

class Embodiment(Protocol):                               # Paddle3, PaddleProxy, G1Tracked
    command_dims: tuple[str, ...]                         # active RacketCommand fields
    def attach(self, spec: mujoco.MjSpec, prefix: str, frame: TeamFrame) -> None: ...
    def apply(self, data, command: np.ndarray) -> None: ...
    def observe(self, data, frame: TeamFrame) -> dict[str, np.ndarray]: ...
    def geoms(self) -> SeatGeoms: ...                     # one event-sampler channel per seat
    def health(self, data) -> Health: ...                 # fall/unsafe -> termination terms

@dataclass(frozen=True)
class RallyEvent: kind: EventKind; player: PlayerId | None; t: float; pos: np.ndarray

class Referee:                                            # MuJoCo-free reference semantics
    def __init__(self, rules: RulesSpec, players: Sequence[PlayerId], regions: CourtRegions): ...
    def reset(self, server: PlayerId, receiver: PlayerId | None) -> None: ...
    def step(self, events: Sequence[RallyEvent]) -> RallyTransition: ...
    # point_over, winner_team, fault_player, reason; team-level legality (a partner's second
    # touch is DOUBLE_HIT; only the designated receiver returns serve). Scorer sits outside.

@dataclass(frozen=True)
class ScenarioSpec:                                       # all reset randomness, one RNG order
    feed: ServeSpec | BallMachineSpec; mid_rally: DrillLibraryRef | None; mid_rally_fraction: float

class TerminationTerm(Protocol):
    kind: Literal["terminal", "truncation", "continue"]   # the GAE contract

class CourtEnv:
    possible_agents: tuple[str, ...]                      # ("A0","B0") | ("A0","A1","B0","B1")
    def layout(self, agent: str) -> ObsLayout: ...
    def reset(self, seed=None, options=None) -> tuple[dict, dict]: ...
    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]: ...
    def state(self) -> np.ndarray: ...                    # team-canonical, for critics

class Controller(Protocol):                               # ScriptedBrain, FrozenPolicy, Pool
    def act(self, obs: NamedObs) -> np.ndarray: ...

def single_agent_view(core, learner: str, controllers: Mapping[str, Controller]) -> gym.Env: ...
class AgentSlotsVecEnv(VecEnv): ...                       # K cores x L learned seats, shared params

@dataclass(frozen=True)
class TaskSpec:                                           # id "courtside/PaddleTennis-v2"
    id: str; arena: ArenaSpec; ball: BallSpec; seats: tuple[SeatSpec, ...]; rules: RulesSpec
    observation: ObsSpec; reward: tuple[RewardTerm, ...]; termination: tuple[TerminationTerm, ...]
    variant_axes: Mapping[str, tuple]; control_hz: int; physics_hz: int
    def fingerprint(self) -> str: ...
```

**Where today's 19 `PaddleTennisEnv` kwargs go.** Each lands in exactly one layer:

| Layer | Kwargs |
|---|---|
| Termination | `episode_len`, `points_per_episode` |
| Rules | `volley_rule` |
| Reward terms | penalties, escrows |
| Scenario | serve, drills |
| Controllers | opponent |
| Metrics | `deep_contact_depth` |
| Render | `court_style` |

`PaddleDoubles-v1` is the same code with four seats and `RulesSpec(format="doubles")`. `G1Doubles-v1` swaps the embodiment, ball and backend.

## 3. Env ladder

| Rung | Adds | Validates | Exit criterion |
|---|---|---|---|
| **R0** Sanity | `ConstantReward-v0`, `TerminalAt64-v0`, `PaddleReach-v1` (one paddle, commanded point, no ball), Pendulum through `train()` | both algorithms learn through the project's wrappers; time-limit bootstrapping | SAC and PPO, 3 seeds, ≤200k steps: Reach <0.05 m, Pendulum >−200, fixtures within 5% of analytic |
| **P1** PaddleTennis-v2, `physics_only` variant, later v3 | locks, named layout, profile variant. v3 adds a forecast group, no truncation-caused reward, one time-limit convention, spin and double-bounce fixes | k=2 interception; task-side vs SAC-side cause | PPO smoke: k=1 ≥50% within 1M. Claim: k=2 ≥10% on 3 held-out seeds by either algorithm, or a lever-budget cap with a recorded falsification. Gates nothing downstream |
| **P2** PaddleDuel-v1 | `CourtEnv`, views, `AgentSlotsVecEnv`, frozen and pool controllers, own-credit option | multi-agent plumbing at CPU cost | the oracle-bound view reproduces v2 locks bit-for-bit; slot-identity test passes; rally bar met against held-out opponents, 3 seeds |
| **P3** PaddleProxy-v1 | humanoid-scale court and ball (from S4); a carriage with G1 speed and acceleration limits, a reach envelope and lag; `RacketCommand`; 100 Hz vs 25 Hz tested as its own lever | positioning under humanoid travel times | scripted brain certifies feasibility and a monotone reward ladder; PPO reaches ≥70% of scripted k=1 and ≥50% of scripted k=2; re-issued after H1 measures the real tracker |
| P3a PaddleAim *(optional)* | racket-face command dims, target-region feeds | placement | ≥60% land in a commanded quarter-court while k=1 holds |
| **P4** PaddleRules-v1 | `PlayerId` referee: struck serve into the box, designated receiver, lets, two serves, attribution, `Scorer` | serve and rule grammar | referee differential tests green; serve-in ≥70%; receiver returns ≥80% of P3's; exploit witnesses score below the oracle |
| **P5** PaddleSingles-v1 | zero-sum reward, snapshot pool plus scripted anchors, opponent-ID critic | self-play without collapse | ≥65% of points against scripted; Elo against fixed anchors rises over ≥5 generations, 3 seeds; rally length above a floor |
| P6a DoublesCoop *(optional)* | four seats, team rally reward | coordination before competition | only used if P6 fails from P5 warm starts |
| **P6** PaddleDoubles-v1 | `n_per_side=2` via MjSpec, alleys and service boxes, MAPPO, partner modes, `PaddleReach2-v1` | who takes the ball; partner credit | scripted team certified; IPPO and MAPPO pass Reach2; ≥60% of points against the scripted team; nearer partner plays ≥80%; interference <5%; cross-play within 20% of self-pairing |
| **H0** G1Footwork (GPU) | G1 with racket welded to the wrist, base commands, falls are terminal | balance while holding a racket | <1 fall per 100 s, 3 seeds; measured speed exported; CPU fall rate within 2× of GPU |
| **H1** G1Striker (GPU) | whole-body tracker for `RacketCommand`; becomes the frozen motor layer | striking while balanced | at strike: ≤10 cm, face ≤15°, ≥70% of commanded speed, falls <1%; otherwise switch to a motion-prior path |
| **H2** G1BallMachine | ball and referee at contact rate; P3/P4 brain transplanted | brain transfer | scripted brain on H1 reaches k=1 ≥50% first; learned brain reaches ≥80% of that within 10⁸ steps; certified on CPU |
| H3 G1Singles *(optional)* | two G1s, P5 brain and pools | competitive transfer | ≥60% of points against a scripted G1; skip if H2 is clean and GPU budget is tight |
| **H4** G1Doubles (goal) | four G1s, P6 brain, MAPPO, pools | the goal | bars pre-registered at H4 entry: rally length, win rate against a scripted team, cross-play |

The humanoid track (H0–H4) starts once spikes S1–S3 pass, and runs in parallel with P2–P6. Every exit criterion includes that rung's PPO smoke test.

## 4. Experiment system

| Layer | Identity | Bumped by |
|---|---|---|
| Task | `courtside/<Family>-vN` + fingerprint + `dyn_fp` lock | any change to observation, action, reward, termination, rules, ball or physics, including bug fixes. Old versions stay registered |
| Variant | a declared axis value (e.g. `obs_profile`) with its own lock | comparing variants needs a declared lever |
| Eval protocol | `paddle-npoint@v2` (the 10-05 instruments) | episodes, seed block, metrics, selection, info schema. A bump means re-score, not retrain |
| Training | the experiment | algorithm and profile, scenario within task bounds, opponents and pools, warm start |

**`dyn_fp`.**
- It hashes obs, reward, terminated/truncated, qpos and qvel over seeded zero, random, oracle and drill streams.
- Info is excluded and gets its own schema digest.
- MuJoCo and numpy versions are pinned. If Colab and local disagree, keep one lock per platform.
- Cost: 2.4 s per 3,000 steps (session scratch probe; not committed).

**Spec.** One TOML per experiment, extending today's run-config format. Old files load as `kind="legacy"`.

```toml
[experiment] id="pt-ppo-ref"; kind="algo_comparison"; parent="pt-sac-adopted"; lever="algo: SAC -> PPO"
[task]   id="courtside/PaddleTennis-v2"; fingerprint="sha256:..."
[eval]   protocol="paddle-npoint@v2"
[algo]   name="PPO"; impl="sb3==2.9.0"; profile="paddle_tennis/ppo_v1"
[seeds]  block="train:paddle-k2/ppo"; n=3
[budget] env_steps=10_000_000; report_at=[3_000_000, 10_000_000]
```

- The launcher refuses any spec that differs from its parent in more than the declared lever.
- Notebooks take a spec path plus a tag.
- `validate_run_config_against_plan` becomes a spec-digest check.

**Harness.**
- `evaluate(ckpt, protocol)`, `rescore(runs, protocol)` and `select(scores, rule)` load checkpoints through the recorded algorithm.
- In-loop eval stays, because early stopping, the degenerate guards and `PerformanceGate` depend on it.

**Comparability.**
- The key is (task fingerprint, protocol version, algorithm, implementation).
- `compare()` and verdicts raise on mixed keys unless the spec is `algo_comparison` or `cross_era`.
- Those two kinds report paired seeds on both env-step and wall-clock axes.

**Seeds.** The ledger moves into `experiments/` with four block kinds:
- `train:` blocks, one per campaign and algorithm family;
- `eval:<protocol>` blocks;
- `heldout:` blocks, which only a claim verdict can open, with every opening logged;
- `probe:` blocks.

The launcher refuses overlapping blocks.

**Artifacts.**
- `run.json` records the resolved spec, every fingerprint, the tag and SHA, library versions, env steps, updates, and wall clock split into collect, update and eval.
- `PolicyArtifact` records weights, normalizer, layout fingerprint, command version and algorithm. It refuses to load on a mismatch.
- Pools are manifests of artifact digests.

**Process.** Only `kind="claim"` needs pre-registration. Pilots get computed verdicts, and `STATUS.md` is generated.

**Archival.**
- The humanoid code goes behind `humanoid-tennis-archive-v0.25.0` after about 29 shared tests are ported.
- Keep the `HUMANOID_*` enum members. `to_info` iterates every member (`tennis_rules.py:296,342`), so removing them changes v2 info keys.
- WallBall stays frozen in place.
- The August runs are back-filled as `version_inferred`.

## 5. Backend and compute

**Paddle rungs: CPU MuJoCo + SB3.**
- I measured SB3 PPO on the recipe env at 632–832 env steps/s on a shared 4-CPU box. Updates took at most 4% of wall time, so throughput scales with cores.
- Probes ran in session scratch space and are not committed.
- Don't build a GPU paddle court unless P5/P6 would need more than about 24 h for 10⁸ agent steps.

**Humanoid rungs: GPU only.**
- 2×G1 runs at 147 raw steps/s per core, while published single-humanoid racket skills used about 10⁹ samples.
- First choice is mjlab (MJWarp + PyTorch + rsl_rl). It reuses the MJCF/MjSpec assets, and its torch weights load into the CPU certification path.
- Fallbacks, in order: MuJoCo Playground (MJX + Brax), then a rented 4090/A100-class GPU.
- Train on GPU, certify on CPU. Every artifact is re-scored in the float64 CPU court under the Python referee.

**Spikes** (off-repo, in parallel with paddle work):

| Spike | When | Pass | On fail |
|---|---|---|---|
| S1: L4 throughput (Playground/mjlab G1 tasks; 4×G1 + ball at 1–2 kHz) | weeks 1–2, ~10 L4-hours | single G1 ≥2×10⁴ steps/s; 4×G1 ≥5×10³ steps/s | re-run on a rented GPU; if that also fails, pause the humanoid track |
| S2: contact fidelity, float32 GPU vs float64 CPU, 10k impacts | weeks 2–3 | exit speed within 2%, direction within 5° | raise the physics rate or retune contact |
| S3: striker learnability; PPO G1 places a welded racket at a commanded pose and time | weeks 2–4 | ≤10 cm at strike, falls <5% | plan a motion-prior path before H1 |
| S4: court-size memo from S1/S3 measurements | before P3 | DECISIONS entry with numbers | use published G1 limits; recalibrate after H1 |

**Control hierarchy:**

```
Brain (per seat; scripted or learned; 100 Hz on Paddle3, 25 Hz variant from P3)
  reads by name: ball, forecast, self_racket, self_base, partner, opponents, rally
  | RacketCommand: racket_pos[3], face_normal[2], swing_speed, base_xy[2]
  v
Paddle3 (today's slide targets) | PaddleProxy (carriage + reach envelope + lag)
| G1Tracked (frozen H1 tracker, 50 Hz -> joint PD, physics 0.5–2 kHz)
```

- The brain never sees joint state, so a paddle brain loads onto the G1 with the same layout fingerprint.
- `BallForecaster` is extracted from the oracle's landing math (`_paddle_court.py:533-537`). It becomes the single source for the oracles, the `forecast` group and diagnostics.

## 6. Migration plan

| # | Step | Gate | Days | Impact |
|---|---|---|---|---|
| M0 | Tag b9585e2; point both notebooks' `REPO_REF` (today `"main"`) at the tag | test that `REPO_REF` is a tag or full SHA | 0.25 | tooling |
| M1 | Golden digest harness (recipe plus 3 variants, BallBalance, BallBounce × {zero, random, oracle} × 5 seeds × 3k steps), plus 2-update SAC and PPO parameter digests | identical across processes; Colab vs local recorded | 1 | tooling |
| M2 | PPO hygiene (§8) | env and SAC trainer digests unchanged, gated on streams rather than `vecnormalize.pkl` bytes (note 1) | 2 | tooling |
| M3 | Harness v0, protocol `paddle-npoint@v2`, rescore script | harness equals a fresh run's in-loop eval row bit-for-bit | 2 | tooling |
| M4 | R0 tasks, GAE test, algorithm-matrix CI | both algorithms pass R0 | 2 | new env |
| M5 | `ObsLayout` with by-name reads (oracle `obs[30]`/`obs[33]`, diagnosis, scripted policies, exclusions, channel permutation); AST lint against integer obs subscripts | v2 digests identical; oracle conversion unchanged | 3 | bit-identical |
| M6 | Registry, versioned IDs (old ID kept as an alias), lock CI, `obs_profile` as a pure projection | v2 locks identical; the variant equals the projection; oracle stats identical | 2.5 | bit-identical |
| M7 | ExperimentSpec, in-package ledger, comparability guard, `STATUS.md` | the August plan re-renders to the same fingerprint; a mixed compare raises | 3 | tooling |
| M8 | Archive the humanoid code (port tests first) | paddle tests green; import no longer touches G1 assets | 1.5 | tooling |
| M9 | PPO second wave (§8) | SAC digest unchanged; PPO digest unchanged with options off | 2.5 | tooling |
| M10 | Geometry from the compiled model; one serve draw | digests identical; serve RNG golden over 10k draws | 1.5 | bit-identical |
| M11 | Decompose `step()` into terms, one `Escrow`, `InfoSchema`, opponent as a `Controller` | `dyn_fp` and info digest identical | 4 | bit-identical |
| M12 | `PaddleTennis-v3` (forecast, no truncation reward, time convention, spin and bounce fixes) | new lock; v2 untouched; oracle ladder monotone | 4 | new version |
| M13 *(conditional)* | PPO demo use (BC pretraining, kickstarting), only if §7's table or the pilot calls for it | default-off equals the stock PPO digest | 3 | tooling |

Note 1: `ret_rms` updates even with `norm_reward=False`. SAC recipes at gamma 0.995 (WallBallBootstrap, `recipes.py:826`) will therefore pickle different stats.

**This week:**
- Day 0: M0, then the zero-code PPO launch.
- Days 1–4: M1, M2, then M3.
- In parallel: spike S1.

The pilot unlocks after M6, about 1.5 weeks in.

**Defer to rung entry:**

| Rung | Deferred work | Days |
|---|---|---|
| P2 | `CourtEnv` and views | 5 |
| P3 | MjSpec composition (a probe already gives a bit-identical 5,000-step rollout), `BallSpec`, PaddleProxy | ~8 |
| P4 | `PlayerId` referee and `Scorer`, differential-tested on recorded event streams, with a v2 info adapter | ~8 |
| P5 | ratings | ~3 |
| P6 | four seats and MAPPO | ~10 |
| Humanoid track | `gpu/` | after S1–S3 |

**Total.** About 70 working days to certified paddle doubles:
- about 31 days for M0–M12;
- about 34 days at rung entries;
- about 5 days for the spikes.

Humanoid rungs take weeks each and are bounded by GPU hours (estimated 100–300 L4-hours, uncertain until S1).

## 7. Training runs before changes

**Yes, start now.** Run only what meets all three conditions: it needs no env change, it launches from the pinned tag, and its result can change a decision.

**1. Today: PPO reference on the unchanged task, with zero code changes.** This config builds and validates at b9585e2:

```python
build_train_config('PaddleTennis', algo='PPO', seed=s, n_envs=8, normalize_reward=False,
    eval_freq=245_760, checkpoint_freq=491_520, early_stop_patience=None, total_timesteps=10_000_000,
    model_kwargs=dict(n_steps=1024, batch_size=2048, n_epochs=5, learning_rate=3e-4, clip_range=0.2,
        target_kl=0.02, gamma=0.995, gae_lambda=0.95, use_sde=True, sde_sample_freq=32, ent_coef=0.0,
        device='cpu', policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0)))
```

- **Why these overrides.** Without them, PPO inherits the SAC recipe's `eval_freq=25_000` and `early_stop_patience=20`. That puts evals off rollout boundaries, makes eval cost about 1.8× the training steps, and stops the run after 500k flat steps.
- **Why `normalize_reward=False`.** It sidesteps the gamma-0.99 normalizer bug. It is also the `ppo_v1` profile default, so the lineage survives M2.
- **Budget.**
  - 3 seeds × 10M steps, read out at 3M (sample parity) and 10M (about wall-clock parity).
  - 3M steps of collection take about 1–1.3 h on 4 cores, against about 20 h for SAC.
  - Record the Colab vCPU count.

**Pre-registered reading at 3M:**

| Result | Reading | Next |
|---|---|---|
| k=1 ≥80%, k=2 ≤1% | the stall is task-side | context-blind pilot in both algorithms; v3 forecast next |
| k=1 ≥80%, k=2 ≥3% on ≥2 seeds | SAC critic or entropy pathology | move P1 to PPO; LD1′ becomes PPO with BC or kickstarting |
| k=1 rising but <80% | slow learner | continue to 10M; read on the wall-clock axis |
| k=1 <50% and flat on all seeds | a problem with the PPO profile, not a verdict on the task | retune `ppo_v1` or kickstart from the oracle; SAC stays the P1 default |

**2. This week: re-score the 2026-08 checkpoints under `paddle-npoint@v2`, instead of a fresh 20-hour SAC reference.**
- This is valid because the instrument batch left env streams bit-identical and the checkpoints carry their VecNormalize stats.
- No PaddleTennis re-score tool exists, so M3 comes first.
- Score `best_model` and every 500k checkpoint first: about 100 checkpoints, a few CPU-hours.

**3. This week: spike S1, off-repo.** It gates the whole humanoid half.

**4. SAC γ=0.995 full-profile arm, 2 seeds × 3M (about 40 L4-hours).** This is the context-blind pilot's control arm, so the profile stays the only lever. Start it now if the L4 is idle; the digest gates keep it valid. Otherwise launch it with the pilot.

**5. After M6: the context-blind pilot.** Two SAC treatment seeds against run 4, plus three PPO seeds against run 1.

**Not worth running:**
- the humanoid Stage 0 gSDE probe: its env is being archived, so the answer transfers nowhere;
- DemoSAC LD1′: it is not frozen, and run 1 may redirect it;
- any WallBall run, or any humanoid run on CPU;
- long (30M-step) PPO runs before the 10M readout.

## 8. PPO as a first-class algorithm

**Already working:**
- PPO rejects SAC kwargs.
- PPO-to-PPO warm start exists.
- `PerformanceGate` already fails loudly if PPO is given SAC-only advance options (`performance_gate.py:467-500`).
- PaddleTennis truncates only at `episode_len` (`paddle_tennis.py:883-893`).
- SB3 2.9 PPO bootstraps γ·V(terminal_obs) on truncation.

**Code changes in M2 (about 2 days):**
- **Gamma.** Pass gamma into all four `SelectiveVecNormalize` constructions (`train.py:1944,1955,2131,2309`). "gamma" appears nowhere in `train.py` today. Also check it on warm-start load.
- **Profiles.** Add per-algorithm profiles.
  - Today `algo='PPO'` on PaddleTennis raises "'target_entropy' is not accepted by PPO".
  - TOML merges only into the resolved algorithm's block.
  - The spec's `[algo]` replaces the TOML ban on `algo` (`run_config.py:74`) and the silent `default_algo="SAC"` (`recipes.py:46`).
- **Rollout geometry.** `AlgoSpec.validate_geometry` checks that:
  - cadences are whole rollouts;
  - the batch divides the rollout;
  - patience is counted in env steps;
  - `n_steps` covers at least 2 exchange cadences.
- **Vec env.** Add a SubprocVecEnv option; none exists in `src/`.
- **Loading.** `AlgoSpec.load` replaces `SAC.load` at `paddle_diagnosis.py:70` and `tools/paddle_tennis_p5_transfer.py:323`.
- **Notebook and records.** The cell-7 guard admits PPO. `config.json` records the algorithm, implementation and comparability key.

**Code changes in M9 (about 2.5 days):**
- Schedules (`env_attr_schedule.py:142`, applied per step today) and gate advances apply at `_on_rollout_start` under PPO.
- PPO warm start carries the actor, value, log_std, gSDE state and normalizer, plus `critic_warmup_rollouts`.
- Add a std-collapse alarm per active action dimension.
- Add mid-leg resume.

**(1) PPO at each rung**

| Rung | Backend | Rollout | γ / λ | Normalization | Exploration |
|---|---|---|---|---|---|
| R0 | SB3 CPU | 8×256 | 0.99 / 0.95 | obs on, reward off | Gaussian, ent 0 |
| P1–P2 | SB3 subproc | 8–16×1024 (5–11 episodes per update); batch 2048–4096; 5 epochs; target_kl 0.02 | 0.995 (horizon ≈ the 126–210-step cadence) / 0.95 | obs by layout mask; reward normalization off in `ppo_v1` (gamma-matched if turned on) | gSDE resampled every 32 steps, log_std −1, std floor. Contact comes from scenarios (mid-rally starts, ball machine) and BC or kickstarting, not from noise |
| P3–P6 (25 Hz) | SB3 `AgentSlotsVecEnv` | cores × seats × 256 | 0.99 / 0.95 | shared across mirrored seats; critic-only fields kept separate | gSDE every 4–8 steps; pools |
| H0–H4 | GPU rsl_rl | 4096×24; adaptive LR, KL 0.01 | 0.99 / 0.95 | empirical | init std 1.0, ent 0.005–0.01, asymmetric critic |

**Throughput.**
- R0–P2: CPU SB3 with SubprocVecEnv. Collection is the bottleneck (632–832 steps/s measured on 4 shared cores, updates ≤4%), so throughput scales with cores.
- P3–P6: four bodies, not measured yet; expect about half the per-core rate.
- H0–H4: GPU only, behind the S1 gate (≥2×10⁴ single-G1 steps/s; Playground reports 1.06×10⁵ on an A100).

**Termination contract.**
- `terminated` is set only for true terminals: a single-point end, an unsafe state, or a fall.
- `truncated` is set only for time limits.
- Points inside an episode are `continue`.

**v2 mixes these conventions, and v3 fixes it.** v2 puts time-to-go in the state (`obs[35]`, `episode_remaining_fraction`) while also bootstrapping timeouts. It also pays escrow clawbacks on the truncation step (`paddle_tennis.py:897-905`). Both bias value targets, for SAC and PPO alike.

**Other notes.**
- rsl_rl bootstraps timeouts with V(s_t), not V(s_T); `algo.impl` records this.
- Eval reports both deterministic and stochastic columns.

**(2) SAC-only pieces**

| Today | Interface | PPO counterpart |
|---|---|---|
| DemoSAC replay mix | `TeacherSource` (oracle, fingerprinted library, artifact) | BC pretraining; a kickstarting loss λ_t·‖μ(s)−a_teacher(s)‖² on the policy's own states, decaying to 0; mid-rally starts. Never off-policy transitions in the surrogate loss |
| `learning_starts` refill | `WarmStart.warmup` | `critic_warmup_rollouts` with the actor frozen (guards against the L2W collapse from 100% to 30%) |
| `log_ent_coef` transfer | `WarmStart.components` | actor, value, log_std, normalizer; across algorithm families, distillation only |
| `ent_coef='auto'` | `ExplorationSpec(active_dims)` | log_std_init, ent_coef, sde_sample_freq, std alarm |
| Gate pause/clear | `Gate.on_advance(family)` | value refit; nothing to clear |
| Per-step `EnvAttrSchedule` | validated setter with read-back | applied at rollout boundaries, so GAE never straddles a change |

**(3) Multi-agent PPO**
- **P2/P5: IPPO** with full parameter sharing through the team-canonical mirror. P4's mirror is bit-exact today.
- **P6/H4: MAPPO.**
  - Shared actor on egocentric observations plus role features.
  - Critic on `state()` plus the opponent's pool ID.
  - In SB3, this is a Dict-observation policy with split extractors (about 150 lines). A test checks that `critic_only` fields never reach the actor.
  - On GPU, seats are the batch dimension.
- **Opponents.**
  - Frozen snapshots every K updates.
  - Each episode draws 50% the latest snapshot and 50% the pool. The pool draw is weighted toward early snapshots and toward opponents the agent loses to.
  - Scripted anchors are always present; the pool holds at most 16.
  - Elo is measured only against fixed anchors.
- **Partners.** Self-copy, frozen or scripted (`PartnerMode`), with cross-play evaluation.
- **Credit and budgets.** Team outcome plus an optional own-credit term from `fault_player`. Env steps and agent steps are counted separately.

**(4) Recording**
- `[algo]` (name, implementation, profile) is required, with no default.
- Profiles are per (family, algorithm). A key from the other family is a parse error.
- `run.json` stores the kwargs read back from the model.
- The algorithm is part of the comparability key.
- Seed blocks, run directories, artifacts, pools and selection bars are all per algorithm.

**(5) Verification**

CI checks, each taking seconds:

| Check | Passes when |
|---|---|
| Truncation fixtures `ConstantReward-v0` and `TerminalAt64-v0`, run through every wrapper and view | V ≈ 1/(1−γ) and V ≈ 1 respectively |
| GAE at truncation | the return equals r + γ·V(s_T) |
| Normalizer gamma | equals the model's gamma |
| Geometry validator | rejects bad rollout geometry |
| Schedules | change only at rollout boundaries |
| Warm-start and load | round-trip correctly |
| BC turned off | equals stock PPO |
| Vector env choice | dummy and subproc digests are equal |
| Algorithm matrix: every task × {SAC, PPO, DEMOSAC-off, PPO_BC-off}, built, 2 updates, saved, loaded | actions are identical |

Learning smoke tests, on Colab with 3 seeds per merge:

| Rung | Smoke test |
|---|---|
| R0 | Reach and Pendulum (oracle-free) |
| P1 | k=1 ≥50% within 1M steps |
| P2 | slot-identity through `AgentSlotsVecEnv` |
| P6 | Reach2 with IPPO and MAPPO |
| H0 | G1 stands within 2×10⁷ steps |

## 9. What not to do

- **Don't change PaddleTennis-v2 in place.** Semantic fixes ship as a new version.
- **Don't launch runs from `main`** while the refactor is merging.
- **Don't build ahead of need.** No `CourtEnv`, `PlayerId`, N-seat scenes or MAPPO before their rung. No GPU paddle court unless CPU throughput fails.
- **Don't churn the layout.** No package re-layer, no move of `docs/`, `tools/` or `notebooks/`, and no offline-only selection.
- **Don't break old artifacts.** Keep WallBall and the `HUMANOID_*` enum members, and add shims for any pickled class that moves.
- **Don't resume the centralized 58-dim humanoid env, and don't train any humanoid on CPU.** Don't attempt humanoid doubles end to end, or before paddle doubles and single-humanoid hitting both work.
- **Don't let PPO inherit SAC kwargs, cadence or patience,** and don't apply schedule changes mid-rollout.
- **Don't copy weights across algorithm families,** and don't put off-policy demos into PPO's surrogate loss.
- **Don't compare SAC and PPO without `algo_comparison`,** and never on env steps alone.
- **Don't pick the humanoid court size before S4.**
- **Don't adopt RLlib, Ray, Isaac Lab or a tracking server,** and don't pre-register pilots.

## 10. Open decisions for the maintainer

| Decision | Recommended default |
|---|---|
| Humanoid court size (13 m vs 23.77 m) | Regulation, unless S4 shows the G1 cannot cover it; then a proportionally scaled court. P1–P2 stay at 13 m |
| GPU platform | Colab L4 if S1 passes; otherwise rent one 4090/A100-class GPU for the humanoid track only |
| GPU stack | mjlab; switch to Playground/Brax only if mjlab fails S1 or S2 |
| Default algorithm for new paddle versions | PPO if the reference reaches k=1 ≥80% at 3M. SAC stays a declared-lever lineage either way |
| Context-blind pilot design | Pure projection (one lever) as a v2 variant. The review's spin and counter-scaling bundle ships as v3 |
| Motion prior / mocap | Not until S3 or H1 fails |
| Doubles reward | Team point outcome plus an own-fault penalty at weight 0.5, declared in the TaskSpec and tested as a lever |
| WallBall | Frozen in-tree; archive at the first refactor that would have to touch it |
| What counts as "trained humanoid doubles" | Pre-register at H4 entry: median rally length, win rate against a scripted G1 team, and a cross-play floor |