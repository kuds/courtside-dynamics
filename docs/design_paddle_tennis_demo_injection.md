# Design: PaddleTennis demonstration injection — the LD1′ mechanism

Status: **Proposed — the mechanism is IMPLEMENTED default-off
(2026-09-02: `DemoSAC`, the demo harvest tool, provenance and plan
pins, SD0 certified bit-identical against stock SAC on a minimal and
a recipe-shaped lockstep; warm-start gates widened to the SAC family
and tested); the LD1′ pilot
(§5) is NOT frozen — its numbers are the maintainer's at freeze, on
the §3a arithmetic.** Routed injection-first by the maintainer's
2026-09-02 bookings
([`paddle_tennis_ld1prime_freeze_brief_20260830.md`](paddle_tennis_ld1prime_freeze_brief_20260830.md),
D-A through D-G), retiring the pure-drill pilot of
[`design_paddle_tennis_k2_drill.md`](design_paddle_tennis_k2_drill.md)
§5 before freeze; the shipped drill stays this design's RE-AIM
escalation. A **training-data change with the eval task bit-frozen**
and the env untouched. Standing doctrine: changes after a
pre-registered run starts void it as campaign evidence.

## 1. Why injection, and why from the policy's own failures

The k=2 deficit is a **joint observation-side context gate** in the
learned policy: the same physics converts 6.9% presented as a fresh
feed and 1.3–2.0% in the real mid-rally context, under both the
deterministic and the behavior policy (freeze brief G3); the oracle
is context-blind (identical 77.5% under both presentations); the
gate is not positional (arm (c): 1/20 from the oracle's own
position), not the clock, not exploration-noise scale (marginal gSDE
std ~0.60, never collapsed). What the policy lacks is **successful
transitions in the failing context** — the one thing neither drill
arm supplies at step 0 (below the literature's 10–20% engagement
floors in both modes) and the thing demonstrations supply by
construction. Demonstrations come from the **policy's own harvested
failure states** completed by the scripted oracle, not from
oracle-native play: the red-team caught that oracle-native demos
would carry the exact distribution mismatch (oracle-conditioned
joint states, 0.89 m vs 3.65 m at the k=2 instant) this campaign
used to reject arm (c).

## 2. Mechanism (shipped default-off)

`DemoSAC(SAC)` (`src/courtside_dynamics/training/demo_sac.py`),
registered as algo `DemoSAC` in `ALGOS` **and** `OFF_POLICY_ALGOS`
(a regression test pins the resolved `gradient_steps = -1`; missing
the off-policy registration would silently train 1 update per 256
transitions). Constructor kwargs, all default-off:

- `demo_library: str | None = None` — a `k2-demo-library-v0`
  artifact from `tools/paddle_tennis_k2_demo_harvest.py`. Its sha256
  is banked **at construction** (the trainer writes `config.json`
  before `learn()` starts) as `demo_library_sha256` into
  `config.json`'s `resolved_model` block by the model probe, pinnable
  via `validate_run_config_against_plan`'s `demo_library_sha256` key;
  the buffer itself builds at the **first `learn()`** (schema,
  per-trajectory array shapes against the env's spaces on the raw
  arrays, non-empty train split — fail-loud; a library whose bytes
  changed under the banked digest is refused), so inference loaders
  never need the file. Half-configured pairs are rejected at
  construction and again after `load()` applies override kwargs.
- `demo_fraction: float = 0.0` in [0, 1): the **exact** share of
  every minibatch drawn from the demo buffer —
  `n_demo = round(fraction × batch_size)`, arithmetic, never a
  random draw, so the share is exact and auditable (a test spies
  both buffers' sample sizes); a fraction that rounds to 0 or to the
  whole batch at the configured batch size is refused. **OFF** makes
  exactly stock SAC's RNG calls (the SD0 property); **ON** interleaves
  the demo buffer's own draws with the live buffer's on the global
  RNG, so an ON run's live sample stream is not a SAC run's
  (expected; recorded here, not claimed away).
- `demo_bc_coef: float = 0.0` — a behavior-cloning term on the demo
  rows of the actor loss: MSE between the actor's deterministic
  action and the demonstrated action (needs `demo_fraction > 0`).
- `demo_bc_filter: "none" | "q"` — `"q"` clones only rows where the
  critics rank the demo action above the policy's (Nair 2018's
  Q-filter). **Default "none" per D-C**: the freeze brief's G1
  measured the kept critics near action-blind at these states, so
  the filter would pass ~half the demos arbitrarily at launch.
- `demo_window: "point" | "to_confirm"` — whole recorded trajectory
  (through the point's end) or through the conversion's confirmation
  step.
- The **D-C arming measurement** ships in two populations:
  `demo_q_ordering()` — the fraction of held-out demo states (every
  transition of every held-out trajectory under the configured
  window) where min-Q ranks the demo action above the policy's
  deterministic action — logged as `train/demo_q_ordering`; and
  `demo_q_ordering_launch()` — the same ordering on the held-out
  trajectories' **launch states only** (one row per held-out failure
  state, the population the freeze brief's G1 measured at 42%) —
  logged as `train/demo_q_ordering_launch`; both every 50th `train()`
  call. The pilot pre-registers its threshold on one named series;
  flipping `demo_bc_filter` to `"q"` when it clears is a
  checkpoint-resume config change, recorded.

**What the buffer holds.** Raw observations (SB3 normalizes at
sample time with the live `VecNormalize` stats, exactly as it does
for live transitions), the oracle's applied env actions mapped
through SB3's `scale_action` exactly as live actions are (identity
on this [−1, 1]³ space), the env's actual reward stream (escrows
included — the pilot's own reward definition), and the env's own
terminal/timeout flags; a cap-ended trajectory's last row is a
non-terminal bootstrap row, like any mid-episode live transition.
Train-split trajectories only; held-out trajectories are kept aside
as ordering-metric material and never sampled.

**Not supported:** `n_steps > 1` (refused at construction — 1-step
demo targets would mix into n-step live targets); Dict observation
spaces.

### 2a. Implementation pins

- **SD0**: `tests/test_demo_sac.py` locksteps `DemoSAC` (off) against
  stock `SAC` on two seeded runs — a minimal one (BallBalance,
  n_envs 1, train_freq 1, 192 updates) and a **recipe-shaped** one
  (gSDE with the 64-step noise hold, `SelectiveVecNormalize` with an
  excluded index, n_envs 4 × train_freq (64, step) with
  gradient_steps −1, batch 256, auto temperature with a target; 768
  updates) — asserting every policy parameter, `log_ent_coef`, the
  normalizer's running stats and the global numpy/torch RNG states
  equal (the explicit global-RNG accounting D-G names). The training
  step makes SAC's calls in SAC's order when the surface is off.
  `train()` is SB3 2.9.0's `SAC.train` verbatim except the minibatch
  source and the BC term; it must be re-synced on any SB3 bump (a
  pinned copy is the price of a subclass fork).
- **Checkpoint compatibility**: the demo buffers are excluded from
  pickling; a `DemoSAC` checkpoint loads as plain `SAC` (the path the
  diagnosis/harvest/step-0 tools use — tested) and as `DemoSAC`
  without the library file present (the buffers rebuild lazily from
  the recorded path at the next `learn()`; load-time overrides are
  re-validated; a load that switches the surface off carries no
  digest — tested).
- **Warm start**: the trainer's warm-start gates were widened from
  the literal `SAC` string to the SAC family (`OFF_POLICY_ALGOS`): a
  `DemoSAC` target warm-starts from a plain-SAC source run (the
  source loads with its own class; the initialization provenance
  records the source's own algo), a PPO source is still refused —
  tested end to end. The transfer is the standing full-policy
  `state_dict` transfer (actor, critics, targets, `log_ent_coef`).
  Critic head-recycling (D-B) is a **separate, not-yet-shipped**
  warm-start option — the next instrument after this one (§5 says
  what happens if the pilot freezes first).
- **Strict kwargs validation survives the subclass**:
  `validate_model_kwargs` walks the MRO, so a typo in a demo kwarg
  fails at config validation, not deep in SB3.

## 3. The demo library (measured, 2026-09-02)

`tools/paddle_tennis_k2_demo_harvest.py` launches each failure-state
entry through the shipped **full-context** drill arm (real rally
flags, real physics), lets `scripted_ground_opponent` play side A,
records the replay tuple every step until the point ends (cap 300),
and **keeps a trajectory only if the oracle's k=2 hit was legal AND
its return confirmed inside the recording**, asserting the
confirmation step paid ≥ `return_reward` (the success signal is in
the buffer by construction, fail-loud). Every 5th source entry is
held out.

| source library | entries | kept | oracle miss | hit, unconfirmed | censored |
|---|---|---|---|---|---|
| registered (seeds 9030–9099, sha `43975265…`) | 102 | 70 | 23 | 9 | 0 |
| extension (seeds **9200–9269**, harvested at `d226a99` with untracked new files present — the harvest tool itself unchanged since `c8bc07d`, so the library reproduces from that tool + seeds) | 100 | 72 | 17 | 3 | 0 |
| **combined demo library** (sha `21fda9dd…`) | 202 | **142** (110 train / 32 held-out) | 40 | 20 | 0 |

**39,591 transitions** (≈30.7k in the train split); keep rate
**70.3%** — below the 77.5% oracle *touch* rate because ~10% of
oracle hits go unconfirmed (returns out/net), which the harvest
correctly excludes. Both source libraries are policy-harvested
failure states (the D-E distribution), one per point, clearance-
filtered at harvest.

### 3a. Replay multiplicity — the arithmetic the fraction is set from

The red-team's warning, now quantified. A 1M-step pilot at the
recipe's UTD 1 (256 updates per 256 collected transitions, batch
256) draws 256M samples in total. With ~30.7k train demo transitions
and a live buffer of 1M:

| `demo_fraction` | demo draws | replays per demo transition | replays per live transition | ratio |
|---|---|---|---|---|
| 0.50 | 128M | ~4,170 | ~128 | ~33× |
| 0.25 | 64M | ~2,080 | ~192 | ~11× |
| 0.10 | 25.6M | ~830 | ~230 | ~3.6× |
| 0.05 | 12.8M | ~420 | ~243 | ~1.7× |
| 0.03 | 7.7M | ~250 | ~248 | ~1.0× |

RLPD's 50/50 is licensed by stabilizers this pilot does not run
(LayerNorm critics, larger ensembles, UTD ~20 — the freeze brief's
D-B dropped LayerNorm as unimplementable in stock SB3); at UTD 1 with
two critics the honest description is Nair/DDPGfD-shaped injection.
**Proposal for the freeze:** `demo_fraction` in the **0.05–0.10**
band (demo transitions replayed ~2–4× as often as live ones — a
DQfD-style priority bonus expressed as a fixed share), with the BC
coefficient set by the SD3 step-0 rows; 0.25+ only as a RE-AIM
escalation with the multiplicity recorded.

## 4. SD battery (proposed; the maintainer freezes it before the pilot)

- **SD0 — bit-identity when off**: shipped (§2a).
- **SD1 — demo fidelity**: by construction at harvest (conversion
  payment asserted per kept trajectory; counts recorded); the
  battery re-runs the harvest at the pinned commit and matches every
  kept trajectory field-for-field (arrays, enders, splits, counts —
  a content match; the pickle header carries the harvest's own git
  sha, so a byte-level file digest is provenance, not the
  certificate).
- **SD2 — composition and provenance**: shipped as tests (exact
  per-minibatch split; `demo_library_sha256` reaches `config.json`
  and the plan validator).
- **SD3 — the ordering baseline and the arming threshold**:
  `demo_q_ordering` on the held-out 32 at step 0 (the freeze brief's
  G1 predicts ~coin-flip: 42–45%); the pilot pre-registers the
  threshold that arms `demo_bc_filter="q"`.
- **SD4 — realized economics at pilot settings**: a short run
  (≥ 20k steps) recording `train/demo_fraction`, `train/demo_bc_loss`,
  the realized replay multiplicity, and — the retention canary —
  `legal_hit_count_a` on the eval stream, before the 1M pilot.
- **Seed block: 6400–6499** (booked). Harvest seeds: the scratch
  ledger (§7).

## 5. LD1′ — pilot shape (proposed, UNFROZEN; inherits LD1's committed rule shape)

Warm start from the **registered 2.4M protected best** (shas
`838997fb…`/`d0502c14…` pinned via `expected_artifact_sha256`),
`algo = "DemoSAC"`, the pinned combined demo library
(`demo_library_sha256` pinned), `demo_fraction` and `demo_bc_coef`
from §3a/SD3, `demo_bc_filter = "none"` at launch with the SD3
arming rule, **drill OFF** (`drill_fraction = 0.0`; the [env] table
carries no drill keys and the plan pins `eval_env_kwargs` drill-free),
**no temperature re-heat** (D-D; `transfer_log_ent_coef = True`,
`ent_coef` and post-swing saturation pre-registered as observables),
critic head-recycling per D-B **once its option ships** (the next
instrument; if the pilot freezes first, it launches on the
transferred critics and the recycling becomes the first RE-AIM
escalation). 1M steps, seed 0,
n_envs 4, eval 25k, checkpoint/diagnosis 100k, recipe defaults
otherwise.

**Decision rule, inherited from LD1 §5 and re-aimed:**

- **KD-primary**: k=2 exchange survival on the STANDARD task at the
  standing bar shape, with the fresh-seed confirm arm (the source's
  1.6% [0.3, 4.5] noise floor makes a lone diagnosis-stream PASS a
  hypothesis).
- **KD-mechanism**: paddle → return-bounce on standard-task k=2
  opportunities (3.65 m at launch / 3.0 m at bounce → toward the
  oracle's 0.9 m); `train/demo_q_ordering` moving off coin-flip;
  `train/demo_bc_loss` falling.
- **KD-retention**: k=1 receiving holds its band; the standing
  degenerate guard (`legal_hit_count_a`) is the mid-run abort, and
  an ABORT before the k=2 observables carry signal is **not a STOP
  verdict** — it is uninformative about injection and routes to the
  fraction/BC re-aim.
- **Branches**: **ADOPT** → pre-register the registered-scale retry
  with injection as recipe convention. **RE-AIM** (mechanism moves,
  KD-primary FAIL) → pre-named escalations, one per run: (1) critic
  head-recycling if not yet in; (2) full critic re-initialization
  (D-B's booked escalation beyond recycling; needs its own transfer
  path — the loader is all-or-nothing); (3) `demo_bc_filter = "q"`
  per the SD3 threshold; (4) `demo_fraction` up one §3a row; (5)
  the drill ON (arm (b), `drill_fraction = 0.5`) as exposure
  vehicle; (6) whole-block context masking — its own design. **STOP** (no
  mechanism movement beyond the SD3/step-0 baselines) → the
  demonstration class is spent at this recipe; SAC-X-style factored
  heads inherit.

## 6. What this does not do, and what it costs

No reward, opponent, dynamics, observation, or action change; the
env is untouched (the drill mechanism stays off). The eval task is
bit-frozen; selection and diagnosis series keep their meaning. What
breaks: the run's `algo` string (comparability of training-side
series across `SAC`/`DemoSAC` runs is by the off-identity
certificate, not by name), and the replay composition (training-side
loss series are not comparable to SAC runs at nonzero fraction).

## 7. Seed ledger

Consumed by this design: **9200–9269** (the extension failure-state
harvest, 70 episodes → 100 entries). Remaining scratch: **9270–9299**
and 9188–9199. Battery block 6400–6499 (booked, unconsumed). 4100–4199
sealed; 5200–5299 stays clean (no demo state derives from it).
