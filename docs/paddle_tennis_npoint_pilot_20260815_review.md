# Review: the n-point L2 pilot (20260815_015143) — verdict and training recommendations

Status: **review complete, 2026-08-15**. The n-point from-scratch
recipe has now failed twice: the spec-conformant L2
(`20260810_211754`, seed 0, n_envs 4, early-stopped at 1.125M) and
the reviewed full-budget replication (`20260815_015143`, seed 1,
n_envs 8, 2M) — both at SHA 2e597b1, both with the same signature.
The reviewed run never acquired the game: zero policy contact at 8 of
20 diagnosis checkpoints including the final five, 21 sporadic
touches total after 100k (one landed in — one shot, all run), every
L2 bar except M at FAIL. Under the frozen decision rule
([`design_paddle_tennis_npoint.md`](design_paddle_tennis_npoint.md)
§4a) the **Stop/pivot branch fires**, seed-independently. This review adds local
validation experiments (seeds 5600–5719, 6100+) that locate the
failure precisely: from scratch under carryover, contact is not
discoverable by exploration noise at any std, and the reward
landscape *pays* the optimizer to go quiet. The design's own §1
mechanism — densely-sampled inter-point credit — was never exercised,
because it routes through k=1 returns the from-scratch policy cannot
produce. The mechanism is **untested, not refuted**; §5's first
recommendation tests it directly by warm-starting from the k=1-mastered
extension checkpoint.

## 1. The run, against its pre-registration

| | pre-registered (§4a) | run 20260815_015143 |
|---|---|---|
| config | `points_per_episode="none"`, `contact_shaping=0.25` | same (config.json provenance, sha 79f6f3a) |
| commit | n-point implementation | 2e597b1 (the implementation commit) |
| seed | **0** | **1** |
| n_envs | **4** | **8** |
| budget / cadence | 2M / 100k | 2M / 100k |

Outcome: 12h22m on an L4 at 44 FPS, completed (no guard fired).
Final eval −3.033 ± 0.180; best-by-headline crowned at 200k on
crossings_ep_mean 3.20 (final 3.00). Training health at end:
`train/ent_coef` 5.34e-5, `train/std` 0.0183, critic_loss 3.9e-5.

Bookkeeping resolves cleanly despite the deviation: the literal,
spec-conformant L2 is **`20260810_211754`** (seed 0, n_envs 4,
npoint_pilot.toml, SHA 2e597b1), which early-stopped at 1.125M with
the same zero-contact collapse (final probe: 1 shot in 214 points,
k=1 0%/1%, `policy_never_reached` 182 + cap 30, inter-point travel
8.40 m). The reviewed run is its seed-1/n_envs-8/full-budget
replication. The verdict books on the 8/10 run; the 8/15 run
upgrades it from "one seed" to "structural".

## 2. The L2 bar table, measured

| criterion | PASS bar | measured (best over 20 checkpoints) | outcome |
|---|---|---|---|
| **K** (headline) | k=2 survival > 0%, either parity | 0% at every checkpoint | **FAIL** |
| R1 | recovery-hold ≤ 5.0 m | 1.35–6.23 m on the handful of swing checkpoints; unmeasurable elsewhere | no signal (no sustained swings) |
| R2 | inter-point recovery ≤ 4.2 m | 7.65–9.28 m, flat, no trend (oracle 1.95–2.10) | **FAIL** |
| P″ | crossings/completed point ≥ 2.5 | 0.42–0.45 all run (oracle 5.35–5.48) | **FAIL** |
| D2″ | touched-after-bounce ≥ 60% | 3% at 100k, ≤4% after, 0% from 1.6M | **FAIL** |
| M | ent-coef anneal + train/std ≥ 5e-3 | std 0.0183 final (≥ 5e-3); coef annealed | intact by the letter (see §4c) |

K FAIL with zero others in PASS-or-middle → **Stop/pivot**: "n-point
stays default-off; the next probed change targets the
opponent/curriculum side."

## 3. What the run did, checkpoint by checkpoint

From the automated diagnosis probes (30 episodes each, seeds 5200+;
full table in the review workpapers):

- **100k**: 6 contacts, 83% crossed, **0% in** — out-depths 13.79 m,
  the exploration pilot's hard-slam signature. k=1 receiving 2%,
  serving 4%.
- **200k–1.5M**: sporadic residue — 21 contacts over ~2,780 probe
  points, all but one on the policy's own serve feed, exactly one
  shot landing in (500k). k=2: never sampled once.
- **1.6M–2.0M**: strictly zero contact. Point enders collapse to two
  categories: `policy_never_reached` 183–184 + cap 30.
- Ready-position error **worsens** over training: ~2.6 m (first
  half) → ~3.55 m (last 800k); oracle 0.89. Inter-point travel flat
  at 7.65–9.28 m (oracle 1.95). Touched-after-bounce decays to hard 0%.
- The opponent row is stationary throughout (crossed 92–95%, in
  88–92%) — every trend above is the policy.
- Eval reward *improved* −3.33 → −3.03 across the same span, and the
  headline (crossings ≈ 3/episode) is entirely the opponent's
  serve-returns on policy-serve points: best-model selection ran on
  opponent noise (lessons_learned #11, verbatim).

## 4. Validation experiments (this review)

Environment at the run's exact commit (2e597b1). The pre-registered
NP1/NP2 battery re-ran locally first — **PASS 30/30, band identical**
(crossings/ep 11.40, bridge 4.46, inter-point recovery 2.10 m) — so
mechanics are certified sound here too and local numbers are
era-comparable. Probe scripts and raw JSON live with the review
workpapers; seeds 5600+ (statue/oracle), 5700+/5900+ (noise arms),
6100+ (coherent noise) — clear of every reserved and burned block.

### 4a. The reward landscape pays competence — and the trained policy never found it

30 episodes each, identical env kwargs to the run:

| policy | reward/ep | returns | faults | net shaping | side-A contacts |
|---|---|---|---|---|---|
| statue (zero action) | −4.37 ± 1.40 | +2.60 | −6.97 | 0 | 0 |
| trained 2M (Colab final eval) | −3.03 ± 0.18 | ~+3.0 | ~−6.2 | 0 | 0 |
| scripted oracle on side A | **+11.64 ± 2.31** | +11.27 | −1.00 | +1.38 | 6.6/ep |

A 16-point gap separates the statue from the oracle **in the reward
the run optimized** — the destination gradient exists and is rich.
The trained policy's edge over the raw statue is reproducible without
touching a ball: constant held actions that park off-center score
−3.15 by slowing point turnover (6.15 vs 6.97 completed points/ep).
Two million GPU steps bought a better-parked statue.

### 4b. The exploration cliff: contact is not discoverable from scratch under carryover

Statue + Gaussian noise, 20 episodes (≈140 points) per arm:

| noise | iid 0.02 | iid 0.1 | iid 0.3 | iid 0.6 | iid 1.0 | held-64 0.3 | held-64 1.0 | held-64 uniform |
|---|---|---|---|---|---|---|---|---|
| side-A contacts | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.15/ep (15% of eps) |
| reward/ep | −4.65 | −4.45 | −4.55 | −4.80 | −4.75 | −4.75 | −3.15 | −3.39 |

- **Zero contacts at every Gaussian operating point**, iid or held at
  the gSDE persistence scale (64 steps). Only full-range uniform held
  actions touch at all — ~1 contact per 10k steps, replicating the
  humanoid C2 constant-action measurement — far beyond any noise
  level a Gaussian policy sustains, and each such touch nets ~0
  (§4d).
- **Reward decreases as noise increases** around the statue. The
  optimizer annealing std toward its floor was the *correct local
  gradient*; the entropy collapse is a symptom of the landscape, not
  an optimizer defect.
- Geometry of the cliff: the feed's first bounce lands mean 4.8 m
  (p90 7.2 m) from the carried-over paddle with ~0.95–1.1 s of
  flight. Kinematically reachable (12.5 m/s paddle ceiling; 0.36–0.64 s
  traverse — measured), but only by *directed* pursuit. Noise
  supplies no direction, and nothing in the reward pays for closing
  the distance.
- Serve easing does not open the cliff: a 7.5 m/s / 27° / laterally
  tight serve shrinks the bounce distance to 3.9 m and contacts stay
  zero (and easier serves die sooner → more completed points → more
  −1s under statue economics). A serve curriculum alone is measured
  insufficient.

### 4c. Why every safeguard slept

- **The escrow held exactly** (NP1 re-run: |paid + clawback − 0.25 ×
  confirms| = 0 across all arms). By design, touch-then-fault pays
  net 0 undiscounted — identical to never touching (−1.0 per point
  either way; the discount-timing crumb is +0.1–0.18). So the 100k
  hard-slam touches were unpaid, decayed, and took k=1 with them —
  the exploration pilot's "a touch that always ends −1 is
  indistinguishable from never touching", now with touches ~10³×
  rarer because of carryover.
- **M is intact by the letter and dead in fact.** The dual update
  defends the *latent* log-prob entropy near the −1.5 target
  (ent_coef_loss −1.38 at end ⇒ the tuner sees entropy ≈ target)
  while the action-space std sits at 0.0183 — a Gaussian whose
  physical entropy is ≈ −7.9. The M2 floor (std ≥ 5e-3) passes while
  exploration is behaviorally extinct (§4b: std 0.02 ≡ statue). The
  mechanism criterion needs a behavioral gauge, not a std floor.
- **The stop/selection machinery was disarmed for PaddleTennis.**
  `best_metric_min_delta` 0.0, `confirm_best_eval` off, degenerate
  guard off (recipes.py:1112–1186) — the exact triple the WallBall
  recipes arm after run 20260714_211111 burned 225k dead steps on a
  1e-8 reward tie-break. Here crossings noise (±0.2 of opponent
  behavior) plus reward drift kept re-crowning best_model and
  resetting patience for 80 straight evaluations. A degenerate guard
  on `legal_hit_count_ep_mean` (already collected in
  `info_eval_keys`) would have stopped the run ~500k in.
- **Every eval surface is deterministic** (reward eval, info eval,
  diagnosis, video): exploration health is invisible to all of them,
  and the diagnosis probes that did record the collapse are
  write-only text files nothing parses.

### 4d. Why this differs from the one-point era that worked

The shaping era learned k=1 from scratch in this same reward scheme
because each episode re-parked the paddle and served into its reach:
touch rates 3–17% under exploration made the escrow collectable. The
n-point amendment removed the re-park (that is its load-bearing
decision) and thereby moved the task from "touch is rare but sampled"
to "touch is ~unsampleable" — while keeping a reward that only pays
*after* touch. From scratch, the era is strictly harder to bootstrap
than the era it replaced; the design's §1 gradient argument silently
assumed a policy that already owns the stroke.

## 5. Recommendations

Ranked. #1 is the run to make next; #2 costs one line each and
should ride along on every future run regardless.

1. **Warm-start the n-point era from the k=1-mastered checkpoint
   (L2W).** Source: `20260809_211147` best model (receiving k=1
   100%, serving 53%, 78% in at 3.73 m, best eval 1.77 *still
   climbing* at 2M). Mechanics are already in place: spaces
   unchanged by the amendment, `warm_start` transfers policy +
   critics + `obs_rms` and validates layout (train.py:718–855,
   1472–1616) — but it also transfers the source's **collapsed**
   `log_ent_coef` (~4e-5) by design. Pair the pilot with an entropy
   re-init (smallest change: a `WarmStartConfig` flag to re-init
   auto-entropy at `auto_0.02`'s floor instead of transferring;
   the gate's `reset_entropy_on_advance` machinery,
   performance_gate.py:154–191, is the in-repo precedent) and the
   docstring's raised `learning_starts`. NP2 measured the oracle
   keeping k=1 92%/88% *under carryover* — a stroke-owning policy
   collects the design's inter-point credit from the first
   boundary, which is precisely the mechanism L2 never exercised.
   Pre-register: K unchanged (k=2 > 0%); a retention floor (k=1
   receiving ≥ 50% by 200k — does the stroke survive the
   distribution shift?); R2/D2″ as before. 1M steps ≈ 6 GPU-hours;
   abort-by-guard if retention fails. No run in the campaign has
   ever warm-started (`warm_start: null` in all seven configs) —
   the mode switch to n-point silently discarded 3M steps of
   accumulated stroke competence, and this is the cheapest way to
   get it back.
   *Note `warm_start` is deliberately not TOML-settable
   (run_config.py:65–66) — the pilot needs a code-side
   `build_train_config(..., warm_start=...)` call like the notebook's.*

2. **Arm the guards the stack already owns** (recipe one-liners,
   no era impact, WallBall precedent recipes.py:626–649):
   `degenerate_guard_keys=("legal_hit_count_ep_mean",)` +
   `early_stop_degenerate_evals≈5` (kills a zero-contact run in
   ~125k steps), `best_metric_min_delta≈0.25` (above opponent
   crossings noise), `confirm_best_eval=True`. Add a
   policy-attributable headline: `crossings` counts both sides, so
   selection tracked the opponent all run — until a side-A crossings
   info key exists, `legal_hit_count` is the honest headline/success
   key for this era.

3. **If from-scratch must stay on the table: a `points_per_episode`
   ladder, not `None`-from-scratch.** The gate callback applies
   arbitrary stage values via `set_wrapper_attr` and can carry
   `reset_entropy_on_advance` + replay-wipe; int stages 1→2→4→8
   keyed to `legal_hit_count_ep_mean`/`crossings` reproduce the
   working one-point bootstrap and then anneal toward continuous
   play (8 points ≥ the ~7 that fit the cap, ≈ `None`). Needs only a
   smoke test that gate stages accept int values; `points_per_episode`
   itself is validated at init but written raw by the gate — pin the
   int contract in the smoke test.

4. **Reward-side amendment for the reach gap (own design + probe
   battery, new comparability era).** The measured hole is *before*
   touch: nothing pays for being near the bounce. Two candidate
   shapes, in the repo's own escrow idiom: (a) once-per-point
   proximity pay at first-bounce time, `s·max(0, 1 − d/3 m)` with
   s ≤ 0.25, escrowed against *touch* (clawed back if the point ends
   untouched) so a camping statue collects nothing; (b) fault
   asymmetry — attempted-return faults −0.5 vs unreturned deaths
   −1.0, making touch strictly dominant undiscounted (the diagnosis
   doc's #3 own-credit direction). Either changes statue economics —
   the NP1 statue witnesses must be re-derived, which is exactly
   what the S-probe pattern is for.

5. **Fix the mechanism criterion (M) before it certifies another
   dead run.** Replace/augment the std floor with a behavioral
   gauge: train-time contact — `rollout` legal-hit rate > 0 by 300k
   — and log per-dim action std. The gSDE latent entropy the dual
   update defends is measured here to diverge by ~6.5 nats from the
   action-space entropy at collapse; `train/ent_coef` magnitude
   stays a non-signal either way (exploration doc §2 was right),
   but so is a latent-entropy target being "met". The collapse
   pattern is chronic, not specific to this run: every run in the
   campaign, under target −3.0 and −1.5 alike, ends at ent_coef
   1e-4–5e-5 with std 0.018–0.036 — the raised target changed
   nothing measurable about where the tuner rests.

6. **What not to spend on**, per the measurements: a serve-easing-only
   curriculum (§4b: insufficient alone); an opponent-softening run
   before #1 (H3 was rejected on measurement, and the opponent row
   was stationary all run); any third from-scratch n-point run (two
   seeds, two n_envs settings, 3.1M cumulative steps, zero contact —
   lessons_learned #8: when two runs agree on a ceiling, change the
   task or reward, not the count).

## 6. Workpapers

Local artifacts backing §4 (session scratchpad, reproducible from
the scripts): `reward_experiments.py/.json` (statue/oracle/noise
arms), `coherent_noise_probe.py/.json` (held-noise arms),
`serve_curriculum_probe.py/.json` (serve-easing arms), NP1/NP2
battery output (verbatim PASS log). Cross-run history, run-metrics
trajectories (ent_coef/std collapse timing), and the branch code
review are appended in
[`paddle_tennis_npoint_pilot_20260815_appendix.md`](paddle_tennis_npoint_pilot_20260815_appendix.md).
