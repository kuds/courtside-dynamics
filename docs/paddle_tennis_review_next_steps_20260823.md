# PaddleTennis campaign review and next steps — 2026-08-23

Review snapshot, pinned to `main`@`902fb33`, 2026-08-23. Diagnosis-side
only: no bars, no verdicts are changed by this document — corrections
are routed to the owning docs (§5.1) and every next step that trains
anything requires its own pre-registration first, per the standing
doctrine.

**What was reviewed.** The three closing documents of the n-point era
(`paddle_tennis_registered_run_prereg_20260816.md` §8,
`design_paddle_tennis_postswing_hold.md` §4a–§4c,
`paddle_tennis_postswing_targets_20260822.md`), the campaign's Drive
artifacts for runs `20260821_013700` (LH1c) and `20260816_235141`
(registered) — evaluations.npz, eval_info.csv, progress.csv, all 31
diagnosis rows per run, config.json, best_model_meta.json, monitor
logs, report plots, replay video — and the instruments themselves,
re-run locally against sha-verified checkpoints. Local stack: Python
3.11.15, MuJoCo **3.12.0** (LH1c's own training/probe version), SB3
2.9.0, Gymnasium 1.3.0, torch 2.13 CPU; the full paddle-tennis test
subset (179 tests) passes on it.

## 1. Verification of the booked verdicts

Every load-bearing number in the three closing documents was re-derived
from the raw artifacts. The verdicts all stand. The details:

**Verified exactly** (to the printed digit unless noted):

- LH1c best eval **+2.483 ± 0.829 at 2,425,000** (evaluations.npz row);
  1M-window best +1.067 at eval 1; eval-info window max +1.187;
  headline `crossings_ep_mean` best 6.17 at 2.3M, final 5.97; the
  entire eval-reward top-5 in the 2.0–2.6M band; `best_model` at step
  2,325,000 (crossings 5.633, meta sha `532ef6e9…`/`0d2db134…`).
- LH1c §4c registered-window scoring: k=2 ≤ 1% at every 100k probe
  with exactly four 1% events (KH1 FAIL confirmed); hold-travel window
  minimum 5.20 m with 8.09/8.32 m at 900k/1M once engagement recovered
  (H1 as read); k=1 receiving window peak 68% (R1 middle confirmed).
- LH1c extension: k=1 receiving 90% at the 3M probe (peak 98% at
  2.6M); 3M hold travel 8.71 m mean / p90 11.37; final ent_coef
  1.724e-4, `train/std` 0.0142 (the transferred temperature, 1.589e-4
  per config.json, never recovered).
- Registered run §8: M PASS at exactly **77.5%** of 120 evals; k=1
  receiving ≥ 80% at consecutive probes with peak 95% (RE3 PASS); k=2
  never above 1% either parity (RK1 FAIL); hold travel 7.46 m at 3M
  (the R2 watch); first-hit shot quality 81% crossed / 71% in at 3M.
- Warm-start provenance: LH1c's config.json records the §4b source
  artifacts by the exact shas the prereg pinned (`838997fb…`,
  `d0502c14…`) and transfers policy + obs_rms + `log_ent_coef`.

**Discrepancies found** (none overturns a verdict; one is material):

1. **§4c's extension observation is materially incomplete.** It
   records "k=2 never left the noise floor: two isolated 2% readings
   (1.3M, 2.9M) among ~19 extension probes otherwise at 0–1%." The
   actual diagnosis series reads 2% at **1.3M, 2.1M, 2.9M** and **5%
   at 2.5M** (4 second-hits / 85 receiving points, one landing in),
   with seven further 1% readings. The missed 5% sits at the frozen
   KH1 PASS bar (≥ 3%) and RK1 PASS bar (≥ 5%) — outside the
   registered 1M window, so KH1 FAIL stands, but the era's strongest
   k=2 reading was booked as noise. §2 below measures it at larger n.
2. Registered-run bookkeeping: k=2 read 1% at **11** checkpoints
   (receiving side), not "nine … and 0% on the other twenty-one"
   (`design_paddle_tennis_postswing_hold.md` §1).
3. The prior campaign record is **+1.822** (npz and stage summary),
   quoted as "+1.85" in §1 of the hold design.
4. Three different "final eval" numbers coexist unlabeled for LH1c:
   +1.438 ± 1.448 (evaluations.npz row at 3M), +1.602 (eval_info at
   3M), +0.976 ± 1.626 (stage summary / §4c "closing eval" — a
   separate end-of-run evaluation). Same direction, different
   instruments; the stage summary should name which it reports.
5. §4c "hold travel settled at 7.5–8.7 m": the extension series
   actually spans 6.38–8.71 m. Direction unchanged.
6. The hold escrow's reward components (`rew_hold`/clawback) appear in
   no logged artifact — eval_info's 54 metrics and the monitor CSVs
   carry no reward decomposition — so "the dose was delivered" is
   verifiable only indirectly (measured travel 5.2–8.7 m against the
   12.0 m budget implies strictly positive pay by the §2 formula, and
   §4b's byte-divergence evidence). §5.5 routes the logging fix.

## 2. New measurements (this review)

**PT1 instrument validated end-to-end.** The probe's code was
adversarially reviewed (ctrl→world mapping exact from the model XML's
world-aligned slide joints; window lifecycle, follow-through split,
determinism and normalizer handling all correct), and both published
rows were replicated locally:

- The **oracle row reproduces to every printed digit** (186 windows;
  2.36/2.31 m commanded/actual, 0.49 m servo gap, 1.8% saturation;
  strike-ended group 164 windows, 2.56/2.55 m, 0.033 m/step saccade).
- The **LH1c-best row reproduces to every printed digit** (52
  strike-ended windows; cmd 22.71 m vs actual 7.55 m; servo 2.23 m;
  saturation 91.4%; saccade 0.258; boundary travel 8.99 m).

Both replications ran under MuJoCo **3.12.0** where the doc's rows ran
under 3.11.0. Identical output on identical seeds means **the
3.11→3.12 physics change is behaviorally nil on this task** for
deterministic replay, oracle and trained checkpoint alike. Two
consequences: PT1's cross-version caveat can be retired, and §4c's
"whether the live hold gradient improved the optimization or MuJoCo
3.12 moved the task is not separable post-hoc" is now separable — the
physics did not move, so LH1c's campaign-record evals belong to the
run itself. (Caveat: in-run diagnosis rows were computed on the GPU;
CPU replays of the same checkpoint match them statistically, not
bit-for-bit — see the 2.5M replication below.)

**PT-K2 — the 2.5M reading at larger n** (checkpoint
`paddle_tennis_sac_2500000_steps.zip`, sha `45a1a80b…`, replayed
through the run's own recorded eval constructor kwargs, n-point
continuous, deterministic):

| sample | seeds | k=2 receiving | 95% CI |
|---|---|---|---|
| in-run probe row (GPU) | 5200–5229 | 5% (≈4/85) | — |
| local replication (CPU) | 5200–5229 | 6.2% (5/81) | [2.0%, 13.8%] |
| fresh seeds | 5230–5299 | 1.6% (3/191) | [0.3%, 4.5%] |
| **pooled, this review** | 5200–5299 | **2.9% (8/272)** | **[1.3%, 5.7%]** |
| best model 2.325M, same instrument | 5200–5299 | 1.1% (3/274) | [0.2%, 3.2%] |

Reading: the 2.5M checkpoint's k=2 is genuinely off the floor — 2.9%
pooled against the registered 1M window's ~0.5% (four 1% probe events
in ~840 receiving points) — but the single 5% reading was partly a
favorable 30-episode draw: the pooled estimate sits **below the KH1
PASS bar (3%)** and well below RK1's 5%, and the elevation over the
crowned best model is not individually significant (Fisher one-sided
p = 0.109 at n ≈ 270 per arm). One fresh-seed second stroke landed in
(2.87 m depth) — a fully completed second exchange; the other second
strokes were wild (out-depth 8.5–30.7 m).

**The elevation is not held-court.** PT1 on the 2.5M checkpoint reads
*more* thrash than the best model (strike-ended cmd path 25.35 m,
saturation 92.4%, saccade 0.290 m/step; hold travel 8.2–8.5 m). §4c's
mechanism verdict — the escrow paid and the policy kept the wander —
is confirmed at this checkpoint too; its k=2 events arrive despite the
wander, not through stillness.

**The serving channel is fully dead in this lineage**: k=1 serving 0%
across 546 local points (0/274 best-model, 0/272 at 2.5M) and ≤ 4% at
every in-run probe. All k=2 signal is receiving-side.

**Selection-instrument note.** Task-metric selection
(crossings/success/reward) crowned 2.325M; the k=2-richest checkpoint
is 2.5M. Nothing in the selection tracks the era headline, so the
protected best is not the k=2 record-holder. The per-checkpoint
diagnosis already measures k=2; a future prereg that cares should name
the k=2-richest checkpoint an era artifact alongside the crowned best.

## 3. Assessment

- The **registered verdicts stand**: RK1 FAIL (registered run), KH1
  FAIL on LH1c's 1M window, and the hold-line closure decision — the
  mechanism was delivered and declined; no re-pair branch remained.
- The **§4c extension record needs an erratum** (§1.1): the honest
  extension story is "k=2 rose off the floor late (to ~2–3% around
  2.5M, peak probe reading 5%) by a non-hold route, below every PASS
  bar," not "never left the noise floor."
- The **binding constraint is unchanged and now fully validated**: the
  post-swing action head emits saturated bang-bang (88–93% of steps)
  under a dead temperature (transferred 1.59e-4, final 1.72e-4, std
  0.014), and `policy_never_reached` ends 60% of points. PT1's routing
  holds: exploration-side first, interface-side second, no further
  escrow scale, no dynamics- or opponent-side work on the wander.
- The **temperature-skip flag does not exist in code**. PT1 §6 calls
  it "already specced"; that is prose only —
  `WarmStartConfig` (`src/courtside_dynamics/training/train.py:131`)
  has exactly `source_run_dir` and `reset_observation_indices`, and
  the loader copies `log_ent_coef` unconditionally when both sides are
  auto (`train.py:1521–1534`). The npoint review says it plainly
  ("there is no skip flag today", §5.1 of that doc). It must be built
  before the routed pilot can launch.
- **Chronic α-collapse is the declared risk for that pilot**: every
  campaign run ends at ent_coef 1e-4–5e-5 regardless of init or
  target (npoint appendix C.1: 0.02 → 9.8e-4 within 12k steps from
  scratch), and appendix D.6's tanh-saturation mechanism — a saturated
  squashed-Gaussian mean inflates latent −log π, annealing α to zero —
  applies with full force to a warm start that begins 92% saturated.
  Restoring α=0.02 buys a window, not a regime; the pilot's mechanism
  observable must be whether any unsaturation happens inside that
  window.

## 4. Next steps, ranked

Each step names its owner-doc and its gate. Order is execution order;
2–4 are independent of each other once 1 is booked.

1. **Book the corrections** (no training, do first):
   - Erratum to `design_paddle_tennis_postswing_hold.md` §4c: the
     corrected extension k=2 series (2% at 1.3M/2.1M/2.9M, 5% at
     2.5M, seven 1% readings), the §2 larger-n measurement (2.9%
     pooled [1.3, 5.7] at 2.5M; below bars; not via stillness), and
     the physics-version retirement (3.11 ≡ 3.12 measured). Plus the
     §1.2–§1.5 bookkeeping fixes in place.
   - Distill the 2026-08-15 → 08-22 era verdicts (L2 stop/pivot, LR1
     ADOPT, RK1 FAIL, hold closure, PT1) into `DECISIONS.md` — the
     journal currently ends at 2026-08-02 and the campaign's actual
     decisions live only in the later docs' status lines.
2. **Implement `transfer_log_ent_coef`** (the routed pilot's missing
   prerequisite; small, default-preserving):
   - `WarmStartConfig.transfer_log_ent_coef: bool = True`
     (`train.py:131–170`, bool-validated); when False, skip the copy
     in the transfer block (`train.py:1521–1534`) so `auto_0.02`'s
     fresh init stands, keeping the auto-vs-fixed mismatch guard;
     record the skip in `initialization` provenance
     (`artifacts.py:258–265`); test sibling of
     `test_train.py:560` asserting `log_ent_coef == log(0.02)` and
     `"log_ent_coef" not in transferred`. Not TOML-settable
     (`run_config.py:58–66` keeps `warm_start` code-side; the pilot's
     TOML records the pairing in its comment header, per convention).
3. **Pre-register and launch LT1 — the temperature-skip pilot** (the
   §4a shelf item, PT1 routing #1). Frozen shape to draft: 1M steps,
   seed 0, n_envs 4, eval 25k, checkpoint/diagnosis 100k, recipe
   defaults (hold off), warm start with `transfer_log_ent_coef=False`.
   Three decisions the prereg must freeze:
   - **Source.** Candidates: the registered 2.4M protected best
     (`838997fb…`, the shelf item's original pairing), LH1c's crowned
     2.325M (`532ef6e9…`, campaign-record eval lineage), or LH1c's
     2.5M (`45a1a80b…`, the k=2-richest measured artifact, 2.9%
     pooled). This review's data supports 2.5M — most k=2 to retain,
     same lineage — with the §2 caveat (its edge over 2.325M is not
     individually significant) recorded in the prereg.
   - **Bars.** KH1-style headline (k=2 ≥ 3% at some checkpoint) and
     R1-style retention, anchored on the source's measured band — for
     a 2.5M source that band is k=1 87–89%, k=2 ~2.9%, so the
     headline bar should demand an *advance* (e.g. ≥ 5% or ≥ 3% at
     two probes), not re-cross a line the source already touches.
     Bars are the maintainer's to freeze; this review supplies the
     anchors.
   - **Mechanism observables, pre-declared:** (a) ent_coef trajectory
     — steps until α < 1e-3 again (the §3 collapse risk; expect ~10k
     from the from-scratch precedent, measure it); (b) post-swing
     saturation and saccade from the PT1 instrument at each 100k
     diagnosis (source band: 91–93% / 0.26–0.29 m/step) — the pilot's
     claim is that live temperature lets the action head unsaturate,
     so measure exactly that. If α re-collapses with no saturation
     movement, the verdict routes to step 4's interface-side treatment
     with no further optimizer-side retry.
4. **Draft the command-rate / low-pass target design doc** (PT1
   routing #2; parallel with 2–3, launches nothing until LT1's
   verdict routes to it). Anchors from the validated PT1 data: the
   servo tracks ≲ 0.1 m/step while the policy commands 0.26–0.29
   m/step at 2.2–2.7 m mean gap; the oracle plays the whole game at
   0.033 m/step. An interface-side rate limit (or first-order lag) on
   the position targets makes stillness the default instead of an
   80-consecutive-action feat. Explicitly a **new comparability era**
   — new probe battery (PH-style witnesses + the PT1 oracle row),
   fresh seed block, its own bars; not a TOML toggle, per the
   standing convention.
5. **Close the tooling gaps this review hit** (before the next pilot
   runs, so its evidence is first-class):
   - aggregate `recovery_travel` in `paddle_campaign_metrics`
     (`notebook_utils.py:1874`) — H1-style hold bars are currently
     unscoreable in the campaign notebook;
   - campaign notebook: a warm-started leg-1 path, env-kwargs/
     config-file plumbing, and the prereg §6 config.json-vs-prereg
     validation cell (currently unimplemented);
   - log the reward decomposition (`rew_*` components) into
     eval_info so a shaping term's delivered dose is auditable from
     artifacts (§1.6);
   - optional `expected_sha256` pinning in `WarmStartConfig` (today
     sha is compute-and-record only);
   - label the stage summary's "Final eval" with its instrument
     (§1.4).
6. **Keep closed and keep sealed:** no further hold-escrow scale (the
   kwargs stay in the env, default off, certified); no dynamics-side
   or opponent-side work on the wander; the serving-side channel
   stays a watch metric (0% k=1 at n=546 — dead, known, not the
   current blocker); seed block **4100–4199 stays sealed** until a
   registered-result branch fires.
7. *(Optional, maintainer-side where the Drive artifacts are
   mounted)* Extend PT-K2 to the 2.3M–2.9M neighbors at n=100 each —
   only if LT1's source selection wants the k=2-richest checkpoint
   resolved more finely than §2's 2.5M-vs-2.325M comparison.

## 5. Seed ledger

Unchanged. All local replays used calibration seeds **5200–5299**
(the diagnosis convention; 5230–5299 had never been run through this
instrument but the block is the calibration block, burned since the
diagnosis era). No reserved block touched; nothing new burned;
**4100–4199 remains sealed.**
