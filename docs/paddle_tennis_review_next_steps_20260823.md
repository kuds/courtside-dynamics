# PaddleTennis campaign review and next steps — 2026-08-23

Review snapshot, pinned to `main`@`902fb33`, 2026-08-23. Diagnosis-side
only: no bars, no verdicts are changed by this document — corrections
are routed to the owning docs (§4 step 1) and every next step that trains
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
  headline `crossings_ep_mean` best 6.17 at 2.3M, final 5.97;
  `best_model` at step 2,325,000 (crossings 5.633, meta sha
  `532ef6e9…`/`0d2db134…`).
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
6. The hold escrow's reward components (`rew_hold`/clawback) are
   absent from the *metrics* artifacts — eval_info's 54 metrics and
   the monitor CSVs carry no reward decomposition — so the eval/train
   record verifies "the dose was delivered" only indirectly (measured
   travel 5.2–8.7 m against the 12.0 m budget implies strictly
   positive pay by the §2 formula, plus §4b's byte-divergence
   evidence). The decomposition *is* logged once per 250k in the
   milestone-video per-step CSVs (`VideoRecordCallback` writes
   `csv_header` incl. `rew_hold`/`rew_hold_clawback`); §2 below audits
   the dose directly from the 2.5M trace. §4 step 5 routes the
   eval-side logging fix.
7. §4c also states "the eval-info channel's entire top-5 sits in the
   2.0–2.6M band." False on both instruments: four of five, never
   five — eval_info's rank 5 is 1,725,000 (+1.606) and the npz's
   rank 2 is 2,925,000 (+1.979).

## 2. New measurements (this review)

**PT1 instrument validated end-to-end.** The probe's code was
adversarially reviewed (ctrl→world mapping exact from the model XML's
world-aligned slide joints; window lifecycle, follow-through split,
determinism and normalizer handling all correct), and two of PT1's
three subject rows were replicated locally (the registered 2.4M row
was not replayed — its checkpoint was not fetched):

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
| **fresh seeds (the scoring sample)** | 5230–5299 | **1.6% (3/191)** | **[0.3%, 4.5%]** |
| best model 2.325M, fresh + calib. | 5200–5299 | 1.1% (3/274) | [0.2%, 3.2%] |

Two readings, kept separate per lesson 13a (an argmax over a logged
series is a hypothesis; score it on seeds the selection never
touched):

- **The 5% reading was real, and seed-conditioned.** The CPU
  replication reproduces it on the same seeds (6.2%; GPU/CPU
  inference differs in fp detail, so statistical, not bit-level,
  agreement is the expected form) — the in-run row was not a probe
  artifact. But 2.5M was singled out *because* that row read high, so
  those seeds cannot score it, and a pooled estimate would inherit
  the selection bias (the `confirm_best` lesson).
- **On fresh seeds the elevation shrinks to 1.6% [0.3, 4.5]** —
  below the KH1 PASS bar (3%), well below RK1's 5%, and statistically
  indistinguishable from the crowned best model's 1.1% [0.2, 3.2].
  The fresh sample does not establish 2.5M over 2.325M; what it does
  establish is nonzero late-run k=2 (3 events in 191 points, versus
  four 1% probe events in ~944 receiving points across the entire
  registered 1M window, ≈ 0.4%). One fresh-seed second stroke landed
  in (2.87 m depth) — a fully completed second exchange; the other
  second strokes were wild (per-arm hit-#2 out-depth means
  13.7–30.7 m).

**The dose is directly audited.** The 2.5M milestone-video per-step
trace (media/videos CSV, the one artifact that logs the reward
decomposition) shows the escrow live on the deterministic policy:
`rew_hold` paid 4 windows for +0.211 total (mean 0.053 of the 0.5
scale ≈ 10.7 m travel inside paying windows) and **every payment was
clawed back — net kept 0.000** (no follow-up hit arrived). Reach and
contact, for contrast, netted +0.93 and +1.00 in the same trace.
"Delivered and declined" (§4c) now rests on a directly logged number,
not only on the §4b byte-divergence argument.

**The elevation is not held-court.** PT1 on the 2.5M checkpoint reads
*more* thrash than the best model (strike-ended cmd path 25.35 m,
saturation 92.4%, saccade 0.290 m/step; hold travel 8.2–8.5 m). §4c's
mechanism verdict — the escrow paid and the policy kept the wander —
is confirmed at this checkpoint too; its k=2 events arrive despite the
wander, not through stillness.

**The serving channel is fully dead in this lineage**: k=1 serving 0%
across 546 local points (0/274 best-model, 0/272 at 2.5M), ≤ 4% at
every LH1c in-run probe and ≤ 5% at every registered-run probe. All
k=2 signal is receiving-side.

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
  extension story is "k=2 rose off the floor late by a non-hold
  route — probe readings up to 5% at 2.5M (at the KH1/RK1 PASS-bar
  values, outside the registered window), 1.6% [0.3, 4.5] on fresh
  seeds — real but small," not "never left the noise floor."
- The **binding constraint is unchanged and now fully validated**: the
  post-swing action head emits saturated bang-bang (88–93% of steps)
  under a dead temperature (transferred 1.59e-4, final 1.72e-4, std
  0.014), and `policy_never_reached` ends 58–64% of local points
  (61% pooled). PT1's routing
  holds: exploration-side first, interface-side second, no further
  escrow scale, no dynamics- or opponent-side work on the wander.
- The **temperature-skip flag does not exist in code**. PT1 §6 calls
  it "already specced"; that is prose only —
  `WarmStartConfig` (`src/courtside_dynamics/training/train.py:131`)
  has exactly `source_run_dir` and `reset_observation_indices`, and
  the loader copies `log_ent_coef` unconditionally when both sides are
  auto (`train.py:1521–1534`). The npoint review says it plainly
  ("there is no skip flag today" — the §6 entropy caveat of that doc).
  It must be built before the routed pilot can launch.
- **Chronic α-collapse is the declared risk for that pilot**: every
  campaign run ends with α collapsed regardless of init or target —
  1.72e-4 / 1.92e-4 final in the two 3M runs, 1e-4–5e-5 across the
  earlier corpus (npoint appendix C.1: 0.02 → 9.8e-4 within 12k steps
  from scratch), and appendix D.6's tanh-saturation mechanism — a saturated
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
     2.5M, seven 1% readings), the §2 fresh-seed measurement (1.6%
     [0.3, 4.5] at 2.5M; below bars; not via stillness; dose paid
     and fully clawed back in the audited trace), the top-5 claim
     (§1.7), and the physics-version retirement (3.11 ≡ 3.12
     measured). Plus the §1.2–§1.5 bookkeeping fixes in place, and
     the `docs/README.md` index row for the hold design, which
     repeats the superseded "k=2 at its 1% floor across 3M" and
     "MuJoCo 3.12 confound noted" language.
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
     record the skip in the `initialization` provenance (assembled at
     `train.py:1541` ff., written via
     `update_run_config_with_initialization`, `artifacts.py:456–469`)
     and add the new field to the config snapshot's `warm_start`
     entry (`artifacts.py:258–267`); test sibling of
     `test_train.py:560` asserting `log_ent_coef == log(0.02)` and
     `"log_ent_coef" not in transferred`. Not TOML-settable
     (`run_config.py:58–66` keeps `warm_start` code-side; the pilot's
     TOML records the pairing in its comment header, per convention).
3. **Pre-register and launch LT1 — the temperature-skip pilot** (the
   §4a shelf item, PT1 routing #1). Frozen shape to draft: 1M steps,
   seed 0, n_envs 4, eval 25k, checkpoint/diagnosis 100k, recipe
   defaults (hold off), warm start with `transfer_log_ent_coef=False`.
   Three decisions the prereg must freeze:
   - **Source.** The default is the registered 2.4M protected best
     (`838997fb…`) — the §4a shelf item's own pairing, and the only
     source with a same-source control already on the books (LH1's
     transferred-temperature run from the identical artifacts), so
     the skip flag stays a one-lever change per the standing
     convention. The LH1c-lineage candidates (crowned 2.325M
     `532ef6e9…`, record-eval band; 2.5M `45a1a80b…`) are alternatives
     the prereg may choose instead, but each is a second lever
     (source + flag) with no temperature-transferred control, and §2's
     fresh-seed data gives 2.5M no measured edge over 2.325M — if a
     lineage change is wanted, pre-register it as its own decision.
   - **Bars.** KH1-style headline (k=2 ≥ 3% at some checkpoint) and
     R1-style retention, anchored on the chosen source's measured
     band. If an LH1c-lineage source is chosen instead of the
     default, note its own probes already brush the bar values (5% at
     2.5M on calibration seeds; 1.6% fresh) — the headline bar should
     then demand an *advance* (e.g. ≥ 3% at two consecutive probes,
     or ≥ 5%), not re-cross a line the source already touches. Bars
     are the maintainer's to freeze; this review supplies the
     anchors.
   - **Mechanism observables, pre-declared:** (a) ent_coef trajectory
     — steps until α < 1e-3 again (the §3 collapse risk; expect ~10k
     from the from-scratch precedent, measure it); (b) post-swing
     saturation and saccade from the PT1 instrument at each 100k
     diagnosis (measured bands: registered 2.4M source 87.6% /
     0.381 m/step per PT1; LH1c lineage 91–93% / 0.26–0.29) — the pilot's
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
   mounted)* Extend PT-K2 to the 2.3M–2.9M neighbors — only if a
   prereg wants an LH1c-lineage source and needs the k=2-richest
   checkpoint resolved beyond §2. Score on seeds no probe-driven
   selection has touched (5230–5299, or a declared fresh use of the
   calibration convention) — pooling with the 5200–5229 readings that
   nominated a checkpoint repeats the bias §2 avoids.

## 5. Seed ledger

Unchanged. All local replays used calibration seeds **5200–5299**
(the diagnosis convention; the block has been burned since the
diagnosis era and the instrument has run its full span before — what
matters here is that no selection over *these* checkpoints ever
touched 5230–5299, which is why §2 scores on that sub-range). No
reserved block touched; nothing new burned; **4100–4199 remains
sealed.**
