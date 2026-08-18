# Design: PaddleTennis escrowed post-swing hold

Status: **design frozen 2026-08-17 before implementation**; the §3
battery and §4 pilot bars are pre-registered here before any probe or
training step. The standing doctrine applies: changes after a
pre-registered run starts void that run as campaign evidence.

## 1. Why this change, and why now

The registered n-point run (`20260816_235141`, from-scratch 3M, seed
1) booked **RK1 FAIL under §5's stop/amend branch**
(`paddle_tennis_registered_run_prereg_20260816.md`): k=2 exchange
survival sat at exactly 1% on nine checkpoints and 0% on the other
twenty-one, while every upstream rung was climbing or solved —
LS1 gate PASS (the from-scratch bootstrap works), RE3 PASS with k=1
receiving at 81–95% across the last third, first-hit shot quality at
81% crossed / 71% in by 3M (the scripted oracle plays 92%/84%), M
PASS at 77.5%, campaign-record evaluations (best +1.85). A channel
that stays flat at 1% for two million steps while its prerequisites
triple is not budget-limited; it is reward-limited.

The instrument names the limiting behavior precisely, and has all
run: **post-swing wander**. After its own legal hit, the policy
travels 7.5 m mean (p90 10.7) while its shot is away, so the
opponent's return arrives to an empty court. Both completed
second-exchange events (and all nine k=2 survivals) happened on the
rare points where it stayed home. This window — from the policy's
hit to the opponent's return strike — is the one segment of the
k=2 causal chain the reward does not price:

| segment | paid by |
|---|---|
| move toward the incoming bounce | reach escrow (at the bounce) |
| touch it legally | contact escrow (at the hit) |
| the return crosses and lands in | +1 confirmed return |
| **hold court while the shot is away** | **nothing** |

R2 (inter-point recovery ≤ 6.0 m) was pre-registered as the watch
metric that routes this design: it read 7.5–9.1 m at every
checkpoint. This document is that route: the **post-swing-hold
escrow**, the third member of the event-escrow family, built to the
same contract as its siblings (default off, bit-identical until
enabled, farm-proof by keep/clawback construction).

## 2. Mechanism (frozen)

Two new constructor kwargs on `PaddleTennisEnv`:

- `hold_shaping: float = 0.0` — payment scale. `0.0` (default)
  leaves the frozen task's reward stream **bit-identical**; validated
  finite and non-negative.
- `hold_shaping_travel: float = 4.0` — the accumulated post-swing
  travel (meters of XY path) at which the pay ramps to zero;
  validated finite and positive.

**Window lifecycle.** A side-A legal hit *arms* a hold window: the
env records the side-A paddle head's XY at that step's end and
starts accumulating the head's per-step XY path length. The window
*pays* at the opponent's next legal racket hit (side B in
`valid_racket_hits` — the event that makes the next incoming ball
real):

```
payment = hold_shaping × max(0, 1 − travel / hold_shaping_travel)
```

and disarms (one payment per window). The window *disarms unpaid* at
every point ending and episode ending — a shot that dies before the
opponent strikes it earns nothing for the wait.

**Escrow.** The payment enters the pending-hold escrow. It is *kept*
by the policy's next side-A legal hit — which, arriving after the
opponent's return strike, **is by construction the k=2 hit** — and
whatever is pending *claws back* at every ending path: episode
terminations, truncation, n-point boundaries, and the forced
non-finite guard, exactly the reach escrow's clawback set. Episode
totals therefore satisfy the family's sum identity: `rew_hold +
rew_hold_clawback` totals exactly the hold pay whose follow-up hit
happened. Camping is unpaid (no window without a hit); hit-once-
then-freeze collects a payment but never keeps it (the point ends
before another legal hit) and nets zero.

**Ordering pins** (the reach §2a lesson, applied at design time):

- A side-A legal hit on the same step both *keeps* the prior pending
  escrow (keep runs first) and *arms* the next window (arm runs
  second). A payment coexisting with a side-A legal hit on one step
  is kept immediately, never escrowed — physically the B strike and
  the A follow-up hit are separated by a court crossing, but the
  edge is pinned rather than assumed, with a unit test.
- Travel accumulates from the arming step's end position, so swing
  follow-through counts against the budget; `hold_shaping_travel`'s
  default absorbs it (§2a below).
- A side-B racket hit with **no armed window** (the opponent
  returning a serve or feed) pays nothing: the escrow is strictly
  post-swing.
- The n-point relaunch teleport never crosses a window: boundaries
  disarm before the next point launches, so serve placement cannot
  leak into travel.

New reward components `rew_hold` / `rew_hold_clawback` extend the
decomposition to nine components in `info` and the CSV schema, all
exact zeros when the kwarg is off.

### 2a. Why travel, and why 4.0 m

The payment variable is accumulated **path length**, not
displacement or distance-to-an-anchor: displacement forgives orbits;
an anchor point presumes a "correct" ready position the design has
no basis to freeze (the reach escrow already prices where the policy
ends up, at bounce time — this escrow prices the flailing itself,
which is what the instrument measures as recovery-hold travel and
what the failed channel exhibits at 7.5 m). Follow-through
(~1–1.5 m of carry inside the window) plus legitimate repositioning
(~1–2 m) lands a disciplined exchange near 2–3 m of travel, paying
0.25–0.5 of scale; the measured wander (≥ 7.5 m instrument-metric,
more with follow-through included) pays zero. `4.0` is that
boundary, frozen from the run's band before any probe.

On the k=2 hit the full ladder now pays: kept hold (≤ 0.25) + reach
at the return's bounce (≤ 0.25) + contact at the hit (0.25) + the
confirmed +1 — every segment of the second exchange carries its own
gradient, each farm-proof under its own keep rule.

## 3. Certification battery (pre-registered; fresh seed block
6200–6299, burned by this battery)

- **PH0 — bit-identity.** Default-off lockstep: `hold_shaping=0.0`
  arm against the pre-change definition, exact observation/reward/
  info equality (the family's standing bar), plus the shaped arm's
  non-hold components identical to the unshaped arm's.
- **PH1 — witness economics** (`tools/paddle_tennis_hold_probe.py`,
  100 episodes × lockstep shaped/unshaped arms, seeds 6200+):
  - `statue` and `volley_patting`: zero hold pay, zero clawback —
    no window ever arms (statue) or survives legally (patting);
  - `camper`: zero everywhere (no hit, no window);
  - a `hit_then_freeze` witness (the farming attempt this design
    must defeat): collects payments, keeps none, **nets exactly
    zero**;
  - `ground_oracle`: collects **and keeps** (the only witness that
    completes second exchanges) — kept pay strictly positive;
  - `hard_slam` and the stacked arm (`hold+reach+contact`): exact
    per-episode sum identities; component streams bit-equal to the
    single-shaping arms.
- **PH2 — ending-path invariants** (`TestHoldShaping` unit tests):
  clawback at episode endings, truncation, n-point boundaries, and
  the non-finite guard; window disarm at boundaries and reset;
  the same-step keep pin; kwarg validation; travel accounting
  against hand-computed paddle paths; stacking identity.

Adoption of any recipe default waits, as always, on a pilot verdict
— nothing in this document changes the frozen task or the recipe.

### 3a. Battery results (recorded 2026-08-17; **PH0–PH2 PASS**)

PH0/PH2 via `TestHoldShaping` (11 tests: lockstep default
bit-identity, formula-exact payments against an externally
re-accumulated travel, every clawback path, boundary disarm,
the same-step keep pin, validation) — all green. PH1
(`tools/paddle_tennis_hold_probe.py`, 100 episodes, seeds
6200–6299 now burned):

| witness | paid | clawed | kept | eps paid | eps kept |
|---|---|---|---|---|---|
| statue | 0.000 | 0.000 | 0.000 | 0/100 | 0/100 |
| camper | 0.000 | 0.000 | 0.000 | 0/100 | 0/100 |
| hit_then_freeze | 2.577 | −2.577 | **0.000** | 41/100 | 0/100 |
| ground_oracle | 21.256 | −4.001 | **17.256** | 89/100 | 83/100 |
| hard_slam | 0.023 | 0.000 | 0.023 | 1/100 | 1/100 |
| volley_patting | 0.000 | 0.000 | 0.000 | 0/100 | 0/100 |
| oracle_stacked | 21.256 | −4.001 | 17.256 | 89/100 | 83/100 |

Worst per-episode identity gap 1.11e-16 (float ulp); every payment
coincided with an opponent return strike; the stacked arm composed
the three escrows exactly. The economics read precisely as designed:
the farming witness collects and keeps nothing; the only witness
that completes second exchanges keeps 81% of what it is paid; the
slam witness's 101 hits earned a single kept payment (its shots are
almost never returnable — hold pay follows genuine rally
continuation, nothing else). The §4 pilot is cleared to launch.

## 4. LH1 — the pilot (pre-registered; frozen before launch)

**Shape.** Warm start from the registered run's protected best
(`training_runs/PaddleTennis/sac/20260816_235141`, best checkpoint
step 2,400,000, `best_model.zip` sha256 `838997fb…`,
`best_vec_normalize.pkl` sha256 `d0502c14…` — the loader validates
the pairing), **seed 0** (pilot convention), **1,000,000 steps**,
n_envs 4, eval 25k, checkpoint/diagnosis 100k. The single change
against the source run's task is the hold escrow. Run config, frozen
verbatim (Drive-side copy as `paddle_tennis_lh1_hold.toml`, sha
recorded by provenance as usual):

```toml
# LH1 — post-swing-hold pilot (docs/design_paddle_tennis_postswing_hold.md §4).
# Code-side pairing: seed 0, total_timesteps 1_000_000, warm start from
# training_runs/PaddleTennis/sac/20260816_235141.

[env]
hold_shaping = 0.25
hold_shaping_travel = 4.0

[train]
n_envs = 4
eval_freq = 25_000
checkpoint_freq = 100_000

[train.model_kwargs]
learning_starts = 25_000
```

**Bars (frozen; anchors: the registered run's measured band).**

| criterion | metric | FAIL | declared middle | PASS |
|---|---|---|---|---|
| KH1 (headline) | k=2 exchange survival, either parity | ≤ 1% at every checkpoint (no advance over the registered band) | one checkpoint at 2% | ≥ 3% at some checkpoint |
| H1 (mechanism) | recovery-hold travel, instrument metric | never < 6.0 m (the registered band's floor was 4.4–7.9 m after 1M) | (3.5, 6.0) everywhere | ≤ 3.5 m at some checkpoint |
| R1 (retention) | k=1 receiving survival | < 60% at every checkpoint (the transferred stroke broke) | [60%, 80%) everywhere | ≥ 80% at some checkpoint |
| guards | recipe defaults | an abort books KH1 FAIL directly | — | — |

**Decision rule.** KH1 PASS ∧ H1 PASS → adopt the hold escrow as a
recipe default (its own adoption commit + pin tests) and pre-register
the era's RK1 retry. KH1 FAIL with H1 PASS → the hold behavior
trained but did not unlock k=2: the block is elsewhere (opponent
return distribution, timing) — book it and re-diagnose before any
further reward change. H1 FAIL → the escrow could not beat the
wander's attractor at 0.25: one re-pair at `hold_shaping = 0.5` is
sanctioned (the same single-knob re-pair convention as the
exploration era), its verdict re-applying this rule without that
branch. Middles → maintainer's call, documented post-hoc.

**Launch checklist (notebook deltas).** `REPO_REF` = the
implementation commit on main; `ENV = "PaddleTennis"`, `SEED = 0`,
`TOTAL_TIMESTEPS = 1_000_000`, `CONFIG_FILE` = the TOML above; §5
config-build cell adds
`overrides["warm_start"] = WarmStartConfig(source_run_dir=".../20260816_235141")`
(the same cell both prior warm-started pilots used). Validate the
run's `config.json` against this section before the first checkpoint.

## 5. Seed ledger

**6200–6299 burned** by the PH1 battery (this document's only new
block). Training seed 0 per the pilot convention; diagnosis stays on
calibration 5200+. **4100–4199 remains sealed** — the registered
run's stop/amend booking did not open it, and it stays reserved for
the first run whose registered-result branch fires.
