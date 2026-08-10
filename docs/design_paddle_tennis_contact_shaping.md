# Design: escrowed contact shaping — paying the touch→in gap

Status: **Implemented; S1 + S2 PASS; L1 complete — the declared
middle (D2′ peak 46%), one budget extension recommended**,
2026-08-09 (§4, §5). The remedy the
exploration-pilot verdict points at
([`paddle_tennis_exploration_20260808.md`](paddle_tennis_exploration_20260808.md)
§3): the pilot proved exploration reaches the ball (27% serving-side
k=1 at its peak) and that reaching pays nothing — every crossed shot
from 300k on landed 9.2–16.0 m deep (2.7–9.5 m past the 6.5 m
baseline), the critic sat converged at ~1e-4 for 900k steps, and
engagement washed away as fast as it formed. The +1 sits on "legal
return that lands in"; everything between touch and in is a flat −1.
This design pays the gap without making the gap farmable.

## 1. Mechanism — the humanoid escrow, ported with its audit

Provenance, stated honestly: the repo **measured the problem** this
pattern answers (humanoid M3: 30/30 episodes at exactly −1.000
reward; the stage oracle got 12/20 hits and 0/20 target returns —
hit-vs-aim entirely unrewarded) and **designed and
accounting-audited the answer** — `valid_hit_shaping=0.25` shipped
in the humanoid Stage 1–2 recipes in 0.16.0, escrow audited for
double-pay paths. **No training run ever exercised it**: the repo
pivoted to WallBall before a humanoid campaign ran. The escrow is
design precedent with a verified accounting core, not a measured
training result; its first learning evidence will be this design's
own L1 pilot. Mechanics (from `humanoid_tennis.py`):

- on a `valid_racket_hit` by a side: pay `+shaping` **now** and
  record it as pending for that side;
- when that side's return **confirms**: clear its pending (the
  advance is kept);
- at episode end — on **every** ending path (termination,
  truncation, and both nonfinite guards; see §2): claw back
  whatever is still pending (`rew_shaping_clawback = −pending`).

The sum over any episode is therefore exactly
`shaping × (confirmed policy returns)` — an unconfirmed hit
contributes **zero** to the undiscounted total, so shaping cannot be
farmed. What it changes is the *contact-time* value: the clawback
lands when the point ends, so the advance surviving at the moment of
the hit is `shaping × (1 − γ^d)` with `d` the point's remaining
length. For a failed shot, `d` is roughly that shot's own remaining
flight — tens of steps — so at γ=0.99 the surviving advance is
≈ 26–45% of the shaping for typical 30–60-step failure flights, and
grows toward 100% only for shots that keep the point alive long.
The gradient is real but modest for the earliest failure modes, and
— usefully — a miss that keeps the ball in play longer pays more
than a miss that ends the point at once. "Hit then miss" and "never
hit" finally have different Q-values, which is precisely the
gradient the pilot showed is missing — while "hit then confirm"
keeps the full advance on top of the +1.

## 2. PaddleTennis specifics

- **Policy-side only (side A).** This deliberately deviates from
  the shared cooperative `+1`: the ground-era diagnosis measured
  the shared stream as the dilution mechanism (serving episodes pay
  regardless of the policy's play). The shaping's entire purpose is
  policy-own credit; opponent hits open no escrow and trigger no
  shaping. The frozen cooperative `+1` for either side's confirmed
  return is untouched. Consequence: a confirmed policy return
  totals `1 + shaping`, an opponent's totals `1` — an intentional,
  documented asymmetry. (Side A is the policy by the frozen P4
  mirror contract, so "side A" and "policy" coincide by
  construction.)
- **Clawback on every ending path — the implementation contract.**
  Unlike the humanoid env's single reward path, `PaddleTennisEnv`
  assembles reward in **three** places: the main step reward
  (currently computed *before* truncation is resolved), the
  forced-nonfinite branch, and the nonfinite guard's early return
  (a NaN action or physics state, reachable mid-point from a
  blown-up policy). The implementation must make the clawback fire
  and the decomposition identity hold on **all three** — including
  a NaN action arriving one step after a hit opened an escrow —
  which requires resolving truncation before the reward is final
  and adding the shaping components to both guard paths. S2
  witnesses each path explicitly. (Without this, an episode ending
  through a guard with a pending escrow keeps the advance and the
  §1 sum identity silently breaks — not profitably exploitable,
  since the escrow is far smaller than the −2 unsafe penalty, but
  false as specified.)
- **The rules stack guards the contact-side exploits.** Verified
  in `tennis_rules.py::_handle_racket_contact`: every fault path
  (WRONG_HITTER, DOUBLE_HIT, PREMATURE_HIT, and the ground-rules
  VOLLEY_RETURN branch) returns before the hit is recorded, so a
  faulting contact can never appear in `valid_racket_hits` and can
  never open an escrow. Strict double-hit means at most **one**
  pending escrow at a time, and every point ends within the
  1500-step cap, so the discount-level gain from deliberately
  unconfirmed hits is bounded at **< 1 × shaping per point** (a
  delay-maximizing shot banks at most the full advance once, then
  the point is over — and a whole point spent doing that forfeits
  the +1s that dominate it).
- **Magnitude: 0.25** — the audited humanoid design value (see §1:
  precedent, not a measurement), carried by the budget analysis: at
  most one escrow per point and `0.25 × confirms` per episode in
  totals — a 25% bonus on the policy's own confirmed returns,
  strictly dominated by the task reward. Not a per-step stream; the
  `ent_coef × episode_len` poison shape does not apply. The L1
  pilot is the value's first measurement.
- **Plumbing.** New env kwarg `contact_shaping: float = 0.0`
  (validated finite non-negative), **default off — the frozen task
  definition is unchanged** until the probe passes. New reward
  components `rew_shaping` and `rew_shaping_clawback` join the
  decomposition identity (`reward == rew_return + rew_fault +
  rew_unsafe + rew_shaping + rew_shaping_clawback` on every step,
  every path), the info dict, and the recipe CSV header.
  `config.json` records the kwarg via constructor provenance
  automatically.
- **Comparability.** Enabling shaping starts a new
  reward-comparability era: learning curves and eval rewards do not
  compare across the boundary. Everything behavioral is unaffected:
  `crossings` metrics, the scripted band (7.78), and the held-out
  certification (7.68 on 4200–4299) are reward-independent and
  remain valid. Selection already follows `crossings`, never eval
  reward.

## 3. Pre-registered probe battery (before the recipe ships it)

Calibration seed block **5300–5399 is reserved for S1** (fresh
block; 5200+ stays the diagnosis block; reserved 4100–4199 is not
touched).

- **S1 — incentive-ordering witnesses** (scripted, no learning;
  100 episodes each on 5300–5399, shaping 0.25 vs 0.0). Because
  shaping is reward-side only, the shaped and unshaped arms of the
  same seed produce **bit-identical trajectories** — so every
  criterion is an exact per-seed identity, not a statistical band:
  - *witness-validity precondition:* the **hard-slam witness**
    (ground oracle driven with the 0.4 swing — measured on the
    bring-up calibration grid to land mean 11.0 m from the net,
    ~4.5 m past the baseline, every stroke long; that grid ran
    from a fixed deep park, not these serve draws) must touch the
    ball in ≥ 50% of episodes and land the majority of its strokes
    out on this block. If not, recalibrate the witness before
    reading the identities.
  - *statue* (zero action): exactly 0 shaping paid, 0 clawed back.
  - *hard-slam*: per episode,
    `rew_shaping + rew_shaping_clawback == 0.25 × side-A confirmed
    returns` exactly, and the per-seed shaped-vs-unshaped total
    reward difference equals the same quantity exactly.
  - *ground oracle* (touch-and-in): per-seed total reward rises by
    exactly `0.25 × its side-A confirmed-return count`.
  - *volley-patting witness* (the ground-rules exploit
    controller): exactly 0 shaping paid — fault contacts open no
    escrow, witnessed at the reward level.
- **S2 — decomposition invariant on every ending path**: the
  existing every-step `reward == Σ rew_*` test extends over the two
  new components, with explicit cases for (a) forced truncation
  with a pending escrow (clawback fires), (b) a confirmed hit then
  cap truncation (advance kept), (c) **a NaN action arriving with
  an escrow pending** (the early-return guard claws back alongside
  the −2), and (d) **a forced-nonfinite ending with an escrow
  pending** (same).
- **L1 — learning pilot** (only if S1/S2 pass): recipe + shaping
  0.25, the pilot convention (seed 0, 1M steps, n_envs 4,
  checkpoint/diagnosis cadence 100k). Criteria, frozen here:
  - **P′: best `crossings_ep_mean` ≥ 2.5** by 1M (unchanged bar).
  - **D2′: touched-after-bounce ≥ 55%** at some checkpoint
    (exploration pilot peak: 17%).
  - **N1: at least one checkpoint ≥ 500k whose 30-episode
    diagnosis row contains ≥ 1 landed-in policy shot** (the
    exploration pilot had zero landed-in policy shots after 200k).
  - **M: mechanism intact** (same two-part check as the
    exploration doc §2 — package keys in artifacts, `train/std`
    ≥ 5e-3).
  - Decision rule, frozen: (1) P′ ∧ D2′ ∧ N1 → pre-register the
    ground-era GPU run (held-out gate on 4100–4199) and hand it to
    Colab. (2) D2′ ≥ 55% ∧ N1 fails → the gap is aim, not credit:
    probe strike-height/velocity control (the P0-measured loft
    channel) before touching rewards again. (3) D2′ < 30% →
    contact shaping alone does not buy reach at this budget:
    combine with n-point episodes (its own probe, new era) before
    re-piloting. **Declared non-forcing middles** (the exploration
    doc's precedent, this time named in advance): D2′ ∈ [30%, 55%)
    — partial reach recovery — and the success-adjacent
    D2′ ∧ N1 with P′ failing — credit repaired, rally length
    lagging. Neither forces a branch; the post-hoc analysis must
    be labeled as such, with the default lean: if the diagnosis
    rows are still improving at 1M, extend the budget once before
    any new change; if they have flattened, combine with n-point
    episodes. In every branch the exploration package stays.

## 4. S1/S2 results — PASS

S1 ran as pre-registered (`tools/paddle_tennis_shaping_probe.py`,
100 episodes per witness per arm on seeds 5300–5399, shaping 0.25
vs 0.0). **Every criterion passed**, with every identity exact to
0.00e+00 and all arms bit-identical:

| witness | hits | confirms | paid | clawed | mean total reward |
|---|---|---|---|---|---|
| statue | 0 | 0 | 0.00 | 0.00 | −0.700 |
| hard-slam | 96 | 1 | 24.00 | −23.75 | −0.517 |
| ground oracle | 393 | 346 | 98.25 | −11.75 | 7.035 |
| volley-patting | 0 | 0 | 0.00 | 0.00 | −1.000 |

- Witness-validity preconditions held: the hard-slam touched in 95%
  of episodes and confirmed 1 of 96 hits — the designed
  touch-then-out prototype, matching its bring-up character.
- The escrow's whole undiscounted effect is exactly
  `0.25 × side-A confirms` for every witness on every seed (the
  hard-slam banked exactly its one confirm; nothing else).
- Fault contacts opened no escrow (patting: 96→0 — zero paid),
  witnessing the exploit seal at the reward level.
- Note on operationalization: the §3 precondition wording is
  measured through two decidable event-level proxies, recorded here
  as deviations-by-refinement. *Touch* is counted as at least one
  **legal** hit per episode (`event_valid_racket_hit_a`; a faulting
  graze counts as no touch — conservative for the precondition).
  *Majority of strokes out* is counted as *majority of legal hits
  unconfirmed* — a strict superset of landed-out that also covers
  net faults and shots still in flight at the cap. On this block the
  hard-slam's misses in fact terminated `out_of_bounds`, so the
  proxy and the literal wording agree on the S1 verdict.

S2 passed as `TestContactShaping` in `tests/test_paddle_tennis.py`:
the every-step decomposition identity over the five components, the
shaped/unshaped bit-identity (lockstep-stepped, observations
asserted equal), and clawback on all four pre-registered ending
cases — truncation-with-pending, confirmed-then-capped (via the
escrow-identity drive), the NaN-action guard, and the
forced-nonfinite branch.

Seed ledger: block **5300–5399 burned** (S1 calibration).
Reserved **4100–4199 remains untouched**. Next: the L1 learning
pilot per §3 (recipe + `contact_shaping=0.25` via `[env]` override,
seed 0, 1M steps, n_envs 4, checkpoint cadence 100k).

## 5. L1 results — the declared middle, with the mechanism visibly working

Run `20260809_161704` (Colab L4, commit `a29883c`, TOML sha
`4c1376ae`, 1M steps in 4 h 04 m at 68 FPS). Criteria:

| criterion | outcome | evidence |
|---|---|---|
| M | **pass** | package + shaping verified in `config.json`/resolved model; `train/std` 0.036 final; the critic is *alive* at 3.0e-4 — three times the exploration pilot's dead floor, the shaping stream finally gave it something to predict |
| N1 (landed-in ≥ 500k) | **pass** | 8 landed-in policy shots per 30-episode row at 500k **and every later row**; receiving strokes 50–64% in at 3.3–4.7 m (oracle: ~100% at 3.9) |
| D2′ (touch ≥ 55%) | **fail — in the declared middle** | peak 46% (800k); 35–46% sustained from 200k |
| P′ (crossings ≥ 2.5) | **fail** | best eval 1.23 (850k), final 1.07 |

Checkpoint rows (30 probe episodes each; 600k row in the run's
`reports/diagnosis/`):

| ckpt | hits | rec k=1 | serv k=1 | touch | landed-in | crossings |
|---|---|---|---|---|---|---|
| 100k | 2 | 0% | 13% | 7% | 0 | 0.50 |
| 200k | 13 | 60% | 27% | 42% | 1 | 0.77 |
| 300k | 9 | 33% | 27% | 29% | 2 | 0.80 |
| 400k | 11 | 53% | 20% | 35% | 1 | 0.77 |
| 500k | 16 | 80% | 27% | 42% | 8 | 1.23 |
| 700k | 14 | 73% | 20% | 39% | 7 | 1.00 |
| 800k | 16 | 87% | 20% | 46% | 8 | 0.93 |
| 900k | 16 | 87% | 20% | 42% | 8 | 1.20 |
| 1M | 16 | 93% | 13% | 43% | 8 | 1.23 |

Against the exploration pilot: touch 46% peak vs 17; landed-in 8/row
vs zero after 200k; receiving k=1 93% vs 13% peak; crossings 1.23 vs
0.67. Against the stock run's 1.37: that number came from one
memorized macro the diagnosis rejected; this 1.23 is real strokes
the opponent must actually play — including the first
`opponent_never_reached` enders in any run (1–2 per row from 800k):
the policy now occasionally hits shots the oracle cannot reach.

**The next bottleneck, measured:** k=2 survival is 0% in every row
of every run so far. The policy's stroke is now good (93% k=1
receiving) but it never survives the opponent's reply — and the
instrument says why: recovery-hold travel *worsened* as the stroke
improved (6.4 → 8.2 m; oracle 2.2), i.e. the policy still wanders
after swinging instead of re-readying, so ball 2 lands unreachable.
Serving-side k=1 also decayed (27% → 13%) as receiving specialized.

**Decision (post-hoc analysis, labeled as such):** D2′ peaked at 46%
— inside the declared non-forcing middle `[30%, 55%)`. The frozen
lean applies: the rows were still improving at 1M (receiving k=1
53 → 93 over the back half; landed-ins steady at 8), so **extend the
budget once before any new change** — a single 2M-step run of the
*identical* configuration (same TOML, seed 0, n_envs 4, checkpoint
cadence 100k). k=2 exposure only began once k=1 became reliable
(~500k), so the extension gives the recovery behavior its first real
training window. If the extension flattens with k=2 still at 0%, the
named follow-up is n-point episodes (its own probe, new era) — the
"once" in the lean is binding; budget does not get extended twice.

Seed ledger: unchanged (run on seed 0; diagnosis calibration 5200+;
4100–4199 untouched).

### The budget extension (run `20260809_211147`) — k=1 mastered on both sides; k=2 exactly 0%

The frozen lean's single extension: identical configuration, 2M
steps (commit `b57c3d2`, same TOML sha, 8 h 17 m, 67 FPS). The
committed branch point **fires**: k=2 survival stayed at exactly 0%
in all twenty diagnosis rows, so the follow-up is **n-point
episodes** — no second extension.

What the doubled budget bought first, though, is close to complete
k=1 mastery (2M row, 30 probe episodes):

- policy hits 23/30 episodes; **96% crossed, 78% in, at 3.73 m** —
  oracle-grade depth (oracle: 3.9);
- **receiving k=1 survival 100%**; **serving k=1 53%** — the
  serving-side decay (27→13% in L1) fully reversed with budget;
- touch 50% (the D2′ middle's upper edge, still climbing at 2M);
- best eval crossings **1.77 at 1,975,000 steps** — still improving
  when the budget ended; final 1.50;
- the opponent now sometimes loses the exchange
  (`opponent_shot_net` 2 in the final row).

And the barrier, sharper than ever: **recovery-hold travel 8.33 m**
(worse than L1's 8.2; oracle 2.2). The policy has never sampled a
k=2 legal return in 3M cumulative steps of this configuration — it
swings well and is then 8 m out of position when the reply arrives,
so the escrow's k=2 payment (which is sitting right there in the
reward) is never collected. The same credit-starvation shape as the
original H1, one rung up the ladder.

**Design requirement this hands the n-point probe:** the wander is
unpaid because nothing after the swing matters until the ball
returns. n-point episodes fix this *only if paddle positions carry
over between points* (no re-park between points): then post-swing
wander in point k directly costs point k+1's serve-return — a +1.25
the policy already knows how to collect at 100%. Carryover turns
the recovery problem from never-sampled k=2 credit into
densely-sampled inter-point credit, and it is therefore the load-
bearing design decision the n-point probe must freeze first.

## 6. What this is not

Not a change to the frozen task semantics (default off; rules,
serve, observations, termination untouched), not the own-credit
restructuring of the shared `+1` (diagnosis §3's #3 — stays behind
this), and not a paddle-pitch actuation change (H2 remains rejected;
only L1 branch 2 could reopen that question, with direct evidence).
