# Diagnosis: PT2 — the command spectrum and the k=2 interception gap

Kind: **Diagnosis snapshot (PT2)**, 2026-08-30. Subjects replayed at
`main`@`a855fcc` (the PR-75 merge); analysis re-derived independently
twice before booking. Commissioned by the maintainer as a
pre-implementation falsification check in front of
[`design_paddle_tennis_command_rate.md`](design_paddle_tennis_command_rate.md)'s
CR0 (session-level decision, 2026-08-29/30; no separate
commissioning document) — the frozen design's §2 mechanism ("stillness by default")
rested on an *assumed* command spectrum and an *unmeasured* k=2
geometry, both measurable from recorded streams for CPU pennies.
Diagnosis-side only: no bars, no verdict — but §6 states plainly
which frozen premise the numbers contradict, and §7 routes.
The disposition of the frozen design lives in that document's §7 —
**booked by the maintainer 2026-08-30: CLOSED without
implementation.**

## 1. Data and provenance

One Colab CPU session (2026-08-30, ~23 min of replay) captured
per-step streams for four subjects — 100 deterministic episodes
each, calibration seeds 5200–5299, the PT1 instrument's exact env
shape (`episode_len=1500`, diagnostic court, `volley_rule="fault"`,
contact/reach 0.25, reach radius 3.0, n-point, hold off) and loader
(`native_checkpoint_policy`; the scripted ground oracle for the
oracle row):

| subject | artifacts | model sha |
|---|---|---|
| `registered_2p4M` | `20260816_235141` protected best | `838997fb…` / `d0502c14…` — **pins verified at capture** |
| `lh1c_2p325M` | `20260821_013700` crowned best | printed at capture |
| `lt1_1M` | `20260828_121324` 1M checkpoint | printed at capture |
| `oracle` | scripted ground opponent | — |

Per step: raw action, side-A `data.ctrl` (the exact stream the
frozen limiter would clamp — post-fence), slide qpos, ball
position, valid-hit flags, `points_played`, plus every rules event
(`processed_events`: kind + position — hits, court bounces,
crossings). Drive-side:
`diagnostics/pt1_ctrl_streams_20260829/` (streams `.npz` + events
`.json` per subject + `MANIFEST.json`; the folder's `pt1_`/`0829`
name is a naming artifact of the capture cell — the contents are
this probe's, captured 2026-08-30). Local analysis copies were
byte-size-verified against Drive; sha256 prefixes of the analyzed
copies: registered `4fdd7ac2…`/`e96d309d…`, lh1c
`5ce6500c…`/`655fdec6…`, lt1 `ff2793ca…`/`378ddb40…`, oracle
`e04555f6…`/`ffc7333c…`. Analysis tool:
[`tools/paddle_tennis_command_spectrum_analysis.py`](../tools/paddle_tennis_command_spectrum_analysis.py)
(windowing mirrors the PT1 probe: open at the side-A valid hit,
close at opponent strike / point boundary / episode end / 300-step
cap, post-follow-through slice at 30 steps).

**Instrument fidelity check:** the oracle's strike-ended row
reproduces PT1 §4's published numbers **digit-exact** (n=164 windows
over the first 30 seeds, cmd 2.56 m / act 2.55 m). The trained
checkpoints' Colab replays do *not* bit-reproduce PT1's local rows
(registered, same seed subset: cmd 27.5 vs 31.04 m, saturation
90.9% vs 87.6% — a torch-build inference divergence; the scripted oracle is pure
numpy, hence exact). Qualitatively the rows are the same policy;
every conclusion below is distributional, not digit-level.

## 2. F1 — the thrash is jump-then-dwell, not per-step alternation

The frozen Δ = 0.15 candidate was priced against "alternating
saturated commands collaps[ing] to a ±0.15 m target dither"
(design §3). The measured per-step XY command delta inside
post-swing windows is nothing like that — it is heavy-tailed
**jump-then-dwell**:

| subject | p50 | p90 | p99 | max | steps ≤ 0.01 m | steps > 1.0 m |
|---|---|---|---|---|---|---|
| registered 2.4M | 0.033 | 0.76 | 5.32 | 8.39 | 44% | 7.1% |
| LH1c 2.325M | 0.003 | 0.51 | 4.96 | 8.39 | 52% | 4.7% |
| LT1 1M | 0.000 | 0.65 | 3.37 | 8.39 | 55% | 6.1% |
| oracle | 0.000 | 0.00 | 1.76 | 4.58 | 98.8% | 1.2% |

Half the steps the command barely moves; the 0.26–0.38 m/step
*mean* saccade is carried by rare multi-meter jumps. The 88–93%
saturation and the near-plant-ceiling actual travel are consistent
with exactly this: the head parks at rails and relocates in bursts.

## 3. F2 — no evidence Δ = 0.15 bites; real bite begins sub-plant

Applying the design's D1 update rule open-loop to the recorded
streams (per-axis, state seeded from qpos, persisting across the
episode), with a per-axis 0.125 m/step rate follower as the plant
proxy on both arms:

| Δ (m/step) | registered | LH1c | LT1 | oracle |
|---|---|---|---|---|
| 0.15 (frozen candidate) | 100% | 100% | 100% | 100% |
| 0.10 | 87% | 88% | 90% | 99% |
| 0.05 | 54% | 56% | 62% | 88% |
| 0.03 | 36% | 39% | 42% | 69% |

(simulated ON-arm window travel as % of the OFF arm)

**Honest instrument limit, established by the independent
re-derivation:** at Δ ≥ the 0.125 plant rate the proxy is
rate-saturated on ~50% of axis-steps in *both* arms, so ratio ≈ 1
is partly baked in — the open-loop check cannot *prove* Δ = 0.15
is transparent to the real second-order servo (and the proxy
overestimates recorded travel by +22–31% in the thrash regime vs
+0.5% for the oracle). What it does establish: the effective
command at Δ = 0.15 still slews at ≥ the plant rate on 47% of
per-axis window steps (38–42% for the other checkpoints; tool
output), so the limiter mostly re-times
jumps the plant already low-passes; **no mechanism of paddle-travel
reduction is in evidence at the frozen candidate**, and material
reduction appears only at sub-plant Δ — where **F4** bites. The
mean-bias hazard is also real: at Δ = 0.15 the effective target
*walks* 3.1–3.3 m net per window (smooth drift, not stillness).

## 4. F3 — the decisive finding: k=2 is an interception gap stillness cannot close

From the event streams, a **k=2 opportunity** = own valid hit →
opponent's valid strike (same point) → the return's first side-A
bounce (same point; the points-counter guard makes every selected
bounce in-bounds). Distances are paddle→bounce at bounce time:

| subject | opps (bounce-cond.) | paddle→bounce mean | ≤ 1.0 m | converts |
|---|---|---|---|---|
| oracle | 445 | **0.89 m** | **99.8%** | **98.2%** (437) |
| registered 2.4M | 134 | 3.25 m | 7.5% | 1.5% (2) |
| LH1c 2.325M | 135 | 2.95 m | 9.6% | 0.7% (1) |
| LT1 1M | 37 | 2.64 m | 13.5% | 0% |

The opportunities are plentiful (~1.3–1.7 per episode for the best
checkpoints) and the ball is interceptable — the oracle converts
98% from 0.89 m. The trained paddle is ~3 m away and converts ~1%.

**The counterfactuals, which falsify the stillness mechanism:**

- **Frozen-at-hit** (paddle parked exactly where it struck — the
  zero-motion end-member): registered 2.59 m mean, **≤ 1 m only
  6.0%**; LH1c 2.79 m / 1.5%; LT1 2.42 m / 2.7%. *No better than
  the thrashing paddle* — on the decisive within-1-m rate, worse.
- **Frozen at hit+30** (the design exempts a 30-step follow-through,
  during which the trained paddles travel another ~3 m
  [verification-pass value] — the faithful anchor for a limited
  paddle): registered **3.7%**, LH1c
  3.0%, LT1 13.5% ≤ 1 m. The corrected counterfactual is *further*
  from closing the gap.
- **Parked-at-home**: 0.7–8.1% ≤ 1 m. Also no.
- **The oracle's own frozen-at-hit counterfactual collapses from
  99.8% to 38.7% ≤ 1 m** (strict k=2 windows — the point's first
  own hit; n=150, tool output): the oracle's second-exchange
  success is not stillness — it is **directed recovery toward where
  the return will land** (every hit-to-bounce leg exceeds 176 steps
  [verification-pass value]: there is ample time to move, and the
  oracle uses it).

Framing note, per the verification pass: a rate limiter constrains
*rate*, not *direction* — Δ = 0.15 permits ~15 m of travel over the
strike→bounce leg, so a policy whose commands *aimed at the bounce*
could reach it. The binding premise is PT1's measured fact that
these commands track no attractor; given that, every realizable
limiter lands between the actual (7.5–13.5% ≤ 1 m) and frozen
(1.5–6%; 3.0–13.5% at the +30 anchor) anchors. **No point on that
interpolation approaches the oracle's 99.8%. The mechanism's k=2
payoff ceiling is approximately zero at any Δ.**

## 5. F4 — the oracle's strike tail bounds Δ from below (CR1)

The oracle's strike demands — unpublished until now (PT1 gave only
its 0.033 mean and 1.8–2.1% saturated steps): per-step target jumps
of **p99 1.76 m, max 4.58 m**. Under the proxy, sub-plant Δ ramps
those jumps and cuts the oracle's window travel to 88% (Δ = 0.05)
and 69% (Δ = 0.03) of its unlimited arm — strike-onset delay of
tens of steps on multi-meter relocations. So the Δ small enough to
still a thrashing head (≤ 0.05) is the Δ that impairs the one
certified-competent player: **the window between "changes the
thrash" and "breaks the strike" is likely empty** for a uniform
always-on per-axis rate limit — and, per F3, even a non-empty
window would buy no k=2.

## 6. What this contradicts, and how it was verified

The frozen design's premise (§1, mechanized by §2) — "stillness
becomes the head's default output … deliberate motion passes
through", paying k=2 via the certified escrow ladder (this probe's
paraphrase of the §1 mechanism) — is contradicted twice over: at the
frozen Δ no travel-reduction mechanism is in evidence (F1/F2), and
at *any* Δ the best achievable geometry leaves the paddle ~2.5–3.3 m
from the return with a ≤ 6% within-reach rate (F3), against an
oracle that wins by moving, not by holding still. The CR2 thrash
witness (symmetric per-step ±1) is the one spectrum the limiter
trivially defeats — it could not have caught F1/F2, and no battery
stage measures F3.

Verification: every headline number was re-derived by two
independent passes writing their own analysis code against the raw
npz/events files. The geometry (F3) was **CONFIRMED** exactly
(every overlapping figure matched), with two conservative
corrections adopted here (the +30 anchor; the 98.2%
bounce-conditioned oracle conversion in place of a mixed-denominator
80%). The transparency reading (F2) was graded **PARTIALLY**: the
spectrum, sub-plant thresholds, and oracle tail confirmed; the
"transparent at 0.15" headline downgraded to "no evidence of bite,
undecidable open-loop" for the proxy-saturation reason stated in §3.

## 7. Routing (diagnosis-side; the disposition decision is the maintainer's)

- **The measured k=2 blocker is directed interception of the
  return-ball class** — getting the paddle from ~3 m to ~1 m of the
  return's bounce, a skill the policy has for serves (k=1 90% on
  the same obs machinery) and has never expressed for returns. Any
  next lever should be scored against that number.
- The lever classes assessed directly behind the limiter in the
  same session-level pre-implementation review (unbooked) map onto
  exactly this blocker, now better targeted:
  **oracle-demonstration / replay-buffer injection of k=2
  transitions** (the buffer has carried essentially zero
  second-exchange experience in 15M+ steps; training-side, no era
  break; needs its own design + battery) and the **k=2 drill
  curriculum** (reset from harvested post-swing states with the
  incoming return; a task-distribution change with the repo's known
  curriculum caveats). This snapshot ranks neither — that is a
  design decision with its own document.
- **Instrument follow-up** (cheap, era-neutral, recommended
  regardless): split the diagnosis instrument's ready-position /
  touched-after-bounce metrics by ball class (incoming feed vs
  opponent return, by exchange index) — this probe had to derive
  the k=2 geometry from raw events because the standing instrument
  is feed-scoped; the split makes F3's number a standing per-run
  observable.
- Seed ledger: calibration 5200–5299 used per the diagnosis
  convention; **no block consumed, nothing new burned; 4100–4199
  stays sealed.** (The CR2 block 6400–6499 assigned at the design's
  freeze has not been used; its disposition follows the design's.)
