# PT1 — post-swing action targets: the wander is commanded thrash

Diagnosis snapshot, 2026-08-22. Instrument:
`tools/paddle_tennis_postswing_target_probe.py` (commit `78eb34c`).
No bars, no verdict — this probe exists to route the design after the
hold-escrow closure (`design_paddle_tennis_postswing_hold.md` §4c).

## 1. The question

The campaign's k=2 blocker is the 7.5+ m the paddle travels after its
own legal hit — measured by the diagnosis instrument on every run,
unmoved by a reward that provably paid for holding (LH1c). §4c closed
the reward-side line on the question this probe answers: is the
wander **commanded** (the policy's position targets themselves move
away) or **drift** (targets hold; the paddle moves anyway)? The two
answers route to different sides of the stack — action/observation
design versus dynamics — and three reward designs were aimed without
this datum.

The paddle is a position servo, so the probe needs no inference:
each step's `data.ctrl` for the side-A actuators IS the commanded
paddle position, in the same joint frame as the slide `qpos`. The
probe replays a checkpoint deterministically (the instrument's own
loader — checkpoint behind its sha-verified normalizer) and records
command, paddle, and ball world-XY for every step of every
post-swing window (side-A legal hit → opponent's next legal strike /
point boundary / episode end), excluding the instrument's 30-step
follow-through.

## 2. Subjects and conditions

| subject | artifacts (sha-verified against the run's meta) | trained under |
|---|---|---|
| ground oracle | scripted (calibration row) | — |
| registered best, step 2,400,000 (`20260816_235141`) | `best_model.zip` `838997fb…`, `best_vec_normalize.pkl` `d0502c14…` | MuJoCo 3.11.0 |
| LH1c best, step 2,325,000 (`20260821_013700`) | `best_model.zip` `532ef6e9…`, `best_vec_normalize.pkl` `0d2db134…` | MuJoCo 3.12.0 |

30 episodes each, calibration seeds 5200–5229 (the diagnosis
convention; no reserved block touched, nothing burned). Env rebuilt
from the runs' recorded evaluation constructor kwargs (n-point
continuous, ground rules, both escrows; LH1c's row adds its hold
pairing — reward-side only, no behavioral effect on a deterministic
replay). Local physics is MuJoCo **3.11.0**: exact for the oracle
and the registered checkpoint; a cross-version replay for the LH1c
policy (trained under 3.12). The cross-checkpoint comparison below
is therefore *cleaner* than the training runs' own histories — both
policies answered the same observations under the same physics on
the same seeds — while LH1c's absolute numbers carry the usual
cross-version grain of salt.

## 3. Validity

The oracle row doubles as the frame check: its commanded path per
window (2.36 m) equals its actual paddle path (2.31 m) with a 0.49 m
mean servo gap and 1.8% action saturation — commands and paddle move
together, so the ctrl/qpos world-frame mapping is exact, and the
oracle's measured hold (2.55 m actual in strike-ended windows) sits
right on the certified band the instrument has always reported for
it. 186 windows over its 30 episodes (the oracle completes rallies);
77 (registered) and 72 (LH1c) for the checkpoints.

## 4. Results

Windows that end in the opponent's return strike — exactly the
windows the hold escrow paid:

| metric (per window, post-follow-through) | oracle | registered 2.4M | LH1c 2.325M |
|---|---|---|---|
| windows | 164 | 48 | 52 |
| steps | 77.8 | 82.8 | 88.7 |
| **commanded XY path** | **2.56 m** | **31.04 m** | **22.71 m** |
| actual paddle XY path | 2.55 m | 7.84 m | 7.55 m |
| servo gap, mean \|cmd − paddle\| | 0.54 m | 2.70 m | 2.23 m |
| \|cmd − hit point\|, mean | 2.15 m | 4.17 m | 3.76 m |
| cmd → ball distance, mean | 3.95 m | 5.21 m | 6.14 m |
| cmd → home distance, mean | 0.91 m | 3.81 m | 4.36 m |
| action saturation (\|a\| > 0.9), steps | 2.1% | 87.6% | 91.4% |
| command saccade, mean per step | 0.033 m | 0.381 m | 0.258 m |

All-window and boundary-window groups tell the same story (full
tables in the probe's JSON outputs; boundary windows are the
inter-point travel channel — LH1c's actual travel there is 8.99 m,
the instrument's familiar number).

## 5. Findings

- **F1 — the wander is commanded.** The command path is 3–4× the
  paddle's actual path (31 vs 7.8 m; oracle ratio 1.0). The paddle
  is not drifting away from a held target; it is low-pass-filtering
  a command signal that thrashes. The dynamics-side hypothesis is
  dead.
- **F2 — and it is thrash, not a destination.** Commands do not
  settle anywhere: 88–91% of steps are saturated (|action| > 0.9)
  and the command jumps 0.26–0.38 m per step (oracle: 0.03). Mean
  command position is ~4 m from the hit point, 4–4.4 m from home,
  5–6 m from the ball — no attractor. This rules out both
  "commanded retreat to a wrong home" and "commanded ball-chase":
  the policy's action head is emitting saturated, high-frequency
  bang-bang in the entire post-swing region.
- **F3 — the oracle shows what trained looks like.** Smooth,
  unsaturated commands that recover ~1 m from home and hold; command
  and paddle indistinguishable. The gap between F2 and this is the
  skill k=2 needs.
- **F4 — why the hold escrow could not win.** To earn the pay, the
  policy needed ~80 consecutive near-zero-displacement commands from
  an action head that is saturated across this whole region while
  exploration is dead (ent_coef ~1.7e-4, std ~0.014). There was no
  noise to *discover* partial stillness and no local gradient to
  unsaturate 80 outputs at once — the escrow's pressure had no
  mechanism to descend. Era I's law refines to: paid windows
  organize **only where exploration can find the paying behavior**.
- **F5 — the hold gradient did leave a fingerprint.** On identical
  physics and seeds, LH1c commands thrash measurably less than the
  registered source (path 22.7 vs 31.0 m; saccade 0.26 vs 0.38
  m/step) while sitting slightly closer to the hit point. Direction
  consistent with the live pay damping the thrash — and consistent
  with LH1c's training-dynamics record (§4c) — but an order of
  magnitude short of stillness.

## 6. Routing (implications, ranked — next design work)

1. **Temperature-skip warm start gains direct support** (the §4a
   shelf item): F4 names dead exploration as the mechanism that
   starved every reward-side fix. One flag (do not transfer
   `log_ent_coef`), a 1M pilot against the standing bars. Cheapest
   probed change, already specced.
2. **A command-rate treatment is now a named design candidate.** The
   action channel physically permits 0.38 m/step target saccades the
   servo cannot follow (2.7 m gap) — the thrash costs the policy
   nothing and the task nothing. A rate-limited or low-passed target
   (interface-side), or an action-smoothness term, would make
   stillness the default rather than an 80-step feat. **Warning:**
   this changes action semantics — a new comparability era with its
   own probe battery and bars, not a TOML toggle. Design doc first,
   per the standing convention.
3. **What this closes:** no further hold-escrow scale (F4 says scale
   was never the binding constraint), no opponent-side explanation
   for the wander, no dynamics-side fix.

Seed ledger: unchanged — calibration 5200+ reused for diagnosis, per
convention; 4100–4199 untouched.
