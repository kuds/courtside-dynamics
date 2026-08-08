# PaddleTennis probe P5 — the transfer shim works, the yield rule is mandatory, the champions await Colab

> **Ground-era recalibration, 2026-08-03**
> ([`paddle_tennis_ground_rules_20260803.md`](paddle_tennis_ground_rules_20260803.md)
> §6): the instrument now runs on the ground-rules default. The
> `scaled + yield` stub row becomes 1.42 crossings / 97% ≥1 — the
> pre-committed admission floors (1.0 / 0.60 / zero unsafe) stand
> unconsumed — but the stub's confirmed returns collapse 0.69 → 0.04
> (wall-ball's flat stroke calibration lands out under ground
> confirmation; native ground oracle on the same block: 6.86
> crossings, 3.38 returns/side). Champion rows on Colab should read
> the returns column beside the binary verdict. The matrix below is
> the volley-era record.

Status: review snapshot, 2026-08-02, of probe P5 (wall-ball champion
transfer) from the pre-committed battery
([`design_paddle_tennis.md`](design_paddle_tennis.md) §6), run against
the registered, held-out-certified `CourtsideDynamics/PaddleTennis`
environment. P5 gates only the phase-P2 opponent-pool decision; the
env definition ([`paddle_tennis_env_20260802.md`](paddle_tennis_env_20260802.md))
is frozen independently of it.

This snapshot covers the **instrument** (the observation/action shim,
`tools/paddle_tennis_p5_transfer.py`) and its **scripted calibration**
(the certified wall-ball lead-charge oracle played through it). The
**champion measurements themselves are still open**: the two
checkpoints (`20260731_132322`, `20260801_144043` — `best_model.zip`
~3.3 MB each) exceed what the Drive connector can transport into this
session, so the champion rows run where the Drive artifacts are
mounted (Colab), one command per champion:

```bash
python tools/paddle_tennis_p5_transfer.py \
  --model  <run>/model/best_model.zip \
  --vec-normalize <run>/model/best_vec_normalize.pkl \
  --baseline
```

Both runs' `best_vec_normalize.pkl` files were pulled and inspected
here: 23-value observation, full-observation normalization (no
exclusions), `clip_obs` 10 — the shim targets exactly that interface.

## 1. The shim

The champions know only the wall-ball world: 23 observation values
against a wall face at x = +3.9, actions through the true-baseline
asymmetric x mapping (low −8.2, home −5.4, high 0.3). The shim renders
the paddle court's side-local state as that world and re-encodes the
champion's action for the court. Two pre-declared x identifications:

- **net** — rigid translation, net plane ↔ wall face, distances
  preserved.
- **scaled** — affine: net ↔ wall face AND paddle-court baseline
  (6.5 m) ↔ wall-ball baseline (11.885 m from the face); x velocities
  scale by the same 1.83 factor.

The y/z action mappings are the shared paddle calibration on both
courts and pass through unchanged. The wall-ball rally tail maps as:
gate flag ← "own shot in flight" (possession flags), floor bounces ←
rules `bounce_count`, stall/curriculum/recovery values ← 0 (none
exist here). Full per-index map in `wall_ball_observation`'s
docstring; pinned by `tests/test_p5_transfer.py`.

Three instrument corrections landed after the pre-push adversarial
review (2026-08-03), before any champion measurement:

1. **The decode now reproduces the trained fence.** The champions'
   env clamped every x target to (−8.2, −2.6) before the actuator saw
   it, so any action ≥ +0.491 physically meant "paddle to −2.6". The
   first instrument spread that saturated interval over unreachable
   forward targets (up to 1.6 m in front of anything the champions
   ever occupied) — undetectable by the stub, whose scripted actions
   are exactly {−1.0, +0.491} by construction, but exactly where a
   saturating SAC policy lives. Stub rows are unchanged by this fix
   for that same reason.
2. **The rendered ball is held at the wall face while on the
   opponent's half.** The far half does not exist in the champion's
   world; unclamped, ~22% of policy-queried steps showed the ball up
   to 6 m beyond the wall at scaled velocities. Held at the face, the
   ball "waits at the wall" until the opponent's return sends it
   back — the rebound moment wall-ball trained on.
3. **The yield park is the champion's own neutral** (wall-ball home
   −5.4 decoded through the shim), not the paddle court's home: the
   release-step self-state stays inside the champion's lifelong
   workspace, and the park sits deep, clear of the serve column.
   This is also why the stub's `scaled + yield` row improved.

## 2. Scripted calibration — the instrument works, and it found a rule

The certified true-baseline oracle (`lead_charge` 2.6, the probe that
certified the champions' own training geometry) played 100 episodes
per arm through the fixed instrument on calibration seeds 5000–5099,
native paddle-court oracle on side B:

| player (side A) | crossings | ≥1 | returns A | returns B | top terminations |
|---|---|---|---|---|---|
| wall-ball oracle via net | 0.00 | 0% | 0.00 | 0.00 | wrong_hitter 50, failed_to_cross 49 |
| wall-ball oracle via net + yield | 0.49 | 49% | 0.00 | 0.23 | failed_to_cross 62, out_of_bounds 37 |
| wall-ball oracle via scaled | 1.00 | 47% | 0.55 | 0.24 | wrong_hitter 50, out_of_bounds 33 |
| **wall-ball oracle via scaled + yield** | **1.71** | **96%** | 0.69 | 0.48 | out_of_bounds 77, ball_net 11, failed_to_cross 10 |
| native paddle-court oracle | 3.22 | 99% | 1.21 | 1.23 | out_of_bounds 92, ball_net 6 |

(The pre-fix instrument measured the same rows at 1.57/96% for
`scaled + yield` and `double_hit`-dominant for `net + yield`; the
deltas are the yield-park correction. Both sweeps are seed-matched.)

Three findings, all structural:

1. **The `net` identification is broken by construction.** With the
   trained fence, the champion's commandable band maps to local
   [−6.4, −3.6] — inside the court, but excluded from the front
   ~3.5 m where the rally is actually played (the native controller
   works around home −1.7; the shimmed player can never command
   forward of −3.6). The `test_net_variant_pins_the_champion_deep`
   test pins the geometry. Real champions need not be measured on
   this arm; it ships for completeness of the pre-declaration.
2. **The serve-yield rule is mandatory, not optional.** Without it,
   every serving point — exactly 50 of 100, the alternation's half —
   ends in `wrong_hitter`: wall-ball serves always arrived *from* the
   wall side, so no wall-ball player ever learned to stand down while
   its own serve is in flight, and the transferred player dives
   straight through the serve's flight path (P3's documented
   self-touch hazard). The overlay (neutral home park unless
   `expected_returner_is_own`) is the minimal rule the native
   controller already follows; with it the serving half survives.
3. **Receiving-side competence survives the scaled shim.** Under
   scaled + yield the wall-ball-frame player returns 96% of points'
   serves and sustains ~53% of the native rally tail; the residual
   taxonomy (out long, strokes dying short) is stroke calibration for
   the shorter court, not disorientation.

Known instrument limitation, kept deliberately: the yield overlay is
**turn-scoped** — it also suppresses between-shot repositioning after
every own hit, a window in which wall-ball did train legitimate
recovery behavior. A champion's competent serve-time and between-shot
play is therefore invisible on the overlay arm. That is why
overlay-off arms are always measured and reported beside it: a
champion that outperforms its own overlay row on the overlay-off arm
is evidence the overlay, not the champion, is binding — and the
admission verdict below must be read against both rows.

## 3. Pre-registered champion decision rule (phase-P2 opponent pool)

Committed before any champion number is seen (amended once, with the
instrument fixes above, still before any champion measurement). A
champion joins the phase-P2 warm opponent pool iff, on the fixed
instrument (100 episodes, seeds 5000–5099, `scaled + yield`, native
oracle on side B):

- mean crossings ≥ **1.0**, and
- ≥1-crossing rate ≥ **0.60**, and
- zero unsafe/nonfinite terminations.

The floors sit at ~60% of the fixed-instrument stub band
(1.71 / 96%), loose enough to admit a differently-styled but
functional learned player and tight enough to exclude a disoriented
one. If neither champion qualifies, the phase-P2 pool starts
scripted-only plus P1's own graduates — a measured outcome, not a
failure of the campaign.

## 4. Seed ledger

This probe burned calibration block **5000–5099** (all arms share
it), plus 1000–1005 in harness bring-up. The reserved held-out block
**4100–4199 remains untouched** (3100–3199 was retired by the env
certification, see the env freeze doc §5). Champion runs on Colab
must reuse 5000–5099 — the comparison to the stub band is
seed-matched by design.
