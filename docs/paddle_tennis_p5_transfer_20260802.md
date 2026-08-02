# PaddleTennis probe P5 — the transfer shim works, the yield rule is mandatory, the champions await Colab

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

## 2. Scripted calibration — the instrument works, and it found a rule

The certified true-baseline oracle (`lead_charge` 2.6, the probe that
certified the champions' own training geometry) played 100 episodes
per arm through the shim on calibration seeds 5000–5099, native
paddle-court oracle on side B:

| player (side A) | crossings | ≥1 | returns A | returns B | top terminations |
|---|---|---|---|---|---|
| wall-ball oracle via net | 0.00 | 0% | 0.00 | 0.00 | wrong_hitter 50, failed_to_cross 49 |
| wall-ball oracle via net + yield | 0.49 | 49% | 0.00 | 0.24 | double_hit 58, out_of_bounds 37 |
| wall-ball oracle via scaled | 1.00 | 47% | 0.55 | 0.24 | wrong_hitter 50, out_of_bounds 33 |
| **wall-ball oracle via scaled + yield** | **1.57** | **96%** | 0.54 | 0.47 | out_of_bounds 71, failed_to_cross 24 |
| native paddle-court oracle | 3.22 | 99% | 1.21 | 1.23 | out_of_bounds 92, ball_net 6 |

Three findings, all structural:

1. **The `net` identification is broken by construction.** The
   champion command range (−8.2 … 0.3) maps entirely behind or at the
   paddle-court baseline (most-forward commandable point: local
   −3.6), pinning the player out of the play band. The
   `test_net_variant_pins_the_champion_deep` test pins the geometry.
   Real champions need not be measured on this arm; it ships for
   completeness of the pre-declaration.
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
   serves and sustains ~49% of the native rally tail; the residual
   taxonomy (out long, strokes dying short) is stroke calibration for
   the shorter court, not disorientation.

## 3. Pre-registered champion decision rule (phase-P2 opponent pool)

Committed here, before any champion number is seen. A champion joins
the phase-P2 warm opponent pool iff, on this same instrument
(100 episodes, seeds 5000–5099, `scaled + yield`, native oracle on
side B):

- mean crossings ≥ **1.0**, and
- ≥1-crossing rate ≥ **0.60**, and
- zero unsafe/nonfinite terminations.

The floors are two-thirds-of-stub-band anchors (stub: 1.57 / 96%),
loose enough to admit a differently-styled but functional learned
player and tight enough to exclude a disoriented one. If neither
champion qualifies, the phase-P2 pool starts scripted-only plus P1's
own graduates — a measured outcome, not a failure of the campaign.

## 4. Seed ledger

This probe burned calibration block **5000–5099** (all arms share
it), plus 1000–1005 in harness bring-up. The reserved held-out block
**4100–4199 remains untouched** (3100–3199 was retired by the env
certification, see the env freeze doc §5). Champion runs on Colab
must reuse 5000–5099 — the comparison to the stub band is
seed-matched by design.
