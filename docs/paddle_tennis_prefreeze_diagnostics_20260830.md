# Pre-freeze checkpoint diagnostics and the arm-(c) roll-in row (2026-08-30)

Status: **Measured input to the unfrozen k=2 design
([`design_paddle_tennis_k2_drill.md`](design_paddle_tennis_k2_drill.md))
— no verdict booked, nothing frozen.** Three literature-recommended
diagnostics on the registered 2.4M best (pins `838997fb…`/`d0502c14…`)
plus the JSRL-style oracle roll-in priced as a third D2 arm. Every
headline below survived, and in four cases was **corrected by**, an
adversarial verification pass (14 confirmed findings, 0 rejected;
every number quoted here is the verifier-reproduced value, and the
first-draft claims the verification overturned are recorded in §6 —
the corrections change the freeze inputs, so they are the point).

State sources: the registered 102-entry k=2 library (banked); a
15,000-state deterministic on-policy batch on the checkpoint-diagnosis
calibration convention (seeds 5200–5209 — a diagnosis-class READ of
the block reserved for exactly this; nothing becomes training data);
roll-in episodes on scratch seeds **9168–9187** (§5 ledger).

## 1. Critic dormancy: heavy on the visited distribution, weights intact

The critics' penultimate layers are severely dormant **on the visited
state distribution**: 55–61% dormant at τ=0.025 and 38–41% exactly
zero across all 15,000 states, versus **<5% dormant / ~0% hard-dead
for freshly initialized critics of the identical 51-256-256-1
architecture on the same batch** — the pathology is real, not
architecture-normal. The actor is mildly affected (14% / 2.7%).
Softeners the verification added, all binding on interpretation: no
unit is unconditionally dead by weights (dead rows keep substantial
incoming mass); roughly half the hard-dead units reactivate under
broadened state inputs (~20% of the layer stays dead even at 3σ-wide
inputs); the τ=0.025 fraction is inflated by a heavy-tailed activation
profile (median-normalized it reads ~39–42%); and the batch is a
deterministic-eval slice, narrower than the replay-like batches ReDo
scores against (the exactly-zero fraction is coverage-sensitive:
0.59 → 0.38 from 1.5k to 15k states; the τ fractions are stable).

**Freeze implication:** critic-head reset / ReDo-style recycling
stays in any fine-tune package, phrased as "dormant on the visited
distribution, weights intact" — and dormancy should be recomputed on
the actual training stream before resets are applied.

## 2. Temperature vs exploration noise: only the entropy TERM is collapsed

`ent_coef` has annealed to **1.6e-4** (target_entropy −1.5) — the
entropy term exerts essentially no pressure on the objective. But the
**gSDE exploration noise is NOT collapsed**: the marginal pre-tanh
action-noise std is **~0.60 mean / 0.73 p90** on the k=2 states
(0.62–0.66 on-policy). The first draft's "gSDE noise 0.020" was the
per-entry exploration-matrix scale (a state-independent parameter,
only ~2.5× below SB3's init of 0.0498), not an action-noise scale —
a ~30× misread the verification caught before it could size anything.

**Freeze implication:** "re-heat" means restoring the entropy term's
pressure (log_alpha re-init / target-entropy raise), not enlarging a
noise scale that never collapsed. And every step-0 row in the design
doc was measured under the deterministic policy while the training
behavior policy actually jitters substantially — the battery's
step-0 conventions should say so.

## 3. Q-calibration: no overestimation, but systematic pessimism

The first draft banked "calibration is sane"; the Monte-Carlo check
refutes the absolute scale. On 10,010 on-policy states ≥500 steps
from truncation (same seeds, raw reward units — `norm_reward=False`,
γ=0.99): realized discounted returns average **+0.068** while
min(Q1,Q2) at tanh(μ) averages **−0.389**; Q < MC on **98.7%** of
states, mean bias **−0.457**, negative in all 10 episodes; the soft-Q
entropy correction is negligible (~−0.02). Rank correlation Q-vs-MC
is 0.87 — relative ordering usable, absolute scale not. The
k=2-below-k=1 class ordering (−0.61 vs −0.48) is demoted to a
descriptive note: the 0.13 gap is small against the 0.46 global bias,
and a calibrated critic would price the near-certain discounted +0.63
pending return a k=2 launch carries (the opponent's return confirms
at the side-A bounce, mean 46 steps out).

**Freeze implication:** there is no pre-existing overestimation for
injection to inflame — but any fine-tune should expect to correct a
~−0.46 absolute value bias, and any Cal-QL-style calibration
reference must be MC returns, not class-mean ordering.

## 4. The context gate at network level: joint, redundant, not decomposable

The joint gate is confirmed and large: swapping the full context
block (dims 24–47) between the full- and feed-context presentations
of bit-identical physics flips the deterministic action by **1.60
mean / 2.56 p90** (of a 3.46 max) across all 102 states; the groups
that read zero (serving/returner/ball-side, bounce count, contact
tail) are bit-identical between presentations — nothing to swap.

The first draft's per-feature ranking (phase 1.39 / rally_count 1.17
/ clock 0.91 / crossing 0.44) is **not a decomposition** and is
withdrawn: the single-group deltas sum to 3.92 against the joint
1.60, and leave-one-group-out marginals are +0.17 (phase), +0.10
(clock), +0.014 (rally), +0.001 (crossing) — the gate is redundantly
encoded across the block, each single swap creates an off-manifold
context never seen in play, and rally_count's sensitivity is generic
(perturbing 1→2 gives 1.10, nearly the 1→0 swap's 1.17). The clock's
0.91 is additionally a probe construction: it paired harvested
mid-episode clocks against a fresh-launch 1.0 — a difference the
**shipped** drill never injects (both arms read the episode's own
clock), and the design doc's env-launch cross-check already
reproduced the 6.9%-vs-2.0% gate through `_launch_drill` with
identical clocks in both arms — behavioral proof the clock does not
carry the gate. Separately and genuinely: the policy conditions on
the clock (±0.1 on-manifold perturbation moves actions 0.26–0.29) —
a shortcut-feature property that matters for per-entry baseline
reproducibility (the KD2 env-body re-anchor), not a gate carrier.

**Freeze implication:** any context-masking/randomization
intervention must target the whole rally-context block jointly
(dims 24–35 as a unit); single-feature masking has no support.

## 5. Arm (c) — oracle roll-in: qualitatively consistent with the gate, statistically indeterminate

Mechanism: the scripted oracle plays side A; control hands to the
checkpoint at the first step-end with the k=2 chain armed and
ball_x ≤ 0.05 (at most one control step before crossing — 34/57
hand-offs fire just pre-crossing; the harvest's snapshot convention
is the last pre-crossing step, so comparability is one step, noted).
Seeds 9168–9187, 57 hand-offs (23 on B-serving slots, the drill's
D3 slot; 34 on A-serving points whose geometry differs).

- **Positioning inherited, like-for-like:** hand-off head **0.89 m**
  from the eventual bounce vs the library's **3.65 m** at launch
  (head_a → continuation bounce, 94/102 entries); by bounce time
  **1.41 m** vs the feed arm's **3.04 m** under identical bounce-time
  semantics. (The first draft paired 0.89 against 3.0 — mismatched
  instants; corrected.)
- **Conversion, censoring-aware:** raw B-slot 1/23; three B-slot
  hand-offs were censored by the 1500-step episode clock and one
  feed died at the net before crossing — all four asymmetrically in
  the headline slot. Excluding the censored rows: **1/20 (5.0%)
  B-slot, 2/54 combined** (further excluding the net-dead feed:
  1/19, 2/53). All 20 recorded B-slot first contacts landed in
  bounds and none ended the point at first contact — the balls were
  playable.
- **Movement after hand-off:** the policy fails to close on the
  intercept — paired bounce-time-minus-hand-off distance median
  +0.26 m, mean +0.52 m (inflated by one truncation-ended +3.18 m
  row); 14/20 ended farther, 6/20 closer. "Fails to track," not
  demonstrable active wandering.
- **Statistics, stated honestly:** 1/20-class results cannot
  discriminate the gate hypothesis (~2%; Fisher p=0.46 vs 2/102)
  from moderate position effects (~6.9%; p=1.0 vs 7/102; Wilson 95%
  CI 0.8–21% covers both). The run **excludes only a large
  position-driven rescue (true rates ≳20%)** — which is itself a
  real result: inheriting near-perfect position does not produce
  feed-context-sized (let alone oracle-sized) conversion. The first
  draft's "decisive negative / position nearly irrelevant" is
  withdrawn as statistical language; a discriminating arm-(c) run
  needs on the order of ~130 B-slot hand-offs.

**Freeze implication:** arm (c) shows no evidence of benefit at this
n and no mechanism by which it trains the binding skill (the
observation-side gate); it should not displace the other candidates,
but it is priced, not falsified — the maintainer can buy the
discriminating run (~45–50 episodes, needs a scratch-block
extension) or drop the arm.

## 6. What the verification overturned (kept for the record)

First-draft claims corrected before banking: "gSDE noise 0.020"
(→ the matrix-entry scale; true marginal std ~0.60); "calibration is
sane" (→ systematic −0.46 pessimism; only no-overestimation
survives); the per-feature gate ranking and the clock as a gate
carrier (→ joint, redundant, clock probe-specific); "decisive
negative, position nearly irrelevant" with 1/23 vs "3.0 m"
(→ indeterminate 1/20 vs like-for-like 3.65 m); "drifts off the
intercept" (→ median +0.26 m, fails-to-close); "critics have lost
major capacity" (→ heavily dormant on-distribution, weights intact).
Minor code notes, none moving a banked number: the k=1-receive mask
carried 4/2857 post-touch states (Q mean shifts −0.4849 → −0.4843);
dormancy hard-dead is coverage-sensitive; the roll-in docstring's
"has crossed" was imprecise for 34/57 pre-crossing hand-offs.

## 7. Routing synthesis (PROPOSAL — the maintainer decides)

Unchanged from the literature note where it survives contact with
these measurements, sharpened where it doesn't:

1. The binding deficit remains the **joint observation-side context
   gate** — now dissociated from position (arm (c)), from the clock
   (env cross-check), and from exploration-noise scale (§2).
2. The package shape the evidence now supports: demonstration
   injection in the real context (§3: no overestimation to inflame,
   but expect the pessimism bias) + whole-block context
   randomization/masking on drilled points (§4) + entropy-term
   re-heat (§2) + critic-head reset with training-stream dormancy
   re-check (§1) — with the drill's real-context launches as the
   exposure vehicle rather than a standalone lever.
3. Arm (c) is priced and parked unless the maintainer buys the
   discriminating sample.

## Seed ledger

Consumed this note: **9168–9187** (roll-in, 20 episodes). The
proposed 9000–9199 scratch block's unconsumed remainder is
**9188–9199 (12 seeds)** — too thin for the ~130-hand-off arm-(c)
run or further probes; this note proposes booking **9200–9299 as a
scratch-block extension** at the maintainer's next booking. Seeds
5200–5209 were used as a diagnosis-class read (the block's purpose;
no training artifact derives from them). 4100–4199 remains sealed.
