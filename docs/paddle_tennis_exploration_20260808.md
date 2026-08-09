# Ground-era exploration package — the H1 remedy, pre-registered

Status: **adopted (recipe change); pilot complete — mechanism
confirmed, stroke acquisition absent**, 2026-08-09 (§3). The remedy
the ground-era pilot diagnosis ranked first
([`paddle_tennis_diagnosis_20260808.md`](paddle_tennis_diagnosis_20260808.md)
§3): sustained exploration, shipped as `PaddleTennis` recipe
`model_kwargs`. §2's falsification criteria were frozen **before**
the pilot ran; §3 records the measured verdict (M/D1/D3 pass, D2/P
fail — the decision rule's middle case) and points the next probed
change at the touch→in credit gap.

> Provenance note: the package's first draft (no `train_freq`, an
> `ent_coef`-based mechanism check, and an overstated provenance
> table) was corrected by the pre-push adversarial review **before
> any pilot verdict existed** — the first pilot launch was killed
> ~10 minutes in, unread beyond liveness checks, because the review
> showed its gSDE noise was still per-step iid (see §1). What
> follows is the corrected package; the pilot below is its first
> real test.

## 1. The change

```python
"model_kwargs": {
    "use_sde": True,
    "ent_coef": "auto_0.02",
    "target_entropy": -1.5,
    "train_freq": (64, "step"),
},
```

Task definition, band (7.78), and held-out certification (7.68 on
4200–4299) are untouched — this is a training-side change inside the
same comparability era. Saved policies remain comparable; learning
curves gain a new training-configuration regime starting here.

Provenance, stated honestly — what is measured, what is design
precedent:

| piece | basis |
|---|---|
| `target_entropy=-1.5` (stock: −3.0) | **Mechanism-level fix for the measured failure.** SB3's dual update anneals the entropy multiplier *down* when policy entropy meets the target and *up* when entropy sags below it — the stock pilot's `ent_coef=5e-5` was the tuner resting at the too-low default target, not a broken tuner. Raising the target raises the level the tuner defends. The −1.5 value is **design precedent, not a measured result**: it comes from the WallBall bootstrap package (DECISIONS.md, 0.11.0), which was designed for the same collapse shape but never trained (its cold-start problem was solved by stock auto-entropy first). The often-cited 0.0005 collapse behind that design was measured on the **legacy 5-action env** (cardinal rule 5 forbids carrying that number across action spaces); on this 3-action interface the directly measured facts are the stock run's 5e-5 amid the H1 plateau, and wall-ball's ~0.0007–0.0011 equilibria on runs that *did* learn. |
| `ent_coef="auto_0.02"` (not a pinned float) | Pinned-float `ent_coef=0.02` is a measured poison (lessons_learned.md: budget `ent_coef × episode_len`; episodes here are 1500 steps, so a permanent bonus is worth tens of reward). Auto from a safe init keeps the warm start without the permanent bonus. |
| `use_sde=True` | Humanoid C2, **measured under PPO**: iid per-step Gaussian noise averages to a near-still effector (0 contacts in 264 random episodes vs ~15% success for a *constant* random action); gSDE holds one noise weight matrix across a rollout. Targets the diagnosis's dominant ender (`policy_never_reached` 83/100). |
| `train_freq=(64, "step")` | **Required for gSDE to work at all under SAC** — this is what the review caught. SB3's off-policy collector calls `actor.reset_noise()` at every rollout start, and SAC's default `train_freq=1` makes every rollout a single vec-step: the noise matrix is resampled every step (verified empirically on the installed SB3 2.9.0 — one reset per step, consecutive matrices independent), i.e. iid noise in a gSDE costume. At 64 vec-steps per rollout one matrix persists through a coherent approach maneuver (half the 126-step exchange cadence). `gradient_steps=-1` (set by `train()`) keeps the 1:1 update-per-transition ratio, so total compute is unchanged. |

## 2. Pre-registered pilot

One local CPU run, launched at adoption time:
`build_train_config("PaddleTennis", log_dir=..., seed=0,
total_timesteps=1_000_000, n_envs=4, checkpoint_freq=100_000)` — the
local-pilot convention
(seed 0 and its derived helper seeds only; the automated diagnosis
uses calibration block 5200+; reserved block 4100–4199 is not
touched). Checkpoint cadence 100k so the per-checkpoint diagnosis
gives ten behavioral rows across the run.

Reference points: the ground-era GPU pilot (`20260808_022106`, stock
recipe) plateaued at best crossings 1.37 over 1.75M steps with final
diagnosis: serving-side exchange survival 0%, touched-after-bounce
37%, ready error 2.83 m.

**Criteria, frozen before the run:**

- **P (primary): best `crossings_ep_mean` ≥ 2.5** by 1M — clear of
  the 1.37 plateau's regime. (A band pass ≥ 6.0 would be a strong
  outcome but is not required of a 1M CPU pilot; the registered GPU
  run carries the band target.)
- **D1 (serving-side unlock): serving-side exchange k=1 survival
  > 0%** at any diagnosis checkpoint from 300k on.
- **D2 (ball-reaching): touched-after-bounce ≥ 55%** at some
  checkpoint by 1M (from 37%; oracle 98%).
- **D3 (positioning): ready-position error mean ≤ 2.0 m** at some
  checkpoint by 1M (from 2.83; oracle 0.89).
- **M (mechanism check), two parts, both decidable from the run's
  artifacts:**
  - **M1 (plumbing):** the run's `config.json` records all four
    package keys, and a checkpoint loaded with `SAC.load` reports
    `target_entropy == -1.5` and `train_freq` of 64 steps — the
    package actually reached the model.
  - **M2 (exploration alive):** `train/std` (logged because
    `use_sde=True`; init e⁻³ ≈ 0.050) must not decay an order of
    magnitude below its init — it stays ≥ 5e-3 across the run.
  - Explicitly **not** a mechanism signal: the magnitude of
    `train/ent_coef`. Under SB3's dual update a low multiplier is
    the raised target binding *cheaply* (healthy); entropy sagging
    below the target manifests as the multiplier *rising*. (The
    first draft of this criterion had that backwards.)

**Decision rule, also frozen:**

1. P and ≥2 of D1–D3 pass (with M intact) → the H1 remedy is
   confirmed at pilot scale: pre-register the ground-era GPU run
   (stock recipe now including this package; held-out gate on
   reserved block 4100–4199) and hand it to Colab.
2. M intact but D1 fails and D2 stays flat → the diagnosis
   signature survived mechanically-real exploration: exploration
   was not the binding constraint, the shared-reward structure is.
   Escalate to the diagnosis doc's #2 (n-point episodes; env
   change, new era, own probe) with #3 (own-credit reward) behind
   it.
3. M1 or M2 fails → fix the package before drawing any conclusion
   about H1 (a package that never reached the model, or exploration
   that died despite it, is a bug to fix, not evidence about the
   hypothesis).

## 3. Results — mechanism confirmed, stroke acquisition absent

Run `20260809_005951` (Colab L4, commit `1c1e8e1`, 1,000,000 steps
in 3 h 58 m at 70 FPS, completed 2026-08-09 04:57 UTC; the packaged
starter TOML rode along via `CONFIG_FILE="auto"` but every
divergent field was overridden by the pre-registered explicit
values, recorded in `config.json` provenance).

**Criteria (frozen in §2) against the measured run:**

| criterion | outcome | evidence |
|---|---|---|
| M1 plumbing | **pass** | `config.json` records all four keys; resolved model reports `TrainFreq(frequency=64, unit=STEP)`, `target_entropy=-1.5`, cuda |
| M2 exploration alive | **pass** | `train/std` 0.0502 → 0.0475 (500k) → 0.0325 (final); floor 5e-3 never approached. (`train/ent_coef` ended at 4e-5 — per §2, not a mechanism signal.) |
| D1 serving-side unlock | **pass** | k=1 survival > 0% at every diagnosis checkpoint from 200k except 900k; 7–13% typical, **27% at 800k** (stock run: hard 0% at 1.75M) |
| D3 positioning | **pass** (by the letter) | ready error ≤ 2.0 m at 100k (1.98), 300k (1.82), 900k (1.90); but oscillating 1.8–2.7, not trending |
| D2 ball-reaching | **fail** | touched-after-bounce peak **17%** (800k) vs the ≥ 55% bar; series 0→10→10→7→7→3→7→17→0→7% |
| P crossings ≥ 2.5 | **fail** | best eval 0.67 (550k), final 0.53; flat ~0.5 across the whole run |

**No decision-rule branch fires.** Rule 1 needs P; rule 2 needs D1
to fail; rule 3 needs M to fail. The run landed in the middle case
the rules deliberately do not force: the exploration mechanism is
real and verified, the stock run's two structural signatures
(entropy collapse; serving-side zero under one memorized
receiving-macro) did not reproduce — and the policy still never
acquired a stroke.

**The checkpoint-by-checkpoint behavior** (30 probe episodes each,
seeds 5200+; oracle reference: ~120 hits, 98% touch, 0.89 m ready,
in-depth 3.9 m):

| ckpt | policy hits | serving k=1 | receiving k=1 | touch | ready m | crossings |
|---|---|---|---|---|---|---|
| 100k | 0 | 0% | 0% | 0% | 1.98 | 0.50 |
| 200k | 3 | 13% | 7% | 10% | 2.50 | 0.57 |
| 300k | 3 | 7% | 13% | 10% | 1.82 | 0.57 |
| 400k | 2 | 13% | 0% | 7% | 2.36 | 0.53 |
| 500k | 2 | 13% | 0% | 7% | 2.70 | 0.50 |
| 600k | 1 | 7% | 0% | 3% | 2.64 | 0.53 |
| 700k | 2 | 13% | 0% | 7% | 2.38 | 0.53 |
| 800k | **5** | **27%** | 7% | **17%** | 2.39 | 0.63 |
| 900k | 0 | 0% | 0% | 0% | 1.90 | 0.50 |
| 1M | 2 | 13% | 0% | 7% | 2.01 | 0.57 |

Two facts carry the mechanism story (post-hoc analysis, clearly
labeled as such):

1. **Engagement oscillates without an attractor.** It builds to the
   800k peak and washes out entirely by 900k. Nothing the policy
   does when it reaches the ball ever pays: from 300k onward,
   **every** policy shot that crossed landed out, 9.2–16.0 m deep
   (the oracle lands at 3.9 m; the same hard-slam failure the
   scripted oracle had at bring-up, fixed there by softening the
   swing). A touch that always ends −1 is indistinguishable from
   never touching, so learned engagement decays as fast as it
   forms.
2. **The macro never formed either.** Unlike the stock run (98%
   receiving k=1 by 1.75M — one memorized serve-return), this run
   never converged on the easy receiving macro. Sustained
   stochasticity prevented both the collapse *and* the
   exploitation.
3. **The optimizer was quiet the whole time.** `train/critic_loss`
   sat at ~1e-4 from 100k to 1M with no spike or drift anywhere —
   including across the 800k engagement peak and its 900k washout —
   and `train/actor_loss` held ~0.24 throughout. The value function
   converged in the first 100k and never saw anything new: the
   environment genuinely returns the same outcome for everything
   the policy can currently do. This rules out off-policy
   value-function instability as the washout's cause (the
   behavioral oscillation happened on a flat landscape, not a
   drifting one) and removes the main motivation for an
   algorithm-control (PPO) arm before the reward landscape is
   changed: both algorithms starve identically on a landscape whose
   critic is already exactly right.

**Reading:** exploration was not the binding constraint at this
budget — *reward attainability* is. The +1 sits on "legal return
that lands in", and the un-shaped gap between touching the ball and
landing it in is where all the remaining difficulty lives; sampling
that gap by noise alone did not produce a single landed-in policy
shot after 300k. This sharpens the diagnosis doc's §3 ranking with
new evidence: n-point episodes multiply *rewarded configurations*
but do not by themselves pay the touch→in gap; the repo's measured
pattern for exactly this shape is escrowed contact shaping
(humanoid 0.16.0's `valid_hit_shaping=0.25`, adopted after "a flat
−1 for all outcomes gives no aim gradient"). The next probed change
should target the touch→in credit gap first — as a reward-side
amendment with its own probe and era bookkeeping — with n-point
episodes as its complement, not its substitute. The exploration
package itself stays: it measurably removed the entropy collapse
and the serving-side zero, and reverting it would reintroduce a
solved failure.

The registered-run pre-registration remains deferred until that
next change is probed; reserved block **4100–4199 stays untouched**.
Seed ledger: unchanged (pilot on seed 0 + diagnosis calibration
5200+ only).
