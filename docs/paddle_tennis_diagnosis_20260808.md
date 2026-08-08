# Ground-era pilot diagnosis — one memorized macro, no general ball-reaching

Status: review snapshot, 2026-08-08, of the behavioral diagnosis of
ground-era pilot run `20260808_022106` (early stop at 1.75M of 6M;
best `crossings_ep_mean` 1.37 vs the scripted band's 7.78), using the
instrument in `tools/paddle_tennis_diagnosis_probe.py`
(core: `training/paddle_diagnosis.py`), 100 episodes on calibration
seeds 5200–5299, ground oracle measured on the same seeds as the
reference row. The run video's impression — "after the first hit the
paddle zips all over the court" — prompted the probe; the numbers
confirm and sharpen it.

## 1. The verdict

**H1 (credit starvation), in its purest form.** The policy learned
exactly one behavior — the receiving-side serve return — and nothing
else:

| metric | checkpoint | ground oracle |
|---|---|---|
| policy legal hits, total | 49 — **all** receiving hit #1 | 397 across five+ exchanges |
| receiving hit #1: crossed / in / depth | 82% / 76% / 3.84 m | 100% / 100% / 3.87 m |
| exchange survival, receiving | k=1: 98%, **k=2: 0%** | 94% → 48% across k=1..5 |
| exchange survival, **serving** | **k=1: 0%** | 96% → 42% |
| point enders | `policy_never_reached` **83**/100 | `opponent_shot_net` 35, `cap` 28 |
| ready-position error at bounce | **2.83 m** (p90 4.19) | 0.89 m (the designed wait margin) |
| touched after bounce | **37%** of 132 | 98% of 405 |
| recovery-hold travel | **7.52 m** | 2.24 m |
| crossings | 1.25 (matches the run's evals) | 7.30 (matches certification) |

Two hypotheses die here:

- **H2 (stroke authority) — rejected.** When the policy strikes, the
  stroke is nearly oracle-grade (76% landed-in at 3.84 m vs 100% at
  3.87). The fixed-pitch face is NOT the binding constraint; **no
  paddle-pitch actuation change is justified by this evidence** (the
  design-doc question stays parked).
- **H3 (opponent asymmetry) — rejected.** The oracle handles the
  policy's returns at 95% in.

## 2. The mechanism

The single trained behavior maps exactly onto where reward gradient
was dense: on receiving points, hit #1 pays +1 immediately and often
(76%). Everything else requires the full unrewarded chain
(position → wait → strike a variable incoming ball) starting from
states the replay buffer rarely visits with success — and by 1.75M
steps SAC's entropy coefficient had collapsed to 5e-5, freezing
exploration around the macro.

The sharpest new fact is the serving-side zero, and it has a
structural component: **under the shared cooperative reward, serving
episodes carry almost no gradient about the policy's own play.** The
opponent's serve-return pays +1 regardless of what the policy does;
the subsequent never-reached fault pays −1; net ≈ 0 for any behavior
the policy can currently produce. Half of all training episodes were
therefore uninformative about its own actions.

## 3. What this points at (for the next probed change — not shipped here)

Ranked, per the evidence and the repo's lessons:

1. **Sustained exploration** — an entropy floor (fixed small
   `ent_coef` or a higher `target_entropy`) and/or temporally
   correlated exploration (gSDE): the humanoid recipes' measured fix
   for this same failure shape (lesson N-F1/C2). Recipe-level, no
   task change, cheapest to falsify.
2. **n-point episodes** — multiple alternating-serve points per
   episode multiply the rewarded configurations each trajectory
   visits (including serving-side receptions). Env-definition change;
   a new comparability era; needs its own probe.
3. **Own-credit reward** (the policy's own confirmed returns only) —
   makes serving episodes informative; a deeper change to the
   cooperative design and only worth reaching for if 1–2 fail
   measurably.

Explicitly NOT next: paddle-pitch actuation (H2 rejected) and any
opponent-side change (H3 rejected).

## 4. The instrument, now automated at checkpoints

`TrainConfig.checkpoint_diagnosis` (wired in the `PaddleTennis`
recipe: 30 episodes, seeds 5200+) runs this instrument against the
live model at every checkpoint save, writing
`reports/diagnosis/diagnosis_probe_<step>.txt` plus a once-per-run
cached oracle reference row. The probe plays on the run's own
evaluation env factory, so a run-config `[env]` override reaches the
diagnosis env too — the instrument measures the task the run actually
trains on, never a silently different stock definition. Exception-isolated (a diagnosis failure
costs one warning, never the run; it disables itself after a
failure). Cost: ~90 s of CPU per checkpoint (7 per 1.75M-step run)
while the GPU idles at the save boundary. With this in place, the
serving-side zero would have been visible at the FIRST checkpoint
(250k) instead of after seven hours.

Instrument provenance: the probe's first draft carried seven
review-confirmed defects — two of which would have inverted this
verdict (never-reached balls attributed to the shot-maker; fault
touches counted as made hits) — fixed with the review's adversarial
cases (statue policy, volley-capable oracle) kept as regression
tests.

## 5. Seed ledger

Diagnosis calibration block **5200–5299** burned (checkpoint row +
oracle row + the automated per-checkpoint reuse going forward — all
calibration, never selection). Reserved block **4100–4199 remains
untouched** and continues to await the ground-era registered run's
held-out gate.
