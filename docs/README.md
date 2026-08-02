# Documentation

Supplementary documentation for Courtside Dynamics. The user-facing overview is
the root [`README.md`](../README.md); per-release migration notes are in the root
[`CHANGELOG.md`](../CHANGELOG.md).

Docs here fall into three kinds, distinguished so a living reference is never
mistaken for a point-in-time snapshot:

- **Living** — kept current with the code (specs, the decisions journal).
- **Proposed** — a design whose work has *not* shipped. Describes intended
  future behavior, so nothing in it may be read as a description of the code.
- **Implemented design** — a proposal whose work has shipped; retained as
  historical rationale, superseded by the code and the journal for current
  behavior.
- **Review snapshot** — an analysis pinned to a specific commit/date; not
  updated. Its durable conclusions are distilled into `DECISIONS.md`.

| Document | Kind | Pinned to | What it covers |
|---|---|---|---|
| [`DECISIONS.md`](DECISIONS.md) | Living | — | Decisions & lessons-learned journal: the non-obvious choices, bugs, and dead ends, distilled from the reviews and version history. **Start here** for "why is it built this way." |
| [`run_config_file_spec.md`](run_config_file_spec.md) | Living | spec v1.1 (impl. 0.13.0) | The per-experiment TOML run-config format: tables, precedence, deep-merge semantics, validation, and provenance. Referenced by the README, the training notebook, and `run_config.py`. |
| [`design_court_and_config_updates.md`](design_court_and_config_updates.md) | Implemented design | 0.13.0 | Design for the WallBallBaseline recalibration, notebook auto-resolution of run configs, and the tennis-court replay style. Shipped in 0.13.0. |
| [`humanoid_env_review.md`](humanoid_env_review.md) | Review snapshot | `main`@`0d294f2`, v0.7.0, 2026-07-13 | Deep review of the humanoid tennis environment and shared infrastructure (rules, curriculum, promotion gate, training, video, learning feasibility). |
| [`wall_ball_baseline_review.md`](wall_ball_baseline_review.md) | Review snapshot | `cdb17d4`, v0.9.0, 2026-07-14/16 (+ addenda) | Post-mortems of two WallBallBaseline SAC runs: the "one-and-done" and "never-touches-the-ball" failures, geometry calibration, and the 0.11.0 bootstrap package. |
| [`design_wall_ball_checkpoint_selection_audit.md`](design_wall_ball_checkpoint_selection_audit.md) | Review snapshot | run `20260727_004014`, v0.21.0, 2026-07-27 | Paired re-scoring of four checkpoints on the goal geometry with 200 fresh seeds. Refutes the claim that the run selected the wrong checkpoint, records the second winner's-curse instance, and pins the seed ledger. |
| [`wall_ball_depth_curriculum_20260727_233859_review.md`](wall_ball_depth_curriculum_20260727_233859_review.md) | Review snapshot | run `20260727_233859`, v0.22.0, 2026-07-28 | First run of the 0.22.0 ladder: one promotion in 5.5M steps, the measured ~525k-step advance-package regression, the stage-0→1 serve-receipt discontinuity, zero goal transfer, and the sweep-uncertified ladder. Code audit of the 0.21/0.22 changes plus fresh scripted probes on the live geometry. |
| [`wall_ball_rally_diagnosis_20260728_review.md`](wall_ball_rally_diagnosis_20260728_review.md) | Review snapshot | campaign-wide, v0.24.0, 2026-07-28 | Why no run rallies from the workspace baseline: ranked diagnosis over the whole wall-ball corpus, the structural verdict retiring the sliding-fence depth ladder, the paired local SAC A/B battery and scripted-probe evidence behind it, the recalibrated certification probes, and the pre-registered `WallBallGoalRally` Colab run. |
| [`wall_ball_goal_rally_20260728_225217_review.md`](wall_ball_goal_rally_20260728_225217_review.md) | Review snapshot | run `20260728_225217`, v0.24.0, 2026-07-29 | The campaign goal is met: first `WallBallGoalRally` run sustains ≥3.0 completed returns at the workspace baseline in 1.25M steps and passes every pre-registered criterion (long-horizon 3.54 mean, 26% ≥5-survival); scorecard, provenance, and the next-phase recommendations. |
| [`wall_ball_goal_rally_replication_20260730_review.md`](wall_ball_goal_rally_replication_20260730_review.md) | Review snapshot | runs `20260729_140112` + `20260730_005134`, v0.24.0, 2026-07-30 | The replication: 2 of 3 seeds pass (seed 2 sets every record: window 3.75, audit 3.76 mean, max 11), all policies play deep with the fixed face, and seed 1's guarded collapse-to-zero-contact is documented as the recipe's open stability question. |
| [`design_wall_ball_true_baseline.md`](design_wall_ball_true_baseline.md) | Implemented design | 0.25.0, 2026-07-31 | The true-baseline extension: probe-verified workspace/serve/mechanics changes behind `WallBallTrueBaseline` (serve-speed depth map, rebound geometry refuting a baseline-only fence, the per-task in-play bound, the certification floor knob), the held-out certification, the local SAC pilot, and the pre-registered first GPU run. |
| [`wall_ball_true_baseline_20260731_132322_review.md`](wall_ball_true_baseline_20260731_132322_review.md) | Review snapshot | run `20260731_132322`, v0.25.0, 2026-08-01 | The era opens: first `WallBallTrueBaseline` run passes every pre-registered primary criterion (window 2.856, uncapped audit 3.02, first contact −6.49), the stretch shortfall is pinned on the 750-step episode cap with three independent measurements, and the seed-1 replication + 0.26.0 cap change are queued. |
| [`wall_ball_true_baseline_replication_20260801_review.md`](wall_ball_true_baseline_replication_20260801_review.md) | Review snapshot | run `20260801_144043`, v0.25.0, 2026-08-02 | The split verdict that closes the wall-ball chapter: seed 1 out-scores seed 0 (first ≥3.0 window; confirmed 3.417) but fails the deep-receive criterion at 0% — it volleys the serve at −3.4. Reliability 2/2, era skill 1/2, loophole-closing rejected ("physics sets the dominant strategy"), campaign pivots to opponent play. |
| [`design_paddle_tennis.md`](design_paddle_tennis.md) | Adopted — phase-P1 env shipped | v0.25.0 → unreleased, 2026-08-02 | PaddleTennis, the missing rung between wall-ball and humanoid tennis: 1v1 rally play on the full court with the calibrated paddles, scripted-oracle then frozen-champion opponents, cooperative rally before scoring, and the pre-committed P0–P5 probe battery that ran before any env code shipped (P0–P4 done; P5 open, gating only the phase-P2 opponent pool). |
| [`repo_infrastructure_review_20260802.md`](repo_infrastructure_review_20260802.md) | Review snapshot | `main`@`4bd715d`, v0.25.0, 2026-08-02 | Between-campaigns infrastructure review before PaddleTennis: 63 verified findings across code cleanup, consolidation, tests, docs, bugs, logging, and CI — headlined by the TOML-validator drift cluster, the video-callback exception-isolation gap, the PaddleTennis reuse blockers in `_tennis_events`/`_tennis_physics`, and the Drive run-corpus inventory. |
| [`paddle_tennis_probes_20260802.md`](paddle_tennis_probes_20260802.md) | Review snapshot | prototype, v0.25.0, 2026-08-02 | P0–P2 results: the pivot's premise is measured (opponent strokes land mean 3.3 m deep, 50% past 3 m — the depth the wall never produced), the full ITF court is infeasible at paddle power and the geometry freezes at 6.5 m / 0.914 m net, the scripted rally band is 2.0 crossings (max 10, ≥4 in 33%), and loft control is identified as the era's core difficulty. |
| [`paddle_tennis_probes_p3_p4_20260802.md`](paddle_tennis_probes_p3_p4_20260802.md) | Review snapshot | committed prototype, post-0.25.0, 2026-08-02 | P3–P4 results on the committed paddle-court prototype: a clean serve band (origin 3.25 m, 9 m/s, 18–24° — 98–100% legal, landing 4.3–4.8 m deep, 92–100% returnable) with a rally tail of 3.15–3.42 crossings, above the P0–P2 scratchpad band; server self-touch identified as a flight-path hazard; alternation exactly fair (36/36 mirrored cell pairs identical); P4's mirroring identity passes bit-for-bit on observations with sign-exact action mirroring. Records two measurement pitfalls (parked-obstacle censoring, the shot-crossing end-state latch) for the certification design. P5 (champion transfer) stays open. |
| [`paddle_tennis_pilot_and_first_run_20260802.md`](paddle_tennis_pilot_and_first_run_20260802.md) | Review snapshot + pre-registration | unreleased, 2026-08-02 | The era's first learned run: a 500k-step local SAC pilot of the stock recipe passes the scripted band by ~190k steps and reaches crossings 6.40 (p90 10, max 11, success rate 1.00, zero timeouts/nonfinite, exact serve alternation) with no plateau — the cooperative rally is directly learnable, unshaped, from a cold start. Pre-registers the first GPU run (seed 1, stock TOML, primary ≥ 6.0, stretch ≥ 10.0, held-out gate on reserved block 4100–4199 at ≥ 85% of the selected window). |
| [`paddle_tennis_p5_transfer_20260802.md`](paddle_tennis_p5_transfer_20260802.md) | Review snapshot | unreleased, 2026-08-02 | P5 (champion transfer), the instrument half: the wall-ball→paddle-court observation/action shim, its scripted calibration matrix (rigid translation broken by command-range geometry; serve-yield overlay mandatory — every unyielded serving point dies `wrong_hitter`; `scaled + yield` reaches 49% of the native rally tail at a 96% returned-serve rate), and the pre-registered champion pool-admission rule. Champion rows run on Colab (checkpoints exceed the session's Drive transport); seeds 5000–5099 burned. |
| [`paddle_tennis_env_20260802.md`](paddle_tennis_env_20260802.md) | Task-definition record | unreleased, 2026-08-02 | The PaddleTennis env freeze: the registered `CourtsideDynamics/PaddleTennis` phase-P1 task definition on the P0–P4 numbers — frozen geometry/serve/reward/observation contract, the `crossings` selection metric and its committed reference band (3.15–3.42), the decisions taken at the freeze (shared cooperative fault penalty, one alternating-serve point per episode, no shaping), and the **held-out certification: PASS** (seeds 3100–3199, mean crossings 3.22 vs floor 2.6, zero unsafe; 4100–4199 stays reserved). Open: P5 and paddle-pitch loft authority. |

| [`design_wall_ball_paddle_orientation.md`](design_wall_ball_paddle_orientation.md) | **Proposed** (not implemented) | v0.22.0, 2026-07-27 | Actuating paddle pitch/yaw: verified MuJoCo change, the action/observation/entropy ripples, why pitch precedes yaw, and the scripted-oracle gate. Explicitly blocked on the depth ladder working first. |

> The table above is incomplete — several depth-campaign documents
> (`design_wall_ball_depth_curriculum.md`, `design_wall_ball_serve_alignment.md`,
> `lessons_learned.md`, `plan_wall_ball_aligned_deep_stages.md`,
> `wall_ball_aligned_patience_review.md`,
> `wall_ball_depth_curriculum_run1_review.md`,
> `design_revert_and_run_layout.md`) are not yet listed.

## Adding a new document

- A recurring lesson, a bug worth remembering, or a "we tried X and rejected it"
  finding → add an entry to [`DECISIONS.md`](DECISIONS.md) (newest first). A full
  investigation can also land as its own **review snapshot** here, but lift its
  durable conclusions into the journal so they don't stay buried.
- A format/contract others depend on → a **living spec** here, kept current.
- A release's behavior/observation/recipe changes → the root
  [`CHANGELOG.md`](../CHANGELOG.md).
- Prefer dating review/design docs and naming the commit or version they were
  written against, so their scope is unambiguous later.
