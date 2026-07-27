# Documentation

Supplementary documentation for Courtside Dynamics. The user-facing overview is
the root [`README.md`](../README.md); per-release migration notes are in the root
[`CHANGELOG.md`](../CHANGELOG.md).

Docs here fall into three kinds, distinguished so a living reference is never
mistaken for a point-in-time snapshot:

- **Living** — kept current with the code (specs, the decisions journal).
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
