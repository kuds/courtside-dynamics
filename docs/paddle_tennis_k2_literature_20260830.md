# Literature input: the k=2 fork against the published record (2026-08-30)

Status: **Literature input to the unfrozen k=2 drill design
([`design_paddle_tennis_k2_drill.md`](design_paddle_tennis_k2_drill.md))
and its §5 routed alternative — no freeze implied, no verdict
booked.** Produced by a six-theme research sweep (reset-state
curricula, demonstration injection into off-policy replay,
collapsed-entropy fine-tuning, context gating, robot table tennis,
sparse-success mechanics); every cited paper was verified against
its primary source (arXiv abstract, and the paper body for each
load-bearing number). The maintainer weighs this alongside the
campaign's own measurements; where the two disagree, the
campaign's measurements are on this task and the literature is
not.

## 1. The question put to the literature

The campaign's live fork: train the second ball by (a) a
**drill** launching points into harvested real k=2 states
(feed-context step-0 6.9%, full-context 2.0%, design §3a), or (b)
**demonstration injection** of the scripted oracle's k=2
transitions (98.2% converter) into the SAC replay buffer — under
a warm start whose temperature is collapsed (ent_coef annealed
low; the LT1 re-heat-alone lever is already adjudicated FAIL).
The deficit is measured context-gated: identical physics converts
6.9% presented as a fresh feed vs 2.0% in the real mid-rally
context.

## 2. The engagement floor: both drill arms sit at or below it

Every reset-to-state method that quantifies engagement keeps its
reset frontier where success is materially higher than ours:

| method | floor / pacing rule | our position |
|---|---|---|
| Reverse Curriculum Generation (Florensa et al., CoRL 2017, [1707.05300](https://arxiv.org/abs/1707.05300)) | trains only "good starts" with success in a 10–90% band; below 10% = uninformative | full arm 2.0% below the band; feed arm 6.9% just below |
| Single-demo Montezuma (Salimans & Chen 2018, [1812.03381](https://arxiv.org/abs/1812.03381)) | reset point advances only at ~20% worker success; backward curriculum "vitally important" | an order of magnitude above our step-0 |
| Backplay (Resnick et al. 2018, [1807.06919](https://arxiv.org/abs/1807.06919)) | too-fast curriculum advancement fails; too-slow never does; windows of states beat single points | direct hard-state launch = the limit of infinitely fast advancement |
| RFCL (Tao et al., ICLR 2024, [2405.03379](https://arxiv.org/abs/2405.03379)) | SAC + per-trajectory reverse frontier advanced on consecutive successes, then a forward curriculum prioritizing intermediate-success starts | the published recipe closest to this exact setting (SAC, state resets, scripted demos) |

Two partial counterexamples, both instructive:

- **Go-Explore** (Ecoffet et al., Nature 2021,
  [2004.12919](https://arxiv.org/abs/2004.12919)) teleports
  straight to hard states with no annealing — but only pays off
  because the explore step from those states is deliberately
  high-entropy. A collapsed-temperature policy launched into k=2
  states is *return without explore*; the prediction is the
  step-0 rate stays flat. (Consistent with the campaign's own
  T1-FAIL from the other side: re-heat without the data was also
  not enough. The literature's claim is that the *package* is
  the mechanism.)
- **JSRL** (Uchendu et al., ICML 2023,
  [2204.02372](https://arxiv.org/abs/2204.02372)): its random
  switch-point ablation reached comparable final performance to
  the ordered curriculum ("good visitation states matter more
  than their order"), at an early sample-efficiency cost — mild
  evidence direct hard-state launches can work off-policy.

**Implication for the D2 fork:** the literature does not endorse
either pure arm at a fixed fraction. The prescribed shape is
difficulty-ordered launches — the feed-like presentations (6.9%,
at the band edge) first, expanding toward the real-context states
as their per-state conversion rises — or a per-trajectory
backward walk from the oracle's contact (harvest the oracle's own
k=2 rollouts, reset near the hit, anneal earlier), which is RFCL
and Salimans–Chen ported to this env. The shipped mechanism's
`drill_context` flag and per-entry provenance already carry the
needed hooks; a difficulty-ordered library is a harvest-time
construction, not an env change.

## 3. Drill vs demonstration injection is a false dichotomy

The single closest precedent — Nair et al., ICRA 2018,
[1709.10089](https://arxiv.org/abs/1709.10089) — uses **both**
levers in one recipe (demonstration replay buffer sampled every
batch + resets into demonstration states) and finds them
complementary. Its Q-filter (clone the demo action only where the
critic says it beats the policy's) is precisely the guard this
campaign needs: cloning pressure confines itself to the k=2
context states while the ~90% k=1 receive is left alone. ~100
demonstrations sufficed there; a 98% oracle generates that in
minutes.

The modern low-surgery recipe for the injection half is **RLPD**
(Ball et al., ICML 2023,
[2302.02948](https://arxiv.org/abs/2302.02948)): built on SAC — a
second replay buffer of demo transitions, symmetric 50/50
minibatch sampling (no ratio scheduling), LayerNorm in the
critic (the cheap fix for Q-overestimation on
out-of-distribution demo actions), automatic entropy tuning kept
on. SB3 surgery is modest. The named failure mode of naive
injection — the actor exploiting overestimated Q on demo states —
is documented in **AWAC** (Nair et al. 2020,
[2006.09359](https://arxiv.org/abs/2006.09359)) and **Cal-QL**
(Nakamoto et al., NeurIPS 2023,
[2303.05479](https://arxiv.org/abs/2303.05479)), which also
predict an early performance dip on fine-tune and prescribe
calibrated/conservative critic targets if it bites; DQfD's
ablations ([1704.03732](https://arxiv.org/abs/1704.03732)) show
naive buffer-dropping without guaranteed demo sampling is the
weakest variant. SQIL
([1905.11108](https://arxiv.org/abs/1905.11108)) is the boundary
marker: use its balanced-sampling machinery, never its
reward-replacement (a working reward and a 90% k=1 policy exist).

**One demo-harvest requirement follows from the campaign's own
context-gate measurement:** oracle k=2 transitions must be
harvested in the real mid-rally context, rally flags intact — a
fresh-feed demo harvest would inject the wrong observation
distribution by our own diagnosis.

## 4. A third arm the design should name: oracle roll-in (JSRL-style)

JSRL needs no state harvesting at all: the oracle plays side A
live through the serve and k=1, and control hands to the policy
at the k=2 ball. This generates the deficit-carrying rally-context
features *natively* (no restore fidelity question), gives fresh
scenario diversity every launch (no library-memorization concern,
no D7 staleness — the roll-in distribution tracks the current
opponent automatically), and is proven with off-policy learners.

Honest caveat, stated before anyone freezes it: the policy then
inherits the **oracle's** paddle position at hand-off (~0.9 m
from the bounce), not the trained policy's own mispositioned
start (~3 m). Roll-in trains *conversion from a good position*;
the harvested-state arms train *repositioning plus conversion* —
which PT2 measured as the actual gap. The two are different
sub-skills, and the choice is a maintainer call the step-0
instruments can price the same way the D2 arms were priced (an
oracle roll-in step-0 row is cheap to add to the battery).

## 5. Temperature: a package term, not a competing lever

The SAC mechanism paper (Haarnoja et al. 2018,
[1812.05905](https://arxiv.org/abs/1812.05905)) explains the
collapsed checkpoint exactly: the dual update anneals ent_coef
low once the policy satisfies its entropy target on the
k=1-dominated distribution, and it stays pinned there. Cheap
re-heat moves with precedent: re-initialize log_alpha at
fine-tune start and/or raise target_entropy — *as part of* a
data intervention, which is compatible with the T1-FAIL booking
(re-heat alone was the falsified lever; Go-Explore's lesson is
that the data intervention alone is the mirror-image half).

Two diagnostics worth running on the checkpoint before any
pilot, both cheap:

- **Dormant-neuron fraction** (Sokar et al., ICML 2023,
  [2302.12902](https://arxiv.org/abs/2302.12902)) on the
  obs-encoder MLP — if capacity is gone, no data intervention
  engages, and ReDo-style recycling or head resets are the fix.
- **Primacy bias** (Nikishin et al., ICML 2022,
  [2205.07802](https://arxiv.org/abs/2205.07802)): the
  prescription of resetting actor/critic output heads while
  keeping the buffer maps to this warm start (shaped almost
  entirely by k=1 evidence); plasticity-loss results (Abbas et
  al. 2023, [2303.07507](https://arxiv.org/abs/2303.07507))
  point the same way.

## 6. The context gate has a name

The measured 6.9%-vs-2.0% gap on identical physics is **causal
confusion / observational overfitting**: a policy that can read
context-indicator features uses them as action gates, and the
deficit lives in the observation side, not the motor skill.

- Causal Confusion in Imitation Learning (de Haan et al.,
  NeurIPS 2019, [1905.11979](https://arxiv.org/abs/1905.11979));
  copycat/nuisance features (Wen et al., NeurIPS 2020,
  [2010.14876](https://arxiv.org/abs/2010.14876));
  Observational Overfitting (Song et al., ICLR 2020,
  [1912.02975](https://arxiv.org/abs/1912.02975)); ZSG survey
  (Kirk et al., JAIR 2023,
  [2111.09794](https://arxiv.org/abs/2111.09794)).
- **The sharpest theory result bears directly on the D2 fork**:
  Invariant Causal Prediction for Block MDPs (Zhang et al., ICML
  2020, [2003.06016](https://arxiv.org/abs/2003.06016)) —
  fresh-feed and mid-rally are two environments of one Block MDP;
  training in ONE context provably cannot disentangle ball-state
  features from context correlates, while training across BOTH
  with an invariance pressure provably can. Prediction: the
  feed-only arm (a) does not zero-shot transfer; the prescribed
  shape mixes both contexts.
- **Cheap precedented add-on**: random masking/dropout of the
  rally-flag observation dims on drilled points (OREO, Park et
  al., NeurIPS 2021,
  [2110.14118](https://arxiv.org/abs/2110.14118) — whose Pong
  score-indicator case is a near-exact analog; de Haan's
  mask-conditioned training). Our obs is already a disentangled
  48-d vector with known semantic groups, so group-dropout of
  indices 24–47 subsets is near-zero-cost to implement — an era
  surface (observation processing) that would need its own
  pricing if proposed.

## 7. Robot table tennis precedents

- **DeepMind 2024** (D'Ambrosio et al.,
  [2408.03906](https://arxiv.org/abs/2408.03906)): the training
  ball distribution is a harvested bank of real states grown
  iteratively from deployment play — strong precedent for the
  drill mechanism, with the explicit warning that mid-rally
  states are policy-conditioned, so a static library (our D7) is
  a named limitation, not a detail.
- **i-Sim2Real** (Abeyruwan et al., CoRL 2022,
  [2207.06572](https://arxiv.org/abs/2207.06572)): the clearest
  published statement that rally play fails from feed-distribution
  training precisely because the rally distribution is conditioned
  on the policy itself — the campaign's context gate, observed in
  hardware; prescribes iterated harvest-retrain loops.
- **HYSR** (Büchler et al., T-RO 2022,
  [2006.05935](https://arxiv.org/abs/2006.05935)): trains
  entirely by resetting into recorded real ball trajectories —
  the drill's core move, validated from scratch (single returns,
  not rally depth).
- **GoalsEye** (Ding et al. 2022,
  [2210.03662](https://arxiv.org/abs/2210.03662)): consumes
  successful transitions through imitation/self-supervised
  practice with continued data expansion, not inert buffer
  filler — a vote for the Q-filtered-BC flavor of injection over
  raw TD on demo transitions.
- **RSS 2023 case study** (D'Ambrosio et al.,
  [2309.03315](https://arxiv.org/abs/2309.03315)): treats
  train-vs-deploy distribution shift as a first-class measured
  failure mode — the published analog of our fresh-feed vs
  in-context gap.

## 8. What carries gradient below the floor

- **SAC-X** (Riedmiller et al., ICML 2018,
  [1802.10567](https://arxiv.org/abs/1802.10567)): auxiliary
  tasks with separate critic heads, not one shaped scalar — the
  principled version of what our escrows approximate; the named
  fallback if both drill and injection underdeliver.
- **HER** (Andrychowicz et al., NeurIPS 2017,
  [1707.01495](https://arxiv.org/abs/1707.01495)): does not
  apply off the shelf (obs is not goal-conditioned); an
  action-relabeling variant (relabel the commanded target to the
  achieved outcome on failed swings) would be its own designed
  era surface.
- **Asymmetric self-play** (OpenAI 2021,
  [2101.04882](https://arxiv.org/abs/2101.04882)): selective
  cloning on currently-failing goals maps onto cloning oracle
  behavior only in the in-context k=2 states.

## 9. Routing implications (PROPOSAL — the maintainer decides)

Weighing the sweep against the campaign's own measurements:

1. **Neither pure D2 arm at a fixed fraction matches the
   literature's shape.** Arm (a) trains the wrong context (§6
   predicts no transfer); arm (b) trains the right context from
   below the engagement floor (§2). The literature-shaped drill
   is: both contexts mixed, difficulty-ordered (feed-like first,
   real-context as per-state conversion rises), with the escrows
   carrying sub-floor gradient — implementable as harvest-time
   library construction plus a fraction schedule, on the shipped
   mechanism.
2. **Injection joins the drill rather than replacing it**
   (Nair): RLPD-style second buffer + 50/50 sampling + LayerNorm
   critic, plus a Q-filtered BC term if actor-side pull is
   needed; harvest demos in the real context.
3. **Temperature re-heat rides along** (log_alpha re-init /
   target-entropy raise at fine-tune start) — a package term,
   compatible with the T1-FAIL booking of re-heat-alone.
4. **Name oracle roll-in as arm (c)** and price its step-0 row
   in the same battery (cheap), with the §4 caveat recorded.
5. **Run the two checkpoint diagnostics first** (dormant-neuron
   fraction, primacy-bias audit) — they are hours of CPU and can
   redirect everything above.

Each numbered item that touches a frozen or priced surface goes
through the standing design-freeze discipline; nothing here
launches anything.
