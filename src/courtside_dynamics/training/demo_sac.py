"""DemoSAC: SAC with demonstration injection from a harvested library.

The LD1′ mechanism (docs/design_paddle_tennis_demo_injection.md):
a second replay buffer holding the ``k2-demo-library-v0`` transitions
harvested by ``tools/paddle_tennis_k2_demo_harvest.py`` (oracle
completions from the policy's own failure states, real rally context,
conversion payment in-buffer), sampled at a fixed per-minibatch
fraction into every gradient step, plus an optional behavior-cloning
term on the demo rows (unfiltered by default; a Q-filter — clone only
where the critics already rank the demo action above the policy's —
is selectable, and the D-C arming measurement it depends on is logged
as ``train/demo_q_ordering`` on held-out demo states).

**Default-off is bit-identical to stock SAC** (SD0): with
``demo_library=None`` the training step makes exactly the calls
``SAC.train`` makes, in the same order — one replay sample per
gradient step, no extra RNG draw, no extra forward pass. The
composition is decided once per gradient step by arithmetic
(``round(demo_fraction * batch_size)``), never by a random draw, so
the demo share is exact. When the surface is ON, the demo buffer's
own sample draws interleave with the live buffer's on the global
numpy RNG (so an ON run's live sample stream differs from a SAC
run's — expected, and recorded in the design); only the OFF case
is stream-identical.

The demo buffer is built lazily at the start of ``learn()`` (and on
first use of the ordering metric), not at construction: a checkpoint
reloaded for inference through an algo-resolving loader must not
need the library file present.

Not supported with n-step returns (``n_steps > 1``): refused at
construction rather than silently mixing 1-step demo targets into
n-step live targets.
"""
from __future__ import annotations

import hashlib
import pickle
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

DEMO_LIBRARY_SCHEMA = "k2-demo-library-v0"
_BC_FILTERS = ("none", "q")
_WINDOWS = ("point", "to_confirm")


class DemoSAC(SAC):
    """SAC + demonstration injection (default-off, bit-identical off)."""

    def __init__(
        self,
        policy: Any,
        env: Any,
        *,
        demo_library: str | None = None,
        demo_fraction: float = 0.0,
        demo_bc_coef: float = 0.0,
        demo_bc_filter: str = "none",
        demo_window: str = "point",
        **sac_kwargs: Any,
    ) -> None:
        fraction = float(demo_fraction)
        bc_coef = float(demo_bc_coef)
        if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError(
                f"demo_fraction must be in [0.0, 1.0), got {demo_fraction!r}"
            )
        if not np.isfinite(bc_coef) or bc_coef < 0.0:
            raise ValueError(
                f"demo_bc_coef must be finite and non-negative, got {demo_bc_coef!r}"
            )
        if demo_bc_filter not in _BC_FILTERS:
            raise ValueError(
                f"demo_bc_filter must be one of {_BC_FILTERS}, got {demo_bc_filter!r}"
            )
        if demo_window not in _WINDOWS:
            raise ValueError(
                f"demo_window must be one of {_WINDOWS}, got {demo_window!r}"
            )
        enabled = fraction > 0.0 or bc_coef > 0.0
        if (demo_library is None) == enabled:
            raise ValueError(
                "demo_library and the demo terms must be enabled together: got "
                f"demo_library={demo_library!r} with demo_fraction="
                f"{demo_fraction!r}, demo_bc_coef={demo_bc_coef!r}"
            )
        if bc_coef > 0.0 and fraction == 0.0:
            raise ValueError(
                "demo_bc_coef > 0 needs demo rows in the minibatch: set "
                "demo_fraction > 0"
            )
        self.demo_library = demo_library
        self.demo_fraction = fraction
        self.demo_bc_coef = bc_coef
        self.demo_bc_filter = demo_bc_filter
        self.demo_window = demo_window
        self.demo_library_sha256: str | None = None
        self.demo_buffer: ReplayBuffer | None = None
        self.demo_holdout: tuple[np.ndarray, np.ndarray] | None = None
        self.demo_transitions: int = 0
        super().__init__(policy, env, **sac_kwargs)

    # -- setup / persistence ------------------------------------------------

    def _setup_model(self) -> None:
        super()._setup_model()
        self.demo_buffer = None
        self.demo_holdout = None
        self.demo_transitions = 0
        if self.demo_library is None:
            return
        if self.n_steps != 1:
            raise ValueError(
                "DemoSAC does not support n_steps > 1 with a demo library "
                "(1-step demo targets would mix into n-step live targets)"
            )
        # The realized composition must actually mix: a fraction that
        # rounds to zero demo rows (or to the whole batch) at this
        # batch size would be a silent no-op (or a pure-demo batch)
        # while config.json records the library as in play.
        n_demo = int(round(self.demo_fraction * self.batch_size))
        if self.demo_fraction > 0.0 and not 1 <= n_demo <= self.batch_size - 1:
            raise ValueError(
                f"demo_fraction={self.demo_fraction} at batch_size="
                f"{self.batch_size} rounds to {n_demo} demo rows per "
                "minibatch; it must round to at least 1 and at most "
                "batch_size - 1"
            )

    def _setup_learn(self, *args: Any, **kwargs: Any) -> tuple[int, BaseCallback]:
        # The demo buffer is built at the first learn(), not at
        # construction: inference-side loaders (eval notebooks,
        # SAC.load of a DemoSAC checkpoint) never need the library file.
        self._ensure_demo_loaded()
        return super()._setup_learn(*args, **kwargs)

    def _ensure_demo_loaded(self) -> None:
        """Build the demo buffer from the library on first need."""
        if self.demo_library is not None and self.demo_buffer is None:
            self._load_demo_library(self.demo_library)

    def _excluded_save_params(self) -> list[str]:
        # The buffers are rebuilt from the library path at load; never
        # pickled into the checkpoint.
        return super()._excluded_save_params() + [  # noqa: RUF005
            "demo_buffer",
            "demo_holdout",
        ]

    def _load_demo_library(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = f.read()
        self.demo_library_sha256 = hashlib.sha256(payload).hexdigest()
        library = pickle.loads(payload)
        schema = library.get("schema") if isinstance(library, dict) else None
        if schema != DEMO_LIBRARY_SCHEMA:
            raise ValueError(
                f"demo_library {path!r} has schema {schema!r}; expected "
                f"{DEMO_LIBRARY_SCHEMA!r}"
            )
        trajectories = list(library["trajectories"])
        obs_dim = int(np.prod(self.observation_space.shape))
        act_dim = int(np.prod(self.action_space.shape))
        train_rows: list[tuple[np.ndarray, ...]] = []
        holdout_obs: list[np.ndarray] = []
        holdout_act: list[np.ndarray] = []
        for position, traj in enumerate(trajectories):
            obs = np.asarray(traj["obs"], dtype=np.float64)
            env_actions = np.asarray(traj["actions"], dtype=np.float64)
            next_obs = np.asarray(traj["next_obs"], dtype=np.float64)
            rewards = np.asarray(traj["rewards"], dtype=np.float64)
            terminated = np.asarray(traj["terminated"], dtype=bool)
            truncated = np.asarray(traj["truncated"], dtype=bool)
            steps = len(rewards)
            # Shapes are checked on the RAW arrays: scale_action would
            # broadcast a wrongly shaped action array back to the env's
            # action width and hide the defect.
            if not (
                obs.shape == (steps, obs_dim)
                and next_obs.shape == (steps, obs_dim)
                and env_actions.shape == (steps, act_dim)
                and terminated.shape == (steps,)
                and truncated.shape == (steps,)
                and steps >= 1
            ):
                raise ValueError(
                    f"demo_library {path!r} trajectory {position} has "
                    "inconsistent array shapes for this env"
                )
            # The live buffer holds policy-space actions (SB3's
            # scale_action of the env action); demos are recorded as
            # env actions, so map them the same way (identity on a
            # [-1, 1] Box, exact on any Box).
            actions = np.asarray(
                self.policy.scale_action(env_actions), dtype=np.float64
            )
            if self.demo_window == "to_confirm":
                end = int(traj["confirm_step"]) + 1
                obs, actions, next_obs = obs[:end], actions[:end], next_obs[:end]
                rewards, terminated, truncated = (
                    rewards[:end], terminated[:end], truncated[:end]
                )
            if traj.get("split") == "heldout":
                holdout_obs.append(obs)
                holdout_act.append(actions)
                continue
            train_rows.append((obs, actions, next_obs, rewards, terminated, truncated))
        n_train = int(sum(len(rows[3]) for rows in train_rows))
        if n_train == 0:
            raise ValueError(f"demo_library {path!r} has no training transitions")
        buffer = ReplayBuffer(
            n_train,
            self.observation_space,
            self.action_space,
            device=self.device,
            n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=True,
        )
        for obs, actions, next_obs, rewards, terminated, truncated in train_rows:
            for t in range(len(rewards)):
                done = bool(terminated[t] or truncated[t])
                buffer.add(
                    obs[t][None, :],
                    next_obs[t][None, :],
                    actions[t][None, :],
                    np.asarray([rewards[t]], dtype=np.float32),
                    np.asarray([done]),
                    [{"TimeLimit.truncated": bool(truncated[t] and not terminated[t])}],
                )
        assert buffer.full and buffer.pos == 0  # exactly n_train adds
        self.demo_buffer = buffer
        self.demo_transitions = n_train
        if holdout_obs:
            self.demo_holdout = (
                np.concatenate(holdout_obs, axis=0),
                np.concatenate(holdout_act, axis=0),
            )

    # -- the D-C arming measurement ----------------------------------------

    def demo_q_ordering(self) -> float | None:
        """Fraction of held-out demo states where min-Q ranks the demo
        action above the policy's deterministic action (the ordering
        the Q-filter needs; Phase 0 measured it at coin-flip)."""
        self._ensure_demo_loaded()
        if self.demo_holdout is None:
            return None
        obs, actions = self.demo_holdout
        if self._vec_normalize_env is not None:
            obs = self._vec_normalize_env.normalize_obs(obs)
        with th.no_grad():
            obs_t = th.as_tensor(obs, dtype=th.float32, device=self.device)
            act_t = th.as_tensor(actions, dtype=th.float32, device=self.device)
            pi = self.actor(obs_t, deterministic=True)
            q_demo = th.min(th.cat(self.critic(obs_t, act_t), dim=1), dim=1).values
            q_pi = th.min(th.cat(self.critic(obs_t, pi), dim=1), dim=1).values
            return float((q_demo > q_pi).float().mean().item())

    # -- training ------------------------------------------------------------

    def _sample_minibatch(self, batch_size: int) -> tuple[ReplayBufferSamples, int]:
        """Live-then-demo minibatch; identical to SAC's call when demos are
        off. When on, the demo sample's own global-RNG draws interleave
        with the live buffer's (the live stream is not SAC's)."""
        n_demo = (
            int(round(self.demo_fraction * batch_size))
            if self.demo_buffer is not None
            else 0
        )
        assert self.replay_buffer is not None
        if n_demo == 0:
            return (
                self.replay_buffer.sample(batch_size, env=self._vec_normalize_env),
                0,
            )
        assert self.demo_buffer is not None
        live = self.replay_buffer.sample(
            batch_size - n_demo, env=self._vec_normalize_env
        )
        demo = self.demo_buffer.sample(n_demo, env=self._vec_normalize_env)
        discounts = None
        if live.discounts is not None:
            demo_discounts = (
                demo.discounts
                if demo.discounts is not None
                else th.full_like(demo.rewards, float(self.gamma))
            )
            discounts = th.cat([live.discounts, demo_discounts], dim=0)
        return (
            ReplayBufferSamples(
                observations=th.cat([live.observations, demo.observations], dim=0),
                actions=th.cat([live.actions, demo.actions], dim=0),
                next_observations=th.cat(
                    [live.next_observations, demo.next_observations], dim=0
                ),
                dones=th.cat([live.dones, demo.dones], dim=0),
                rewards=th.cat([live.rewards, demo.rewards], dim=0),
                discounts=discounts,
            ),
            n_demo,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # SAC.train (SB3 2.9.0) verbatim, except: the minibatch comes
        # from _sample_minibatch, and the demo rows (the LAST n_demo of
        # the batch) may carry a behavior-cloning term on the actor.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        bc_losses: list[float] = []
        filter_pass: list[float] = []

        for gradient_step in range(gradient_steps):
            replay_data, n_demo = self._sample_minibatch(batch_size)
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

            if n_demo and self.demo_bc_coef > 0.0:
                demo_obs = replay_data.observations[-n_demo:]
                demo_actions = replay_data.actions[-n_demo:]
                bc_actions = self.actor(demo_obs, deterministic=True)
                per_row = ((bc_actions - demo_actions) ** 2).mean(dim=1)
                if self.demo_bc_filter == "q":
                    with th.no_grad():
                        q_demo = th.min(
                            th.cat(self.critic(demo_obs, demo_actions), dim=1), dim=1
                        ).values
                        q_pi = th.min(
                            th.cat(self.critic(demo_obs, bc_actions.detach()), dim=1),
                            dim=1,
                        ).values
                        mask = (q_demo > q_pi).float()
                    filter_pass.append(float(mask.mean().item()))
                    bc_loss = (per_row * mask).sum() / th.clamp(mask.sum(), min=1.0)
                else:
                    bc_loss = per_row.mean()
                bc_losses.append(bc_loss.item())
                actor_loss = actor_loss + self.demo_bc_coef * bc_loss
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        if self.demo_buffer is not None:
            self.logger.record("train/demo_fraction", self.demo_fraction)
            if bc_losses:
                self.logger.record("train/demo_bc_loss", np.mean(bc_losses))
            if filter_pass:
                self.logger.record("train/demo_q_filter_pass", np.mean(filter_pass))
            if self._n_updates // gradient_steps % 50 == 0:
                ordering = self.demo_q_ordering()
                if ordering is not None:
                    self.logger.record("train/demo_q_ordering", ordering)


__all__ = ["DEMO_LIBRARY_SCHEMA", "DemoSAC"]
