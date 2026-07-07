"""Shared plumbing for the Courtside Dynamics MuJoCo environments.

Every env in the curriculum renders the same way (100 fps, the standard
mode trio), truncates on an internal ``episode_len`` counter, applies the
same small uniform state noise on reset, and builds its observation from
interleaved per-joint ``qpos``/``qvel`` blocks. Those pieces live here so
each env file only contains what is actually unique to it: the reward
function, the termination rule, and the observation layout.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class CourtsideMujocoEnv(MujocoEnv):
    """Base class for the curriculum envs.

    Subclasses implement ``step`` / ``reset_model`` / ``_get_obs`` as
    usual, call :meth:`_noisy_init_state` from ``reset_model``, and can
    use :meth:`_joints_obs` to assemble the per-joint part of the
    observation vector.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        model_path: str,
        episode_len: int,
        obs_dim: int,
        frame_skip: int = 5,
        **kwargs: Any,
    ) -> None:
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip,
            observation_space=observation_space,
            **kwargs,
        )
        self.episode_len = int(episode_len)
        self.step_number = 0

    def _noisy_init_state(
        self, noise: float = 0.01
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(qpos, qvel)``: the model keyframe plus uniform noise."""
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-noise, high=noise
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-noise, high=noise
        )
        return qpos, qvel

    def _joints_obs(self, *names: str) -> np.ndarray:
        """Concatenate ``qpos``/``qvel`` pairs for each named joint, in order."""
        parts: list[np.ndarray] = []
        for name in names:
            joint = self.data.joint(name)
            parts.append(np.asarray(joint.qpos))
            parts.append(np.asarray(joint.qvel))
        return np.concatenate(parts)
