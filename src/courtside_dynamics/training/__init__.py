"""Shared training entry points for the Courtside Dynamics curriculum."""
from courtside_dynamics.training.monitor_log import (
    MonitorBundle,
    load_monitor_episodes,
)
from courtside_dynamics.training.train import TrainConfig, train

__all__ = [
    "MonitorBundle",
    "TrainConfig",
    "load_monitor_episodes",
    "train",
]
