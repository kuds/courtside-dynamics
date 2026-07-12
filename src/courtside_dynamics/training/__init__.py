"""Shared training entry points for the Courtside Dynamics curriculum."""
from courtside_dynamics.training.monitor_log import (
    MonitorBundle,
    load_monitor_episodes,
)
from courtside_dynamics.training.tennis_curriculum import (
    CurriculumEpisodeResult,
    CurriculumEvaluationSummary,
    EvaluationProvenance,
    HeldOutCondition,
    HeldOutSeedSuite,
    PromotionConfig,
    PromotionReport,
    RateSummary,
    assess_curriculum_promotion,
    evaluate_curriculum_stage,
    summarize_curriculum_episodes,
)
from courtside_dynamics.training.train import TrainConfig, train

__all__ = [
    "MonitorBundle",
    "CurriculumEpisodeResult",
    "CurriculumEvaluationSummary",
    "EvaluationProvenance",
    "HeldOutCondition",
    "HeldOutSeedSuite",
    "PromotionConfig",
    "PromotionReport",
    "RateSummary",
    "TrainConfig",
    "assess_curriculum_promotion",
    "evaluate_curriculum_stage",
    "load_monitor_episodes",
    "summarize_curriculum_episodes",
    "train",
]
