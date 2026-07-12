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
    SideSuccessSummary,
    assess_curriculum_promotion,
    evaluate_curriculum_stage,
    summarize_curriculum_episodes,
)
from courtside_dynamics.training.train import (
    SelectiveVecNormalize,
    TrainConfig,
    WarmStartConfig,
    train,
)

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
    "SelectiveVecNormalize",
    "SideSuccessSummary",
    "TrainConfig",
    "WarmStartConfig",
    "assess_curriculum_promotion",
    "evaluate_curriculum_stage",
    "load_monitor_episodes",
    "summarize_curriculum_episodes",
    "train",
]
