"""Side-A-local feed geometry shared by every serve/launch config.

Both the free-standing serve (:class:`TennisServeConfig`) and the
curriculum launch (``CurriculumLaunchConfig``) describe the initial
feed in side-A court coordinates and mirror it 180 degrees about z for
side B. The geometric contract they share -- start on one court half,
keep noise off the net plane, stay inside the court width, keep the
reset height in the 1.1-1.5 m window -- previously lived as two
diverging ``__post_init__`` copies in two modules; it lives once here
so the configs (and the PaddleTennis serve-rules probe, which needs
exactly this machinery) validate one implementation, and the side-B
mirroring is one helper instead of per-sampler ``[:2] *= -1`` lines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from courtside_dynamics.envs._tennis_physics import (
    COURT_HALF_LENGTH,
    COURT_HALF_WIDTH,
)
from courtside_dynamics.envs.tennis_rules import CourtSide

#: The reset-height band every feed must keep the ball inside, noise
#: included: below it the ball clips the court on the first frame,
#: above it the feed stops resembling a struck ball.
FEED_HEIGHT_WINDOW = (1.1, 1.5)


def validate_feed_geometry(
    *,
    kind: str,
    start_distance_from_net: float,
    lateral_position: float,
    height: float,
    position_noise: tuple[float, float, float],
    require_inside_baseline: bool = False,
    half_length: float = COURT_HALF_LENGTH,
    half_width: float = COURT_HALF_WIDTH,
) -> None:
    """Raise unless a side-A-local feed can legally start anywhere in
    its noise box.

    ``kind`` prefixes the error messages (``"serve"`` / ``"launch"``)
    so a failure names the config that carried the bad value.
    ``require_inside_baseline`` additionally bounds the longitudinal
    noise by the baseline (the curriculum launch's stricter contract).
    ``half_length``/``half_width`` default to the regulation court; a
    smaller court passes its own dimensions.
    """
    scalars = (
        start_distance_from_net,
        lateral_position,
        height,
        *position_noise,
    )
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError(
            f"{kind} feed geometry must contain only finite values"
        )
    if any(value < 0.0 for value in position_noise):
        raise ValueError(
            f"{kind} position noise values must be non-negative"
        )
    if not 0.0 < start_distance_from_net <= half_length:
        raise ValueError(
            f"{kind} start_distance_from_net must lie on one court half"
        )
    if start_distance_from_net <= position_noise[0]:
        raise ValueError(
            f"{kind} longitudinal position noise must keep the ball on "
            f"its serving side"
        )
    if require_inside_baseline and (
        start_distance_from_net + position_noise[0] > half_length
    ):
        raise ValueError(
            f"{kind} position noise must keep the launch inside the "
            f"baseline"
        )
    if abs(lateral_position) + position_noise[1] > half_width:
        raise ValueError(
            f"{kind} lateral position/noise must start inside the court "
            f"width"
        )
    height_low, height_high = FEED_HEIGHT_WINDOW
    if not (
        height_low <= height - position_noise[2]
        and height + position_noise[2] <= height_high
    ):
        raise ValueError(
            f"{kind} height and noise must keep the reset height within "
            f"{height_low}–{height_high} m"
        )


def mirror_for_side(vector: np.ndarray, side: CourtSide) -> np.ndarray:
    """Mirror a side-A-local court vector for ``side``, in place.

    Side B is the exact 180-degree rotation of side A about the z
    axis: x and y negate, z is invariant -- and because that is a
    proper rotation, the same rule applies to positions, linear
    velocities, and angular velocities alike. Side A returns the array
    untouched. Mutates and returns ``vector`` (callers pass freshly
    built arrays).
    """
    if CourtSide(side) is CourtSide.B:
        vector[:2] *= -1.0
    return vector


@dataclass(frozen=True, slots=True)
class TennisServeConfig:
    """Seeded initial-feed distribution in side-A-local court coordinates.

    Side B is the exact 180-degree court rotation of side A.  The default ball
    starts 1.635 m inside the baseline, just courtward of the serving-side G1;
    a centered feed launched from behind that robot would immediately hit its
    torso rather than enter the rally.
    """

    start_distance_from_net: float = 10.25
    lateral_position: float = 0.0
    height: float = 1.3
    local_velocity: tuple[float, float, float] = (16.0, 0.0, 4.5)
    position_noise: tuple[float, float, float] = (0.1, 0.25, 0.1)
    velocity_noise: tuple[float, float, float] = (0.5, 0.25, 0.25)
    local_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name, values in (
            ("local_velocity", self.local_velocity),
            ("position_noise", self.position_noise),
            ("velocity_noise", self.velocity_noise),
            ("local_angular_velocity", self.local_angular_velocity),
        ):
            if len(values) != 3:
                raise ValueError(f"{name} must contain exactly three values")
        scalar_values = (
            *self.local_velocity,
            *self.velocity_noise,
            *self.local_angular_velocity,
        )
        if not all(math.isfinite(float(value)) for value in scalar_values):
            raise ValueError("serve configuration must contain only finite values")
        validate_feed_geometry(
            kind="serve",
            start_distance_from_net=self.start_distance_from_net,
            lateral_position=self.lateral_position,
            height=self.height,
            position_noise=self.position_noise,
        )
        if self.height <= 0.0:
            raise ValueError("serve height must be positive")
        if any(value < 0.0 for value in self.velocity_noise):
            raise ValueError("velocity_noise values must be non-negative")
        if self.local_velocity[0] <= self.velocity_noise[0]:
            raise ValueError(
                "forward serve velocity must remain positive across its noise range"
            )
