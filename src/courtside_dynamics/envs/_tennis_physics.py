"""Geometry constants and predicates shared by humanoid-tennis physics.

The court coordinate frame is fixed for every curriculum stage: ``x`` runs
along the court length, ``y`` across its width, and the net lies in the
``x = 0`` plane.  Court measurements are the regulation singles dimensions
and are measured to the *outside* edge of each line, so a contact on the
boundary is in bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np

COURT_LENGTH: Final = 23.77
COURT_WIDTH: Final = 8.23
COURT_HALF_LENGTH: Final = COURT_LENGTH / 2.0
COURT_HALF_WIDTH: Final = COURT_WIDTH / 2.0

NET_CENTER_HEIGHT: Final = 0.914
NET_POST_HEIGHT: Final = 1.07
NET_POST_OFFSET_FROM_SIDELINE: Final = 0.914
NET_POST_Y: Final = COURT_HALF_WIDTH + NET_POST_OFFSET_FROM_SIDELINE

BALL_MASS: Final = 0.0577
BALL_RADIUS: Final = 0.0335
AIR_DENSITY: Final = 1.225
BALL_DRAG_COEFFICIENT: Final = 0.55
BALL_CROSS_SECTION: Final = np.pi * BALL_RADIUS**2

# Stable semantic collision categories shared by MJCF, contact sampling, and
# tests. Affinity masks are composed from these one-hot category bits.
COLLISION_COURT: Final = 1 << 0
COLLISION_BALL: Final = 1 << 1
COLLISION_NET: Final = 1 << 2
COLLISION_RACKET: Final = 1 << 3
COLLISION_ROBOT: Final = 1 << 4

# A tiny numerical tolerance is enough for deterministic analytic tests while
# still treating the outside edge of each painted line as in bounds.
LINE_TOLERANCE: Final = 1e-6


@dataclass(frozen=True, slots=True)
class CourtGeometry:
    """The playing-surface dimensions the rules machinery measures against.

    Defaults are the regulation singles court described by this module's
    constants, so every existing consumer is byte-identical. A
    non-regulation court (the PaddleTennis probes froze 6.5 m
    half-length) constructs its own instance and threads it through
    ``RallyStateMachine`` and ``SubstepTennisEventSampler`` instead of
    forking the reducer.
    """

    half_length: float = COURT_HALF_LENGTH
    half_width: float = COURT_HALF_WIDTH
    tolerance: float = LINE_TOLERANCE

    def __post_init__(self) -> None:
        for name in ("half_length", "half_width"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(
                    f"CourtGeometry.{name} must be a finite positive "
                    f"number, got {value!r}"
                )
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (int, float))
            or not math.isfinite(self.tolerance)
            or self.tolerance < 0.0
        ):
            raise ValueError(
                f"CourtGeometry.tolerance must be a finite non-negative "
                f"number, got {self.tolerance!r}"
            )

    def is_in_bounds(self, x: float, y: float) -> bool:
        """Whether a court-contact point is on or inside the lines."""
        return (
            abs(float(x)) <= self.half_length + self.tolerance
            and abs(float(y)) <= self.half_width + self.tolerance
        )


#: The regulation singles court every consumer uses by default.
REGULATION_COURT: Final = CourtGeometry()


def is_in_bounds(
    x: float,
    y: float,
    *,
    tolerance: float = LINE_TOLERANCE,
) -> bool:
    """Return whether a court-contact point is on or inside the singles lines."""
    return (
        abs(float(x)) <= COURT_HALF_LENGTH + tolerance
        and abs(float(y)) <= COURT_HALF_WIDTH + tolerance
    )


def net_height_at(y: float) -> float:
    """Return the linear top-cable height at lateral coordinate ``y``.

    The physical XML approximates this profile with collision panels plus a
    sloped cable.  Clamping outside the posts keeps this helper well-defined
    for diagnostics without implying that the net extends beyond the posts.
    """
    lateral_fraction = min(abs(float(y)) / NET_POST_Y, 1.0)
    return NET_CENTER_HEIGHT + lateral_fraction * (NET_POST_HEIGHT - NET_CENTER_HEIGHT)


def ball_drag_force(linear_velocity: np.ndarray) -> np.ndarray:
    """Return quadratic aerodynamic drag for the regulation-scale ball.

    MuJoCo's generic ellipsoid-fluid approximation is not calibrated for a
    felt tennis ball. The environment applies this explicit force at every
    physics substep so its behavior can be tested independently and changed
    without affecting robot aerodynamics.
    """
    velocity = np.asarray(linear_velocity, dtype=np.float64)
    if velocity.shape != (3,):
        raise ValueError(f"linear_velocity must have shape (3,), got {velocity.shape}")
    speed = float(np.linalg.norm(velocity))
    if speed == 0.0:
        return np.zeros(3, dtype=np.float64)
    scale = -0.5 * AIR_DENSITY * BALL_DRAG_COEFFICIENT
    return scale * BALL_CROSS_SECTION * speed * velocity


__all__ = [
    "AIR_DENSITY",
    "BALL_CROSS_SECTION",
    "BALL_DRAG_COEFFICIENT",
    "BALL_MASS",
    "BALL_RADIUS",
    "COLLISION_BALL",
    "COLLISION_COURT",
    "COLLISION_NET",
    "COLLISION_RACKET",
    "COLLISION_ROBOT",
    "COURT_HALF_LENGTH",
    "COURT_HALF_WIDTH",
    "COURT_LENGTH",
    "COURT_WIDTH",
    "LINE_TOLERANCE",
    "CourtGeometry",
    "REGULATION_COURT",
    "NET_CENTER_HEIGHT",
    "NET_POST_HEIGHT",
    "NET_POST_OFFSET_FROM_SIDELINE",
    "NET_POST_Y",
    "ball_drag_force",
    "is_in_bounds",
    "net_height_at",
]
