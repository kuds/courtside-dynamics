"""The calibrated 3-DoF paddle interface, as a composable unit.

WallBall's paddle earned its calibrations one campaign at a time — the
piecewise normalized-target mapping around a movable home pivot, the
world-space x coordinate contract, the target-clamping fence, and the
schedulable slide damping whose terminal velocity (force cap / damping)
IS the era's power ceiling. The PaddleTennis design doc says this
paddle "transfers as-is", and the two-paddle court needs TWO instances;
this module packages the machinery so the paddle court composes it
instead of copy-pasting ~300 lines out of ``wall_ball.py``.

Semantics are deliberately identical to ``WallBallEnv``'s paddle
handling (same validation rules, same fence-clamps-the-target contract,
same damping restore-on-None), and a drift test pins this class's
control mapping bit-for-bit against the live ``WallBallEnv`` so the two
implementations cannot diverge while ``WallBallEnv``'s internals await
the mechanical delegation onto this class.

Coordinate contract: everything public is **world-space x** (and raw
joint coordinates for y/z, whose XML ranges are symmetric); the
``qpos``-space conversion through the compiled body origin happens once
at construction, exactly like ``WallBallEnv`` resolves
``paddle_home_x``.
"""
from __future__ import annotations

import numpy as np

from courtside_dynamics.envs._base import piecewise_targets


class PaddleInterface:
    """One calibrated paddle: normalized 3-action targets in, ctrl out.

    Parameters mirror the WallBall kwargs they were extracted from:

    - ``home_x``: world x that a zero x-action targets (the mapping
      pivot; movable later via :meth:`set_home_x`).
    - ``x_target_range``: world-space span the normalized x action maps
      onto; ``None`` uses the full physical workspace. Must lie within
      the physical workspace, and ``home_x`` within it.
    - ``x_fence``: optional world-space clamp applied to the *target*
      (not the mapping): in-fence actions mean the same thing whatever
      the fence, only where out-of-fence targets saturate moves.
    - ``joint_damping``: per-instance slide-damping override; ``None``
      keeps the XML value (captured at construction and restorable).

    The instance owns its actuator/joint indices, so two paddles in one
    model each write only their own ``ctrl`` entries.
    """

    def __init__(
        self,
        model,
        data,
        *,
        home_x: float,
        joint_names: tuple[str, str, str] = (
            "paddle_slide_x",
            "paddle_slide_y",
            "paddle_slide_z",
        ),
        actuator_names: tuple[str, str, str] = (
            "paddle_target_x",
            "paddle_target_y",
            "paddle_target_z",
        ),
        head_body_name: str = "paddle_head",
        x_target_range: tuple[float, float] | None = None,
        x_fence: tuple[float, float] | None = None,
        joint_damping: float | None = None,
    ) -> None:
        self.model = model
        self.joint_names = tuple(joint_names)
        self.actuator_names = tuple(actuator_names)
        self.head_body_name = head_body_name

        self._actuator_indices = np.array(
            [int(model.actuator(name).id) for name in self.actuator_names],
            dtype=np.intp,
        )
        self._x_qposadr = int(model.joint(self.joint_names[0]).qposadr[0])
        self._dofadrs = tuple(
            int(model.joint(name).dofadr[0]) for name in self.joint_names
        )
        # Baseline damping captured AT CONSTRUCTION, so a later
        # ``set_damping(None)`` restores it exactly. On a fresh model
        # this is the XML value (the WallBall contract); on a model
        # whose damping was already overridden before construction --
        # e.g. an interface built over a live WallBallEnv with
        # ``paddle_joint_damping`` set -- the baseline is that
        # override, and the delegation must construct the interface
        # BEFORE applying the env's own override to preserve
        # restore-to-XML semantics.
        self._xml_damping = tuple(
            float(model.dof_damping[dofadr]) for dofadr in self._dofadrs
        )

        # World-space origin of the x slide: body x at the CURRENT
        # (assumed home/keyframe) pose minus the joint's current qpos —
        # the same resolution WallBallEnv performs after mj_forward.
        # Callers must have run mj_forward before constructing.
        x_joint_qpos = float(data.qpos[self._x_qposadr])
        self._x_origin = (
            float(data.body(head_body_name).xpos[0]) - x_joint_qpos
        )

        ctrl_low = np.array(
            [
                float(model.actuator_ctrlrange[index, 0])
                for index in self._actuator_indices
            ]
        )
        ctrl_high = np.array(
            [
                float(model.actuator_ctrlrange[index, 1])
                for index in self._actuator_indices
            ]
        )
        self._physical_world_range = (
            self._x_origin + ctrl_low[0],
            self._x_origin + ctrl_high[0],
        )
        self._control_low = ctrl_low
        self._control_high = ctrl_high
        self._control_home = np.zeros(3, dtype=np.float64)

        self._home_x = float(home_x)
        self._x_fence: tuple[float, float] | None = None
        self._joint_damping: float | None = None
        self.set_x_target_range(x_target_range)  # validates home too
        self.set_fence(x_fence)
        self.set_damping(joint_damping)

    # -- world-space views ------------------------------------------------

    @property
    def home_x(self) -> float:
        """World x that a zero x-action maps to (the mapping pivot)."""
        return self._home_x

    @property
    def x_target_range(self) -> tuple[float, float]:
        """World-space span the normalized x action maps onto."""
        return self._x_target_range

    @property
    def x_fence(self) -> tuple[float, float] | None:
        """World-space clamp on the x position target; None disables."""
        return self._x_fence

    @property
    def physical_world_range(self) -> tuple[float, float]:
        """World-space span the paddle can physically reach."""
        return self._physical_world_range

    @property
    def joint_damping(self) -> float | None:
        """Per-instance slide damping; ``None`` keeps the XML value."""
        return self._joint_damping

    def world_x(self, data) -> float:
        """Current world x of the paddle head."""
        return self._x_origin + float(data.qpos[self._x_qposadr])

    def qpos_x_for_world(self, world_x: float) -> float:
        """Convert a world x to the x slide's qpos coordinate."""
        return float(world_x) - self._x_origin

    # -- schedulable knobs (the WallBall property-setter semantics) -------

    def set_x_target_range(
        self, value: tuple[float, float] | None
    ) -> None:
        """Re-map the x action span (``None`` = full physical range).

        Also re-validates home containment, mirroring the WallBall
        constructor's ordering.

        DIVERGENCE NOTE for the planned WallBall delegation:
        ``WallBallEnv`` resolves ``paddle_x_target_range=None`` to its
        FROZEN historical default (-4.7, 0.3), deliberately NOT the
        physical workspace (the 0.25.0 extension widened the XML while
        freezing action semantics). Here ``None`` means the full
        physical range -- the right contract for a fresh court. The
        delegation must therefore resolve WallBall's default
        explicitly before constructing this interface; forwarding
        ``None`` through would silently rescale every recipe's x
        mapping.
        """
        if value is not None:
            if len(value) != 2:
                raise ValueError(
                    "paddle_x_target_range must contain two values"
                )
            if not (float(value[0]) < float(value[1])):
                raise ValueError(
                    "paddle_x_target_range must be strictly increasing"
                )
        tolerance = 1e-9
        physical = self._physical_world_range
        target = (
            physical
            if value is None
            else (float(value[0]), float(value[1]))
        )
        if (
            target[0] < physical[0] - tolerance
            or target[1] > physical[1] + tolerance
        ):
            raise ValueError(
                "paddle_x_target_range must stay within the physical "
                f"world-space range {physical}, got {target}"
            )
        if not (
            target[0] - tolerance <= self._home_x <= target[1] + tolerance
        ):
            raise ValueError(
                "paddle_home_x must lie inside paddle_x_target_range; "
                f"got home {self._home_x} and range {target}"
            )
        self._x_target_range = target
        self._control_low[0] = target[0] - self._x_origin
        self._control_high[0] = target[1] - self._x_origin
        self._control_home[0] = self._home_x - self._x_origin

    def set_home_x(self, value: float) -> None:
        """Move the mapping pivot, keeping the derived arrays in step.

        The 0.22.0 lesson behind WallBall's property: a home the
        mapping does not follow is a silent no-op that collapses part
        of the action range onto a fence edge.
        """
        home = float(value)
        if not np.isfinite(home):
            raise ValueError("paddle_home_x must be finite")
        tolerance = 1e-9
        target = self._x_target_range
        if not (
            target[0] - tolerance <= home <= target[1] + tolerance
        ):
            raise ValueError(
                "paddle_home_x must lie inside paddle_x_target_range; "
                f"got home {home} and range {target}"
            )
        self._home_x = home
        self._control_home[0] = home - self._x_origin

    def set_fence(self, value: tuple[float, float] | None) -> None:
        """Clamp x position targets to a world-space window."""
        if value is None:
            self._x_fence = None
            return
        if len(value) != 2:
            raise ValueError("paddle_x_fence must contain two values")
        fence = (float(value[0]), float(value[1]))
        if not np.isfinite(fence).all():
            raise ValueError("paddle_x_fence must be finite")
        if fence[0] >= fence[1]:
            raise ValueError("paddle_x_fence must be strictly increasing")
        tolerance = 1e-9
        physical = self._physical_world_range
        if (
            fence[0] < physical[0] - tolerance
            or fence[1] > physical[1] + tolerance
        ):
            raise ValueError(
                "paddle_x_fence must stay within the physical "
                f"world-space range {physical}, got {fence}"
            )
        self._x_fence = fence

    def set_damping(self, value: float | None) -> None:
        """Override the slide damping (``None`` restores the XML value).

        Terminal velocity under the position servo's force cap is
        (force cap / damping), so this knob IS the paddle's calibrated
        power ceiling — the probe battery froze damping 8 for the
        paddle court.
        """
        if value is not None and (
            not np.isfinite(value) or float(value) <= 0.0
        ):
            raise ValueError(
                "paddle_joint_damping must be None or finite and positive"
            )
        self._joint_damping = None if value is None else float(value)
        for dofadr, xml_default in zip(
            self._dofadrs, self._xml_damping, strict=True
        ):
            self.model.dof_damping[dofadr] = (
                xml_default
                if self._joint_damping is None
                else self._joint_damping
            )

    # -- the control mapping ----------------------------------------------

    def targets(self, normalized: np.ndarray) -> np.ndarray:
        """Map a normalized [-1, 1]^3 action to this paddle's ctrl values.

        Bit-identical to ``WallBallEnv._action_to_controls`` (pinned by
        a drift test): clip, piecewise map around home, then fence-clamp
        the x *target*.
        """
        normalized = np.asarray(normalized, dtype=np.float64)
        if normalized.shape != (3,):
            raise ValueError(
                f"action must have shape (3,), got {normalized.shape}"
            )
        if not bool(np.isfinite(normalized).all()):
            raise ValueError("action must contain only finite values")
        normalized = np.clip(normalized, -1.0, 1.0)
        targets = piecewise_targets(
            normalized,
            self._control_low,
            self._control_home,
            self._control_high,
        )
        if self._x_fence is not None:
            targets[0] = np.clip(
                targets[0],
                self._x_fence[0] - self._x_origin,
                self._x_fence[1] - self._x_origin,
            )
        return targets

    def apply(self, data, normalized: np.ndarray) -> np.ndarray:
        """Write :meth:`targets` into this paddle's ctrl entries."""
        targets = self.targets(normalized)
        data.ctrl[self._actuator_indices] = targets
        return targets
