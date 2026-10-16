"""Static contracts for the package metadata in ``pyproject.toml``.

Each bound pinned here was measured, not chosen: it names the release that
broke the package, or the first one that reads its metadata. An edit that
drops one fails here, with the reason, instead of on the next fresh Colab
resolve.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

REPOSITORY_ROOT = Path(__file__).parents[1]


def _pyproject() -> dict[str, Any]:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement(name: str, requirements: list[str]) -> Requirement:
    matches = [
        requirement
        for requirement in map(Requirement, requirements)
        if requirement.name == name
    ]
    assert len(matches) == 1, (name, requirements)
    return matches[0]


def test_mujoco_range_stops_before_the_release_that_moved_the_humanoid_oracle() -> None:
    """MuJoCo 3.16.0's ``mj_setConst`` refreshes the racket body's bounds.

    The humanoid Stage-1 oracle was calibrated on the stale pre-3.16
    bounds and misses its target on 3.16. Lift the cap only once the
    humanoid code is archived or its oracle is recalibrated (DECISIONS.md,
    "Humanoid tennis environment").
    """
    dependencies = _pyproject()["project"]["dependencies"]
    specifier = _requirement("mujoco", dependencies).specifier
    assert specifier.contains("3.15.0")
    assert not specifier.contains("3.16.0")


def test_gymnasium_floor_skips_the_release_that_cannot_build_any_env() -> None:
    """gymnasium 1.1.0's ``MujocoEnv`` asserts a four-mode render list."""
    dependencies = _pyproject()["project"]["dependencies"]
    specifier = _requirement("gymnasium", dependencies).specifier
    assert not specifier.contains("1.1.0")
    assert specifier.contains("1.1.1")


def test_license_is_an_spdx_expression_without_a_license_classifier() -> None:
    """The table form plus a classifier is slated for removal (2027-02-18)."""
    pyproject = _pyproject()
    assert pyproject["project"]["license"] == "MIT"
    assert not [
        classifier
        for classifier in pyproject["project"]["classifiers"]
        if classifier.startswith("License ::")
    ]
    # setuptools reads the SPDX string from 77.0.0 on.
    build_requires = pyproject["build-system"]["requires"]
    specifier = _requirement("setuptools", build_requires).specifier
    assert not specifier.contains("76.1.0")
    assert specifier.contains("77.0.0")
