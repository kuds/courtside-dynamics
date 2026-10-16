"""Static contracts for the package metadata in ``pyproject.toml``.

Each bound pinned here was measured, not chosen: it names the release that
broke the package, or the first one that reads its metadata. An edit that
drops one fails here, with the reason, instead of on the next fresh Colab
resolve. The CI workflow's CPU-torch ordering is pinned the same way.
"""

from __future__ import annotations

import re
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


def test_mujoco_floor_is_the_oldest_release_that_reproduces_the_goldens() -> None:
    """3.12.0 through 3.15 reproduce every golden digest; 3.11 does not.

    On 3.11.0 the humanoid stream moves; on 3.10.0 every PaddleTennis,
    WallBall, BallBounce and BallBalance stream does too, and on 3.3.0
    the Stage-1 oracle takes 127 steps instead of 129.
    """
    dependencies = _pyproject()["project"]["dependencies"]
    specifier = _requirement("mujoco", dependencies).specifier
    assert not specifier.contains("3.11.0")
    assert specifier.contains("3.12.0")


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


def _workflow_jobs() -> dict[str, list[str]]:
    """Each CI job's ``run:`` commands, in step order.

    A deliberately small reader for this workflow's own layout (jobs at
    two spaces, ``run:`` values on one line or as a ``|``/``>`` block),
    so the check needs no YAML library.
    """
    lines = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text().splitlines()
    jobs: dict[str, list[str]] = {}
    in_jobs = False
    current: list[str] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        job = re.match(r"^  ([\w-]+):\s*$", line)
        if job:
            current = jobs.setdefault(job.group(1), [])
            continue
        run = re.match(r"^(\s*)(?:- )?run: ?(.*)$", line)
        if run and current is not None:
            value = run.group(2).strip()
            if value in ("|", ">"):
                indent = len(run.group(1))
                block = []
                while index < len(lines) and (
                    not lines[index].strip()
                    or len(lines[index]) - len(lines[index].lstrip()) > indent
                ):
                    block.append(lines[index].strip())
                    index += 1
                value = " ".join(part for part in block if part)
            current.append(value)
    return jobs


def test_ci_installs_cpu_torch_before_the_train_extra() -> None:
    """PyPI's default torch is the CUDA build: about 2.7 GB of nvidia-*
    wheels a CPU runner never uses. Every job that installs the train
    extra must install CPU torch first, so the editable install keeps it."""
    cpu_torch = "pip install torch --index-url https://download.pytorch.org/whl/cpu"
    jobs = _workflow_jobs()
    training = {
        name: runs
        for name, runs in jobs.items()
        if any('pip install -e ".[train' in run for run in runs)
    }
    assert {"typecheck", "test"} <= set(training), sorted(jobs)
    for name, runs in training.items():
        extra = next(i for i, run in enumerate(runs) if 'pip install -e ".[train' in run)
        assert cpu_torch in runs[:extra], (name, runs)
