"""Static contracts for the package metadata in ``pyproject.toml``.

Each bound pinned here was measured, not chosen: it names the release that
broke the package, or the first one that reads its metadata. An edit that
drops one fails here, with the reason, instead of on the next fresh Colab
resolve. The CI workflow's CPU-torch ordering and its ``python -m mypy``
are pinned the same way. The last tests check the running environment
against the declared ranges, so a stale install fails with its version
named instead of deep inside a calibrated test.
"""

from __future__ import annotations

import importlib.metadata
import re
import shlex
import tomllib
from pathlib import Path
from typing import Any

import pytest
from packaging.requirements import InvalidRequirement, Requirement

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

    On 3.11.0 the humanoid stream moves; on 3.10.0 every CI case but
    WallBallTrueBaseline's random stream moves (21 of 22), and on 3.3.0
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


#: pip options that take a value as the next token (or after ``=``).
#: ``-e``/``--editable`` is handled apart: its value is a requirement.
_PIP_VALUE_OPTIONS = frozenset(
    {
        "-i", "--index-url", "--extra-index-url", "-f", "--find-links",
        "-c", "--constraint", "-r", "--requirement", "-t", "--target",
        "--prefix", "--root", "--upgrade-strategy", "--python-version",
        "--platform", "--implementation", "--abi", "--only-binary",
        "--no-binary", "-C", "--config-settings", "--progress-bar",
        "--trusted-host", "--cache-dir", "--log", "--timeout", "--retries",
        "--proxy", "--src", "--report",
    }
)
_PIP_EDITABLE = frozenset({"-e", "--editable"})
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def _workflow_jobs(text: str | None = None) -> dict[str, list[str]]:
    """Each CI job's shell commands, in step order.

    A deliberately small reader for this workflow's own layout, so the
    check needs no YAML library: jobs at two spaces; ``run:`` values on
    one line (bare or quoted) or as a ``|``/``>`` block scalar, with or
    without a chomping indicator (``|-``, ``>+``). A ``|`` block keeps
    its lines and a ``>`` block folds them into one. Backslash line
    continuations are joined, and every line is split into commands at
    ``&&``, ``||`` and ``;``.
    """
    if text is None:
        text = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    lines = text.splitlines()
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
        if not run or current is None:
            continue
        value = run.group(2).strip()
        block = re.fullmatch(r"([|>])[+-]?", value)
        if block:
            indent = len(run.group(1))
            body = []
            while index < len(lines) and (
                not lines[index].strip()
                or len(lines[index]) - len(lines[index].lstrip()) > indent
            ):
                body.append(lines[index].strip())
                index += 1
            body = [part for part in body if part]
            value = ("\n" if block.group(1) == "|" else " ").join(body)
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        value = re.sub(r"\\\n\s*", " ", value)
        for command_line in value.splitlines():
            current.extend(
                command.strip()
                for command in re.split(r"&&|\|\||;", command_line)
                if command.strip()
            )
    return jobs


def _pip_install(command: str) -> tuple[list[str], dict[str, list[str]]] | None:
    """``(requirements, options)`` of a ``pip install`` command, else None.

    Recognizes ``pip``/``pip3`` (any path), ``python -m pip`` and ``uv
    pip``. Requirements include ``-e``/``--editable`` targets; options
    map each value-taking option to its values (``--opt value`` and
    ``--opt=value`` alike, ``-i`` read as ``--index-url``).
    """
    tokens = shlex.split(command)
    pip = next(
        (i for i, token in enumerate(tokens) if Path(token).name in ("pip", "pip3")),
        None,
    )
    if pip is None:
        return None
    rest = tokens[pip + 1 :]
    if not rest or rest[0] != "install":
        return None
    requirements: list[str] = []
    options: dict[str, list[str]] = {}
    args = iter(rest[1:])
    for token in args:
        name, _, inline = token.partition("=")
        if token.startswith("-") and name in _PIP_EDITABLE:
            requirements.append(inline if inline else next(args, ""))
        elif token.startswith("-") and name in _PIP_VALUE_OPTIONS:
            key = "--index-url" if name == "-i" else name
            options.setdefault(key, []).append(inline if inline else next(args, ""))
        elif not token.startswith("-"):
            requirements.append(token)
    return requirements, options


def _installs_train_extra(command: str) -> bool:
    """A ``pip install`` of anything carrying the ``train`` extra."""
    parsed = _pip_install(command)
    if parsed is None:
        return False
    return any(
        "train" in (extra.strip() for extra in match.group(1).split(","))
        for requirement in parsed[0]
        for match in re.finditer(r"\[([^\]]*)\]", requirement)
    )


def _installs_cpu_torch(command: str) -> bool:
    """A ``pip install torch`` whose primary index is the CPU wheel index.

    ``--extra-index-url`` does not count: pip then picks the best
    candidate across both indexes, which can be PyPI's CUDA build.
    """
    parsed = _pip_install(command)
    if parsed is None:
        return False
    requirements, options = parsed
    names = set()
    for requirement in requirements:
        try:
            names.add(Requirement(requirement).name)
        except InvalidRequirement:  # a path or URL
            continue
    return "torch" in names and any(
        url.rstrip("/") == CPU_TORCH_INDEX for url in options.get("--index-url", [])
    )


def _cpu_torch_violations(jobs: dict[str, list[str]]) -> list[str]:
    """Jobs that install the train extra without CPU torch first."""
    violations = []
    for name, commands in jobs.items():
        extra = next(
            (i for i, command in enumerate(commands) if _installs_train_extra(command)),
            None,
        )
        if extra is not None and not any(map(_installs_cpu_torch, commands[:extra])):
            violations.append(f"{name}: {commands[extra]!r} without CPU torch before it")
    return violations


def test_ci_installs_cpu_torch_before_the_train_extra() -> None:
    """PyPI's default torch is the CUDA build: about 2.7 GB of nvidia-*
    wheels a CPU runner never uses. Every job that installs the train
    extra must install CPU torch first, so the editable install keeps it."""
    jobs = _workflow_jobs()
    training = {
        name for name, commands in jobs.items() if any(map(_installs_train_extra, commands))
    }
    assert {"typecheck", "test"} <= training, sorted(jobs)
    assert _cpu_torch_violations(jobs) == []


_TORCH = f"pip install torch --index-url {CPU_TORCH_INDEX}"


def _workflow(*steps: str) -> str:
    return "jobs:\n  job:\n    steps:\n" + "".join(steps)


def test_cpu_torch_check_reads_quoting_and_flag_variants() -> None:
    """The ordering check must not go blind on a reworded install line."""
    train_variants = [
        'pip install -e ".[train,dev]"',
        "pip install -e '.[train,dev]'",
        "pip install -e .[dev,train]",
        'python -m pip install --upgrade --editable ".[dev, train]"',
        "pip install --editable=.[train]",
        'pip3 install -q ".[train]"',
        "uv pip install -e '.[train]'",
    ]
    torch_variants = [
        _TORCH,
        f"python -m pip install --index-url={CPU_TORCH_INDEX}/ torch==2.5.1",
        f"pip3 install -i {CPU_TORCH_INDEX} 'torch>=2.3'",
    ]
    for train in train_variants:
        assert _installs_train_extra(train), train
        for torch in torch_variants:
            assert _installs_cpu_torch(torch), torch
            # One step each, one block, a folded block, a quoted value.
            for text in (
                _workflow(f"      - run: {torch}\n", f"      - run: {train}\n"),
                _workflow(f"      - run: |\n          {torch}\n          {train}\n"),
                _workflow(f"      - run: |-\n          {torch} && {train}\n"),
                _workflow(f"      - run: >\n          {torch}\n          ; {train}\n"),
            ):
                jobs = _workflow_jobs(text)
                assert _cpu_torch_violations(jobs) == [], (text, jobs)
            backwards = _workflow(f"      - run: {train}\n", f"      - run: {torch}\n")
            assert _cpu_torch_violations(_workflow_jobs(backwards)), backwards
        assert _cpu_torch_violations(_workflow_jobs(_workflow(f"      - run: {train}\n")))
    quoted = _workflow(f"      - run: \"{_TORCH}\"\n", "      - run: pip install -e .[train]\n")
    assert _cpu_torch_violations(_workflow_jobs(quoted)) == []
    continued = _workflow(
        "      - run: |\n          pip install torch \\\n"
        f"            --index-url {CPU_TORCH_INDEX}\n          pip install -e .[train]\n"
    )
    assert _cpu_torch_violations(_workflow_jobs(continued)) == []
    # Not CPU torch: an extra index lets pip pick PyPI's CUDA build, and
    # the CPU index serving some other package is no torch install.
    for not_cpu in (
        f"pip install torch --extra-index-url {CPU_TORCH_INDEX}",
        "pip install torch",
        f"pip install torchvision --index-url {CPU_TORCH_INDEX}",
        f"pip download torch --index-url {CPU_TORCH_INDEX}",
    ):
        assert not _installs_cpu_torch(not_cpu), not_cpu
    assert not _installs_train_extra("pip install 'stable-baselines3[extra]'")


def test_ci_typechecks_with_python_m_mypy_after_the_train_extra() -> None:
    """mypy without numpy, gymnasium and SB3 installed beside it passes
    silently (``ignore_missing_imports``): the typecheck job must run it
    as ``python -m mypy``, in the interpreter that has the train extra."""
    commands = _workflow_jobs()["typecheck"]
    extra = next(i for i, command in enumerate(commands) if _installs_train_extra(command))
    mypy = [i for i, command in enumerate(commands) if "mypy" in shlex.split(command)]
    assert mypy, commands
    for i in mypy:
        assert i > extra, commands
        assert shlex.split(commands[i])[:3] == ["python", "-m", "mypy"], commands[i]


@pytest.mark.parametrize(
    "requirement", _pyproject()["project"]["dependencies"], ids=lambda r: Requirement(r).name
)
def test_installed_dependency_satisfies_its_declared_range(requirement: str) -> None:
    """A stale environment fails here with the reason, not downstream.

    The golden digests and the calibrated tests (the humanoid Stage-1
    oracle's 129 steps, for one) hold on the declared range only. A
    MuJoCo 3.16 left over from before the cap otherwise surfaces as
    ``assert 138 == 129``; here it reads "mujoco 3.16.0 violates the
    declared range".
    """
    declared = Requirement(requirement)
    installed = importlib.metadata.version(declared.name)
    assert declared.specifier.contains(installed, prereleases=True), (
        f"{declared.name} {installed} violates the declared range "
        f"{declared.name}{declared.specifier} (pyproject.toml); reinstall "
        f"the package (pip install -e .) so the environment matches it"
    )


def test_imported_mujoco_satisfies_the_declared_range() -> None:
    """The module that runs, not just the distribution metadata: a
    ``PYTHONPATH`` or a second site-packages can shadow the install."""
    import mujoco

    declared = _requirement("mujoco", _pyproject()["project"]["dependencies"])
    assert declared.specifier.contains(mujoco.__version__, prereleases=True), (
        f"mujoco {mujoco.__version__} (imported from {mujoco.__file__}) "
        f"violates the declared range mujoco{declared.specifier}"
    )
