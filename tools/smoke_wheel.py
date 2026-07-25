"""Smoke-test an *installed* courtside-dynamics distribution.

``python -m build`` proves the wheel builds; it does not prove the wheel
works. The gap matters here because the package's whole distribution path
is ``pip install "courtside-dynamics @ git+https://..."`` on Colab, and it
ships its simulation assets as package *data*: 35 STL meshes, 7 MJCF XMLs,
and 11 starter run-config TOMLs, selected by hand-written globs in
``[tool.setuptools.package-data]``. Drop a glob and every environment
constructor fails at ``mj_loadXML`` for anyone who installed from a wheel
-- while a source checkout, where the files sit next to the code
regardless, keeps passing the entire test suite.

So this script deliberately exercises the things a source tree cannot
distinguish:

* the import resolves to an installed distribution, not a local ``src/``
  tree (the check is worthless otherwise, and with a src layout the
  shadowing is quiet);
* every registered environment id builds, resets, and steps -- which is
  what actually reads the MJCF and, transitively, its meshes;
* the packaged run configs are discoverable and copyable, the path Colab
  uses to bootstrap a config without cloning the repo.

Usage (after installing a built wheel into the active interpreter)::

    python -m build --wheel
    pip install dist/*.whl
    python tools/smoke_wheel.py

Exits non-zero with a specific message on the first failure.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Registered ids are unversioned as of 0.6.0; see the README's table.
EXPECTED_ENV_IDS = (
    "CourtsideDynamics/BallBalance",
    "CourtsideDynamics/BallBounce",
    "CourtsideDynamics/WallBall",
    "CourtsideDynamics/HumanoidTennisCoop",
)


def _check_installed_not_shadowed() -> str:
    """Fail unless ``courtside_dynamics`` came from an installed dist."""
    import courtside_dynamics

    location = os.path.dirname(os.path.abspath(courtside_dynamics.__file__))
    # A src layout means the repo root does not normally shadow an
    # installed package -- but an editable install, a stray PYTHONPATH, or
    # running from inside src/ all would, and then this script would be
    # testing the working tree while claiming to test the wheel.
    if not any(
        marker in location.replace(os.sep, "/")
        for marker in ("site-packages", "dist-packages")
    ):
        raise SystemExit(
            f"smoke_wheel: courtside_dynamics resolved to {location}, which "
            f"is not an installed distribution -- the working tree is "
            f"shadowing the wheel, so this check would prove nothing. Run "
            f"from outside the repo, or uninstall the editable install."
        )
    return location


def _check_environments() -> None:
    """Build, reset, and step every registered env id.

    This is the step that actually loads the packaged MJCF and its meshes:
    a missing asset surfaces here as a MuJoCo load error, not as an import
    error, which is why importing the package is not a sufficient check.
    """
    import gymnasium as gym

    import courtside_dynamics  # noqa: F401 - registers the env ids

    for env_id in EXPECTED_ENV_IDS:
        try:
            env = gym.make(env_id)
        except Exception as error:  # noqa: BLE001 - report and fail loudly
            raise SystemExit(
                f"smoke_wheel: gym.make({env_id!r}) failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        try:
            env.reset(seed=0)
            env.step(env.action_space.sample())
        except Exception as error:  # noqa: BLE001
            raise SystemExit(
                f"smoke_wheel: {env_id} built but could not be stepped: "
                f"{type(error).__name__}: {error}"
            ) from error
        finally:
            env.close()
        print(f"  ok  {env_id}")


def _check_packaged_run_configs() -> None:
    """Assert the starter TOMLs are installed files, and are readable.

    The starter configs are how a pip-installed Colab session bootstraps a
    run config without cloning, so they have to be real installed data
    rather than just an importable module name.

    Checked through ``importlib.resources`` rather than
    ``run_config.available_run_configs()`` on purpose: that helper imports
    ``recipes`` -> ``training`` -> ``pandas``, so it needs the ``train``
    extra. Routing this check through it would make the smoke test drag in
    torch, and would fail on a perfectly well-packaged base install.
    """
    from importlib.resources import files

    root = files("courtside_dynamics.run_configs")
    tomls = sorted(
        entry.name for entry in root.iterdir() if entry.name.endswith(".toml")
    )
    if not tomls:
        raise SystemExit(
            "smoke_wheel: no *.toml found in courtside_dynamics.run_configs "
            "-- the packaged starter configs are missing from the "
            "distribution"
        )
    # Read one end to end: an entry can exist in the RECORD while the file
    # itself is absent or truncated.
    sample = root.joinpath(tomls[0])
    text = sample.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit(f"smoke_wheel: packaged {tomls[0]} is empty")
    with tempfile.TemporaryDirectory() as destination:
        copy = os.path.join(destination, tomls[0])
        with open(copy, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(f"  ok  {len(tomls)} packaged run configs (read {tomls[0]})")


def main() -> int:
    print("smoke-testing the installed courtside-dynamics distribution")
    location = _check_installed_not_shadowed()
    import courtside_dynamics

    print(f"  version {courtside_dynamics.__version__} from {location}")
    _check_environments()
    _check_packaged_run_configs()
    print("smoke_wheel: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
