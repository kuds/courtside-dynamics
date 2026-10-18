"""Contracts for the feasibility-spike notebooks under ``notebooks/spikes/``.

The spikes (design ``design_env_ladder_and_structure_20261016.md``,
section 5) run on Colab GPUs, so CI cannot execute them. What CI can
check: every notebook is clean and compiles, installs a pinned ref, and
S1's bars, verdict rules, MjSpec scene and results folder do what the
notebook says, executed from the notebook's own cells.
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from courtside_dynamics.assets import asset_path
from courtside_dynamics.envs.robot_models import (
    UNITREE_G1_STAND_BASE_HEIGHT,
    get_robot_model_spec,
)
from tests.test_notebooks import _IMMUTABLE_REF, _PACKAGE_URL, _source

REPOSITORY_ROOT = Path(__file__).parents[1]
SPIKE_NOTEBOOKS = sorted((REPOSITORY_ROOT / "notebooks" / "spikes").glob("*.ipynb"))
S1_NOTEBOOK = REPOSITORY_ROOT / "notebooks" / "spikes" / "s1_gpu_throughput.ipynb"
#: S1's code cells in execution order.
_S1_CODE_CELLS = (
    "s1-install",
    "s1-settings",
    "s1-env",
    "s1-paths",
    "s1-rules",
    "s1-helpers",
    "s1-mjlab-script",
    "s1-mjlab",
    "s1-playground",
    "s1-scene-build",
    "s1-scene",
    "s1-results",
    "s1-verdict",
)


def _compilable(source: str) -> str:
    """The cell with each shell (!) and magic (%) line replaced by ``pass``."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            line = line[: len(line) - len(stripped)] + "pass"
        lines.append(line)
    return "\n".join(lines)


def _s1_cells() -> dict[str, str]:
    notebook = json.loads(S1_NOTEBOOK.read_text())
    return {cell["id"]: _source(cell) for cell in notebook["cells"]}


def _s1_settings(monkeypatch: pytest.MonkeyPatch, **replace: str) -> dict[str, Any]:
    """Run the settings cell; it sets XLA variables, which monkeypatch restores."""
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.setenv("XLA_FLAGS", "")
    source = _s1_cells()["s1-settings"]
    for old, new in replace.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace: dict[str, Any] = {}
    exec(compile(source, f"{S1_NOTEBOOK}:s1-settings", "exec"), namespace)
    return namespace


def _run_cell(cell_id: str, namespace: dict[str, Any]) -> dict[str, Any]:
    exec(compile(_s1_cells()[cell_id], f"{S1_NOTEBOOK}:{cell_id}", "exec"), namespace)
    return namespace


# -- Every spike notebook -----------------------------------------------------


def test_the_spike_notebooks_include_s1() -> None:
    assert S1_NOTEBOOK in SPIKE_NOTEBOOKS


@pytest.mark.parametrize("path", SPIKE_NOTEBOOKS, ids=lambda path: path.name)
def test_spike_notebook_is_clean_and_code_cells_compile(path: Path) -> None:
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    cell_ids = [cell["id"] for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile(_compilable(_source(cell)), f"{path}:{cell['id']}", "exec")


@pytest.mark.parametrize("path", SPIKE_NOTEBOOKS, ids=lambda path: path.name)
def test_spike_notebook_installs_a_pinned_immutable_ref(path: Path) -> None:
    """The rule tests/test_notebooks.py applies to the training notebooks."""
    notebook = json.loads(path.read_text())
    install_cells = [
        _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and _PACKAGE_URL in _source(cell)
    ]
    assert len(install_cells) == 1
    [source] = install_cells
    refs = re.findall(r'^REPO_REF = "([^"]*)"$', source, flags=re.MULTILINE)
    assert len(refs) == 1
    assert _IMMUTABLE_REF.fullmatch(refs[0]), refs[0]
    installs = [
        line
        for line in source.splitlines()
        if line.lstrip().startswith("!pip install") and _PACKAGE_URL in line
    ]
    assert installs
    for line in installs:
        assert line.rstrip().endswith(f'{_PACKAGE_URL}@{{REPO_REF}}"'), line


# -- S1: bars, settings, pins -------------------------------------------------


def test_s1_cells_run_in_the_documented_order() -> None:
    notebook = json.loads(S1_NOTEBOOK.read_text())
    code_ids = [cell["id"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert tuple(code_ids) == _S1_CODE_CELLS


def test_s1_states_the_design_bars_and_evaluates_them(monkeypatch) -> None:
    cells = _s1_cells()
    intro = cells["s1-intro"]
    assert "**>= 2x10^4 steps/s**" in intro
    assert "**>= 5x10^3 steps/s**" in intro
    assert "rented" in intro and "pause the humanoid track" in intro
    settings = _s1_settings(monkeypatch)
    assert settings["SINGLE_G1_BAR"] == 2e4
    assert settings["FOUR_G1_BAR"] == 5e3
    assert settings["PHYSICS_HZ"] == 1000  # the low end of the design's 1-2 kHz
    assert settings["CONTROL_HZ"] == 50
    assert settings["SMOKE"] is False
    verdict = cells["s1-verdict"]
    assert "single_bar=SINGLE_G1_BAR" in verdict
    assert "four_bar=FOUR_G1_BAR" in verdict


def test_s1_pins_the_gpu_stack() -> None:
    install = _s1_cells()["s1-install"]
    for pin in (
        '"jax[cuda12]==0.11.2"',
        '"mujoco==3.15.0"',
        '"mujoco-mjx==3.15.0"',
        '"warp-lang==1.18.0"',
        '"playground==0.2.0"',
        '"brax==0.14.2"',
    ):
        assert pin in install
    settings = _s1_cells()["s1-settings"]
    assert 'MJLAB_PIN = "mjlab==1.6.0"' in settings


def test_s1_smoke_shrinks_every_sweep(monkeypatch) -> None:
    smoke = _s1_settings(monkeypatch, **{"SMOKE = False": "SMOKE = True"})
    for name in ("MJLAB_NUM_ENVS", "PLAYGROUND_BATCH_SIZES", "SCENE_BATCH_SIZES"):
        assert smoke[name] == (2,), name
    assert smoke["MAX_TIMED_CALLS"] == 1
    assert smoke["MIN_TIMED_S"] == 0.0


@pytest.mark.parametrize(
    ("replace", "message"),
    [
        ({"PHYSICS_HZ = 1000": "PHYSICS_HZ = 500"}, "defined at 1-2 kHz"),
        ({"EXTRA_PHYSICS_HZ = (2000,)": "EXTRA_PHYSICS_HZ = (1030,)"}, "whole number"),
        (
            {'SIMPLIFY_COLLISION = "auto"': 'SIMPLIFY_COLLISION = "never"'},
            "'auto' or 'always'",
        ),
        ({'SCENE_IMPLS = ("warp", "jax")': 'SCENE_IMPLS = ("cpp",)'}, "unknown MJX"),
    ],
)
def test_s1_settings_refuse_what_the_bars_do_not_define(
    monkeypatch, replace: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        _s1_settings(monkeypatch, **replace)


# -- S1: the verdict ----------------------------------------------------------


def _rules() -> dict[str, Any]:
    return _run_cell("s1-rules", {})


def _row(stack: str, rate: float | None, **fields: Any) -> dict[str, Any]:
    return {
        "stack": stack,
        "impl": "warp",
        "batch": 4096,
        "bar": True,
        "steps_per_s": rate,
        **fields,
    }


def _verdict(single, scene, *, gpu="NVIDIA L4", smoke=False) -> dict[str, Any]:
    return _rules()["s1_verdict"](
        single, scene, single_bar=2e4, four_bar=5e3, gpu_name=gpu, smoke=smoke
    )


def test_s1_verdict_passes_at_the_bars_and_plans_h0_on_the_l4() -> None:
    verdict = _verdict(
        [_row("playground", 2e4), _row("mjlab", 3e4)], [_row("mjx-scene", 5e3)]
    )
    assert verdict["overall"] == "PASS"
    assert verdict["bars"]["single_g1"]["status"] == "PASS"
    assert verdict["bars"]["single_g1"]["best"]["stack"] == "mjlab"
    assert verdict["bars"]["four_g1"]["status"] == "PASS"
    assert "plan H0 (G1Footwork) on Colab L4 GPUs" in verdict["next_step"]
    assert "mjlab, the design's first choice" in verdict["next_step"]


def test_s1_verdict_names_the_playground_fallback_without_mjlab() -> None:
    verdict = _verdict([_row("playground", 5e4)], [_row("mjx-scene", 6e3)])
    assert verdict["overall"] == "PASS"
    assert "not measured" in verdict["next_step"]
    assert "Playground/Brax" in verdict["next_step"]


def test_s1_verdict_sends_an_l4_failure_to_a_rented_gpu() -> None:
    verdict = _verdict([_row("playground", 5e4)], [_row("mjx-scene", 4999.0)])
    assert verdict["overall"] == "FAIL"
    assert verdict["bars"]["single_g1"]["status"] == "PASS"
    assert verdict["bars"]["four_g1"]["status"] == "FAIL"
    assert (
        "re-run this notebook unchanged on a rented A100/4090-class GPU"
        in verdict["next_step"]
    )


def test_s1_verdict_pauses_the_humanoid_track_when_the_rented_run_fails() -> None:
    verdict = _verdict(
        [_row("playground", 1e4)], [_row("mjx-scene", 6e3)], gpu="NVIDIA A100-SXM4-80GB"
    )
    assert verdict["overall"] == "FAIL"
    assert "pause the humanoid track" in verdict["next_step"]
    assert "rented A100" not in verdict["next_step"]


def test_s1_verdict_rents_a_gpu_class_that_passes_off_the_l4() -> None:
    verdict = _verdict(
        [_row("playground", 5e4)], [_row("mjx-scene", 6e3)], gpu="NVIDIA A100-SXM4-80GB"
    )
    assert verdict["overall"] == "PASS"
    assert "not an L4" in verdict["next_step"]


def test_s1_verdict_counts_only_clean_rows_at_the_bars_settings() -> None:
    scene = [
        _row("mjx-scene", 9e3, overflow={"NARROWPHASE": 3}),
        _row("mjx-scene", 9e3, nonfinite_worlds=1),
        _row("mjx-scene", None, error="XlaRuntimeError: RESOURCE_EXHAUSTED"),
        _row("mjx-scene", 9e3, bar=False, physics_hz=2000),
        _row("mjx-scene", 4e3),
    ]
    verdict = _verdict([_row("playground", 5e4)], scene)
    assert verdict["bars"]["four_g1"]["best"]["steps_per_s"] == 4e3
    assert verdict["overall"] == "FAIL"
    assert verdict["excluded_rows"] == 3  # the informational 2 kHz row is not a bar row
    assert "did not count" in verdict["next_step"]


def test_s1_verdict_has_no_result_without_a_clean_row() -> None:
    verdict = _verdict(
        [_row("playground", 5e4)], [_row("mjx-scene", None, error="boom")]
    )
    assert verdict["overall"] == "NO RESULT"
    assert verdict["bars"]["four_g1"]["status"] == "NO RESULT"
    assert "S1 has no verdict yet" in verdict["next_step"]
    assert "raise RuntimeError" in _s1_cells()["s1-verdict"]


def test_s1_smoke_run_never_yields_a_verdict() -> None:
    verdict = _verdict([_row("playground", 5e4)], [_row("mjx-scene", 6e3)], smoke=True)
    assert verdict["overall"] == "SMOKE"
    assert {entry["status"] for entry in verdict["bars"].values()} == {"SMOKE"}
    assert "decides nothing" in verdict["next_step"]


# -- S1: the four-G1 scene ------------------------------------------------------


def _scene_namespace(monkeypatch) -> dict[str, Any]:
    return _run_cell("s1-scene-build", _s1_settings(monkeypatch))


def test_s1_scene_composes_four_packaged_g1s_and_the_court_ball(monkeypatch) -> None:
    namespace = _scene_namespace(monkeypatch)
    model, changes = namespace["build_scene"](1000)
    g1 = get_robot_model_spec("unitree_g1")
    seats = tuple(namespace["SEATS"])
    assert seats == ("a0", "a1", "b0", "b1")
    assert changes == []
    assert model.nu == 4 * g1.nu
    assert int(np.sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)) == 5
    for seat in seats:
        for name in g1.actuator_names:
            model.actuator(f"{seat}_{name}")
        model.geom(f"{seat}_{g1.racket_stringbed_geom}")
    assert model.opt.timestep == pytest.approx(0.001)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert (model.opt.iterations, model.opt.ls_iterations) == (10, 20)

    court = mujoco.MjModel.from_xml_path(asset_path("humanoid_tennis.xml"))
    for scene_geom, court_geom in (
        ("ball_geom", "ball_geom"),
        ("floor", "court_surface"),
    ):
        mine, theirs = model.geom(scene_geom), court.geom(court_geom)
        for field in (
            "contype",
            "conaffinity",
            "condim",
            "priority",
            "friction",
            "solref",
            "solimp",
            "margin",
        ):
            np.testing.assert_array_equal(
                getattr(mine, field), getattr(theirs, field), err_msg=field
            )
    np.testing.assert_array_equal(
        model.geom("ball_geom").size, court.geom("ball_geom").size
    )
    np.testing.assert_allclose(model.body("ball").mass, court.body("ball").mass)
    np.testing.assert_allclose(model.body("ball").inertia, court.body("ball").inertia)
    pair, court_pair = model.pair("ball_floor_pair"), court.pair("ball_court_pair")
    assert {int(pair.geom1[0]), int(pair.geom2[0])} == {
        model.geom("ball_geom").id,
        model.geom("floor").id,
    }
    for field in ("dim", "friction", "solref", "solimp", "margin", "gap"):
        np.testing.assert_array_equal(
            getattr(pair, field), getattr(court_pair, field), err_msg=field
        )


def test_s1_scene_starts_every_seat_at_the_standing_reference(monkeypatch) -> None:
    namespace = _scene_namespace(monkeypatch)
    model, _ = namespace["build_scene"](1000)
    qpos, qvel, ctrl = namespace["scene_home"](model)
    g1 = get_robot_model_spec("unitree_g1")
    for seat, (spawn, _quat) in namespace["SEATS"].items():
        base = int(model.joint(f"{seat}_floating_base_joint").qposadr[0])
        np.testing.assert_allclose(
            qpos[base : base + 3], (*spawn[:2], UNITREE_G1_STAND_BASE_HEIGHT)
        )
        for name, target in zip(
            g1.actuator_names, g1.stand_joint_positions, strict=True
        ):
            assert qpos[int(model.joint(f"{seat}_{name}").qposadr[0])] == target
            assert ctrl[model.actuator(f"{seat}_{name}").id] == target
    ball = int(model.joint("ball_free").dofadr[0])
    np.testing.assert_array_equal(qvel[ball : ball + 3], namespace["BALL_VEL"])
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    for seat in namespace["SEATS"]:
        base = int(model.joint(f"{seat}_floating_base_joint").qposadr[0])
        assert data.qpos[base + 2] > 0.7  # 0.1 s at the standing targets: still up


def _colliding_categories(model: mujoco.MjModel) -> list[tuple[int, int]]:
    mask = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    pairs = zip(model.geom_contype[mask], model.geom_conaffinity[mask], strict=True)
    return sorted((int(kind), int(affinity)) for kind, affinity in pairs)


def test_s1_simplified_collision_set_drops_only_what_it_reports(monkeypatch) -> None:
    namespace = _scene_namespace(monkeypatch)
    full, _ = namespace["build_scene"](1000)
    simple, changes = namespace["build_scene"](1000, simplified=True)
    assert not np.any(simple.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    assert simple.nmesh == 0
    assert not np.any(simple.geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    assert int(np.sum(full.geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID)) == 4
    assert simple.pair("ball_floor_pair").dim[0] == 3
    assert "ball-floor pair condim 6 -> 3" in changes
    assert sum("deleted 35 visual mesh geoms" in change for change in changes) == 4
    # Every colliding geom survives, in its contact categories.
    assert _colliding_categories(simple) == _colliding_categories(full)
    assert (simple.nu, simple.nq, simple.nv) == (full.nu, full.nq, full.nv)
    assert simple.npair == full.npair


# -- S1: the results folder -----------------------------------------------------


def _paths_namespace(monkeypatch, tmp_path: Path, **replace: str) -> dict[str, Any]:
    namespace = _s1_settings(
        monkeypatch, **{"USE_DRIVE = True": "USE_DRIVE = False", **replace}
    )
    monkeypatch.chdir(tmp_path)
    namespace.update(
        json=json,
        re=re,
        time=time,
        datetime=datetime,
        UTC=UTC,
        STARTED=time.monotonic(),
        STARTED_UTC="20261017T000000Z",
        GPU_NAME="NVIDIA L4",
        ENVIRONMENT={"gpu": "NVIDIA L4"},
        REPO_REF="0" * 40,
    )
    return _run_cell("s1-paths", namespace)


def test_s1_results_go_under_diagnostics_and_are_saved_whole(
    monkeypatch, tmp_path
) -> None:
    namespace = _paths_namespace(monkeypatch, tmp_path)
    path = namespace["RESULTS_PATH"]
    assert path == Path(
        "diagnostics/s1_gpu_throughput/s1_nvidia-l4_20261017T000000Z.json"
    )
    saved = json.loads(path.read_text())
    assert saved["bars"] == {
        "single_g1_env_steps_per_s": 2e4,
        "four_g1_control_steps_per_s": 5e3,
    }
    assert saved["settings"]["PHYSICS_HZ"] == 1000
    # Written whole: no partial file is left beside it.
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]


def test_s1_refuses_a_results_folder_inside_training_runs(
    monkeypatch, tmp_path
) -> None:
    inside = {
        'RESULTS_SUBDIR = "diagnostics/s1_gpu_throughput"': (
            'RESULTS_SUBDIR = "training_runs/s1"'
        )
    }
    with pytest.raises(ValueError, match="training_runs"):
        _paths_namespace(monkeypatch, tmp_path, **inside)
    assert not (tmp_path / "training_runs").exists()


# -- S1: what the results record ----------------------------------------------


def test_s1_records_versions_timesteps_and_wall_times() -> None:
    cells = _s1_cells()
    for package in (
        "jax",
        "jax-cuda12-plugin",
        "mujoco",
        "mujoco-mjx",
        "warp-lang",
        "playground",
        "brax",
    ):
        assert f'"{package}"' in cells["s1-env"], package
    assert '"nvidia_smi": nvidia_smi()' in cells["s1-env"]
    for cell_id in ("s1-playground", "s1-scene"):
        for field in (
            '"batch"',
            '"timestep_s"',
            '"substeps"',
            '"physics_hz"',
            "**timing",
        ):
            assert field in cells[cell_id], (cell_id, field)
    for field in ("compile_s", "warmup_s", "timed_s"):
        assert f'"{field}"' in cells["s1-helpers"], field
    assert 'RESULTS["wall_s"]' in cells["s1-paths"]
    # Warp's own view of the GPU, and warp rows refuse to run without it.
    assert 'ENVIRONMENT["warp"] = {' in cells["s1-helpers"]
    for cell_id in ("s1-playground", "s1-scene"):
        assert "require_warp_gpu(impl)" in cells[cell_id], cell_id
