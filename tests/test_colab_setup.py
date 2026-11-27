"""Tests for ``courtside_dynamics.colab_setup``.

``setup_colab`` is the notebooks' first package call. On a GPU runtime it
points MuJoCo at EGL on the NVIDIA driver; on a CPU runtime (a PPO run,
which trains on CPU by design) it must neither refuse nor force an EGL
backend that cannot work there; outside Colab it is a no-op.
"""
from __future__ import annotations

import importlib.util
import os

# Imported before any test sets MUJOCO_GL=egl: setup_colab's own
# ``import mujoco`` is then the cached module. A first import of mujoco
# under MUJOCO_GL=egl on a host without an EGL library fails inside
# mujoco/__init__ (PyOpenGL's EGL binding is None) -- the very failure the
# CPU-runtime path avoids by leaving MUJOCO_GL alone -- and would leave a
# half-imported mujoco behind for the rest of the worker.
import mujoco  # noqa: F401
import pytest

from courtside_dynamics import colab_setup


@pytest.fixture
def _restore_mujoco_gl():
    """``setup_colab`` writes ``MUJOCO_GL``; put it back afterwards."""
    original = os.environ.get("MUJOCO_GL")
    yield
    if original is None:
        os.environ.pop("MUJOCO_GL", None)
    else:
        os.environ["MUJOCO_GL"] = original


def _colab(monkeypatch, *, gpu: bool) -> list[str]:
    """Pretend to run on Colab with or without a GPU; record ICD writes."""
    icd_writes: list[str] = []
    monkeypatch.setattr(colab_setup, "_in_colab", lambda: True)
    monkeypatch.setattr(colab_setup, "_check_gpu", lambda: gpu)
    monkeypatch.setattr(
        colab_setup, "_ensure_nvidia_icd", lambda: icd_writes.append("icd")
    )
    return icd_writes


def test_setup_colab_is_a_no_op_outside_colab(monkeypatch, _restore_mujoco_gl):
    monkeypatch.setattr(colab_setup, "_in_colab", lambda: False)
    monkeypatch.setattr(
        colab_setup, "_check_gpu", lambda: pytest.fail("probed the GPU off Colab")
    )
    os.environ.pop("MUJOCO_GL", None)
    assert colab_setup.setup_colab() is None
    assert "MUJOCO_GL" not in os.environ


def test_setup_colab_refuses_a_cpu_runtime_when_a_gpu_is_required(
    monkeypatch, _restore_mujoco_gl
):
    icd_writes = _colab(monkeypatch, gpu=False)
    with pytest.raises(RuntimeError, match="Cannot communicate with GPU"):
        colab_setup.setup_colab()
    assert icd_writes == []


def test_setup_colab_admits_a_cpu_runtime_without_forcing_egl(
    monkeypatch, capsys, _restore_mujoco_gl
):
    """A PPO notebook on a CPU runtime: no refusal, no NVIDIA ICD, and
    MUJOCO_GL left as it was (forcing EGL without an EGL library would
    break ``import mujoco``). The recorders then skip video: no display."""
    icd_writes = _colab(monkeypatch, gpu=False)
    os.environ.pop("MUJOCO_GL", None)
    assert colab_setup.setup_colab(require_gpu=False) is False
    assert icd_writes == []
    assert "MUJOCO_GL" not in os.environ
    assert "video recording is skipped" in capsys.readouterr().out

    os.environ["MUJOCO_GL"] = "osmesa"  # a user's own choice is kept
    assert colab_setup.setup_colab(require_gpu=False) is False
    assert os.environ["MUJOCO_GL"] == "osmesa"


@pytest.mark.parametrize("require_gpu", [True, False])
def test_setup_colab_configures_egl_on_a_gpu_runtime(
    monkeypatch, require_gpu, _restore_mujoco_gl
):
    icd_writes = _colab(monkeypatch, gpu=True)
    assert colab_setup.setup_colab(require_gpu=require_gpu) is True
    assert icd_writes == ["icd"]
    assert os.environ["MUJOCO_GL"] == "egl"


def test_in_colab_is_false_without_a_google_package(monkeypatch):
    """``find_spec("google.colab")`` imports the parent package, so with
    no ``google`` installed it raises instead of returning ``None``."""

    def no_google(name, *args, **kwargs):
        raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")

    monkeypatch.setattr(importlib.util, "find_spec", no_google)
    assert colab_setup._in_colab() is False
