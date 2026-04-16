"""Colab bootstrap helpers.

The original notebooks each carried a copy of the same Colab / EGL / ffmpeg
setup block. This module consolidates that logic so a notebook only needs::

    from courtside_dynamics.colab_setup import setup_colab
    setup_colab()

It is a no-op outside of Colab.
"""
from __future__ import annotations

import importlib
import os
import subprocess


NVIDIA_ICD_CONFIG_PATH = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
_NVIDIA_ICD_CONTENTS = """{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
"""


def _in_colab() -> bool:
    return importlib.util.find_spec("google.colab") is not None


def _ensure_nvidia_icd() -> None:
    if os.path.exists(NVIDIA_ICD_CONFIG_PATH):
        return
    try:
        os.makedirs(os.path.dirname(NVIDIA_ICD_CONFIG_PATH), exist_ok=True)
        with open(NVIDIA_ICD_CONFIG_PATH, "w") as f:
            f.write(_NVIDIA_ICD_CONTENTS)
    except PermissionError:
        # Non-root environments can't write here; MuJoCo will fall back to
        # another GL backend.
        pass


def _check_gpu() -> bool:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def setup_colab(require_gpu: bool = True) -> None:
    """Configure MuJoCo for GPU rendering inside Google Colab.

    Parameters
    ----------
    require_gpu:
        Raise ``RuntimeError`` if no GPU is available. Set to ``False`` if
        you want to run the notebook on CPU.
    """
    if not _in_colab():
        return

    if require_gpu and not _check_gpu():
        raise RuntimeError(
            "Cannot communicate with GPU. Go to Runtime -> Change runtime type "
            "and select a GPU accelerator."
        )

    _ensure_nvidia_icd()
    os.environ["MUJOCO_GL"] = "egl"

    try:
        import mujoco

        mujoco.MjModel.from_xml_string("<mujoco/>")
    except Exception as exc:  # pragma: no cover - Colab-only
        raise RuntimeError(
            "MuJoCo failed to initialize after Colab setup; check the GPU runtime."
        ) from exc
