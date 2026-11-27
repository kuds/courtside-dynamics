"""Colab bootstrap helpers.

The original notebooks each carried a copy of the same Colab / EGL / ffmpeg
setup block. This module consolidates that logic so a notebook only needs::

    from courtside_dynamics.colab_setup import setup_colab
    setup_colab()

It is a no-op outside of Colab.
"""
from __future__ import annotations

import importlib.util
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
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        # find_spec imports the parent package: no ``google`` at all.
        return False


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


def setup_colab(require_gpu: bool = True) -> bool | None:
    """Configure MuJoCo rendering inside Google Colab.

    On a GPU runtime, MuJoCo renders off-screen through EGL on the NVIDIA
    driver. On a CPU runtime (``require_gpu=False``) ``MUJOCO_GL`` is left
    alone: the NVIDIA EGL setup cannot work there, and forcing EGL without
    an EGL library would break ``import mujoco`` itself. Colab has no
    display, so the recorders skip video without opening a render context
    (``callbacks.video_record``). Training and evaluation never render.

    Parameters
    ----------
    require_gpu:
        Raise ``RuntimeError`` when the runtime has no GPU. The SAC family
        needs one. PPO trains on CPU by design (its profiles pin
        ``device="cpu"``), so a PPO notebook passes ``False``.

    Returns
    -------
    ``True`` or ``False`` on Colab, saying whether ``nvidia-smi`` found a
    GPU; ``None`` outside Colab, where this is a no-op. The nvidia-smi
    check is a fast first gate, not the device guarantee:
    ``TrainConfig.require_device`` is the check that torch can see the
    GPU.
    """
    if not _in_colab():
        return None

    has_gpu = _check_gpu()
    if not has_gpu:
        if require_gpu:
            raise RuntimeError(
                "Cannot communicate with GPU. Go to Runtime -> Change runtime "
                "type and select a GPU accelerator."
            )
        print(
            "[setup_colab] CPU runtime: MuJoCo is left without EGL, so video "
            "recording is skipped (Colab has no display). Training and "
            "evaluation do not render."
        )
        return False

    _ensure_nvidia_icd()
    os.environ["MUJOCO_GL"] = "egl"

    try:
        import mujoco

        mujoco.MjModel.from_xml_string("<mujoco/>")
    except Exception as exc:  # pragma: no cover - Colab-only
        raise RuntimeError(
            "MuJoCo failed to initialize after Colab setup; check the GPU runtime."
        ) from exc
    return True
