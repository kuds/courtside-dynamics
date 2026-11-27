"""Pinned CPU numerics: torch's CPU float32 matmuls give the same bits on every host.

torch's CPU ``Linear`` layers call MKL's SGEMM, and MKL picks its kernel
from the host's instruction set at run time: an AVX-512 host and an AVX2
host round a policy's float32 outputs differently in the last bits.
PaddleTennis is chaotic, so a one-ulp action difference re-rolls whole
episodes: the same snapshot, seeds and protocol fingerprint scored 53 of
132 metrics differently on the two kernels (the 2026-11-27 review, N2).

MKL's conditional numerical reproducibility (CNR) fixes the code path.
``MKL_CBWR=COMPATIBLE`` is the branch Intel documents as reproducible
across processors, Intel and non-Intel alike. Measured on an AVX-512
host, single-threaded: batch-1 inference (a re-score's policy calls)
costs the same; a 256-wide MLP's batch-30 forward pass and its 64- to
256-batch training steps run 2-2.7x slower; a PaddleTennis PPO
iteration (SB3's 64x64 nets) took the same 4.5-4.9 s either way. GPU
training is untouched (cuBLAS, not MKL).

MKL reads the variable once, at its first call, and a read of the branch
fixes it too. So :func:`pin_cpu_numerics` runs when the package is
imported, before anything here touches torch, and an operator's own
non-empty ``MKL_CBWR`` wins. Whether the pin took effect is read from
MKL itself (:func:`cpu_numerics`): a process that ran an MKL call before
importing this package keeps MKL's default branch, and the offline
evaluation harness refuses to score a CPU protocol there. So does a
build whose branch cannot be read: the pin cannot be shown there.

Nothing here imports torch at module import time.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

#: The MKL CNR branch the package pins for every process that imports it.
PINNED_MKL_CBWR = "COMPATIBLE"

#: ``mkl_cbwr_get``'s option asking for the active branch (MKL_CBWR_BRANCH).
_MKL_CBWR_BRANCH = 1

#: MKL's branch codes (mkl_cbwr.h), as ``mkl_cbwr_get(MKL_CBWR_BRANCH)``
#: returns them. 1 is "no CNR branch": MKL dispatches on the host.
_MKL_BRANCH_NAMES: dict[int, str] = {
    1: "OFF",
    2: "AUTO",
    3: "COMPATIBLE",
    4: "SSE2",
    6: "SSSE3",
    7: "SSE4_1",
    8: "SSE4_2",
    9: "AVX",
    10: "AVX2",
    11: "AVX512_MIC",
    12: "AVX512",
    13: "AVX512_MIC_E1",
    14: "AVX512_E1",
}

#: ``mkl_cbwr_get`` under the names torch's builds export it: the public
#: symbol where MKL is a shared library, its service twin where torch
#: links MKL statically (the PyPI wheels).
_CBWR_GET_SYMBOLS = ("mkl_cbwr_get", "mkl_serv_cbwr_get")


def pin_cpu_numerics() -> None:
    """Pin MKL's CNR branch for this process unless the operator chose one.

    Runs at package import. Effective only if no MKL call has happened
    yet in this process; :func:`cpu_numerics` reports whether it did. An
    empty ``MKL_CBWR`` counts as unset (MKL ignores it).
    """
    if not os.environ.get("MKL_CBWR", "").strip():
        os.environ["MKL_CBWR"] = PINNED_MKL_CBWR


#: Loaded libraries that may export ``mkl_cbwr_get``: torch's own (the
#: PyPI wheels link MKL into it) and MKL's (conda builds load it shared).
_MKL_LIBRARY_MARKERS = ("libtorch_cpu", "libmkl_rt", "libmkl_core")


def _mkl_libraries(torch_module: Any) -> list[Path]:
    """Libraries to ask for MKL's branch: torch's ``lib`` dir, then the
    MKL-bearing libraries this process has loaded (Linux)."""
    lib_dir = Path(torch_module.__file__).parent / "lib"
    candidates = [
        lib_dir / name
        for name in ("libtorch_cpu.so", "libtorch_cpu.dylib", "torch_cpu.dll")
    ]
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].startswith("/"):
                    path = Path(fields[5].strip())
                    if path.name.startswith(_MKL_LIBRARY_MARKERS):
                        candidates.append(path)
    except OSError:
        pass  # no procfs: macOS, Windows
    found: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() not in found:
            found.append(candidate.resolve())
    return found


def _mkl_cbwr_branch(torch_module: Any) -> int | None:
    """The active MKL CNR branch code, or ``None`` if MKL cannot say.

    Reading it fixes the branch for the rest of the process, like any
    first MKL call: call this only after :func:`pin_cpu_numerics`.
    """
    for library in _mkl_libraries(torch_module):
        try:
            handle = ctypes.CDLL(str(library))
        except OSError:
            continue
        for symbol in _CBWR_GET_SYMBOLS:
            function = getattr(handle, symbol, None)
            if function is None:
                continue
            function.restype = ctypes.c_int
            function.argtypes = [ctypes.c_int]
            return int(function(_MKL_CBWR_BRANCH))
    return None


def cpu_numerics() -> dict[str, Any]:
    """How torch's CPU float32 matmuls are computed in this process.

    ``blas``: ``"mkl"`` (torch's CPU BLAS is MKL), ``"other"`` (a build
    without MKL, e.g. Apple silicon), or ``"no-torch"``. ``mkl_cbwr``:
    the active MKL CNR branch by name (``None`` when unreadable);
    ``mkl_cbwr_env``: the ``MKL_CBWR`` variable; ``torch_cpu_capability``:
    torch's own vectorized-kernel level. ``pinned`` is true when MKL
    runs the pinned branch, the only state in which a CPU score is the
    same on every host.
    """
    info: dict[str, Any] = {
        "blas": "no-torch",
        "mkl_cbwr": None,
        "mkl_cbwr_env": os.environ.get("MKL_CBWR"),
        "torch_cpu_capability": None,
        "pinned": False,
    }
    try:
        import torch
    except ImportError:
        return info
    try:
        info["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    except (AttributeError, RuntimeError):
        pass
    if not torch.backends.mkl.is_available():
        info["blas"] = "other"
        return info
    info["blas"] = "mkl"
    code = _mkl_cbwr_branch(torch)
    if code is not None:
        info["mkl_cbwr"] = _MKL_BRANCH_NAMES.get(code, f"branch {code}")
    info["pinned"] = info["mkl_cbwr"] == PINNED_MKL_CBWR
    return info


def cpu_numerics_problem() -> str | None:
    """Why a CPU score in this process would not match another host's.

    ``None`` when MKL runs the pinned branch. A build without MKL (or
    without torch) returns ``None`` too: its kernels cannot be pinned
    from here, and :func:`cpu_numerics` records what ran.
    """
    info = cpu_numerics()
    if info["pinned"] or info["blas"] != "mkl":
        return None
    if info["mkl_cbwr"] is None:
        return (
            f"torch is built with MKL, but its CNR branch cannot be read "
            f"here (no {' or '.join(_CBWR_GET_SYMBOLS)} in torch's "
            f"libraries or a loaded MKL library), so the {PINNED_MKL_CBWR} "
            f"pin cannot be shown and a CPU score here might not be the "
            f"protocol's. Score with the PyPI torch wheel on Linux (Colab, "
            f"CI) (see docs/DECISIONS.md, 'CPU scores pin MKL's numerics')"
        )
    active = info["mkl_cbwr"]
    return (
        f"torch's MKL runs its {active} CNR branch, not {PINNED_MKL_CBWR} "
        f"(MKL_CBWR={info['mkl_cbwr_env']!r}), so its CPU float32 matmuls "
        f"round differently on AVX2 and AVX-512 hosts and a score here "
        f"would not be the protocol's. courtside_dynamics pins "
        f"MKL_CBWR={PINNED_MKL_CBWR} when it is imported, which takes "
        f"effect only before torch's first MKL call: start Python with "
        f"MKL_CBWR={PINNED_MKL_CBWR}, or import courtside_dynamics before "
        f"running anything in torch (see docs/DECISIONS.md, 'CPU scores "
        f"pin MKL's numerics')"
    )
