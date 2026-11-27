"""Tests for ``courtside_dynamics.numerics``: pinned CPU float32 numerics.

MKL reads its CNR branch once, at its first call, so each case runs in a
fresh interpreter: the pin, an operator's override, a too-late import, and
the property the pin exists for -- a policy's CPU outputs are the same
bits whichever kernels the host's MKL would pick (review 2026-11-27, N2).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mkl.is_available(), reason="torch built without MKL"
)

#: A policy-shaped MLP's outputs on fixed inputs, hashed: the thing that
#: differs between MKL's AVX2 and AVX-512 kernels when unpinned.
_POLICY_OUTPUTS = """
import hashlib, json, os, sys
{prelude}
import torch
torch.set_num_threads(1)
generator = torch.Generator().manual_seed(0)
layers = [torch.nn.Linear(48, 256), torch.nn.Linear(256, 256), torch.nn.Linear(256, 3)]
for layer in layers:
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.1)
        layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator) * 0.1)
digest = hashlib.sha256()
for batch in (1, 8, 256, 2048):
    x = torch.randn(batch, 48, generator=generator)
    with torch.no_grad():
        digest.update(layers[2](torch.tanh(layers[1](torch.tanh(layers[0](x))))).numpy().tobytes())
from courtside_dynamics.numerics import cpu_numerics
print(json.dumps({{"hash": digest.hexdigest(), "numerics": cpu_numerics()}}))
"""


def _run(script: str, **env_overrides: str | None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in {"MKL_CBWR", "MKL_ENABLE_INSTRUCTIONS"}}
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def _outputs(prelude: str = "import courtside_dynamics", **env: str | None) -> dict:
    return _run(_POLICY_OUTPUTS.format(prelude=prelude), **env)


def test_importing_the_package_pins_mkl():
    numerics = _outputs()["numerics"]
    assert numerics["blas"] == "mkl"
    assert numerics["mkl_cbwr"] == "COMPATIBLE"
    assert numerics["pinned"] is True


def test_an_operators_own_branch_wins_and_is_reported_unpinned():
    numerics = _outputs(MKL_CBWR="AUTO")["numerics"]
    assert numerics["mkl_cbwr"] == "AUTO"
    assert numerics["mkl_cbwr_env"] == "AUTO"
    assert numerics["pinned"] is False


def test_an_mkl_call_before_the_import_leaves_mkl_unpinned():
    """MKL fixes its branch at its first call: a process that multiplies
    in torch before importing the package keeps MKL's default, and says
    so instead of claiming the pin."""
    report = _run(
        """
        import json, torch
        torch.ones(8, 8) @ torch.ones(8, 8)
        import courtside_dynamics
        from courtside_dynamics.numerics import cpu_numerics, cpu_numerics_problem
        print(json.dumps({"numerics": cpu_numerics(), "problem": cpu_numerics_problem()}))
        """
    )
    assert report["numerics"]["mkl_cbwr"] == "OFF"
    assert report["numerics"]["pinned"] is False
    assert "import courtside_dynamics before" in report["problem"]


def test_pinned_policy_outputs_are_the_same_under_every_mkl_kernel():
    """The property the pin exists for. ``MKL_ENABLE_INSTRUCTIONS=AVX2``
    makes this host's MKL run the kernels an AVX2-only host would: pinned,
    the policy's float32 outputs are bit-identical either way. On an
    AVX-512 host the unpinned outputs differ, which shows the comparison
    can see the problem at all."""
    pinned_default = _outputs()["hash"]
    pinned_avx2 = _outputs(MKL_ENABLE_INSTRUCTIONS="AVX2")["hash"]
    assert pinned_default == pinned_avx2
    if torch.backends.cpu.get_cpu_capability() == "AVX512":
        unpinned_default = _outputs(MKL_CBWR="AUTO")["hash"]
        unpinned_avx2 = _outputs(MKL_CBWR="AUTO", MKL_ENABLE_INSTRUCTIONS="AVX2")["hash"]
        assert unpinned_default != unpinned_avx2


def test_cpu_numerics_problem_names_the_fix(monkeypatch):
    from courtside_dynamics import numerics

    monkeypatch.setattr(
        numerics,
        "cpu_numerics",
        lambda: {
            "blas": "mkl",
            "mkl_cbwr": "AVX2",
            "mkl_cbwr_env": "AVX2",
            "torch_cpu_capability": "AVX2",
            "pinned": False,
        },
    )
    problem = numerics.cpu_numerics_problem()
    assert "AVX2 CNR branch, not COMPATIBLE" in problem
    assert "MKL_CBWR=COMPATIBLE" in problem
    for blas in ("other", "no-torch"):
        monkeypatch.setattr(
            numerics,
            "cpu_numerics",
            lambda blas=blas: {"blas": blas, "mkl_cbwr": None, "pinned": False},
        )
        assert numerics.cpu_numerics_problem() is None  # nothing to pin here
