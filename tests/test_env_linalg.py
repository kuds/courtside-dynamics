"""The env streams' small-vector math rounds the same on every CPU.

numpy hands float64 ``np.linalg.norm``, ``np.dot`` and ``@`` to an
OpenBLAS that picks its kernels from the CPU, and the kernels round
3-vectors differently, so the env code does that math in
:mod:`courtside_dynamics.envs._linalg`. These tests pin the helpers'
arithmetic and keep BLAS-routed calls out of the stream code;
``tests/test_golden_digests.py`` replays CI cases under each kernel.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from courtside_dynamics.envs import _linalg

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "courtside_dynamics"

#: The modules whose code runs inside a digested env stream: the envs,
#: the scripted policies that drive them, and the digest harness.
_STREAM_MODULES = (
    *sorted((PACKAGE / "envs").glob("*.py")),
    PACKAGE / "scripted_policies.py",
    PACKAGE / "digests.py",
)

#: numpy entry points that reach BLAS for float64 arrays.
_BLAS_CALLS = frozenset({"dot", "matmul", "inner", "vdot", "tensordot", "einsum"})


def _left_to_right(left: list[float], right: list[float]) -> float:
    total = 0.0
    for a, b in zip(left, right, strict=True):
        total += a * b
    return total


def test_dot_and_norm_sum_left_to_right_from_zero():
    rng = np.random.default_rng(0)
    for _ in range(200):
        u = rng.standard_normal(3) * rng.uniform(0.01, 10.0)
        v = rng.standard_normal(3)
        assert _linalg.dot(u, v) == _left_to_right(u.tolist(), v.tolist())
        assert _linalg.norm(u) == math.sqrt(_left_to_right(u.tolist(), u.tolist()))
        assert _linalg.norm(u) == pytest.approx(float(np.linalg.norm(u)), rel=1e-15)


def test_norm_pins_a_value_the_avx512_ddot_rounds_up():
    # numpy's AVX-512 (SkylakeX) OpenBLAS kernel gives 16.174162073787652
    # here; the AVX2 and pre-AVX2 kernels, and these helpers, give ...765.
    vector = [15.490675797091775, 0.15637415233345517, 4.649519250742807]
    assert _linalg.norm(vector) == 16.17416207378765


def test_matvec_takes_each_row_left_to_right():
    rng = np.random.default_rng(1)
    for _ in range(200):
        matrix = rng.standard_normal((3, 3))
        vector = rng.standard_normal(3)
        for operand in (matrix, matrix.T):
            product = _linalg.matvec(operand, vector)
            assert product.dtype == np.float64 and product.shape == (3,)
            expected = [_left_to_right(row, vector.tolist()) for row in operand.tolist()]
            assert product.tolist() == expected
            np.testing.assert_allclose(product, operand @ vector, rtol=1e-14, atol=1e-15)


def test_shapes_are_checked():
    with pytest.raises(ValueError, match="1-D"):
        _linalg.norm(np.zeros((3, 1)))
    with pytest.raises(ValueError, match="lengths differ"):
        _linalg.dot([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="2-D"):
        _linalg.matvec(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="lengths differ"):
        _linalg.matvec(np.zeros((3, 2)), np.zeros(3))


def test_nonfinite_values_pass_through():
    assert math.isnan(_linalg.norm([math.nan, 0.0, 0.0]))
    assert _linalg.norm([math.inf, 1.0, 0.0]) == math.inf


def _blas_routed_calls(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            found.append(f"{path.name}:{node.lineno}: @")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            owner = ast.unparse(node.func.value)
            if owner == "_linalg":
                continue
            if name in _BLAS_CALLS or owner.split(".")[-1] == "linalg":
                found.append(f"{path.name}:{node.lineno}: {owner}.{name}")
    return found


def test_stream_code_never_reaches_blas():
    """Stream code must not call numpy's BLAS-backed products and norms.

    The kernel replay in ``tests/test_golden_digests.py`` sees only the
    paths its CI cases reach; this sees every line.
    """
    found = [hit for path in _STREAM_MODULES for hit in _blas_routed_calls(path)]
    assert found == [], "use courtside_dynamics.envs._linalg:\n" + "\n".join(found)
