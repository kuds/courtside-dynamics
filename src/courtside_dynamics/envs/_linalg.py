"""Small-vector float64 math that rounds the same on every CPU.

numpy hands ``np.linalg.norm(v)``, ``np.dot(u, v)``, ``u @ v`` and
``A @ v`` on float64 arrays to OpenBLAS, which picks its kernels from the
CPU at run time, and the kernels do not round alike even on 3-vectors.
With numpy 2.5's OpenBLAS 0.3.34, the AVX-512 (SkylakeX) ``ddot`` and
``A.T @ v`` give other last bits than the AVX2 (Haswell, Zen) kernels,
and the AVX2-and-later ``A @ v`` differs from the pre-AVX2 one. A seeded
rollout that calls them reproduces only on CPUs that pick the same
kernels, and GitHub's runners and Colab's hosts vary (docs/DECISIONS.md,
"Golden digests and the freeze harness gate every refactor").

These helpers do the same arithmetic on Python floats instead: products
summed left to right from 0.0, which is what the pre-AVX2 kernels
compute, so the env streams give the same bits whatever the CPU.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def _vector(value: ArrayLike) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape {array.shape}")
    return array.tolist()


def _sum_of_products(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"vector lengths differ: {len(left)} and {len(right)}")
    total = 0.0
    for a, b in zip(left, right, strict=True):
        total += a * b
    return total


def dot(left: ArrayLike, right: ArrayLike) -> float:
    """``float(np.dot(left, right))`` for two 1-D vectors."""
    return _sum_of_products(_vector(left), _vector(right))


def norm(value: ArrayLike) -> float:
    """``float(np.linalg.norm(value))`` for a 1-D vector."""
    vector = _vector(value)
    return math.sqrt(_sum_of_products(vector, vector))


def matvec(matrix: ArrayLike, vector: ArrayLike) -> np.ndarray:
    """``matrix @ vector`` for a 2-D matrix; pass ``matrix.T`` to transpose."""
    rows = np.asarray(matrix, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {rows.shape}")
    values = _vector(vector)
    return np.array(
        [_sum_of_products(row, values) for row in rows.tolist()], dtype=np.float64
    )
