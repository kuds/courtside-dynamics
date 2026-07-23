from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from tools.depth_stage_sweep import (
    STAGES,
    TELEMETRY_KEYS,
    _episode_telemetry,
    _oracle,
    _validate_ladder,
)


def test_sliding_sweep_ladder_has_required_static_geometry():
    assert [stage["paddle_x_fence"] for stage in STAGES] == [
        (-2.7, 0.3),
        (-3.2, -0.8),
        (-3.7, -1.6),
        (-4.2, -2.4),
        (-4.7, -3.0),
    ]
    assert _validate_ladder(STAGES) == []
    assert [stage.get("oracle_charge_gap") for stage in STAGES] == [
        None,
        1.0,
        1.0,
        1.8,
        1.7,
    ]
    assert all(
        (stage.get("oracle_run_up") is None)
        != (stage.get("oracle_charge_gap") is None)
        for stage in STAGES
    )


def test_sweep_oracle_requires_one_stage_probe_mode():
    obs = np.zeros(23, dtype=np.float64)
    with pytest.raises(ValueError, match="exactly one"):
        _oracle(obs, (-4.7, -3.0))
    with pytest.raises(ValueError, match="exactly one"):
        _oracle(obs, (-4.7, -3.0), run_up=1.4, charge_gap=1.7)


def test_sweep_static_validation_rejects_each_shortcut_shape():
    outside = deepcopy(STAGES)
    outside[0]["paddle_start_x"] = -3.0
    assert any("outside fence" in error for error in _validate_ladder(outside))

    disjoint = deepcopy(STAGES)
    disjoint[1]["paddle_x_fence"] = (-4.0, -3.5)
    assert any("do not overlap" in error for error in _validate_ladder(disjoint))

    common_refuge = deepcopy(STAGES)
    for stage in common_refuge:
        stage["paddle_x_fence"] = (-4.7, 0.3)
    assert any(
        "all stages share" in error
        for error in _validate_ladder(common_refuge)
    )


def test_sweep_telemetry_contract_and_missing_key_handling():
    valid = {
        "pre_bounce_legal_paddle_hit_count": 1,
        "post_bounce_legal_paddle_hit_count": 2,
        "opening_volley_count": 1,
        "post_bounce_completed_return_count": 2,
        "legal_paddle_hit_count": 3,
        "bounce_count": 3,
    }
    values, missing, errors = _episode_telemetry(valid)
    assert values is not None
    assert missing == ()
    assert errors == ()

    invalid = {
        **valid,
        "pre_bounce_legal_paddle_hit_count": 0,
        "opening_volley_count": 1,
        "post_bounce_completed_return_count": 4,
    }
    _, _, errors = _episode_telemetry(invalid)
    assert set(errors) == {
        "pre + post legal hits != legal hits",
        "opening volleys > pre-bounce legal hits",
        "post-bounce completed returns > completed returns",
        "post-bounce completed returns > post-bounce legal hits",
    }

    values, missing, errors = _episode_telemetry({})
    assert values is None
    assert set(missing) == set(TELEMETRY_KEYS)
    assert errors == ()
