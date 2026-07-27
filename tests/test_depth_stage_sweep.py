from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from tools.depth_stage_sweep import (
    ALIGNED_STAGES,
    BASELINE_STAGES,
    LADDERS,
    LANDING_MEAN_TOLERANCE,
    LANDING_REQUIRED_FRACTION,
    LANDING_WINDOW,
    STAGES,
    TELEMETRY_KEYS,
    _apply_oracle_probe,
    _blend_serve_origins,
    _episode_telemetry,
    _landing_alignment_failures,
    _landing_statistics,
    _oracle,
    _paddle_contact_before_first_floor,
    _parked,
    _parked_invariant_failures,
    _parse_oracle_probe,
    _validate_ladder,
    _wilson_interval,
    _write_json_report,
)


def test_sliding_sweep_ladder_has_required_static_geometry():
    assert STAGES is ALIGNED_STAGES
    assert LADDERS == {
        "aligned": ALIGNED_STAGES,
        "baseline": BASELINE_STAGES,
    }
    assert [stage["paddle_x_fence"] for stage in STAGES] == [
        (-2.7, 0.3),
        (-3.2, -0.8),
        (-3.7, -1.6),
        (-4.2, -2.4),
        (-4.7, -3.0),
    ]
    assert [stage["serve_start_x"] for stage in STAGES] == [
        1.0,
        0.69,
        0.34,
        -0.01,
        -0.35,
    ]
    assert [stage["serve_start_x"] for stage in BASELINE_STAGES] == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert ALIGNED_STAGES[0] == BASELINE_STAGES[0]
    assert _validate_ladder(BASELINE_STAGES) == []
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


def test_parked_controller_targets_each_stage_start_not_global_home():
    for stage in STAGES:
        action = _parked(stage["paddle_start_x"])
        assert action.shape == (3,)
        assert action.dtype == np.float32
        assert action[1:].tolist() == [0.0, 0.0]

    assert _parked(STAGES[0]["paddle_start_x"])[0] == pytest.approx(0.05)
    assert _parked(STAGES[-1]["paddle_start_x"])[0] == pytest.approx(
        -2.2 / 3.0
    )
    assert not np.array_equal(
        _parked(STAGES[-1]["paddle_start_x"]),
        np.zeros(3, dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("events", "expected"),
    (
        ([(0, 1, "paddle"), (1, 0, "floor")], True),
        ([(0, 0, "floor"), (1, 1, "paddle")], False),
        ([(2, 1, "paddle"), (2, 0, "floor")], False),
        ([(0, 2, "wall"), (1, 0, "floor")], False),
        ([(0, 1, "paddle")], True),
        ([], False),
    ),
)
def test_landing_probe_uses_physical_substep_order(events, expected):
    assert _paddle_contact_before_first_floor(events) is expected


def test_hold_start_invariant_blocks_contacts_and_returns():
    passing = {
        "stage": 3,
        "contact": 0.0,
        "mean": 0.0,
        "ge2_successes": 0,
    }
    assert _parked_invariant_failures(passing) == []

    failures = _parked_invariant_failures({
        **passing,
        "contact": 0.005,
        "mean": 0.01,
        "ge2_successes": 1,
    })
    assert len(failures) == 2
    assert any("contacted" in failure for failure in failures)
    assert any("scored returns" in failure for failure in failures)


def test_landing_statistics_report_raw_offset_distribution():
    stats = _landing_statistics(
        [-2.50, -2.25, -2.10, -1.95, -1.70],
        paddle_start_x=-2.10,
    )

    assert stats["observed"] == 5
    assert stats["landing_mean"] == pytest.approx(-2.10)
    assert stats["mean_offset"] == pytest.approx(0.0)
    assert stats["offset_std"] == pytest.approx(
        np.std([-0.4, -0.15, 0, 0.15, 0.4])
    )
    assert stats["offset_p05"] == pytest.approx(-0.35)
    assert stats["offset_median"] == pytest.approx(0.0)
    assert stats["offset_p95"] == pytest.approx(0.35)
    assert stats["offset_min"] == pytest.approx(-0.4)
    assert stats["offset_max"] == pytest.approx(0.4)
    assert stats["offset_abs_max"] == pytest.approx(0.4)
    assert stats["within_window"] == pytest.approx(3 / 5)

    assert _landing_statistics([], -2.1) == {
        "observed": 0,
        "landing_mean": None,
        "target_offset": 0.0,
        "mean_offset": None,
        "mean_offset_from_start": None,
        "offset_std": None,
        "offset_p05": None,
        "offset_median": None,
        "offset_p95": None,
        "offset_min": None,
        "offset_max": None,
        "offset_abs_max": None,
        "within_window": None,
    }


def test_landing_statistics_reject_malformed_samples():
    with pytest.raises(ValueError, match="one-dimensional"):
        _landing_statistics([[-2.1]], -2.1)
    with pytest.raises(ValueError, match="finite"):
        _landing_statistics([np.nan], -2.1)
    with pytest.raises(ValueError, match="finite"):
        _landing_statistics([-2.1], np.inf)


def test_wilson_interval_handles_boundaries_and_rejects_bad_counts():
    assert _wilson_interval(0, 200) == pytest.approx(
        (0.0, 0.01884532637726656)
    )
    assert _wilson_interval(200, 200) == pytest.approx(
        (0.9811546736227335, 1.0)
    )
    low, high = _wilson_interval(168, 200)
    assert low < 0.84 < high

    with pytest.raises(ValueError, match="positive"):
        _wilson_interval(0, 0)
    with pytest.raises(ValueError, match="between"):
        _wilson_interval(2, 1)
    with pytest.raises(ValueError, match="positive finite"):
        _wilson_interval(1, 1, z=np.nan)


def test_landing_alignment_contract_is_blocking_and_inclusive():
    passing = {
        "stage": 2,
        "episodes": 200,
        "observed": 200,
        "pre_bounce_contact_episodes": 0,
        "mean_offset": LANDING_MEAN_TOLERANCE,
        "within_window": LANDING_REQUIRED_FRACTION,
    }
    assert _landing_alignment_failures(passing) == []

    failures = _landing_alignment_failures({
        **passing,
        "observed": 199,
        "pre_bounce_contact_episodes": 1,
        "mean_offset": LANDING_MEAN_TOLERANCE + 0.001,
        "within_window": LANDING_REQUIRED_FRACTION - 0.001,
    })
    assert len(failures) == 4
    assert any("199/200" in failure for failure in failures)
    assert any("contacted" in failure for failure in failures)
    assert any("mean serve-landing offset" in failure for failure in failures)
    assert any(
        f"within +/-{LANDING_WINDOW:.2f} m" in failure
        for failure in failures
    )
    baseline_failures = _landing_alignment_failures(
        {
            **passing,
            "mean_offset": 2.0,
            "within_window": 0.0,
        },
        require_alignment=False,
    )
    assert baseline_failures == []


def test_json_report_preserves_exact_paired_per_seed_outcomes(tmp_path):
    controller_rows = [
        {
            "stage": 0,
            "policy": "parked",
            "reward": -1.0,
            "samples": [
                {
                    "seed": 7,
                    "reward": -1.0123456789,
                    "bounce_count": 0,
                }
            ],
        }
    ]
    landing_rows = [
        {
            "stage": 0,
            "samples": [
                {
                    "seed": 7,
                    "landing_x": -1.6123456789,
                    "offset": -0.0123456789,
                }
            ],
        }
    ]
    output = tmp_path / "nested" / "sweep.json"

    _write_json_report(
        output,
        episodes=1,
        seed_start=7,
        rows=controller_rows,
        landing_rows=landing_rows,
        failures=["expected calibration failure"],
        ladder_name="aligned",
        stages=ALIGNED_STAGES,
        require_alignment=True,
        calibration_seed_start=0,
        calibration_episodes=1,
        calibration_artifact="calibration.json",
        provenance={"git_head": "abc123"},
    )

    report = json.loads(output.read_text())
    assert report["schema_version"] == 3
    assert report["ladder"] == {
        "name": "aligned",
        "serve_origin_blend": None,
        "paired_variant": "baseline",
        "pairing_key": "wall-ball-serve-alignment:7:8",
        "stage_zero_equivalent": True,
        "alignment_thresholds_blocking": True,
    }
    assert report["episodes_per_cell"] == 1
    assert report["seed_range"] == {
        "start": 7,
        "stop": 8,
        "stop_is_exclusive": True,
    }
    assert report["passed"] is False
    assert report["calibration"] == {
        "seed_range": {
            "start": 0,
            "stop": 1,
            "stop_is_exclusive": True,
        },
        "artifact": "calibration.json",
    }
    assert report["provenance"] == {"git_head": "abc123"}
    assert report["stages"] == json.loads(json.dumps(ALIGNED_STAGES))
    assert report["controller_cells"][0]["samples"][0] == {
        "seed": 7,
        "reward": -1.0123456789,
        "bounce_count": 0,
    }
    assert report["serve_landings"][0]["samples"][0]["seed"] == 7
    assert report["serve_landings"][0]["samples"][0]["landing_x"] == pytest.approx(
        -1.6123456789
    )


def test_sweep_oracle_requires_one_stage_probe_mode():
    obs = np.zeros(23, dtype=np.float64)
    with pytest.raises(ValueError, match="exactly one"):
        _oracle(obs, (-4.7, -3.0))
    with pytest.raises(ValueError, match="exactly one"):
        _oracle(obs, (-4.7, -3.0), run_up=1.4, charge_gap=1.7)


def test_serve_origin_blend_endpoints_reproduce_both_ladders_exactly():
    """The blend must be a no-op at both ends, bit-for-bit.

    Written as a lerp rather than 1.0 + blend * (aligned - 1.0) precisely
    so blend 1.0 does not drift off the aligned table by float error.
    """
    baseline = _blend_serve_origins(ALIGNED_STAGES, 0.0)
    aligned = _blend_serve_origins(ALIGNED_STAGES, 1.0)

    assert [s["serve_start_x"] for s in baseline] == [
        s["serve_start_x"] for s in BASELINE_STAGES
    ]
    assert [s["serve_start_x"] for s in aligned] == [
        s["serve_start_x"] for s in ALIGNED_STAGES
    ]
    # The fully aligned arm targets the paddle start itself, so its
    # declared offset is exactly zero and the contract is unchanged.
    assert all(s["landing_target_offset"] == 0.0 for s in aligned)
    # The fixed origin declares the full gap it leaves in front.
    assert baseline[4]["landing_target_offset"] == pytest.approx(1.35)
    assert baseline[0]["landing_target_offset"] == pytest.approx(0.0)
    # Tables are never mutated.
    assert "landing_target_offset" not in ALIGNED_STAGES[4]
    assert ALIGNED_STAGES[4]["serve_start_x"] == -0.35


def test_serve_origin_blend_interpolates_origin_and_target_together():
    half = _blend_serve_origins(ALIGNED_STAGES, 0.5)
    assert [s["serve_start_x"] for s in half] == pytest.approx(
        [1.0, 0.845, 0.67, 0.495, 0.325]
    )
    # Landing moves one-for-one with the origin, so a half blend leaves
    # half the fixed-origin gap in front of the paddle.
    assert [s["landing_target_offset"] for s in half] == pytest.approx(
        [0.0, 0.155, 0.33, 0.505, 0.675]
    )
    assert _blend_serve_origins(ALIGNED_STAGES, None) is ALIGNED_STAGES


def test_landing_contract_measures_against_the_declared_target():
    """A blended candidate is gated at its own target, not paddle_start_x."""
    # Landings sitting 0.70 m in front of a start at -2.70.
    landings = [-2.0, -2.0, -2.0, -2.0]

    against_start = _landing_statistics(landings, -2.70)
    assert against_start["mean_offset"] == pytest.approx(0.70)
    assert against_start["target_offset"] == 0.0
    # Would fail the contract if the target were the paddle start.
    assert _landing_alignment_failures(
        {**against_start, "stage": 2, "episodes": 4,
         "pre_bounce_contact_episodes": 0},
        require_alignment=True,
    )

    against_target = _landing_statistics(landings, -2.70, 0.70)
    assert against_target["mean_offset"] == pytest.approx(0.0)
    assert against_target["target_offset"] == pytest.approx(0.70)
    # The physical distance from the paddle survives the retarget.
    assert against_target["mean_offset_from_start"] == pytest.approx(0.70)
    assert against_target["within_window"] == 1.0
    assert not _landing_alignment_failures(
        {**against_target, "stage": 2, "episodes": 4,
         "pre_bounce_contact_episodes": 0},
        require_alignment=True,
    )


def test_landing_statistics_rejects_a_nonfinite_target():
    with pytest.raises(ValueError, match="target_offset must be finite"):
        _landing_statistics([-2.0], -2.7, float("nan"))


def test_oracle_probe_override_replaces_the_stage_probe_mode():
    """A grid search must be able to switch probe mode, not just its value.

    ``_oracle`` accepts exactly one of run_up / charge_gap, so overriding
    run_up at a charge_gap stage has to clear the other key outright.
    """
    overrides = _parse_oracle_probe(
        ["3=run_up:0.9", "4=charge_gap:2.4"], len(ALIGNED_STAGES)
    )
    assert overrides == {3: ("run_up", 0.9), 4: ("charge_gap", 2.4)}

    resolved = _apply_oracle_probe(ALIGNED_STAGES, overrides)
    # Stage 3 ships charge_gap; the override must swap the key, not add one.
    assert resolved[3]["oracle_run_up"] == 0.9
    assert "oracle_charge_gap" not in resolved[3]
    assert resolved[4]["oracle_charge_gap"] == 2.4
    assert "oracle_run_up" not in resolved[4]
    # Untouched stages keep their shipped probe, and the module tables are
    # never mutated -- a later ladder lookup must not inherit the override.
    assert resolved[0]["oracle_run_up"] == ALIGNED_STAGES[0]["oracle_run_up"]
    assert ALIGNED_STAGES[3]["oracle_charge_gap"] == 1.8
    assert "oracle_run_up" not in ALIGNED_STAGES[3]
    assert BASELINE_STAGES[3]["oracle_charge_gap"] == 1.8


@pytest.mark.parametrize(
    "spec, message",
    [
        ("9=charge_gap:1.4", "outside the ladder"),
        ("3=bogus:1.4", "expects STAGE="),
        ("3charge_gap1.4", "expects STAGE="),
        ("3=charge_gap:x", "integer stage and a float"),
        ("3=charge_gap:-1", "positive and finite"),
        ("3=charge_gap:0", "positive and finite"),
        ("3=charge_gap:nan", "positive and finite"),
    ],
)
def test_oracle_probe_override_rejects_malformed_specs(spec, message):
    with pytest.raises(SystemExit, match=message):
        _parse_oracle_probe([spec], len(ALIGNED_STAGES))


def test_oracle_probe_override_rejects_a_repeated_stage():
    with pytest.raises(SystemExit, match="more than once"):
        _parse_oracle_probe(
            ["3=charge_gap:1.4", "3=run_up:0.9"], len(ALIGNED_STAGES)
        )


def test_oracle_probe_override_is_a_no_op_when_unset():
    assert _apply_oracle_probe(ALIGNED_STAGES, {}) is ALIGNED_STAGES
    assert _parse_oracle_probe(None, len(ALIGNED_STAGES)) == {}


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
