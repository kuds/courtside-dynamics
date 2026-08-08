"""The ground-era diagnosis instrument: event walk and attribution.

Smoke-validates ``tools/paddle_tennis_diagnosis_probe.py`` on the
ground oracle (calibration seeds only; the instrument's real subject
-- the learned checkpoint -- runs on Colab). The oracle row doubles as
a numeric self-check: its designed constants must be recoverable from
the instrumented observations.
"""
from __future__ import annotations

import numpy as np

from courtside_dynamics.envs._paddle_court import (
    GROUND_WAIT_MARGIN,
    scripted_ground_opponent,
)
from courtside_dynamics.envs.tennis_rules import CourtSide
from tools.paddle_tennis_diagnosis_probe import (
    report,
    run_player,
)


class TestDiagnosisInstrument:
    def test_oracle_reference_row_is_coherent(self):
        traces = run_player(
            scripted_ground_opponent, episodes=6, seed_start=1000
        )
        assert len(traces) == 6
        # Serve alternation reaches the instrument.
        assert {t.serve_side_is_policy for t in traces} == {True, False}
        for trace in traces:
            # Both sides hit in a real rally trace; every shot record
            # belongs to a side and closed with a known outcome.
            assert trace.shots, "no shots instrumented"
            for shot in trace.shots:
                assert shot.hitter in (CourtSide.A, CourtSide.B)
                assert shot.outcome in (
                    "in",
                    "out",
                    "net",
                    "opponent_hit",
                    "open",
                    "terminal",
                )
            # A recovery window exists for each policy hit whose
            # follow-through completed before the point ended.
            assert len(trace.recovery_travel) <= trace.policy_hits
            assert len(trace.touched_after_bounce) >= 0
            assert trace.ender != ""

    def test_oracle_recovers_designed_wait_margin(self):
        """The bounce-time ready error must reproduce the ground
        oracle's designed wait margin -- the instrument's numeric
        self-check (bring-up measured 0.90 vs the 0.9 constant)."""
        traces = run_player(
            scripted_ground_opponent, episodes=6, seed_start=1000
        )
        errors = [e for t in traces for e in t.ready_errors]
        assert errors, "no ready-position snapshots instrumented"
        assert abs(float(np.mean(errors)) - GROUND_WAIT_MARGIN) < 0.25
        touched = [x for t in traces for x in t.touched_after_bounce]
        assert touched and np.mean(touched) > 0.9

    def test_statue_policy_reads_as_never_reached(self):
        """Adversarial regression (review 2026-08-08): a policy that
        never moves must be attributed policy_never_reached -- the
        first draft credited the opponent's good shots (or the feed)
        with going out when the untouched ball's second bounce
        skipped past the baseline, inverting H1 evidence into H3."""
        statue = lambda observation: np.zeros(3)  # noqa: E731
        traces = run_player(statue, episodes=6, seed_start=1000)
        receiving = [t for t in traces if not t.serve_side_is_policy]
        assert receiving
        for trace in receiving:
            assert trace.policy_hits == 0
            assert trace.ender == "policy_never_reached", trace.ender
            assert trace.touched_after_bounce == [False]

    def test_fault_touches_are_not_hits(self):
        """Adversarial regression (review 2026-08-08): the frozen
        volley-capable oracle's touches are terminal VOLLEY_RETURN
        faults under ground rules -- they must not count as made hits
        or the exchange-survival table overstates rally exposure."""
        from courtside_dynamics.envs._paddle_court import (
            scripted_lead_charge_opponent,
        )

        traces = run_player(
            scripted_lead_charge_opponent, episodes=4, seed_start=1000
        )
        for trace in traces:
            if trace.termination == "volley_return":
                policy_fault = trace.ender == "policy_volley_fault"
                if policy_fault:
                    assert trace.policy_hits == 0
                assert all(
                    s.hitter.name != "A" or s.outcome != "open"
                    for s in trace.shots
                    if s.exchange_index > trace.policy_hits
                )

    def test_report_renders(self):
        traces = run_player(
            scripted_ground_opponent, episodes=2, seed_start=1000
        )
        text = report(traces, "smoke")
        assert "shot ledger" in text
        assert "exchange survival" in text
        assert "ready position" in text
        assert "recovery hold" in text
