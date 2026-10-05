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
        traces, _travels = run_player(
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
        traces, _travels = run_player(
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
        traces, _travels = run_player(statue, episodes=6, seed_start=1000)
        receiving = [t for t in traces if not t.serve_side_is_policy]
        assert receiving
        for trace in receiving:
            assert trace.policy_hits == 0
            assert trace.ender == "policy_never_reached", trace.ender
            assert trace.touched_after_bounce == [False]
            # The rules now label the untouched skip-out a second
            # bounce themselves (it was out_of_bounds, which the
            # attribution had to repair from the shot ledger).
            assert trace.termination == "second_bounce", trace.termination

    def test_forced_nonfinite_point_reads_nonfinite(self):
        """The env's forced-nonfinite backstop (a nonfinite observation
        after a finite physics step) leaves the rules snapshot at
        "none"; the trace must still name the unsafe ending -- as its
        termination AND its ender (the ender used to read "cap")."""
        from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
        from courtside_dynamics.training.paddle_diagnosis import run_episode

        env = PaddleTennisEnv()
        try:
            real_get_obs = env._get_obs
            calls = []

            def poisoned():
                calls.append(1)
                obs = real_get_obs().copy()
                if len(calls) > 20:  # finite reset, then blow up mid-point
                    obs[0] = np.nan
                return obs

            env._get_obs = poisoned
            traces, _travels = run_episode(env, scripted_ground_opponent, 1000)
        finally:
            env.close()
        assert len(traces) == 1
        assert traces[0].termination == "nonfinite_state"
        assert traces[0].ender == "nonfinite_state"

    def test_fault_touches_are_not_hits(self):
        """Adversarial regression (review 2026-08-08): the frozen
        volley-capable oracle's touches are terminal VOLLEY_RETURN
        faults under ground rules -- they must not count as made hits
        or the exchange-survival table overstates rally exposure."""
        from courtside_dynamics.envs._paddle_court import (
            scripted_lead_charge_opponent,
        )

        traces, _travels = run_player(
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
        traces, _travels = run_player(
            scripted_ground_opponent, episodes=2, seed_start=1000
        )
        text = report(traces, "smoke")
        assert "shot ledger" in text
        assert "exchange survival" in text
        assert "ready position" in text
        assert "recovery hold" in text


class TestDiagnosisProbeCallback:
    """Checkpoint-cadence automation: reports written, failures isolated."""

    @staticmethod
    def _stub_model(fail: bool = False):
        class _Stub:
            def get_vec_normalize_env(self):
                return None

            def predict(self, observation, deterministic=True):
                if fail:
                    raise RuntimeError("boom")
                return np.zeros(3), None

        return _Stub()

    def test_writes_reports_at_cadence_and_caches_oracle(self, tmp_path):
        from courtside_dynamics.training.paddle_diagnosis import (
            DiagnosisProbeCallback,
        )

        callback = DiagnosisProbeCallback(
            save_dir=str(tmp_path),
            save_freq=2,
            episodes=2,
            seed_start=1000,
        )
        callback.model = self._stub_model()
        callback.n_calls = 1
        callback.num_timesteps = 100
        assert callback._on_step() is True
        assert not list(tmp_path.iterdir())  # off-cadence: nothing

        callback.n_calls = 2
        callback.num_timesteps = 200
        assert callback._on_step() is True
        names = {p.name for p in tmp_path.iterdir()}
        assert names == {
            "diagnosis_probe_oracle.txt",
            "diagnosis_probe_200.txt",
        }
        oracle_text = (tmp_path / "diagnosis_probe_oracle.txt").read_text()
        assert "ground oracle" in oracle_text

        # The oracle row is measured once; later triggers add only the
        # new checkpoint file.
        callback.n_calls = 4
        callback.num_timesteps = 400
        assert callback._on_step() is True
        names = {p.name for p in tmp_path.iterdir()}
        assert "diagnosis_probe_400.txt" in names
        assert len(names) == 3

    def test_probes_through_the_provided_env_factory(self, tmp_path):
        """Both rows (oracle + checkpoint) run on env_fn's env, so a
        run-config [env] override reaches the diagnosis measurements
        instead of a silently different stock env."""
        from courtside_dynamics.envs.paddle_tennis import PaddleTennisEnv
        from courtside_dynamics.training.paddle_diagnosis import (
            DiagnosisProbeCallback,
        )

        constructed = []

        def factory():
            constructed.append(1)
            return PaddleTennisEnv()

        callback = DiagnosisProbeCallback(
            save_dir=str(tmp_path),
            save_freq=1,
            episodes=1,
            seed_start=1000,
            env_fn=factory,
        )
        callback.model = self._stub_model()
        callback.n_calls = 1
        callback.num_timesteps = 100
        assert callback._on_step() is True
        assert not callback._disabled
        assert len(constructed) == 2  # oracle row + checkpoint row

    def test_failure_skips_only_its_checkpoint(self, tmp_path, capsys):
        """Review 2026-08-28 §3: one failure used to disable the probe
        for the rest of the run. A failure now costs only its own
        report (logged to the console and a durable failures file),
        and a later success resets the consecutive count."""
        from courtside_dynamics.training.paddle_diagnosis import (
            DiagnosisProbeCallback,
        )

        callback = DiagnosisProbeCallback(
            save_dir=str(tmp_path),
            save_freq=1,
            episodes=1,
            seed_start=1000,
        )
        outcomes = (True, True, False, True, True)
        for step, fail in enumerate(outcomes, start=1):
            callback.model = self._stub_model(fail=fail)
            callback.n_calls = step
            callback.num_timesteps = 100 * step
            assert callback._on_step() is True  # isolated, not raised
            assert not callback._disabled
        names = {p.name for p in tmp_path.iterdir()}
        # Only the successful checkpoint wrote a report.
        assert {n for n in names if n.startswith("diagnosis_probe_")} >= {
            "diagnosis_probe_oracle.txt",
            "diagnosis_probe_300.txt",
        }
        assert not names & {
            "diagnosis_probe_100.txt",
            "diagnosis_probe_200.txt",
            "diagnosis_probe_400.txt",
            "diagnosis_probe_500.txt",
        }
        failures = (tmp_path / "diagnosis_probe_failures.txt").read_text()
        assert failures.count("probe failed") == 4
        assert "at 400 steps (1/3 consecutive)" in failures
        assert "skipping this checkpoint" in capsys.readouterr().out

    def test_three_consecutive_failures_disable(self, tmp_path):
        from courtside_dynamics.training.paddle_diagnosis import (
            DiagnosisProbeCallback,
        )

        callback = DiagnosisProbeCallback(
            save_dir=str(tmp_path),
            save_freq=1,
            episodes=1,
            seed_start=1000,
        )
        callback.model = self._stub_model(fail=True)
        for step in range(1, callback.MAX_CONSECUTIVE_FAILURES + 1):
            assert not callback._disabled
            callback.n_calls = step
            callback.num_timesteps = 100 * step
            assert callback._on_step() is True  # isolated, not raised
        assert callback._disabled
        assert "disabled for the rest of the run" in (
            tmp_path / "diagnosis_probe_failures.txt"
        ).read_text()
        assert not any(
            p.name.startswith("diagnosis_probe_1")
            for p in tmp_path.iterdir()
        )
        # Disabled means quiet: a working model is not probed again.
        callback.model = self._stub_model()
        callback.n_calls = 10
        callback.num_timesteps = 1000
        assert callback._on_step() is True
        assert not (tmp_path / "diagnosis_probe_1000.txt").exists()
