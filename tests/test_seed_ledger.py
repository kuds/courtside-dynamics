"""The probe tools' shared seed ledger and verdict exit status.

Both close the same 2026-08-28 review finding group (section 3,
"Ladder envs + tools"): five probes carried drifted private copies of
the reserved-block table (one accepted a burned block), and three
probes printed a FAIL verdict yet exited 0, so automation gating on
the exit status read a failed battery as a pass. No test here draws a
seed from any block: the refusal checks run before any env exists, and
the exit-status checks replace the batteries with canned results.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tools import _seed_ledger
from tools._seed_ledger import RESERVED_BLOCKS, refuse_reserved

REPOSITORY_ROOT = Path(__file__).parents[1]
TOOLS_DIR = REPOSITORY_ROOT / "tools"

#: Every tool that guards its --seed-start, with the ledger blocks it is
#: sanctioned to draw from (its own burned calibration block, or the
#: shared diagnosis block for the diagnosis-side probe).
_GUARDED_TOOLS = {
    "paddle_tennis_hold_probe": ((6200, 6299),),
    "paddle_tennis_npoint_probe": ((5400, 5499),),
    "paddle_tennis_postswing_target_probe": ((5200, 5299),),
    "paddle_tennis_reach_probe": ((5500, 5599),),
    "paddle_tennis_k2_harvest": (),
}


# --- the ledger itself ------------------------------------------------


def test_ledger_blocks_are_sorted_disjoint_and_cited():
    previous_high = -1
    for block in RESERVED_BLOCKS:
        assert block.low <= block.high
        assert block.low > previous_high, f"{block} overlaps or is unsorted"
        assert block.note
        previous_high = block.high
    source = Path(_seed_ledger.__file__).read_text()
    # Every DECISIONS-booked block is cited by its DECISIONS line.
    for citation in (
        "DECISIONS.md:554",
        "DECISIONS.md:420",
        "DECISIONS.md:285",
        "DECISIONS.md:125",
        "DECISIONS.md:126",
    ):
        assert citation in source


def test_ledger_is_the_union_of_the_former_tool_tables():
    """Every block any tool's private table refused is still refused --
    the shared ledger only ever widened a tool's protection."""
    former_tables = {
        "hold": [
            (4100, 4199), (4300, 4399), (5200, 5299), (5300, 5399),
            (5400, 5499), (5500, 5599), (5600, 6199),
        ],
        "npoint": [(4100, 4199), (4300, 4399)],
        "postswing": [
            (4100, 4199), (4300, 4399), (5300, 5399), (5400, 5499),
            (5500, 5599), (5600, 6199), (6200, 6299), (6300, 6399),
        ],
        "reach": [
            (4100, 4199), (4300, 4399), (5200, 5299), (5300, 5399),
            (5400, 5499),
        ],
        "k2_harvest": [
            (3000, 3099), (3100, 3199), (4000, 4099), (4100, 4199),
            (4200, 4299), (4300, 4399), (5000, 5099), (5100, 5199),
            (5200, 5299), (5300, 5399), (5400, 5499), (5500, 5599),
            (5600, 6199), (6200, 6299), (6300, 6399), (9000, 9029),
            (9100, 9146), (9147, 9147),
        ],
    }
    ledger = {(block.low, block.high) for block in RESERVED_BLOCKS}
    for tool, table in former_tables.items():
        assert set(table) <= ledger, tool
    # Plus the DECISIONS bookings: the re-booked LD1' battery and the
    # consumed part of the D-G scratch extension (its 9270-9299
    # remainder is still free scratch).
    assert {(6400, 6499), (9200, 9269)} <= ledger
    assert all(not (b.low <= 9270 <= b.high) for b in RESERVED_BLOCKS)


@pytest.mark.parametrize(
    ("seed_start", "episodes", "block"),
    [
        (4100, 1, "4100-4199"),  # first seed of the sealed gate
        (4099, 2, "4000-4099"),  # straddles 4099/4100: the lower hit wins
        (4199, 1, "4100-4199"),  # last seed of a block
        (4050, 200, "4000-4099"),  # a span covering whole blocks
        (6300, 1, "6300-6399"),  # the burned block the hold probe missed
        (9147, 1, "9147-9147"),  # a single-seed block
    ],
)
def test_refuse_reserved_exits_on_any_overlap(seed_start, episodes, block):
    with pytest.raises(SystemExit, match=block):
        refuse_reserved(seed_start, episodes)


@pytest.mark.parametrize(
    ("seed_start", "episodes"),
    [
        (2900, 100),  # ends at 2999, one short of 3000-3099
        (3200, 800),  # the 3200-3999 gap exactly
        (6500, 2500),  # 6500-8999, between the battery and the scratch
        (9030, 70),  # the registered k=2 library's reproduction range
        (9270, 30),  # the unconsumed D-G scratch remainder
        (4100, 0),  # an empty span draws no seed
    ],
)
def test_refuse_reserved_accepts_free_ranges(seed_start, episodes):
    refuse_reserved(seed_start, episodes)


def test_refuse_reserved_allow_sanctions_only_the_named_block():
    refuse_reserved(5400, 100, allow=((5400, 5499),))
    # The allowance is per block: spilling past it is still refused.
    with pytest.raises(SystemExit, match="5500-5599"):
        refuse_reserved(5450, 100, allow=((5400, 5499),))
    with pytest.raises(SystemExit, match="4100-4199"):
        refuse_reserved(4100, 1, allow=((5400, 5499),))


def test_refuse_reserved_rejects_an_allowance_outside_the_ledger():
    # A typo'd allowance must not read like an exemption that does nothing.
    with pytest.raises(ValueError, match="not in the seed ledger"):
        refuse_reserved(5400, 1, allow=((5400, 5498),))


# --- the tools use it ---------------------------------------------------


def _tool_sources():
    for path in sorted(TOOLS_DIR.glob("*.py")):
        if path.name in ("__init__.py", "_seed_ledger.py"):
            continue
        yield path, ast.parse(path.read_text(), filename=str(path))


def test_no_tool_defines_its_own_reserved_table_or_refusal():
    """Static: no tool carries a private block table or refusal helper,
    so the drift channel the review found cannot reopen."""
    offenders = []
    for path, tree in _tool_sources():
        for node in tree.body:
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.lstrip(
                    "_"
                ) in ("RESERVED_BLOCKS", "REFUSED_BLOCKS"):
                    offenders.append(f"{path.name}: {target.id}")
            if isinstance(node, ast.FunctionDef) and "refuse" in node.name:
                offenders.append(f"{path.name}: def {node.name}")
    assert offenders == []


@pytest.mark.parametrize(("module_name", "allowed"), sorted(_GUARDED_TOOLS.items()))
def test_guarded_tools_refuse_through_the_shared_ledger(module_name, allowed):
    """Import-level: each guarded tool's refusal IS the ledger's, and
    every allowance it passes names a ledger block (the helper would
    raise on an unknown one at the tool's first run)."""
    import importlib

    module = importlib.import_module(f"tools.{module_name}")
    assert module.refuse_reserved is refuse_reserved
    for name in ("RESERVED_BLOCKS", "_RESERVED_BLOCKS", "_REFUSED_BLOCKS"):
        assert getattr(module, name, None) is None, name
    source = (TOOLS_DIR / f"{module_name}.py").read_text()
    assert "refuse_reserved(" in source
    ledger = {(block.low, block.high) for block in RESERVED_BLOCKS}
    assert set(allowed) <= ledger


def test_hold_probe_script_refuses_the_block_it_used_to_accept():
    """Script mode (``python tools/<name>.py``, tools/ on sys.path
    instead of the repo root) resolves the ledger too, and the hold
    probe now refuses its own re-battery block 6300-6399."""
    env = dict(os.environ, MUJOCO_GL="disable")
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS_DIR / "paddle_tennis_hold_probe.py"),
            "--seed-start",
            "6300",
            "--episodes",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        env=env,
        timeout=240,
    )
    assert result.returncode == 1
    assert "reserved/burned block 6300-6399" in result.stderr


# --- verdict exit status --------------------------------------------------


def test_npoint_probe_exits_nonzero_on_an_np1_fail(monkeypatch):
    from tools import paddle_tennis_npoint_probe as probe

    monkeypatch.setattr(probe, "run_pass", lambda *args, **kwargs: [])
    verdicts = {"value": False}
    monkeypatch.setattr(
        probe,
        "evaluate_np1",
        lambda passes: [("carryover", verdicts["value"], "canned")],
    )
    argv = ["--skip-np2", "--np1-episodes", "1", "--parker-episodes", "1"]
    assert probe.main(argv) == 1
    verdicts["value"] = True
    assert probe.main(argv) == 0


def test_shaping_probe_exits_nonzero_on_an_s1_fail(monkeypatch, capsys):
    from tools import paddle_tennis_shaping_probe as probe

    row = probe.EpisodeRow(
        seed=0,
        steps=1,
        total_reward=0.0,
        shaping_paid=0.0,
        clawback=0.0,
        hits_a=0,
        confirms_a=0,
    )
    monkeypatch.setattr(probe, "run_witness", lambda *args, **kwargs: [row])
    verdicts = {"value": False}
    monkeypatch.setattr(
        probe,
        "evaluate_criteria",
        lambda results: [("identity", verdicts["value"], "canned")],
    )
    assert probe.main(["--episodes", "1"]) == 1
    assert "S1 verdict: FAIL" in capsys.readouterr().out
    verdicts["value"] = True
    assert probe.main(["--episodes", "1"]) == 0
    assert "S1 verdict: PASS" in capsys.readouterr().out


def _volley_cell(player, volley_rule, *, crossings, volley_faults):
    from tools.paddle_tennis_volley_probe import CellResult

    return CellResult(
        player=player,
        volley_rule=volley_rule,
        episodes=100,
        mean_crossings=crossings,
        std_crossings=1.0,
        ge1_rate=0.9,
        mean_returns=crossings,
        cadence_steps_per_crossing=100.0,
        volley_fault_fraction=volley_faults,
        truncated=0,
        terminations=Counter({"volley_return": 1}),
    )


@pytest.mark.parametrize(("ground_crossings", "expected"), [(3.0, 0), (1.0, 1)])
def test_volley_probe_exits_nonzero_on_do_not_adopt(
    monkeypatch, capsys, ground_crossings, expected
):
    """The pre-registered criteria themselves decide: a ground row
    below the 2.6 feasibility floor is DO NOT ADOPT, and exits 1."""
    from tools import paddle_tennis_volley_probe as probe

    def fake_cell(player, volley_rule, *, episodes, seed_start):
        if player == "ground":
            return _volley_cell(
                player, volley_rule, crossings=ground_crossings, volley_faults=0.0
            )
        return _volley_cell(player, volley_rule, crossings=0.0, volley_faults=1.0)

    monkeypatch.setattr(probe, "run_cell", fake_cell)
    assert probe.main(["--quick"]) == expected
    out = capsys.readouterr().out
    assert ("DO NOT ADOPT" in out) == bool(expected)
