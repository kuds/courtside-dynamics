"""The probe tools' seed ledger: every currently reserved or burned block.

Every scripted probe and harvest instrument draws its episodes from a
fixed seed block, and a block that a verdict has already consumed (or
that a pre-registration holds sealed) must never be drawn again by a
mistyped ``--seed-start``. Each tool used to carry its own copy of the
block table; the copies drifted to 2/5/7/8/18 entries and one burned
block (6300-6399) slipped through the hold probe's guard
(``docs/rl_pipeline_review_20260828.md`` section 3, "Seed-ledger drift
across probes"). This module is now the single source of truth: the
union of those five tables plus the blocks booked in
``docs/DECISIONS.md``, keeping only blocks that are reserved or
consumed today (a booked scratch range's unconsumed remainder stays
free scratch).

A tool refuses through :func:`refuse_reserved`. Drawing from a ledger
block is sanctioned only where a tool names it in ``allow``: a probe
re-running on its own calibration block (to reproduce the numbers it
burned the block for), a diagnosis tool on the shared calibration
block, or a certification's single sanctioned opening of its reserved
block. A new booking is one entry here plus its citation.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple


class SeedBlock(NamedTuple):
    """One inclusive ``[low, high]`` seed block and why it is held."""

    low: int
    high: int
    note: str


#: Every seed block that is reserved (sealed for a future verdict) or
#: burned (consumed by an adjudicated measurement) today, ascending.
#: Each entry cites its booking: the ``docs/DECISIONS.md`` line where
#: the ledger decision lives there, else the doc section that burned
#: it, plus the tool tables it was merged from.
RESERVED_BLOCKS: tuple[SeedBlock, ...] = (
    # docs/DECISIONS.md:554 -- the scripted WallBall probes' held-out
    # certification ("once on held-out seeds 3000-3099"); k2_harvest.
    SeedBlock(3000, 3099, "WallBall scripted-probe held-out certification"),
    # paddle_tennis_env_20260802.md section 5 -- the volley-era env
    # certification's single use of its reserved block; k2_harvest.
    SeedBlock(3100, 3199, "volley-era PaddleTennis held-out certification"),
    # docs/DECISIONS.md:420 -- the true-baseline certification ("on
    # fresh seeds 4000-4099"); k2_harvest.
    SeedBlock(4000, 4099, "WallBall true-baseline certification"),
    # docs/DECISIONS.md:285 -- "Held-out block 4100-4199 not opened;
    # stays sealed" for the first registered-result branch; all five
    # tool tables.
    SeedBlock(4100, 4199, "SEALED registered-result held-out gate"),
    # paddle_tennis_ground_rules_20260803.md section 7 -- the
    # ground-era certification (paddle_tennis_probes.py --certify);
    # k2_harvest.
    SeedBlock(4200, 4299, "ground-rules-era held-out certification"),
    # design_paddle_tennis_npoint.md section 4a -- NP3 certification,
    # consumed (paddle_tennis_npoint_probe.py --certify); all five
    # tool tables.
    SeedBlock(4300, 4399, "n-point NP3 held-out certification"),
    # paddle_tennis_p5_transfer_20260802.md section 4 -- P5 transfer
    # calibration, re-burned by the ground-rules re-run
    # (paddle_tennis_ground_rules_20260803.md section 7); k2_harvest.
    SeedBlock(5000, 5099, "P5 transfer calibration"),
    # paddle_tennis_ground_rules_20260803.md section 7 -- the
    # ground-rules probe matrix; k2_harvest.
    SeedBlock(5100, 5199, "ground-rules probe calibration"),
    # paddle_tennis_diagnosis_20260808.md section 5 -- the diagnosis
    # instrument's calibration block. The in-run checkpoint diagnosis
    # and the campaign gate read it, so only diagnosis-side tools may
    # draw from it; hold, reach, k2_harvest.
    SeedBlock(5200, 5299, "diagnosis calibration (diagnosis tools only)"),
    # design_paddle_tennis_contact_shaping.md section 4 -- S1; hold,
    # reach, postswing, k2_harvest.
    SeedBlock(5300, 5399, "S1 contact-shaping probe calibration"),
    # design_paddle_tennis_npoint.md section 3 -- NP1/NP2 calibration;
    # hold, reach, postswing, k2_harvest.
    SeedBlock(5400, 5499, "n-point NP1/NP2 calibration"),
    # design_paddle_tennis_reach_shaping.md section 3a -- RS1; hold,
    # postswing, k2_harvest.
    SeedBlock(5500, 5599, "RS1 reach-shaping probe calibration"),
    # design_paddle_tennis_reach_shaping.md section 3 -- the review's
    # ad-hoc workpaper blocks, recorded burned; hold, postswing,
    # k2_harvest.
    SeedBlock(5600, 6199, "reach-shaping review workpaper blocks"),
    # design_paddle_tennis_postswing_hold.md section 5 -- the PH1
    # battery at (0.25, 4.0); postswing, k2_harvest.
    SeedBlock(6200, 6299, "PH1 hold-shaping battery"),
    # design_paddle_tennis_postswing_hold.md section 5 -- the section 4b
    # re-battery at (0.5, 12.0), the block the hold probe's own table
    # omitted; postswing, k2_harvest.
    SeedBlock(6300, 6399, "PH1 hold-shaping re-battery"),
    # docs/DECISIONS.md:125 -- (D-G) the LD1' battery block, booked
    # 2026-09-02 and unconsumed. DECISIONS.md:188 had returned it to the
    # pool on 2026-08-30 (the closed command-rate design's CR2 block,
    # DECISIONS.md:209); the later D-G booking re-reserved it.
    SeedBlock(6400, 6499, "LD1' battery (booked, unconsumed)"),
    # design_paddle_tennis_k2_drill.md section 7 -- consumed entries of
    # the 9000-9199 scratch block; k2_harvest.
    SeedBlock(9000, 9029, "k=2 drill feasibility probe (consumed scratch)"),
    SeedBlock(9100, 9146, "k=2 drill review probes (consumed scratch)"),
    SeedBlock(9147, 9147, "k=2 step-0 replay reset seed (consumed scratch)"),
    # docs/DECISIONS.md:126 -- (D-G) books the 9200-9299 scratch
    # extension; design_paddle_tennis_demo_injection.md section 7
    # records 9200-9269 consumed by the demo harvest. The remainder
    # 9270-9299 is unconsumed scratch, deliberately not held here.
    SeedBlock(9200, 9269, "LD1' demo harvest (consumed scratch)"),
)


def refuse_reserved(
    seed_start: int,
    episodes: int,
    *,
    allow: Iterable[tuple[int, int]] = (),
) -> None:
    """Exit loudly when ``[seed_start, seed_start + episodes)`` hits the ledger.

    ``allow`` names the ledger blocks this call is sanctioned to draw
    from, each as its exact ``(low, high)`` -- a tool's own calibration
    block, or a certification's single sanctioned opening. An allowance
    that is not a ledger block raises ``ValueError``: a typo there would
    otherwise sanction nothing while reading like an exemption. An
    empty span (``episodes < 1``) draws no seed and is never refused.
    """
    ledger = {(block.low, block.high) for block in RESERVED_BLOCKS}
    allowed = {(int(low), int(high)) for low, high in allow}
    unknown = sorted(allowed - ledger)
    if unknown:
        raise ValueError(
            f"allow names blocks that are not in the seed ledger: {unknown}"
        )
    if episodes < 1:
        return
    last = seed_start + episodes - 1
    for block in RESERVED_BLOCKS:
        if (block.low, block.high) in allowed:
            continue
        if seed_start <= block.high and last >= block.low:
            raise SystemExit(
                f"seed range [{seed_start}, {seed_start + episodes}) "
                f"intersects reserved/burned block {block.low}-{block.high} "
                f"({block.note}); refuse to run"
            )
