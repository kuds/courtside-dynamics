"""Experiment bookkeeping and offline evaluation of saved policies.

``docs/design_campaign_structure_20261016.md`` step L1. Every
``train()`` run directory carries ``run_record.json`` (what ran, how it
ended) and, once the run has ended, ``sync_manifest.json`` (the bytes
and sha256 of every other file in the directory, written last). A run
directory without a manifest, or whose manifest does not verify, is
incomplete by definition: :func:`records.verify_run_dir` and its CLI
(``python -m courtside_dynamics.experiments.verify <run_dir>``) say
which. The run index (``python -m courtside_dynamics.experiments.index
--root <runs root>``) regenerates ``<root>/index/runs.jsonl`` from the
run directories, backfilling the ones that predate run records.

The offline evaluation layer (design M3 and L2) re-scores what a run
saved. ``train()`` keeps an evaluation snapshot (policy plus
normalizer) at every evaluation when ``TrainConfig.snapshot_evals`` is
on (:mod:`snapshots`); :mod:`protocols` declares what a score is a
score of (``paddle-npoint@v3``, the PaddleTennis selection protocol
since design L3, the earlier ``paddle-npoint@v2``, and one in-loop
protocol per other recipe) and what a run records of it;
:mod:`harness` scores any model under any protocol with the in-loop
evaluator's own code (``python -m
courtside_dynamics.experiments.rescore``; ``notebooks/rescore_runs.ipynb``
drives it over a Drive runs folder); :mod:`select` picks a best
from the rows, purely; and :mod:`probe` scores the scripted witnesses
(``python -m courtside_dynamics.experiments.probe``).

Importing this package needs neither Stable-Baselines3 nor torch, so a
record, a manifest, a Drive mirror or a score file can be read and
checked anywhere the bare package installs; ``train()`` produces the
run files, and the harness loads the training stack only when it
scores a model.
"""
