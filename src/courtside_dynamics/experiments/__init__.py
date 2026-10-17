"""Experiment bookkeeping: run records, sync manifests, run-dir trust.

``docs/design_campaign_structure_20261016.md`` step L1. Every
``train()`` run directory carries ``run_record.json`` (what ran, how it
ended) and, once the run has ended, ``sync_manifest.json`` (the bytes
and sha256 of every other file in the directory, written last). A run
directory without a manifest, or whose manifest does not verify, is
incomplete by definition: :func:`records.verify_run_dir` and its CLI
(``python -m courtside_dynamics.experiments.verify <run_dir>``) say
which.

Nothing in this package imports Stable-Baselines3 or torch, so a
record, a manifest or a Drive mirror can be read and checked anywhere
the bare package installs; ``train()`` produces them.
"""
