"""Repository tools: probes, harvests, and release checks.

Each module runs as a script (``python tools/<name>.py``); the package
marker lets the test suite import them as ``tools.<name>`` and gives
mypy one module name per file, so the shared :mod:`tools._seed_ledger`
resolves the same way in both modes.
"""
