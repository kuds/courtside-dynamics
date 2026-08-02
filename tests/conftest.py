"""Suite-wide pytest configuration.

No fixtures yet -- shared plain-function helpers (ball-state
teleports, contact inspection, event-batch injection, the
``get_env``-shaped fake model) live in :mod:`tests._helpers`, imported
explicitly by the files that use them so each test's dependencies stay
visible at its imports. Add fixtures here only when they need pytest's
lifecycle (caching, teardown, parametrization); a helper a test calls
with arguments belongs in ``_helpers``.
"""
