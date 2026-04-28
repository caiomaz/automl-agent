"""Shared test fixtures and module-level mocks."""

import sys
from unittest.mock import MagicMock

import pytest

# Prevent kaggle from calling authenticate() at import time in all tests
_kaggle_mock = MagicMock()
for mod in ("kaggle", "kaggle.api", "kaggle.api.kaggle_api_extended"):
    sys.modules.setdefault(mod, _kaggle_mock)


@pytest.fixture(autouse=True)
def _reset_active_run():
    """Ensure the process-local active-run registry is clean for every test.

    Without this, a test that calls ``prepare_new_run`` without a paired
    ``finalize_run`` would leak state and cause unrelated tests to raise
    ``ActiveRunError`` purely because of execution order.
    """
    try:
        from utils.run_context import clear_active_run
        clear_active_run()
    except Exception:
        pass
    yield
    try:
        from utils.run_context import clear_active_run
        clear_active_run()
    except Exception:
        pass
