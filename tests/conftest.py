"""Shared test fixtures and module-level mocks."""

import sys
from unittest.mock import MagicMock

# Prevent kaggle from calling authenticate() at import time in all tests
_kaggle_mock = MagicMock()
for mod in ("kaggle", "kaggle.api", "kaggle.api.kaggle_api_extended"):
    sys.modules.setdefault(mod, _kaggle_mock)
