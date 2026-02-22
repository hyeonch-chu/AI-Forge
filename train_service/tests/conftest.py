"""Pytest configuration for train_service tests.

Injects a mock ``ultralytics`` module into sys.modules so that tests can run
without the real ultralytics package installed.  The actual YOLO class is
mocked via fixtures in test_train.py.
"""
import sys
from unittest.mock import MagicMock

# Stub out ultralytics before train.py (or any test module) imports it.
_mock_ultralytics = MagicMock()
sys.modules.setdefault("ultralytics", _mock_ultralytics)
