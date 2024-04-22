# file src/coverup/coverup.py:254-256
# lines [256]
# branches []

import pytest
import typing as T
from unittest.mock import patch, MagicMock
from coverup.coverup import get_required_modules

@pytest.fixture
def mock_module_available():
    return {
        'module1': 1,
        'module2': 0,
        'module3': 1,
        'module4': 0
    }

def test_get_required_modules(mock_module_available):
    with patch('coverup.coverup.module_available', new=mock_module_available):
        result = get_required_modules()
        expected = ['module2', 'module4']
        assert sorted(result) == sorted(expected)
