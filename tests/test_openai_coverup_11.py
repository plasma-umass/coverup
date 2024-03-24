# file src/coverup/delta.py:5-32
# lines [16, 18]
# branches ['12->18', '13->16']

import pytest
from pathlib import Path
from src.coverup.delta import _compact

def test_compact_with_non_path_objects(monkeypatch):
    # Mocking Path to avoid filesystem dependency using monkeypatch
    monkeypatch.setattr('src.coverup.delta.Path', Path)

    # Create a test set with non-Path objects
    test_set = {42, 'test_name', 3.14}

    # Call the _compact function with the test set
    result = _compact(test_set)

    # Verify that the non-Path objects are converted to strings and included in the result
    assert '42' in result
    assert 'test_name' in result
    assert '3.14' in result

    # Verify that the result does not contain any range representation
    assert '-' not in result
