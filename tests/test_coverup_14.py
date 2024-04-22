# file src/coverup/pytest_plugin.py:67-70
# lines [67, 68, 69, 70]
# branches ['68->exit', '68->69']

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Assuming the CoverUpPlugin class is part of a module named coverup.pytest_plugin
from coverup.pytest_plugin import CoverUpPlugin

# Test function to cover the write_outcomes method
def test_write_outcomes(tmp_path, monkeypatch):
    # Setup a temporary file for outcomes
    outcomes_file = tmp_path / "outcomes.json"
    
    # Mock the config for CoverUpPlugin
    mock_config = MagicMock()
    
    # Create a CoverUpPlugin instance with a mocked _outcomes_file attribute
    plugin = CoverUpPlugin(mock_config)
    monkeypatch.setattr(plugin, '_outcomes_file', outcomes_file)
    monkeypatch.setattr(plugin, '_outcomes', {1: 'passed', 2: 'failed'})
    
    # Call the method under test
    plugin.write_outcomes()
    
    # Verify the file was created and contains the correct content
    assert outcomes_file.exists()
    with outcomes_file.open() as f:
        data = json.load(f)
        assert data == {'1': 'passed', '2': 'failed'}
    
    # Cleanup is handled by pytest's tmp_path fixture, which creates a new temporary directory for each test function

# Test function to cover the branch when _outcomes_file is None
def test_write_outcomes_no_file(monkeypatch):
    # Mock the config for CoverUpPlugin
    mock_config = MagicMock()
    
    # Create a CoverUpPlugin instance with a mocked _outcomes_file attribute set to None
    plugin = CoverUpPlugin(mock_config)
    monkeypatch.setattr(plugin, '_outcomes_file', None)
    
    # Mock the open method to ensure it is not called
    open_mock = MagicMock()
    monkeypatch.setattr(Path, 'open', open_mock)
    
    # Call the method under test
    plugin.write_outcomes()
    
    # Verify that the open method was not called
    open_mock.assert_not_called()
