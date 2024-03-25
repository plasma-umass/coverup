# file src/coverup/coverup.py:112-115
# lines [115]
# branches []

import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Assuming the 'args' object is part of a larger context not shown here,
# we will need to mock it to ensure the test can run independently.
# We will also assume that 'args.tests_dir' is supposed to be a Path object.

# Mock the 'args' object with a 'tests_dir' attribute
@pytest.fixture
def mock_args():
    mock_args = MagicMock()
    mock_args.tests_dir = Path("/tmp")
    return mock_args

# Test function to cover line 115
def test_test_file_path_executes_line_115(mock_args):
    from src.coverup.coverup import test_file_path

    # Assuming PREFIX is defined elsewhere in the module, we need to mock it as well
    PREFIX = "example"
    test_seq = 1  # Example test sequence number

    # Mock the 'args' object in the module
    module = pytest.importorskip("src.coverup.coverup")
    module.args = mock_args
    module.PREFIX = PREFIX

    expected_path = mock_args.tests_dir / f"test_{PREFIX}_{test_seq}.py"
    actual_path = test_file_path(test_seq)

    assert actual_path == expected_path, "The function did not return the expected file path."

    # Cleanup by deleting the mock attributes
    del module.args
    del module.PREFIX
